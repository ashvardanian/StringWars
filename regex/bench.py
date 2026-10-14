"""Multi-pattern regular-expression benchmarks in Python, over every document. Mirrors `regex/bench.rs`.

- stringzilla.RegexEngine: counts, find, extract and replace over one compiled vocabulary
- re and the `regex` package, over one alternation of the vocabulary
- google-re2 and hyperscan, which refuse lookarounds and note those rows as unavailable
- cudf.Series.str.count, str.contains and str.extract on each selected GPU, where cuDF is installed

Three vocabularies, spelled identically in `regex/bench.rs`: literal keywords, everyday patterns (emails,
URLs, ISO dates, numbers, word-bounded keywords), and lookarounds. `re` and `regex` run with their ASCII
flags, so `\\b` and the classes mean what they mean in RE2 and StringZilla. Hyperscan reports every match
end, not a leftmost-first cover, and `str.contains` answers whether a document matches, not how often.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group regex regex/bench.py
"""

import argparse
import functools
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from typing import Any

import hyperscan
import numpy as np
import re2
import regex
import stringzilla as sz

from stringwars import (
    Dataset,
    MeasureSpec,
    Settings,
    Threads,
    finish,
    gpu_managed_array,
    gpu_ordinals,
    gpu_require,
    log_dataset,
    log_timing_overhead,
    measure,
    note_unavailable,
    pass_over,
    print_machine,
    print_settings,
    read_settings,
    resolve_dataset,
    thread_pool,
)

# RAPIDS cuDF is outside the suite's dependency group, so its GPU rows are optional.
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

# Fixed vocabularies, none holding a capturing group, so wrapping each pattern in one numbers the
# alternatives of a rival's rewrite. Identical to `regex/bench.rs`.
VOCABULARIES = {
    "literal": ["government", "minister", "president", "police"],
    "plain": [
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"https?://[A-Za-z0-9.-]+(?:/[A-Za-z0-9._~%/-]*)?",
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
        r"[0-9]+(?:[.,][0-9]+)*",
        r"\b(?:government|minister|president|police)\b",
    ],
    "lookaround": [
        r"(?<=\$)[0-9]+(?:\.[0-9]{2})?",
        r"[0-9]+(?=%)",
        r"[A-Z][a-z]+(?= [A-Z][a-z]+)",
        r"(?<![A-Za-z])[A-Z]{2,}(?![a-z])",
    ],
}

# The one pattern `extract` reads the groups of: an ISO date's year, month and day.
DATE_GROUPS = r"([0-9]{4})-([0-9]{2})-([0-9]{2})"

# The concurrency of every single-threaded row.
ONE_THREAD = Threads(1)


@dataclass(frozen=True)
class Device:
    """Where one row runs, in the order the rows run. Mirrors `bench.rs`."""

    label: str
    concurrency: Threads = ONE_THREAD
    ordinal: int | None = None


def selected_devices(settings: Settings) -> list[Device]:
    """The single core, every core, then each GPU the run selects."""
    devices = [Device("1cpu"), Device(f"{settings.threads}cpu", settings.threads)]
    for ordinal in gpu_ordinals(settings):
        if not gpu_require(settings, ordinal):
            continue
        try:
            sz.cuda_capabilities_enabled(ordinal)
        except RuntimeError as error:
            if settings.devices is not None:
                raise SystemExit(f"Requested CUDA device cuda:{ordinal} is unavailable: {error}") from error
            continue
        devices.append(Device(f"cuda:{ordinal}", ordinal=ordinal))
    return devices


@dataclass(frozen=True)
class Corpus:
    """Every engine walks the same documents, so one pass over all of them is every row's work."""

    texts: list[str]
    tape: sz.Strings
    dataset: Dataset

    def spec(self, concurrency: Threads = ONE_THREAD) -> MeasureSpec:
        return MeasureSpec(
            unit="bytes",
            elements=self.dataset.token_count,
            total_bytes=self.dataset.token_bytes,
            concurrency=concurrency,
        )

    def share(self, thread: int, threads: int) -> slice:
        """The documents one of `threads` engines walks, contiguous so its outputs are too."""
        chunk = -(-len(self.texts) // threads)
        first = min(thread * chunk, len(self.texts))
        return slice(first, min(first + chunk, len(self.texts)))


class Round:
    """One round of a `RegexEngine` verb over fixed haystacks and the outputs it writes, the variable-length
    ones sized by an empty round whose last offset is what the filled one needs. `allocate` places every
    buffer where the engine's device reaches."""

    def __init__(
        self,
        verb: str,
        engine: Any,
        haystacks: sz.Strings,
        replacements: sz.Strings,
        allocate: Callable[[tuple[int, ...], Any], np.ndarray],
        capabilities: Any = None,
        stream: int | None = None,
    ) -> None:
        self.verb, self.engine, self.haystacks, self.replacements = verb, engine, haystacks, replacements
        self.capabilities, self.stream = capabilities, stream
        self.placement = {} if stream is None else {"stream": stream}
        count = len(haystacks)
        self.counts = allocate((count,), np.uintp)
        self.offsets = allocate((count + 1,), np.uintp)
        self.matches = self.target = None
        if verb == "find":
            engine.find(haystacks, None, self.offsets, **self.placement)
        elif verb == "extract":
            engine.extract(haystacks, 0, None, self.offsets, **self.placement)
        elif verb == "replace":
            engine.replace(haystacks, replacements, None, self.offsets)
        self.join()
        total = max(1, int(self.offsets[count]))
        if verb in ("find", "extract"):
            self.matches = allocate((total, 4), np.uintp)
        elif verb == "replace":
            self.target = allocate((total,), np.uint8)

    def join(self) -> None:
        if self.stream is not None:
            sz.synchronize(self.capabilities, self.stream)

    def __call__(self) -> None:
        if self.verb == "counts":
            self.engine.counts(self.haystacks, self.counts, **self.placement)
        elif self.verb == "find":
            self.engine.find(self.haystacks, self.matches, self.offsets, **self.placement)
        elif self.verb == "extract":
            self.engine.extract(self.haystacks, 0, self.matches, self.offsets, **self.placement)
        else:
            self.engine.replace(self.haystacks, self.replacements, self.target, self.offsets)
        self.join()

    def agrees(self, host: "Round") -> bool:
        """Whether a device round wrote what the host engine did."""
        if self.verb == "counts":
            return np.array_equal(self.counts, host.counts)
        total = int(host.offsets[-1])
        return np.array_equal(self.offsets, host.offsets) and np.array_equal(self.matches[:total], host.matches[:total])


def sweep(
    settings: Settings,
    corpus: Corpus,
    group: str,
    patterns: list[str],
    replacements: list[str],
    devices: list[Device],
    verb: str,
) -> None:
    """Times `verb` on every device: one host engine, one engine per pool thread over a contiguous share of
    the documents, and one engine per GPU whose first round is checked against the host. A rewrite runs on
    the CPU alone, so its GPU rows are noted instead."""
    vocabulary = sz.Strings(patterns)
    replacement_tape = sz.Strings(replacements)
    for device in devices:
        name = f"{group}/stringzilla.RegexEngine.{verb}<{device.label}>"
        if not settings.selects(name):
            continue
        if device.ordinal is not None:
            if verb == "replace":
                note_unavailable(name, "RegexEngine.replace runs on the CPU only")
                continue
            stream = sz.cuda_stream_init(device.ordinal)
            try:
                sweep_gpu(settings, corpus, name, vocabulary, device.ordinal, stream, verb)
            finally:
                sz.cuda_stream_free(stream)
            continue
        threads = device.concurrency
        try:
            engines = [sz.RegexEngine(vocabulary) for _ in range(threads)]
        except ValueError as error:  # a pattern this engine refuses skips its own row
            note_unavailable(name, str(error))
            continue
        if threads == 1:
            measure(settings, name, corpus.spec(), Round(verb, engines[0], corpus.tape, replacement_tape, np.zeros))
            continue
        rounds = [
            Round(verb, engine, sz.Strings(corpus.texts[corpus.share(thread, threads)]), replacement_tape, np.zeros)
            for thread, engine in enumerate(engines)
        ]
        configure = functools.partial(sz.cpu_configure_thread, sz.cpu_capabilities_enabled())
        with thread_pool(threads, configure) as run_all:
            measure(settings, name, corpus.spec(threads), functools.partial(run_all, rounds))


def sweep_gpu(
    settings: Settings, corpus: Corpus, name: str, vocabulary: sz.Strings, ordinal: int, stream: int, verb: str
) -> None:
    """One GPU row, every buffer in managed memory and the documents copied into the device's own tape."""
    capabilities = sz.cuda_capabilities_enabled(ordinal)
    try:
        engine = sz.RegexEngine(vocabulary, capabilities=capabilities, stream=stream)
    except Exception as error:  # a device tier that declines this vocabulary skips its own row
        note_unavailable(name, str(error))
        return
    haystacks = corpus.tape.copy(capabilities, stream)
    device_round = Round(verb, engine, haystacks, sz.Strings([]), gpu_managed_array, capabilities, stream)
    device_round()
    host = Round(verb, sz.RegexEngine(vocabulary), corpus.tape, sz.Strings([]), np.zeros)
    host()
    if not device_round.agrees(host):
        raise SystemExit(f"{name} disagrees with the host engine on its first round")
    measure(settings, name, corpus.spec(), device_round)


def measure_rival(
    settings: Settings, corpus: Corpus, name: str, build: Callable[[], Any], walk: Callable[[Any], Callable[[], object]]
) -> None:
    """Builds a single-threaded rival and times the pass `walk` returns, or notes the row when the rival
    cannot express the vocabulary."""
    if not settings.selects(name):
        return
    try:
        engine = build()
    except Exception as error:  # an engine that refuses a pattern, as RE2 and Hyperscan refuse lookarounds
        note_unavailable(name, str(error))
        return
    measure(settings, name, corpus.spec(), walk(engine))


def alternation(patterns: list[str]) -> str:
    """The vocabulary as one leftmost-first alternation, every pattern its own capturing group."""
    return "|".join(f"({pattern})" for pattern in patterns)


def build_hyperscan(patterns: list[str]) -> Any:
    database = hyperscan.Database()
    database.compile(
        expressions=[pattern.encode() for pattern in patterns],
        ids=list(range(len(patterns))),
        elements=len(patterns),
        flags=[0] * len(patterns),
    )
    return database


def bench_counting(
    settings: Settings, corpus: Corpus, vocabulary: str, patterns: list[str], devices: list[Device]
) -> None:
    """Counts every leftmost-first match of the vocabulary, the walk every engine offers."""
    group = f"count-{vocabulary}"
    sweep(settings, corpus, group, patterns, [], devices, "counts")

    def count_finditer(expression: Any) -> Callable[[], object]:
        return pass_over(lambda text: deque(expression.finditer(text), maxlen=0), corpus.texts)

    measure_rival(
        settings, corpus, f"{group}/re.finditer", lambda: re.compile(alternation(patterns), re.ASCII), count_finditer
    )
    measure_rival(
        settings,
        corpus,
        f"{group}/regex.finditer",
        lambda: regex.compile(alternation(patterns), regex.ASCII),
        count_finditer,
    )
    measure_rival(settings, corpus, f"{group}/re2.finditer", lambda: re2.compile(alternation(patterns)), count_finditer)

    def scan_hyperscan(database: Any) -> Callable[[], object]:
        events = [0]

        def on_match(pattern_id: int, start: int, end: int, flags: int, context: Any) -> None:
            events[0] += 1

        return pass_over(lambda document: database.scan(document, match_event_handler=on_match), corpus.dataset.tokens)

    measure_rival(
        settings, corpus, f"{group}/hyperscan.Database.scan", lambda: build_hyperscan(patterns), scan_hyperscan
    )

    for ordinal in gpu_ordinals(settings):
        for call in ("count", "contains"):
            name = f"{group}/cudf.Series.str.{call}<cuda:{ordinal}>"
            if not CUDF_AVAILABLE:
                note_unavailable(name, "cudf not installed")
                continue
            if not settings.selects(name) or not gpu_require(settings, ordinal):
                continue
            series = cudf.Series(corpus.texts)
            kernel = getattr(series.str, call)
            try:
                kernel(alternation(patterns))
            except Exception as error:  # cuDF's regex engine refuses lookarounds
                note_unavailable(name, str(error))
                continue
            measure(settings, name, corpus.spec(), functools.partial(kernel, alternation(patterns)))


def bench_extracting(settings: Settings, corpus: Corpus, devices: list[Device]) -> None:
    """Reads the year, month and day of each document's first ISO date."""
    group = "extract-date"
    sweep(settings, corpus, group, [DATE_GROUPS], [], devices, "extract")

    def search_each(expression: Any) -> Callable[[], object]:
        return pass_over(expression.search, corpus.texts)

    measure_rival(settings, corpus, f"{group}/re.search", lambda: re.compile(DATE_GROUPS, re.ASCII), search_each)
    measure_rival(
        settings, corpus, f"{group}/regex.search", lambda: regex.compile(DATE_GROUPS, regex.ASCII), search_each
    )
    measure_rival(settings, corpus, f"{group}/re2.search", lambda: re2.compile(DATE_GROUPS), search_each)

    for ordinal in gpu_ordinals(settings):
        name = f"{group}/cudf.Series.str.extract<cuda:{ordinal}>"
        if not CUDF_AVAILABLE:
            note_unavailable(name, "cudf not installed")
            continue
        if not settings.selects(name) or not gpu_require(settings, ordinal):
            continue
        series = cudf.Series(corpus.texts)
        measure(settings, name, corpus.spec(), functools.partial(series.str.extract, DATE_GROUPS))


def bench_rewriting(
    settings: Settings, corpus: Corpus, vocabulary: str, patterns: list[str], devices: list[Device]
) -> None:
    """Rewrites every match with its pattern's replacement, the one verb whose output is another corpus."""
    group = f"replace-{vocabulary}"
    replacements = [f"<{index}>" for index in range(len(patterns))]
    sweep(settings, corpus, group, patterns, replacements, devices, "replace")

    def substitute_each(expression: Any) -> Callable[[], object]:
        rewrite = functools.partial(expression.sub, lambda found: replacements[found.lastindex - 1])
        return pass_over(rewrite, corpus.texts)

    measure_rival(
        settings, corpus, f"{group}/re.sub", lambda: re.compile(alternation(patterns), re.ASCII), substitute_each
    )
    measure_rival(
        settings,
        corpus,
        f"{group}/regex.sub",
        lambda: regex.compile(alternation(patterns), regex.ASCII),
        substitute_each,
    )


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    libraries = {
        "StringZilla": f"{sz.__version__} with {sz.cpu_capabilities_enabled().name}",
        **{name: pkg_version(name) for name in ("regex", "google-re2", "hyperscan")},
    }
    print_machine(libraries)
    settings = read_settings("regex")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    log_dataset(dataset)
    log_timing_overhead(settings)

    texts = dataset.text_tokens()
    corpus = Corpus(texts=texts, tape=sz.Strings(texts), dataset=dataset)
    devices = selected_devices(settings)

    print("Benchmark configuration:")
    print(f"- Corpus: {dataset.token_count} documents, {dataset.token_bytes} bytes")
    for device in devices:
        print(f"- Device: {device.label}")

    for vocabulary, patterns in VOCABULARIES.items():
        print(f"\n# {vocabulary} - {len(patterns)} patterns")
        bench_counting(settings, corpus, vocabulary, patterns, devices)
        sweep(settings, corpus, f"find-{vocabulary}", patterns, [], devices, "find")
        bench_rewriting(settings, corpus, vocabulary, patterns, devices)

    print("\n# extract - 1 pattern")
    bench_extracting(settings, corpus, devices)

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
