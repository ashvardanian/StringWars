"""Multi-pattern search benchmarks in Python: one dictionary walked over every document. Mirrors `substrings/bench.rs`.

- stringzilla.SubstringsEngine: counts, find, replace and bm25_scores over one compiled automaton
- ahocorasick_rs: the `aho-corasick` crate bound to Python
- pyahocorasick: a C automaton with an overlapping walk
- re: an alternation of escaped literals, the baseline most people actually write
- bm25s, rank_bm25 and sklearn.TfidfVectorizer: index-then-query scorers, for the scoring group only

Every rival is single-threaded and CPU-only, so the head-to-head row is `<1cpu>`. Scoring is scan versus
index: a build tokenizes the corpus and never sees the dictionary, so only the first slice measures it,
while a query is the slice itself.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group substrings substrings/bench.py
"""

import argparse
import functools
import re
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from typing import Any

import ahocorasick as pyahocorasick
import ahocorasick_rs
import bm25s
import numpy as np
import rank_bm25
import stringzilla as sz
from sklearn.feature_extraction.text import TfidfVectorizer

from stringwars import (
    Dataset,
    MeasureSpec,
    Settings,
    Threads,
    Tokenization,
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
    tokenize_dataset,
)

# RAPIDS cuML and cuDF are outside the suite's dependency group, so the GPU TF-IDF rows are optional.
try:
    import cudf
    from cuml.feature_extraction.text import TfidfVectorizer as CumlTfidfVectorizer

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Shortest word admitted into the dictionary: anything under three bytes matches at nearly every
# position, so the benchmark would measure match materialization rather than the automaton walk.
VOCABULARY_MIN_WORD_BYTES = 3

# Longest word admitted. Longer whitespace-cut tokens are unsegmented CJK runs and URLs rather than
# words, and the longest needle sets the warm-up every GPU chunk re-walks.
VOCABULARY_MAX_WORD_BYTES = 32

# Which end of the frequency-ordered vocabulary a dictionary is drawn from, and the share of it taken.
VOCABULARY_SHARES = {
    "most_frequent_1%": 100,
    "most_frequent_10%": 10,
    "least_frequent_1%": 100,
    "least_frequent_10%": 10,
    "entire": 1,
}
VOCABULARY_SLICES = list(VOCABULARY_SHARES)

# The concurrency of every single-threaded row.
ONE_THREAD = Threads(1)


def build_vocabulary(documents: list[bytes]) -> list[bytes]:
    """Counts word frequencies over every document and drops both noisy ends, returning the survivors
    ordered by descending frequency with ties broken by content so the ranking is reproducible."""
    frequencies = Counter(
        word
        for document in documents
        for word in tokenize_dataset(document, Tokenization.WORDS, "keep")
        if VOCABULARY_MIN_WORD_BYTES <= len(word) <= VOCABULARY_MAX_WORD_BYTES
    )
    ranked = sorted(
        ((term, count) for term, count in frequencies.items() if count >= 2),
        key=lambda entry: (-entry[1], entry[0]),
    )
    # The top cutoff keeps its base: a fraction of all distinct terms, hapax included.
    return [term for term, _ in ranked[len(frequencies) // 100 :]]


def needle_slice_of(vocabulary: list[bytes], slice_name: str) -> list[bytes]:
    """One contiguous slice of a non-empty frequency-ordered vocabulary, never empty itself."""
    available = len(vocabulary)
    count = min(max(available // VOCABULARY_SHARES[slice_name], 1), available)
    first = available - count if slice_name.startswith("least_frequent") else 0
    return vocabulary[first : first + count]


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

    documents: list[str]
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
        chunk = -(-len(self.documents) // threads)
        first = min(thread * chunk, len(self.documents))
        return slice(first, min(first + chunk, len(self.documents)))


@dataclass(frozen=True)
class Verb:
    """One verb a compiled dictionary answers: its group, its call, and the overlap policy its engine is
    built with, since covers and rewrites need a leftmost one."""

    group: str
    call: str
    policy: str


COUNT = Verb("count", "counts", "overlapping")
COVER = Verb("cover", "counts", "leftmost-longest")
FIND = Verb("find", "find", "overlapping")
REPLACE = Verb("replace", "replace", "leftmost-longest")
SCORE = Verb("bm25", "bm25_scores", "overlapping")


class Round:
    """One round of a verb over fixed haystacks and the outputs it writes, sized by an empty round that
    reports what the filled one needs. `allocate` places every buffer where the engine's device reaches."""

    def __init__(
        self,
        verb: Verb,
        engine: Any,
        haystacks: sz.Strings,
        replacements: sz.Strings,
        mean_length: float,
        allocate: Callable[[tuple[int, ...], Any], np.ndarray],
        capabilities: Any = None,
        stream: int | None = None,
    ) -> None:
        self.verb, self.engine, self.haystacks, self.replacements = verb, engine, haystacks, replacements
        self.mean_length, self.capabilities, self.stream = mean_length, capabilities, stream
        self.placement = {} if stream is None else {"stream": stream}
        count = len(haystacks)
        self.counts = allocate((count,), np.uintp)
        self.offsets = allocate((count + 1,), np.uintp)
        self.scores = allocate((count,), np.float32)
        self.weights = allocate((len(replacements),), np.float32)
        self.weights[:] = 1.0
        self.matches = self.target = None
        if verb is FIND:
            engine.find(haystacks, None, self.offsets, **self.placement)
            self.join()
            self.matches = allocate((max(1, engine.report["matches_emitted"]), 4), np.uintp)
        if verb is REPLACE:
            engine.replace(haystacks, replacements, None, self.offsets, **self.placement)
            self.join()
            self.target = allocate((max(1, engine.report["target_length"]),), np.uint8)

    def join(self) -> None:
        if self.stream is not None:
            sz.synchronize(self.capabilities, self.stream)

    def __call__(self) -> None:
        if self.verb.call == "counts":
            self.engine.counts(self.haystacks, self.counts, **self.placement)
        elif self.verb is FIND:
            self.engine.find(self.haystacks, self.matches, self.offsets, **self.placement)
        elif self.verb is REPLACE:
            self.engine.replace(self.haystacks, self.replacements, self.target, self.offsets, **self.placement)
        else:
            self.engine.bm25_scores(
                self.haystacks, self.weights, self.scores, average_document_length=self.mean_length, **self.placement
            )
        self.join()

    def agrees(self, host: "Round") -> bool:
        """Whether a device round wrote what the host engine did, scores to a float tolerance."""
        if self.verb.call == "counts":
            return np.array_equal(self.counts, host.counts)
        if self.verb is FIND:
            return np.array_equal(self.offsets, host.offsets)
        if self.verb is REPLACE:
            return np.array_equal(self.offsets, host.offsets) and np.array_equal(self.target, host.target)
        return np.allclose(self.scores, host.scores, rtol=1e-3, atol=1e-3)


def compile_engine(needles: list[str], verb: Verb, **placement: Any) -> Any:
    """Compiles `needles` for `verb`; a device tier also sizes one round for `haystacks_budget` documents."""
    return sz.SubstringsEngine(sz.Strings(needles), "cased", verb.policy, **placement)


def sweep(settings: Settings, corpus: Corpus, needles: list[str], slice_name: str, devices: list[Device], verb: Verb):
    """Times `verb` on every device: one host engine, one engine per pool thread over a contiguous share of
    the documents, and one engine per GPU whose first round is checked against the host."""
    replacements = sz.Strings(needles)
    mean_length = corpus.dataset.token_bytes / max(1, corpus.dataset.token_count)
    for device in devices:
        name = f"{verb.group}-{slice_name}/stringzilla.SubstringsEngine.{verb.call}<{device.label}>"
        if not settings.selects(name):
            continue
        if device.ordinal is not None:
            stream = sz.cuda_stream_init(device.ordinal)
            try:
                sweep_gpu(settings, corpus, needles, name, device.ordinal, stream, verb)
            finally:
                sz.cuda_stream_free(stream)
            continue
        threads = device.concurrency
        try:
            engines = [compile_engine(needles, verb) for _ in range(threads)]
        except Exception as error:  # an engine that declines this dictionary skips its own row
            note_unavailable(name, str(error))
            continue
        if threads == 1:
            measure(
                settings, name, corpus.spec(), Round(verb, engines[0], corpus.tape, replacements, mean_length, np.zeros)
            )
            continue
        rounds = [
            Round(
                verb,
                engine,
                sz.Strings(corpus.documents[corpus.share(thread, threads)]),
                replacements,
                mean_length,
                np.zeros,
            )
            for thread, engine in enumerate(engines)
        ]
        configure = functools.partial(sz.cpu_configure_thread, sz.cpu_capabilities_enabled())
        with thread_pool(threads, configure) as run_all:
            measure(settings, name, corpus.spec(threads), functools.partial(run_all, rounds))


def sweep_gpu(
    settings: Settings, corpus: Corpus, needles: list[str], name: str, ordinal: int, stream: int, verb: Verb
) -> None:
    """One GPU row, every buffer in managed memory and every input copied into the device's own tape."""
    capabilities = sz.cuda_capabilities_enabled(ordinal)
    replacements = sz.Strings(needles)
    mean_length = corpus.dataset.token_bytes / max(1, corpus.dataset.token_count)
    try:
        engine = compile_engine(
            needles, verb, haystacks_budget=len(corpus.documents), capabilities=capabilities, stream=stream
        )
    except Exception as error:  # a device tier that declines this dictionary skips its own row
        note_unavailable(name, str(error))
        return
    haystacks = corpus.tape.copy(capabilities, stream)
    device_round = Round(
        verb,
        engine,
        haystacks,
        replacements.copy(capabilities, stream),
        mean_length,
        gpu_managed_array,
        capabilities,
        stream,
    )
    device_round()
    host = Round(verb, compile_engine(needles, verb), corpus.tape, replacements, mean_length, np.zeros)
    host()
    if not device_round.agrees(host):
        raise SystemExit(f"{name} disagrees with the host engine on its first round")
    measure(settings, name, corpus.spec(), device_round)


def measure_rival(
    settings: Settings, corpus: Corpus, name: str, build: Callable[[], Any], walk: Callable[[Any], Callable[[], object]]
) -> None:
    """Builds a single-threaded rival and times the pass `walk` returns for it."""
    if settings.selects(name):
        measure(settings, name, corpus.spec(), walk(build()))


def bench_counting(settings: Settings, corpus: Corpus, needles: list[str], slice_name: str, devices: list[Device]):
    """Counts every overlapping match of every needle, the cheapest walk each engine offers."""
    sweep(settings, corpus, needles, slice_name, devices, COUNT)

    measure_rival(
        settings,
        corpus,
        f"count-{slice_name}/ahocorasick_rs.find_matches_as_indexes",
        lambda: ahocorasick_rs.AhoCorasick(needles, matchkind=ahocorasick_rs.MatchKind.Standard),
        lambda automaton: pass_over(
            functools.partial(automaton.find_matches_as_indexes, overlapping=True), corpus.documents
        ),
    )

    def build_pyahocorasick() -> Any:
        automaton = pyahocorasick.Automaton()
        for index, needle in enumerate(needles):
            automaton.add_word(needle, index)
        automaton.make_automaton()
        return automaton

    measure_rival(
        settings,
        corpus,
        f"count-{slice_name}/pyahocorasick.Automaton.iter",
        build_pyahocorasick,
        lambda automaton: pass_over(lambda document: deque(automaton.iter(document), maxlen=0), corpus.documents),
    )


def bench_cover(settings: Settings, corpus: Corpus, needles: list[str], slice_name: str, devices: list[Device]):
    """Counts a leftmost cover, the walk a rewrite needs and the one every engine spells differently."""
    sweep(settings, corpus, needles, slice_name, devices, COVER)

    measure_rival(
        settings,
        corpus,
        f"cover-{slice_name}/ahocorasick_rs.find_matches_as_indexes<LeftmostLongest>",
        lambda: ahocorasick_rs.AhoCorasick(needles, matchkind=ahocorasick_rs.MatchKind.LeftmostLongest),
        lambda automaton: pass_over(automaton.find_matches_as_indexes, corpus.documents),
    )

    # An alternation of escaped literals resolves leftmost-first, so it sits beside the covers. `re` scans
    # it branch by branch with no literal prefilter, so one pass runs for minutes at a few thousand
    # needles; only the narrowest dictionary measures it.
    if slice_name != VOCABULARY_SLICES[0]:
        return
    measure_rival(
        settings,
        corpus,
        f"cover-{slice_name}/re.finditer",
        lambda: re.compile("|".join(re.escape(needle) for needle in needles)),
        lambda pattern: pass_over(lambda document: deque(pattern.finditer(document), maxlen=0), corpus.documents),
    )


def bench_locating(settings: Settings, corpus: Corpus, needles: list[str], slice_name: str, devices: list[Device]):
    """Materializes every match rather than tallying it, which is where the output bandwidth shows."""
    sweep(settings, corpus, needles, slice_name, devices, FIND)

    measure_rival(
        settings,
        corpus,
        f"find-{slice_name}/ahocorasick_rs.find_matches_as_indexes",
        lambda: ahocorasick_rs.AhoCorasick(needles, matchkind=ahocorasick_rs.MatchKind.Standard),
        lambda automaton: pass_over(
            functools.partial(automaton.find_matches_as_indexes, overlapping=True), corpus.documents
        ),
    )


def bench_scoring_index(settings: Settings, corpus: Corpus, needles: list[str], slice_name: str) -> None:
    """The index-then-query rivals. A build tokenizes the corpus and never sees the dictionary, so only
    the first slice measures it; the query is the slice itself, so it widens with the dictionary."""
    documents = corpus.documents
    query_text = " ".join(needles)

    def build_bm25s() -> Any:
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(documents, show_progress=False), show_progress=False)
        return retriever

    def build_rank_bm25() -> Any:
        return rank_bm25.BM25Okapi([document.split() for document in documents])

    def build_tfidf() -> Any:
        vectorizer = TfidfVectorizer()
        return vectorizer, vectorizer.fit_transform(documents)

    if slice_name == VOCABULARY_SLICES[0]:
        for name, build in [
            ("bm25s.BM25.index", build_bm25s),
            ("rank_bm25.BM25Okapi", build_rank_bm25),
            ("sklearn.TfidfVectorizer.fit", build_tfidf),
        ]:
            name = f"bm25-{slice_name}/{name}"
            if settings.selects(name):
                measure(settings, name, corpus.spec(), build)

    def query_bm25s(retriever: Any) -> Callable[[], object]:
        query_tokens = bm25s.tokenize([query_text], show_progress=False)
        return lambda: retriever.retrieve(query_tokens, k=min(10, len(documents)), show_progress=False)

    def query_tfidf(index: Any) -> Callable[[], object]:
        vectorizer, document_matrix = index
        query_vector = vectorizer.transform([query_text])
        return lambda: document_matrix @ query_vector.T

    measure_rival(settings, corpus, f"bm25-{slice_name}/bm25s.BM25.retrieve", build_bm25s, query_bm25s)
    measure_rival(
        settings,
        corpus,
        f"bm25-{slice_name}/rank_bm25.get_scores",
        build_rank_bm25,
        lambda okapi: functools.partial(okapi.get_scores, query_text.split()),
    )
    measure_rival(settings, corpus, f"bm25-{slice_name}/sklearn.TfidfVectorizer.transform", build_tfidf, query_tfidf)

    for ordinal in gpu_ordinals(settings):
        build_name = f"bm25-{slice_name}/cuml.TfidfVectorizer.fit<cuda:{ordinal}>"
        query_name = f"bm25-{slice_name}/cuml.TfidfVectorizer.transform<cuda:{ordinal}>"
        if not CUML_AVAILABLE:
            if slice_name == VOCABULARY_SLICES[0]:
                note_unavailable(build_name, "cuml not installed")
            note_unavailable(query_name, "cuml not installed")
            continue
        if not gpu_require(settings, ordinal):
            continue
        device_documents = cudf.Series(documents)
        if slice_name == VOCABULARY_SLICES[0] and settings.selects(build_name):
            measure(
                settings,
                build_name,
                corpus.spec(),
                lambda device_documents=device_documents: CumlTfidfVectorizer().fit_transform(device_documents),
            )
        if settings.selects(query_name):
            device_vectorizer = CumlTfidfVectorizer()
            device_matrix = device_vectorizer.fit_transform(device_documents)
            device_query = device_vectorizer.transform(cudf.Series([query_text]))
            measure(
                settings,
                query_name,
                corpus.spec(),
                lambda device_matrix=device_matrix, device_query=device_query: device_matrix @ device_query.T,
            )


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    libraries = {
        "StringZilla": f"{sz.__version__} with {sz.cpu_capabilities_enabled().name}",
        **{
            name: pkg_version(name)
            for name in ("pyahocorasick", "ahocorasick-rs", "bm25s", "rank-bm25", "scikit-learn")
        },
    }
    print_machine(libraries)
    settings = read_settings("substrings")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    log_dataset(dataset)
    log_timing_overhead(settings)

    vocabulary = build_vocabulary(dataset.tokens)
    if not vocabulary:
        raise SystemExit("The working set yielded no vocabulary; is the dataset path right?")
    documents = dataset.text_tokens()
    corpus = Corpus(documents=documents, tape=sz.Strings(documents), dataset=dataset)
    devices = selected_devices(settings)

    print("Benchmark configuration:")
    print(f"- Vocabulary: {len(vocabulary)} terms after dropping hapax and the top one percent")
    print(f"- Corpus: {dataset.token_count} documents, {dataset.token_bytes} bytes")
    for device in devices:
        print(f"- Device: {device.label}")

    for slice_name in VOCABULARY_SLICES:
        needle_bytes = needle_slice_of(vocabulary, slice_name)
        needles = [needle.decode() for needle in needle_bytes]
        print(f"\n# {slice_name} - {len(needles)} needles, {sum(map(len, needle_bytes))} bytes")

        bench_counting(settings, corpus, needles, slice_name, devices)
        bench_cover(settings, corpus, needles, slice_name, devices)
        bench_locating(settings, corpus, needles, slice_name, devices)
        sweep(settings, corpus, needles, slice_name, devices, REPLACE)
        sweep(settings, corpus, needles, slice_name, devices, SCORE)
        bench_scoring_index(settings, corpus, needles, slice_name)

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
