"""Batched JSONPath and XPath benchmarks in Python, over every document. Mirrors `subdocuments/bench.rs`.

- stringzilla.SubdocumentsEngine: counts and find over one compiled vocabulary of paths
- jsonpath-ng over `orjson` documents, for JSONPath
- orjson with a manual walk and pysimdjson pointers, for the paths a JSON pointer can spell
- lxml, for XPath
- cudf.Series.str.get_json_object on each selected GPU, where cuDF is installed

One document per line. A `.xml` dataset runs the XPath vocabulary over Stack Exchange `<row>` elements,
anything else the JSONPath vocabularies over JSON events; lines that are not whole documents are left out
by the same rule as in `bench.rs`. A rival that cannot spell a vocabulary notes its row as unavailable.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=data/gharchive/2024-01-01-15.json uv run --group subdocuments subdocuments/bench.py
"""

import argparse
import functools
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from typing import Any

import jsonpath_ng
import lxml.etree
import numpy as np
import orjson
import simdjson
import stringzilla as sz

from stringwars import (
    Bytes,
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


@dataclass(frozen=True)
class Vocabulary:
    """One fixed vocabulary of paths, identical in `bench.rs`, with the JSON pointers that spell the same
    selection where pointers can."""

    name: str
    paths: list[str]
    pointers: list[str] | None = None


JSON_VOCABULARIES = [
    Vocabulary("json-pointer", ["$.type", "$.actor.login", "$.repo.name"], ["/type", "/actor/login", "/repo/name"]),
    Vocabulary(
        "json-query",
        ["$.payload.commits[*].sha", "$..url", "$.payload.commits[?@.distinct == true].message"],
    ),
]
XML_VOCABULARY = Vocabulary("xml-attributes", ["/row/@Id", "/row/@Score", "/row/@Tags", "/row/@OwnerUserId"])

# The concurrency of every single-threaded row.
ONE_THREAD = Threads(1)


def is_document(line: bytes, format_name: str) -> bool:
    """Whether a line is one whole document: a JSON object, or a self-closing XML `<row>` element."""
    if format_name == "json":
        return line.startswith(b"{") and line.endswith(b"}")
    return line.startswith(b"<row") and line.endswith(b"/>")


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

    documents: list[bytes]
    tape: sz.Strings
    format_name: str

    def spec(self, concurrency: Threads = ONE_THREAD) -> MeasureSpec:
        return MeasureSpec(
            unit="bytes",
            elements=len(self.documents),
            total_bytes=Bytes(sum(map(len, self.documents))),
            concurrency=concurrency,
        )

    def share(self, thread: int, threads: int) -> slice:
        """The documents one of `threads` engines walks, contiguous so its outputs are too."""
        chunk = -(-len(self.documents) // threads)
        first = min(thread * chunk, len(self.documents))
        return slice(first, min(first + chunk, len(self.documents)))


class Round:
    """One round of a `SubdocumentsEngine` verb over fixed documents and the outputs it writes, the matches
    sized by an empty round whose last offset is what the filled one needs. `allocate` places every buffer
    where the engine's device reaches."""

    def __init__(
        self,
        verb: str,
        engine: Any,
        documents: sz.Strings,
        allocate: Callable[[tuple[int, ...], Any], np.ndarray],
        capabilities: Any = None,
        stream: int | None = None,
    ) -> None:
        self.verb, self.engine, self.documents = verb, engine, documents
        self.capabilities, self.stream = capabilities, stream
        self.placement = {} if stream is None else {"stream": stream}
        count = len(documents)
        self.counts = allocate((count,), np.uintp)
        self.offsets = allocate((count + 1,), np.uintp)
        self.matches = None
        if verb == "find":
            engine.find(documents, None, self.offsets, **self.placement)
            self.join()
            self.matches = allocate((max(1, int(self.offsets[count])), 4), np.uintp)

    def join(self) -> None:
        if self.stream is not None:
            sz.synchronize(self.capabilities, self.stream)

    def __call__(self) -> None:
        if self.verb == "counts":
            self.engine.counts(self.documents, self.counts, **self.placement)
        else:
            self.engine.find(self.documents, self.matches, self.offsets, **self.placement)
        self.join()

    def agrees(self, host: "Round") -> bool:
        """Whether a device round wrote what the host engine did."""
        if self.verb == "counts":
            return np.array_equal(self.counts, host.counts)
        total = int(host.offsets[-1])
        return np.array_equal(self.offsets, host.offsets) and np.array_equal(self.matches[:total], host.matches[:total])


def sweep(settings: Settings, corpus: Corpus, group: str, paths: list[str], devices: list[Device], verb: str) -> None:
    """Times `verb` on every device: one host engine, one engine per pool thread over a contiguous share of
    the documents, and one engine per GPU whose first round is checked against the host."""
    vocabulary = sz.Strings(paths)
    for device in devices:
        name = f"{group}/stringzilla.SubdocumentsEngine.{verb}<{device.label}>"
        if not settings.selects(name):
            continue
        if device.ordinal is not None:
            stream = sz.cuda_stream_init(device.ordinal)
            try:
                sweep_gpu(settings, corpus, name, vocabulary, device.ordinal, stream, verb)
            finally:
                sz.cuda_stream_free(stream)
            continue
        threads = device.concurrency
        try:
            engines = [sz.SubdocumentsEngine(vocabulary, format=corpus.format_name) for _ in range(threads)]
        except ValueError as error:  # a path this engine refuses skips its own row
            note_unavailable(name, str(error))
            continue
        if threads == 1:
            measure(settings, name, corpus.spec(), Round(verb, engines[0], corpus.tape, np.zeros))
            continue
        rounds = [
            Round(verb, engine, sz.Strings(corpus.documents[corpus.share(thread, threads)]), np.zeros)
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
        engine = sz.SubdocumentsEngine(vocabulary, format=corpus.format_name, capabilities=capabilities, stream=stream)
    except Exception as error:  # a device tier that declines this vocabulary skips its own row
        note_unavailable(name, str(error))
        return
    documents = corpus.tape.copy(capabilities, stream)
    device_round = Round(verb, engine, documents, gpu_managed_array, capabilities, stream)
    device_round()
    host = Round(verb, sz.SubdocumentsEngine(vocabulary, format=corpus.format_name), corpus.tape, np.zeros)
    host()
    if not device_round.agrees(host):
        raise SystemExit(f"{name} disagrees with the host engine on its first round")
    measure(settings, name, corpus.spec(), device_round)


def measure_rival(
    settings: Settings, corpus: Corpus, name: str, build: Callable[[], Any], walk: Callable[[Any], Callable[[], object]]
) -> None:
    """Builds a single-threaded rival and times the pass `walk` returns, or notes the row when the rival
    cannot spell the vocabulary."""
    if not settings.selects(name):
        return
    try:
        engine = build()
    except Exception as error:  # a parser that refuses a path, as jsonpath-ng refuses RFC 9535 filters
        note_unavailable(name, str(error))
        return
    measure(settings, name, corpus.spec(), walk(engine))


def walk_pointer(document: Any, keys: list[str]) -> Any:
    """Follows `keys` down a parsed document, or None where a key is missing."""
    for key in keys:
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


def bench_json_rivals(settings: Settings, corpus: Corpus, group: str, vocabulary: Vocabulary) -> None:
    """The JSON rivals: jsonpath-ng for every vocabulary, the pointer readers only where it is pointers."""

    def query_each(queries: list[Any]) -> Callable[[], object]:
        def select(document: bytes) -> int:
            parsed = orjson.loads(document)
            return sum(len(query.find(parsed)) for query in queries)

        return pass_over(select, corpus.documents)

    measure_rival(
        settings,
        corpus,
        f"{group}/jsonpath_ng.JSONPath.find",
        lambda: [jsonpath_ng.parse(path) for path in vocabulary.paths],
        query_each,
    )

    pointer_rows = ["orjson.loads", "simdjson.Parser.parse"]
    if vocabulary.pointers is None:
        for call in pointer_rows:
            name = f"{group}/{call}"
            if settings.selects(name):
                note_unavailable(name, "a JSON pointer cannot spell wildcards, descendants or filters")
    else:
        keys = [pointer.split("/")[1:] for pointer in vocabulary.pointers]

        def walk_orjson(document: bytes) -> int:
            parsed = orjson.loads(document)
            return sum(walk_pointer(parsed, path) is not None for path in keys)

        name = f"{group}/orjson.loads"
        if settings.selects(name):
            measure(settings, name, corpus.spec(), pass_over(walk_orjson, corpus.documents))

        parser = simdjson.Parser()

        def walk_simdjson(document: bytes) -> int:
            parsed = parser.parse(document)
            found = 0
            for pointer in vocabulary.pointers:
                try:
                    parsed.at_pointer(pointer)
                    found += 1
                except (KeyError, IndexError, ValueError):
                    pass
            return found

        name = f"{group}/simdjson.Parser.parse"
        if settings.selects(name):
            measure(settings, name, corpus.spec(), pass_over(walk_simdjson, corpus.documents))

    for ordinal in gpu_ordinals(settings):
        name = f"{group}/cudf.Series.str.get_json_object<cuda:{ordinal}>"
        if not CUDF_AVAILABLE:
            note_unavailable(name, "cudf not installed")
            continue
        if not settings.selects(name) or not gpu_require(settings, ordinal):
            continue
        series = cudf.Series([document.decode() for document in corpus.documents])
        try:
            for path in vocabulary.paths:
                series.str.get_json_object(path)
        except Exception as error:  # cuDF reads a subset of JSONPath
            note_unavailable(name, str(error))
            continue

        def select_all(series: Any = series) -> None:
            for path in vocabulary.paths:
                series.str.get_json_object(path)

        measure(settings, name, corpus.spec(), select_all)


def bench_xml_rivals(settings: Settings, corpus: Corpus, group: str, vocabulary: Vocabulary) -> None:
    """The XML rival: compiled lxml XPath queries over each parsed document."""

    def query_each(queries: list[Any]) -> Callable[[], object]:
        def select(document: bytes) -> int:
            root = lxml.etree.fromstring(document)
            return sum(len(query(root)) for query in queries)

        return pass_over(select, corpus.documents)

    measure_rival(
        settings,
        corpus,
        f"{group}/lxml.etree.XPath",
        lambda: [lxml.etree.XPath(path) for path in vocabulary.paths],
        query_each,
    )


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    libraries = {
        "StringZilla": f"{sz.__version__} with {sz.cpu_capabilities_enabled().name}",
        **{name: pkg_version(name) for name in ("jsonpath-ng", "orjson", "pysimdjson", "lxml")},
    }
    print_machine(libraries)
    settings = read_settings("subdocuments")
    print_settings(settings)
    dataset: Dataset = resolve_dataset(settings)
    log_dataset(dataset)
    log_timing_overhead(settings)

    format_name = "xml" if settings.dataset_path.endswith(".xml") else "json"
    # The whitespace Rust's `trim_ascii` strips, so both languages keep the same documents.
    documents = [line.strip(b" \t\n\x0c\r") for line in dataset.tokens]
    documents = [line for line in documents if is_document(line, format_name)]
    if not documents:
        raise SystemExit("The working set holds no whole document; is the dataset path right?")
    corpus = Corpus(documents=documents, tape=sz.Strings(documents), format_name=format_name)
    devices = selected_devices(settings)

    print("Benchmark configuration:")
    print(f"- Corpus: {len(documents)} documents, {sum(map(len, documents))} bytes")
    for device in devices:
        print(f"- Device: {device.label}")

    vocabularies = JSON_VOCABULARIES if format_name == "json" else [XML_VOCABULARY]
    for vocabulary in vocabularies:
        print(f"\n# {vocabulary.name} - {len(vocabulary.paths)} paths")
        count_group = f"count-{vocabulary.name}"
        sweep(settings, corpus, count_group, vocabulary.paths, devices, "counts")
        if format_name == "json":
            bench_json_rivals(settings, corpus, count_group, vocabulary)
        else:
            bench_xml_rivals(settings, corpus, count_group, vocabulary)
        sweep(settings, corpus, f"find-{vocabulary.name}", vocabulary.paths, devices, "find")

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
