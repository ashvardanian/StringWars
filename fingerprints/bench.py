"""MinHash fingerprinting and exact window-overlap benchmarks in Python. Mirrors `fingerprints/bench.rs`.

StringZilla v6 has no MinHash engine, so its MinHash rows are noted as unavailable, and its
`OverlapEngine` scores the exact n-gram overlap that MinHash estimates.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group fingerprints fingerprints/bench.py
"""

import argparse
import functools
from collections.abc import Callable, Sequence
from dataclasses import replace
from importlib.metadata import version as pkg_version
from typing import Any

import numpy as np
import stringzilla as sz
from datasketch import MinHash

from stringwars import (
    Bytes,
    Dataset,
    MeasureSpec,
    Settings,
    Threads,
    batch_size,
    finish,
    gpu_batch_size,
    gpu_managed_array,
    gpu_ordinals,
    gpu_require,
    log_dataset,
    log_timing_overhead,
    measure,
    note_unavailable,
    print_machine,
    print_settings,
    read_settings,
    resolve_dataset,
    thread_pool,
)

# RAPIDS cuDF is outside the suite's dependency group, so its rows are optional.
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

# Fixed n-gram widths for multi-scale fingerprinting (matching Rust benchmark)
NGRAM_WIDTHS = [5, 9, 17, 33]


def bench_fingerprint(
    settings: Settings,
    name: str,
    documents: Sequence[Any],
    kernel: Callable[[Any], None],
    document_byte_lengths: np.ndarray,
    dimensions: int,
    documents_per_batch: int,
    concurrency: Threads,
) -> None:
    """One pass sketches every document, in batches; reports hashes/s and bytes/s."""
    count = len(documents)
    total_bytes = Bytes(int(document_byte_lengths.sum()))
    # Hash operations mirror the Rust harness: `dimensions` hash updates per byte.
    work = MeasureSpec(
        unit="hashes", elements=dimensions * total_bytes, total_bytes=total_bytes, concurrency=concurrency
    )

    def one_pass() -> None:
        for low in range(0, count, documents_per_batch):
            kernel(documents[low : low + documents_per_batch])

    measure(settings, name, work, one_pass)


def benchmark_overlap(settings: Settings, dataset: Dataset) -> None:
    """Scores the first `batch_per_core` documents as queries against every document as a candidate, at
    every n-gram width, the exact overlap a MinHash sketch estimates. One pass streams every candidate
    past every query, so the byte rate is candidate bytes per query batch. Mirrors `bench.rs`."""
    documents = dataset.text_tokens()
    queries = sz.Strings(documents[: min(batch_size(settings, 1), len(documents))])
    candidates = sz.Strings(documents)
    widths = len(NGRAM_WIDTHS)
    work = MeasureSpec(unit="bytes", elements=len(queries) * len(documents), total_bytes=dataset.token_bytes)
    call = "overlap/stringzilla.OverlapEngine.scores"
    print(f"\n# overlap - {len(queries)} queries, {len(documents)} candidates")

    name = f"{call}<1cpu>"
    if settings.selects(name):
        engine = sz.OverlapEngine(queries, NGRAM_WIDTHS)
        scores = np.zeros((len(queries), len(documents), widths), dtype=np.float32)
        measure(settings, name, work, lambda: engine.scores(candidates, scores))

    # One engine per pool thread, each holding every query against its share of the candidates.
    threads = settings.threads
    name = f"{call}<{threads}cpu>"
    if settings.selects(name):
        chunk = -(-len(documents) // threads)
        tasks = []
        for first in range(0, len(documents), chunk):
            share = sz.Strings(documents[first : first + chunk])
            engine = sz.OverlapEngine(queries, NGRAM_WIDTHS)
            scores = np.zeros((len(queries), len(share), widths), dtype=np.float32)
            tasks.append(functools.partial(engine.scores, share, scores))
        configure = functools.partial(sz.cpu_configure_thread, sz.cpu_capabilities_enabled())
        with thread_pool(threads, configure) as run_all:
            measure(settings, name, replace(work, concurrency=threads), functools.partial(run_all, tasks))

    # One engine per GPU, its first round checked against the host engine.
    for ordinal in gpu_ordinals(settings):
        name = f"{call}<cuda:{ordinal}>"
        if not settings.selects(name) or not gpu_require(settings, ordinal):
            continue
        stream = sz.cuda_stream_init(ordinal)
        try:
            benchmark_overlap_gpu(settings, name, ordinal, stream, queries, candidates, work)
        finally:
            sz.cuda_stream_free(stream)


def benchmark_overlap_gpu(
    settings: Settings,
    name: str,
    ordinal: int,
    stream: int,
    queries: sz.Strings,
    candidates: sz.Strings,
    work: MeasureSpec,
) -> None:
    """One GPU overlap row, the candidates copied into the device's tape and the scores in managed memory."""
    capabilities = sz.cuda_capabilities_enabled(ordinal)
    try:
        engine = sz.OverlapEngine(
            queries, NGRAM_WIDTHS, candidates_budget=len(candidates), capabilities=capabilities, stream=stream
        )
    except Exception as error:  # a device that declines these widths skips its own row
        note_unavailable(name, str(error))
        return
    device_candidates = candidates.copy(capabilities, stream)
    shape = (len(queries), len(candidates), len(NGRAM_WIDTHS))
    scores = gpu_managed_array(shape, np.float32)

    def one_round() -> None:
        engine.scores(device_candidates, scores, stream=stream)
        sz.synchronize(capabilities, stream)

    one_round()
    expected = np.zeros(shape, dtype=np.float32)
    sz.OverlapEngine(queries, NGRAM_WIDTHS).scores(candidates, expected)
    if not np.allclose(scores, expected, atol=1e-4):
        raise SystemExit(f"{name} disagrees with the host engine on its first round")
    measure(settings, name, work, one_round)


def benchmark_datasketch(
    settings: Settings, documents: list[bytes], document_byte_lengths: np.ndarray, dimensions: int
) -> None:
    """datasketch MinHash on CPU: the common data-science baseline, n-grams built in Python."""
    name = f"minhash-{dimensions}/datasketch.MinHash"
    if not settings.selects(name):
        return
    per_width = max(1, dimensions // len(NGRAM_WIDTHS))
    # A fresh sketch per document is what the algorithm requires; a fresh permutation
    # table is not, and regenerating one per document per width is pure setup cost.
    prototype = MinHash(num_perm=per_width)
    permutations, scheme = prototype.permutations, prototype.scheme

    def kernel(slice_of_documents: list[bytes]) -> None:
        for data in slice_of_documents:
            for width in NGRAM_WIDTHS:
                signature = MinHash(num_perm=per_width, permutations=permutations, scheme=scheme)
                signature.update_batch(data[offset : offset + width] for offset in range(len(data) - width + 1))

    bench_fingerprint(
        settings, name, documents, kernel, document_byte_lengths, dimensions, batch_size(settings, 1), Threads(1)
    )


def benchmark_cudf(
    settings: Settings, documents: list[str], document_byte_lengths: np.ndarray, dimensions: int
) -> None:
    """cuDF MinHash on the GPU: the CUDA first-party comparison (optional, best-effort)."""
    ordinals = gpu_ordinals(settings)
    names = [f"minhash-{dimensions}/cudf.minhash<cuda:{ordinal}>" for ordinal in ordinals]
    if not any(settings.selects(name) for name in names):
        return
    try:
        import cupy as cp
    except ImportError:
        for name in names:
            note_unavailable(name, "cupy not installed")
        return

    per_width = max(1, dimensions // len(NGRAM_WIDTHS))

    def kernel(series_slice: Any) -> None:
        for width in NGRAM_WIDTHS:
            series_slice.str.minhash(seed=0, a=parameters_a, b=parameters_b, width=width)

    for ordinal, name in zip(ordinals, names, strict=True):
        if not settings.selects(name):
            continue
        if not gpu_require(settings, ordinal):
            continue
        # Built after the device is current, so both the parameters and the series live on it.
        parameters_a = cp.arange(1, per_width + 1, dtype=cp.uint32)
        parameters_b = cp.arange(1, per_width + 1, dtype=cp.uint32)
        series = cudf.Series(documents)
        try:
            bench_fingerprint(
                settings,
                name,
                series,
                kernel,
                document_byte_lengths,
                dimensions,
                gpu_batch_size(settings, ordinal),
                Threads(1),
            )
        except Exception as error:  # cuDF raises its own types for an unusable device or kernel
            note_unavailable(name, f"{type(error).__name__}: {error}")


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    libraries = {
        "StringZilla": f"{sz.__version__} with {sz.cpu_capabilities_enabled().name}",
        "DataSketch": pkg_version("datasketch"),
    }
    if CUDF_AVAILABLE:
        libraries["cuDF"] = cudf.__version__
    print_machine(libraries)
    settings = read_settings("fingerprints")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    texts = dataset.text_tokens()
    log_dataset(dataset)
    log_timing_overhead(settings)

    document_byte_lengths = np.fromiter(map(len, dataset.tokens), dtype=np.int64, count=dataset.token_count)

    # Sweep the same widths as `bench.rs`, both read from STRINGWARS_DIMS, so both
    # languages emit rows at the same scales.
    for dimensions in settings.dims:
        print(f"\n# minhash-{dimensions}")
        stringzilla_name = f"minhash-{dimensions}/stringzillas.Fingerprints"
        if settings.selects(stringzilla_name):
            note_unavailable(stringzilla_name, "StringZilla v6 has no MinHash engine")
        benchmark_datasketch(settings, dataset.tokens, document_byte_lengths, dimensions)
        if not CUDF_AVAILABLE:
            for ordinal in gpu_ordinals(settings):
                note_unavailable(f"minhash-{dimensions}/cudf.minhash<cuda:{ordinal}>", "cudf not installed")
        else:
            benchmark_cudf(settings, texts, document_byte_lengths, dimensions)

    benchmark_overlap(settings, dataset)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
