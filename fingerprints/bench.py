"""MinHash fingerprinting benchmarks in Python across CPU and GPU. Mirrors `fingerprints/bench.rs`.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group fingerprints fingerprints/bench.py
"""

import argparse
from collections.abc import Callable, Sequence
from importlib.metadata import version as pkg_version
from typing import Any

import numpy as np
import stringzilla as sz
import stringzillas as szs
from datasketch import MinHash

from stringwars import (
    FALLBACK_GPU_MULTIPROCESSORS,
    Bytes,
    MeasureSpec,
    Settings,
    Threads,
    batch_size,
    finish,
    gpu_multiprocessor_count,
    log_dataset,
    log_timing_overhead,
    measure,
    note_unavailable,
    print_machine,
    print_settings,
    read_settings,
    resolve_dataset,
)

# RAPIDS cuDF is outside the suite's dependency group, so its rows are optional.
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

# Fixed n-gram widths for multi-scale fingerprinting (matching Rust benchmark)
NGRAM_WIDTHS = [5, 9, 17, 33]
NGRAM_WIDTHS_ARRAY = np.array(NGRAM_WIDTHS, dtype=np.uint64)


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


def benchmark_stringzillas(
    settings: Settings, documents: list[str], document_byte_lengths: np.ndarray, dimensions: int
) -> None:
    """StringZilla Fingerprints on 1 core, all cores, and the GPU (if present)."""
    cpu_cores = settings.threads
    default_scope = szs.DeviceScope()
    cpu_scope = szs.DeviceScope(cpu_cores=cpu_cores)
    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:  # any failure here means there is no usable GPU
        gpu_scope = None

    moved = sz.Strs(documents)

    def run_variant(name: str, scope: szs.DeviceScope, documents_per_batch: int, concurrency: Threads) -> None:
        engine = szs.Fingerprints(ndim=dimensions, window_widths=NGRAM_WIDTHS_ARRAY, capabilities=scope)

        def kernel(strs_slice: sz.Strs) -> None:
            engine(strs_slice, device=scope)  # returns (hashes, counts); discarded for throughput

        bench_fingerprint(
            settings, name, moved, kernel, document_byte_lengths, dimensions, documents_per_batch, concurrency
        )

    # Row names carry no batch size, matching what bench.rs prints, so a filter selects the
    # same rows in both harnesses.
    single_cpu_name = f"minhash-{dimensions}/stringzillas.Fingerprints<1cpu>"
    all_cpu_name = f"minhash-{dimensions}/stringzillas.Fingerprints<{cpu_cores}cpu>"
    gpu_name = f"minhash-{dimensions}/stringzillas.Fingerprints<1gpu>"

    if settings.selects(single_cpu_name):
        run_variant(single_cpu_name, default_scope, batch_size(settings, 1), Threads(1))
    if settings.selects(all_cpu_name):
        run_variant(all_cpu_name, cpu_scope, batch_size(settings, cpu_cores), cpu_cores)
    if gpu_scope is not None and settings.selects(gpu_name):
        gpu_cores = gpu_multiprocessor_count(0) or FALLBACK_GPU_MULTIPROCESSORS
        run_variant(gpu_name, gpu_scope, batch_size(settings, gpu_cores), Threads(1))


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
    name = f"minhash-{dimensions}/cudf.minhash<1gpu>"
    if not settings.selects(name):
        return
    try:
        import cupy as cp
    except ImportError:
        note_unavailable(name, "cupy not installed")
        return

    per_width = max(1, dimensions // len(NGRAM_WIDTHS))
    parameters_a = cp.arange(1, per_width + 1, dtype=cp.uint32)
    parameters_b = cp.arange(1, per_width + 1, dtype=cp.uint32)
    series = cudf.Series(documents)

    def kernel(series_slice: Any) -> None:
        for width in NGRAM_WIDTHS:
            series_slice.str.minhash(seed=0, a=parameters_a, b=parameters_b, width=width)

    gpu_cores = gpu_multiprocessor_count(0) or FALLBACK_GPU_MULTIPROCESSORS
    try:
        bench_fingerprint(
            settings,
            name,
            series,
            kernel,
            document_byte_lengths,
            dimensions,
            batch_size(settings, gpu_cores),
            Threads(1),
        )
    except Exception as error:  # cuDF raises its own types for an unusable device or kernel
        note_unavailable(name, f"{type(error).__name__}: {error}")


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    libraries = {
        "StringZilla": f"{sz.__version__} with {sz.__capabilities_str__}",
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
        benchmark_stringzillas(settings, texts, document_byte_lengths, dimensions)
        benchmark_datasketch(settings, dataset.tokens, document_byte_lengths, dimensions)
        if not CUDF_AVAILABLE:
            note_unavailable(f"minhash-{dimensions}/cudf.minhash<1gpu>", "cudf not installed")
        else:
            benchmark_cudf(settings, texts, document_byte_lengths, dimensions)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
