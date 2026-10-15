"""String-similarity benchmarks in Python, reported in CUPS. Mirrors `similarities/bench.rs`.

StringZilla v6 scores unit-cost Levenshtein only, through `LevenshteinEngine`, so the Needleman-Wunsch
and Smith-Waterman groups keep their BioPython baselines and note the StringZilla rows as unavailable.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group similarities similarities/bench.py
"""

import argparse
import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import editdistance
import edlib
import jellyfish
import Levenshtein as python_levenshtein
import numpy as np
import polyleven
import stringzilla as sz
from Bio import Align
from nltk.metrics.distance import edit_distance as nltk_edit_distance
from rapidfuzz.distance import Levenshtein as rapidfuzz_levenshtein

from stringwars import (
    Bytes,
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
    pass_over,
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

Category = Literal["uniform", "linear", "affine"]
AlignmentMode = Literal["global", "local"]


def crossproduct_side(pairs_per_batch: int, token_count: int) -> int:
    """Square cross-product side for `pairs_per_batch` pairs and `token_count` available tokens.

    A ``side × side`` cross-product holds about `pairs_per_batch` pairs, clamped so the query slice
    ``[0, side)`` and candidate slice ``[side, 2×side)`` are disjoint, i.e. ``2 * side <= token_count``.
    """
    target = max(1, round(pairs_per_batch**0.5))
    max_side = token_count // 2
    return max(1, min(target, max_side))


def crossproduct_metrics(
    query_lengths: np.ndarray,
    candidate_lengths: np.ndarray,
    query_byte_lengths: np.ndarray,
    candidate_byte_lengths: np.ndarray,
    side: int,
) -> tuple[int, Bytes]:
    """True aggregate cells and bytes spanned by a ``side × side`` cross-product.

    Cells = ``sum(query_lengths[:side]) * sum(candidate_lengths[:side])`` — the real number of
    matrix cells the dense cross-product fills. Bytes = the UTF-8 bytes fed to the kernel,
    ``sum(query_byte_lengths[:side]) + sum(candidate_byte_lengths[:side])``. The length arrays are
    either codepoint counts (UTF-8 metric) or byte counts (binary metric).
    """
    sum_query = int(query_lengths[:side].sum())
    sum_candidate = int(candidate_lengths[:side].sum())
    total_cells = sum_query * sum_candidate
    total_bytes = Bytes(int(query_byte_lengths[:side].sum()) + int(candidate_byte_lengths[:side].sum()))
    return total_cells, total_bytes


def measure_pairwise_baseline(
    settings: Settings,
    name: str,
    scalar_function: Callable[[Any, Any], Any],
    queries: Sequence[str],
    candidates: Sequence[str],
    total_cells: int,
    total_bytes: Bytes,
) -> None:
    """One pass scores every query/candidate pair, so the pair mixture cancels exactly."""
    work = MeasureSpec(unit="cups", elements=total_cells, total_bytes=total_bytes)
    measure(settings, name, work, pass_over(scalar_function, queries, candidates))


@dataclass(frozen=True)
class DeviceVariant:
    """A named benchmark variant for one device configuration, with its cross-product side."""

    label: str
    side: int
    concurrency: Threads = Threads(1)
    ordinal: int | None = None


def build_device_variants(settings: Settings, token_count: int) -> list[DeviceVariant]:
    """Single-core, all-cores, and each selected GPU, with per-variant sides.

    Each variant scales ``STRINGWARS_BATCH_PER_CORE`` by its own core count: a CPU core is one core, a
    GPU streaming multiprocessor is one core. The cross-product side is ``round(sqrt(pairs_per_batch))``
    clamped to the available tokens. Mirrors the Rust side derivation.
    """
    cpu_cores = settings.threads
    variants = [
        DeviceVariant("1cpu", crossproduct_side(batch_size(settings, 1), token_count)),
        DeviceVariant(f"{cpu_cores}cpu", crossproduct_side(batch_size(settings, cpu_cores), token_count), cpu_cores),
    ]
    for ordinal in gpu_ordinals(settings):
        if not gpu_require(settings, ordinal):
            continue
        try:
            sz.cuda_capabilities_enabled(ordinal)
        except RuntimeError as error:
            if settings.devices is not None:
                raise SystemExit(f"Requested CUDA device cuda:{ordinal} is unavailable: {error}") from error
            continue
        print(f"- GPU device: cuda:{ordinal}")
        side = crossproduct_side(gpu_batch_size(settings, ordinal), token_count)
        variants.append(DeviceVariant(f"cuda:{ordinal}", side, ordinal=ordinal))
    return variants


def benchmark_levenshtein(
    settings: Settings,
    tokens: Sequence[str],
    device_variants: list[DeviceVariant],
    symbol: str,
    byte_lengths: np.ndarray,
    metric_lengths: np.ndarray,
) -> None:
    """Cross-product rows of `LevenshteinEngine.distances` for one symbol, `bytes` or `runes`.

    Each variant prepares the `[0, side)` queries once and scores the `[side, 2×side)` candidates into
    a preallocated `(side, side)` matrix. The multi-core row runs one engine per pool thread, each holding
    its share of the queries; a GPU row checks its first round against a host engine before timing.
    """
    for variant in device_variants:
        name = f"uniform/stringzilla.LevenshteinEngine.distances<{symbol},{variant.label}>"
        if not settings.selects(name):
            continue
        side = variant.side
        queries = tokens[0:side]
        candidates = sz.Strings(tokens[side : 2 * side])
        cells, total_bytes = crossproduct_metrics(
            metric_lengths, metric_lengths[side:], byte_lengths, byte_lengths[side:], side
        )
        work = MeasureSpec(unit="cups", elements=cells, total_bytes=total_bytes, concurrency=variant.concurrency)

        if variant.ordinal is not None:
            stream = sz.cuda_stream_init(variant.ordinal)
            try:
                benchmark_levenshtein_gpu(settings, name, variant.ordinal, stream, queries, candidates, symbol, work)
            finally:
                sz.cuda_stream_free(stream)
            continue

        threads = variant.concurrency
        chunk = -(-side // threads)
        tasks = []
        for first in range(0, side, chunk):
            share = queries[first : first + chunk]
            engine = sz.LevenshteinEngine(sz.Strings(share), symbol=symbol)
            distances = np.zeros((len(share), side), dtype=np.uintp)
            tasks.append(functools.partial(engine.distances, candidates, distances))
        if threads == 1:
            measure(settings, name, work, tasks[0])
            continue
        configure = functools.partial(sz.cpu_configure_thread, sz.cpu_capabilities_enabled())
        with thread_pool(threads, configure) as run_all:
            measure(settings, name, work, functools.partial(run_all, tasks))


def benchmark_levenshtein_gpu(
    settings: Settings,
    name: str,
    ordinal: int,
    stream: int,
    queries: Sequence[str],
    candidates: sz.Strings,
    symbol: str,
    work: MeasureSpec,
) -> None:
    """One GPU distances row, the candidates copied into the device's tape and the matrix in managed memory."""
    capabilities = sz.cuda_capabilities_enabled(ordinal)
    try:
        engine = sz.LevenshteinEngine(sz.Strings(queries), symbol=symbol, capabilities=capabilities, stream=stream)
    except Exception as error:  # a device that declines these inputs skips its own row
        note_unavailable(name, str(error))
        return
    device_candidates = candidates.copy(capabilities, stream)
    distances = gpu_managed_array((len(queries), len(candidates)), np.uintp)

    def one_round() -> None:
        engine.distances(device_candidates, distances, stream=stream)
        sz.synchronize(capabilities, stream)

    one_round()
    expected = np.zeros((len(queries), len(candidates)), dtype=np.uintp)
    sz.LevenshteinEngine(sz.Strings(queries), symbol=symbol).distances(candidates, expected)
    if not np.array_equal(distances, expected):
        raise SystemExit(f"{name} disagrees with the host engine on its first round")
    measure(settings, name, work, one_round)


def benchmark_edit_distance_baselines(
    settings: Settings,
    tokens: Sequence[str],
    baseline_side: int,
    codepoint_lengths: np.ndarray,
    byte_lengths: np.ndarray,
) -> None:
    """Third-party edit-distance baselines along the single-CPU cross-product diagonal."""

    queries = tokens[0:baseline_side]
    candidates = tokens[baseline_side : 2 * baseline_side]
    query_codepoints = codepoint_lengths[:baseline_side]
    candidate_codepoints = codepoint_lengths[baseline_side : 2 * baseline_side]
    query_bytes = byte_lengths[:baseline_side]
    candidate_bytes = byte_lengths[baseline_side : 2 * baseline_side]

    def run(
        name: str, scalar_function: Callable[[Any, Any], int], length_metric: tuple[np.ndarray, np.ndarray]
    ) -> None:
        name = f"levenshtein/{name}"
        if not settings.selects(name):
            return
        # These baselines score the diagonal pairs, not a cross-product, so the work is
        # summed pairwise: cells are len(query_i) * len(candidate_i) under whichever
        # length metric the library uses, bytes are what is actually fed to the kernel.
        query_lengths, candidate_lengths = length_metric
        total_cells = int((query_lengths * candidate_lengths).sum())
        total_bytes = Bytes(int(query_bytes.sum()) + int(candidate_bytes.sum()))
        # A baseline that declines these inputs skips its own row rather than aborting the suite:
        # `edlib` rejects any pair whose combined alphabet exceeds 256 symbols, as multilingual corpora do.
        try:
            measure_pairwise_baseline(
                settings,
                name,
                scalar_function,
                queries,
                candidates,
                total_cells,
                total_bytes,
            )
        except Exception as baseline_error:
            note_unavailable(name, str(baseline_error))

    codepoint_metric = (query_codepoints, candidate_codepoints)
    byte_metric = (query_bytes, candidate_bytes)

    def edlib_distance(first_string: str, second_string: str) -> int:
        distance: int = edlib.align(first_string, second_string, mode="NW", task="distance")["editDistance"]
        return distance

    run("rapidfuzz.Levenshtein.distance", rapidfuzz_levenshtein.distance, codepoint_metric)
    run("Levenshtein.distance", python_levenshtein.distance, codepoint_metric)
    run("jellyfish.levenshtein_distance", jellyfish.levenshtein_distance, codepoint_metric)
    run("editdistance.eval", editdistance.eval, codepoint_metric)
    run("nltk.edit_distance", nltk_edit_distance, codepoint_metric)
    run("edlib.align", edlib_distance, byte_metric)
    run("polyleven.levenshtein", polyleven.levenshtein, byte_metric)

    # cuDF batched GPU edit distance: it scores a whole batch per call, but exposes no cross-product,
    # so it is benchmarked over the diagonal pairs as a batched array kernel.
    for ordinal in gpu_ordinals(settings):
        if not CUDF_AVAILABLE:
            note_unavailable(f"levenshtein/cudf.edit_distance<cuda:{ordinal}>", "cudf not installed")
            continue
        if not gpu_require(settings, ordinal):
            continue
        batch = gpu_batch_size(settings, ordinal)
        name = f"levenshtein/cudf.edit_distance<cuda:{ordinal},batch={batch}>"
        if settings.selects(name):
            _benchmark_cudf_edit_distance(
                settings,
                name,
                queries,
                candidates,
                query_codepoints,
                candidate_codepoints,
                query_bytes,
                candidate_bytes,
            )


def _benchmark_cudf_edit_distance(
    settings: Settings,
    name: str,
    queries: Sequence[str],
    candidates: Sequence[str],
    query_codepoints: np.ndarray,
    candidate_codepoints: np.ndarray,
    query_bytes: np.ndarray,
    candidate_bytes: np.ndarray,
) -> None:
    """cuDF GPU edit-distance baseline over the diagonal pairs (one batched call per iteration)."""
    query_series = cudf.Series(queries)
    candidate_series = cudf.Series(candidates)
    # Diagonal-pair cells = sum over pairs of (q_i * c_i); cudf scores element-wise pairs, not a matrix.
    diagonal_cells = int((query_codepoints * candidate_codepoints).sum())
    diagonal_bytes = Bytes(int(query_bytes.sum() + candidate_bytes.sum()))

    def compute() -> int:
        results = query_series.str.edit_distance(candidate_series)
        return int(results.to_arrow().to_numpy().sum())

    work = MeasureSpec(unit="cups", elements=diagonal_cells, total_bytes=diagonal_bytes)
    measure(settings, name, work, compute)


def benchmark_biopython_baseline(
    settings: Settings,
    tokens: Sequence[str],
    baseline_side: int,
    byte_lengths: np.ndarray,
    gap_open: int,
    gap_extend: int,
    category: Category,
    mode: AlignmentMode,
) -> None:
    """BioPython PairwiseAligner baseline (global or local) over the cross-product diagonal.

    Uses unary match=+2 / mismatch=-1 scoring, as `bench.rs` does for `bio`. `mode` selects global
    (Needleman-Wunsch) or local (Smith-Waterman) alignment.
    """
    name = f"{category}/biopython.PairwiseAligner.{mode}"
    if not settings.selects(name):
        return

    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    queries = tokens[0:baseline_side]
    candidates = tokens[baseline_side : 2 * baseline_side]
    query_bytes = byte_lengths[:baseline_side]
    candidate_bytes = byte_lengths[baseline_side : 2 * baseline_side]

    total_cells = int((query_bytes * candidate_bytes).sum())
    total_bytes = Bytes(int(query_bytes.sum()) + int(candidate_bytes.sum()))
    measure_pairwise_baseline(settings, name, aligner.score, queries, candidates, total_cells, total_bytes)


def perform_uniform_benchmarks(
    settings: Settings,
    tokens: Sequence[str],
    device_variants: list[DeviceVariant],
    codepoint_lengths: np.ndarray,
    byte_lengths: np.ndarray,
) -> None:
    """Uniform-cost group: classic Levenshtein (match=0, mismatch=1, open=1, extend=1)."""
    benchmark_edit_distance_baselines(settings, tokens, device_variants[0].side, codepoint_lengths, byte_lengths)
    # The binary metric counts byte cells, the rune metric codepoint cells.
    benchmark_levenshtein(settings, tokens, device_variants, "bytes", byte_lengths, byte_lengths)
    benchmark_levenshtein(settings, tokens, device_variants, "runes", byte_lengths, codepoint_lengths)


def perform_score_benchmarks(
    settings: Settings,
    tokens: Sequence[str],
    device_variants: list[DeviceVariant],
    byte_lengths: np.ndarray,
    group_name: Category,
    gap_open: int,
    gap_extend: int,
) -> None:
    """NW/SW score group (linear or affine) with unary match=+2 / mismatch=-1 scoring."""
    for mode in ("global", "local"):
        benchmark_biopython_baseline(
            settings, tokens, device_variants[0].side, byte_lengths, gap_open, gap_extend, group_name, mode
        )
    for engine in ("NeedlemanWunschScores", "SmithWatermanScores"):
        name = f"{group_name}/stringzilla.{engine}"
        if settings.selects(name):
            note_unavailable(name, "StringZilla v6 scores unit-cost Levenshtein only")


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    print_machine({"StringZilla": f"{sz.__version__} with {sz.cpu_capabilities_enabled().name}"})
    settings = read_settings("similarities")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    tokens = dataset.text_tokens()
    log_dataset(dataset)
    log_timing_overhead(settings)

    if len(tokens) < 2:
        raise SystemExit("Dataset must contain at least two tokens for the cross-product")

    token_count = len(tokens)

    # Per-token length metrics, computed once and sliced per variant.
    codepoint_lengths = np.fromiter((len(token) for token in tokens), dtype=np.int64, count=token_count)
    byte_lengths = np.fromiter(map(len, dataset.tokens), dtype=np.int64, count=token_count)

    device_variants = build_device_variants(settings, token_count)

    print("Benchmark configuration (all-pairs cross-product):")
    for variant in device_variants:
        side = variant.side
        print(f"- <{variant.label}>: {side}×{side} cross-product ({side * side:,} pairs)")
    print(f"- Tokens available: {token_count:,}")
    print()

    print("# uniform")
    perform_uniform_benchmarks(settings, tokens, device_variants, codepoint_lengths, byte_lengths)

    print("\n# linear")
    perform_score_benchmarks(settings, tokens, device_variants, byte_lengths, "linear", gap_open=-2, gap_extend=-2)

    print("\n# affine")
    perform_score_benchmarks(settings, tokens, device_variants, byte_lengths, "affine", gap_open=-5, gap_extend=-1)

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
