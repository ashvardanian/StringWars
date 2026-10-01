"""String-similarity benchmarks in Python, reported in CUPS. Mirrors `similarities/bench.rs`.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group similarities similarities/bench.py
"""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from typing import Any, Literal

import editdistance
import edlib
import jellyfish
import Levenshtein as python_levenshtein
import numpy as np
import polyleven
import stringzilla as sz
import stringzillas as szs
from Bio import Align
from nltk.metrics.distance import edit_distance as nltk_edit_distance
from rapidfuzz.distance import Levenshtein as rapidfuzz_levenshtein

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
    pass_over,
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

Category = Literal["uniform", "linear", "affine"]
AlignmentMode = Literal["global", "local"]


def crossproduct_side(pairs_per_batch: int, token_count: int) -> int:
    """Square cross-product side for `pairs_per_batch` pairs and `token_count` available tokens.

    A ``side x side`` cross-product holds about `pairs_per_batch` pairs, clamped so the query slice
    ``[0, side)`` and candidate slice ``[side, 2*side)`` are disjoint, i.e. ``2 * side <= token_count``.
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
    """True aggregate cells and bytes spanned by a ``side x side`` cross-product.

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


# Deliberately unequal lengths: that is the shape a binding without the cross-product call rejects.
_PROBE_QUERIES = sz.Strs(["ab", "cde", "f"])
_PROBE_CANDIDATES = sz.Strs(["gh", "ij"])


def _crossproduct_supported(engine: Any) -> bool:
    """Probe whether the installed binding exposes the queries x candidates cross-product call.

    A supporting binding accepts two disjoint, differently-sized collections and returns a 2-D
    matrix; a binding without the cross-product call raises on unequal lengths. The probe uses a
    tiny mismatched pair so an unsupported binding degrades to a clear SKIP instead of a crash,
    and so the check costs nothing next to the row it guards.
    """
    try:
        probe = engine(_PROBE_QUERIES, _PROBE_CANDIDATES)
    except Exception:
        return False
    return getattr(np.asarray(probe), "ndim", 0) == 2


def measure_crossproduct(
    settings: Settings,
    name: str,
    compute: Callable[[], None],
    total_cells: int,
    total_bytes: Bytes,
    concurrency: Threads,
) -> None:
    """One pass is one full cross-product; the output buffer lives in `compute`'s closure."""
    work = MeasureSpec(unit="cups", elements=total_cells, total_bytes=total_bytes, concurrency=concurrency)
    measure(settings, name, work, compute)


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


def unary_class_costs(match_cost: int, mismatch_cost: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the 32-class substitution table for classic unary scoring (mirrors bench.rs).

    Each byte folds into one of 32 classes via ``byte % 32``, keeping the table compact; the cost is
    `match_cost` on the diagonal and `mismatch_cost` off it. Throughput (CUPS) is invariant to the
    actual cost values, so this stays apples-to-apples with the Rust harness.
    """
    byte_to_class = np.arange(256, dtype=np.uint8) % 32
    class_substitution_costs = np.full((32, 32), mismatch_cost, dtype=np.int8)
    np.fill_diagonal(class_substitution_costs, match_cost)
    return byte_to_class, class_substitution_costs


@dataclass(frozen=True)
class DeviceVariant:
    """A named benchmark variant for one device configuration, with its cross-product side."""

    label: str
    scope: szs.DeviceScope
    side: int
    concurrency: Threads = Threads(1)


def build_device_variants(settings: Settings, token_count: int) -> list[DeviceVariant]:
    """Single-core, all-cores, and (when present) single-GPU variants with per-variant sides.

    Each variant scales ``STRINGWARS_BATCH_PER_CORE`` by its own core count: a CPU core is one core, a
    GPU streaming multiprocessor is one core. The cross-product side is ``round(sqrt(pairs_per_batch))``
    clamped to the available tokens. The GPU variant is included only when a GPU DeviceScope can be
    created. Mirrors the Rust side derivation.
    """
    cpu_cores = settings.threads
    variants: list[DeviceVariant] = []

    variants.append(
        DeviceVariant(
            "<1cpu>",
            szs.DeviceScope(cpu_cores=1),
            crossproduct_side(batch_size(settings, 1), token_count),
        ),
    )
    variants.append(
        DeviceVariant(
            f"<{cpu_cores}cpu>",
            szs.DeviceScope(cpu_cores=cpu_cores),
            crossproduct_side(batch_size(settings, cpu_cores), token_count),
            cpu_cores,
        ),
    )

    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:  # any failure here means there is no usable GPU
        gpu_scope = None
    if gpu_scope is not None:
        gpu_cores = gpu_multiprocessor_count(0) or FALLBACK_GPU_MULTIPROCESSORS
        variants.append(
            DeviceVariant("<1gpu>", gpu_scope, crossproduct_side(batch_size(settings, gpu_cores), token_count))
        )

    return variants


def benchmark_stringzillas_distances(
    settings: Settings,
    tokens: Sequence[str],
    device_variants: list[DeviceVariant],
    category: Category,
    engine_name: str,
    engine_class: Any,
    result_dtype: Any,
    byte_lengths: np.ndarray,
    metric_lengths: np.ndarray,
) -> None:
    """Cross-product benchmark for a StringZilla edit-distance engine (Levenshtein / UTF-8).

    For each device variant the disjoint query slice ``[0, side)`` and candidate slice
    ``[side, 2*side)`` are wrapped as ``sz.Strs`` once, a 2-D output matrix is preallocated, and the
    engine is invoked as ``engine(queries, candidates, device, out=matrix)`` each iteration so the
    matrix is reused.
    """
    for variant in device_variants:
        full_name = f"{category}/{engine_name}{variant.label}"
        if not settings.selects(full_name):
            continue

        side = variant.side
        queries = sz.Strs(tokens[0:side])
        candidates = sz.Strs(tokens[side : 2 * side])

        try:
            engine = engine_class(capabilities=variant.scope)
        except Exception as creation_error:
            note_unavailable(full_name, str(creation_error))
            continue

        if not _crossproduct_supported(engine):
            note_unavailable(full_name, "installed stringzillas lacks the queries x candidates cross-product API")
            continue

        total_cells, total_bytes = crossproduct_metrics(
            metric_lengths,
            metric_lengths[side:],
            byte_lengths,
            byte_lengths[side:],
            side,
        )
        matrix = np.zeros((side, side), dtype=result_dtype)

        def compute(
            engine: Any = engine,
            queries: sz.Strs = queries,
            candidates: sz.Strs = candidates,
            scope: szs.DeviceScope = variant.scope,
            matrix: np.ndarray = matrix,
        ) -> None:
            engine(queries, candidates, scope, out=matrix)

        # Mirror bench.rs: attempt the kernel once; a backend that declines (e.g. no working GPU
        # path in the installed wheel for these inputs) surfaces as an exception we SKIP on rather
        # than aborting the whole suite.
        try:
            compute()
        except Exception as compute_error:
            note_unavailable(full_name, str(compute_error))
            continue

        measure_crossproduct(settings, full_name, compute, total_cells, total_bytes, variant.concurrency)


def benchmark_stringzillas_scores(
    settings: Settings,
    tokens: Sequence[str],
    device_variants: list[DeviceVariant],
    category: Category,
    engine_name: str,
    engine_class: Any,
    byte_to_class: np.ndarray,
    class_substitution_costs: np.ndarray,
    gap_open: int,
    gap_extend: int,
    byte_lengths: np.ndarray,
) -> None:
    """Cross-product benchmark for a StringZilla scoring engine (Needleman-Wunsch / Smith-Waterman).

    Same structure as `benchmark_stringzillas_distances`, but the engine is built from the 32-class
    unary cost model plus affine gap penalties, and the output matrix is int64 (signed scores). The
    throughput denominator uses byte lengths (the binary cells), matching bench.rs.
    """
    for variant in device_variants:
        full_name = f"{category}/{engine_name}{variant.label}"
        if not settings.selects(full_name):
            continue

        side = variant.side
        queries = sz.Strs(tokens[0:side])
        candidates = sz.Strs(tokens[side : 2 * side])

        try:
            engine = engine_class(
                byte_to_class,
                class_substitution_costs,
                open=gap_open,
                extend=gap_extend,
                capabilities=variant.scope,
            )
        except Exception as creation_error:
            note_unavailable(full_name, str(creation_error))
            continue

        if not _crossproduct_supported(engine):
            note_unavailable(full_name, "installed stringzillas lacks the queries x candidates cross-product API")
            continue

        total_cells, total_bytes = crossproduct_metrics(
            byte_lengths,
            byte_lengths[side:],
            byte_lengths,
            byte_lengths[side:],
            side,
        )
        matrix = np.zeros((side, side), dtype=np.int64)

        def compute(
            engine: Any = engine,
            queries: sz.Strs = queries,
            candidates: sz.Strs = candidates,
            scope: szs.DeviceScope = variant.scope,
            matrix: np.ndarray = matrix,
        ) -> None:
            engine(queries, candidates, scope, out=matrix)

        # Mirror bench.rs: attempt the kernel once; a backend that declines (e.g. no working GPU
        # path in the installed wheel for these inputs) surfaces as an exception we SKIP on rather
        # than aborting the whole suite.
        try:
            compute()
        except Exception as compute_error:
            note_unavailable(full_name, str(compute_error))
            continue

        measure_crossproduct(settings, full_name, compute, total_cells, total_bytes, variant.concurrency)


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
        measure_pairwise_baseline(
            settings,
            name,
            scalar_function,
            queries,
            candidates,
            total_cells,
            total_bytes,
        )

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
    if not CUDF_AVAILABLE:
        note_unavailable("levenshtein/cudf.edit_distance<1gpu>", "cudf not installed")
    else:
        gpu_cores = gpu_multiprocessor_count(0) or FALLBACK_GPU_MULTIPROCESSORS
        gpu_batch_size = batch_size(settings, gpu_cores)
        name = f"levenshtein/cudf.edit_distance<1gpu,batch={gpu_batch_size}>"
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

    Uses the same unary match=+2 / mismatch=-1 scoring as the StringZilla score engines so the CUPS
    are comparable. `mode` selects global (Needleman-Wunsch) or local (Smith-Waterman) alignment.
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
    baseline_side = device_variants[0].side

    benchmark_edit_distance_baselines(
        settings,
        tokens,
        baseline_side,
        codepoint_lengths,
        byte_lengths,
    )

    benchmark_stringzillas_distances(
        settings,
        tokens,
        device_variants,
        "uniform",
        "stringzillas.LevenshteinDistances",
        szs.LevenshteinDistances,
        np.uint64,
        byte_lengths,
        byte_lengths,  # binary metric: cells = byte_length product
    )

    benchmark_stringzillas_distances(
        settings,
        tokens,
        device_variants,
        "uniform",
        "stringzillas.LevenshteinDistancesUTF8",
        szs.LevenshteinDistancesUTF8,
        np.uint64,
        byte_lengths,
        codepoint_lengths,  # UTF-8 metric: cells = codepoint_length product
    )


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
    byte_to_class, class_substitution_costs = unary_class_costs(2, -1)

    benchmark_biopython_baseline(
        settings,
        tokens,
        device_variants[0].side,
        byte_lengths,
        gap_open,
        gap_extend,
        group_name,
        "global",
    )
    benchmark_stringzillas_scores(
        settings,
        tokens,
        device_variants,
        group_name,
        "stringzillas.NeedlemanWunschScores",
        szs.NeedlemanWunschScores,
        byte_to_class,
        class_substitution_costs,
        gap_open,
        gap_extend,
        byte_lengths,
    )

    benchmark_biopython_baseline(
        settings,
        tokens,
        device_variants[0].side,
        byte_lengths,
        gap_open,
        gap_extend,
        group_name,
        "local",
    )
    benchmark_stringzillas_scores(
        settings,
        tokens,
        device_variants,
        group_name,
        "stringzillas.SmithWatermanScores",
        szs.SmithWatermanScores,
        byte_to_class,
        class_substitution_costs,
        gap_open,
        gap_extend,
        byte_lengths,
    )


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    print_machine(
        {
            "StringZilla": f"{sz.__version__} with {sz.__capabilities_str__}",
            "StringZillas": pkg_version("stringzillas-cpus"),
        }
    )
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
        print(f"- {variant.label}: {side}x{side} cross-product ({side * side:,} pairs)")
    print(f"- Tokens available: {token_count:,}")
    print()

    print("# uniform")
    perform_uniform_benchmarks(
        settings,
        tokens,
        device_variants,
        codepoint_lengths,
        byte_lengths,
    )

    print("\n# linear")
    perform_score_benchmarks(
        settings,
        tokens,
        device_variants,
        byte_lengths,
        "linear",
        gap_open=-2,
        gap_extend=-2,
    )

    print("\n# affine")
    perform_score_benchmarks(
        settings,
        tokens,
        device_variants,
        byte_lengths,
        "affine",
        gap_open=-5,
        gap_extend=-1,
    )

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
