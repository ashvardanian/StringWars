# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla",
#   "stringzillas-cpus",
#   "rapidfuzz",
#   "python-Levenshtein",
#   "levenshtein",
#   "jellyfish",
#   "editdistance",
#   "polyleven",
#   "edlib",
#   "nltk",
#   "biopython",
#   "numpy",
#   "tqdm",
# ]
# ///
"""
Similarity benchmarks in Python: MCUPS for string similarity operations.

- Edit Distance: rapidfuzz, python-Levenshtein, jellyfish, editdistance, nltk, edlib
- StringZilla: szs.LevenshteinDistances, szs.NeedlemanWunschScores, szs.SmithWatermanScores
- BioPython: PairwiseAligner with BLOSUM62 matrix
- cuDF: GPU-accelerated edit distance (optional)

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')
- STRINGWARS_BATCH_PER_CORE: Items processed per core (default: 256)

The only batch knob is STRINGWARS_BATCH_PER_CORE (items per core); the per-device batch is
auto-derived from the hardware core count — one CPU core is a core, one GPU streaming
multiprocessor (SM) is a core — so each device is fed enough work without manual scaling.
The --batch-size flag overrides STRINGWARS_BATCH_PER_CORE as the per-core base.

Examples:
  uv run --with stringzillas-cpus similarities/bench.py --dataset README.md --max-pairs 1000
  uv run --with stringzillas-cpus similarities/bench.py --dataset xlsum.csv --bio -k "biopython"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines uv run --with stringzillas-cpus similarities/bench.py
"""

import argparse
import os
import random
import re
import sys
from collections.abc import Callable, Sequence
from typing import Any

import editdistance as ed
import edlib
import jellyfish as jf
import Levenshtein as le
import numpy as np

# String similarity libraries
import stringzilla as sz
import stringzillas as szs
from nltk.metrics.distance import edit_distance as nltk_ed
from rapidfuzz.distance import Levenshtein as rf
from tqdm import tqdm

from utils import (
    add_common_args,
    auto_batch_size,
    clamped_subranges,
    gpu_multiprocessor_count,
    load_dataset,
    now_nanoseconds,
    reduce_in_windows,
    report_stats,
    resolve_tokens,
    should_run,
    tokenize_dataset,
)

try:
    import polyleven

    POLYLEVEN_AVAILABLE = True
except ImportError:
    POLYLEVEN_AVAILABLE = False

# For Needleman-Wunsch alignment
try:
    from Bio import Align
    from Bio.Align import substitution_matrices

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# For RAPIDS cuDF GPU-accelerated edit distance
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

# Default per-core batch base for similarity benchmarks (items processed per core).
# 256 is the measured GPU knee for short-word edit distance.
DEFAULT_BATCH_PER_CORE = 256


def _report(
    name: str,
    start_nanoseconds: int,
    processed_pairs: int,
    checksum: int,
    cells_per_pair: np.ndarray,
    bytes_per_pair: np.ndarray,
):
    """Print the throughput summary shared by the pairwise and batched paths."""
    elapsed_seconds = (now_nanoseconds() - start_nanoseconds) / 1e9
    if processed_pairs <= 0:
        print(f"{name}: No pairs processed")
        return
    # Cells and bytes are summed once, over exactly the pairs we got through, instead of
    # being accumulated (and numpy-boxed) on every iteration. The MCUPS metric is cells/sec,
    # so elements = total matrix cells; total_bytes = bytes spanned by the processed pairs.
    processed_cells = int(cells_per_pair[:processed_pairs].sum())
    processed_bytes = int(bytes_per_pair[:processed_pairs].sum())
    report_stats(name, "cups", elapsed_seconds, processed_cells, processed_bytes)
    print(f"  {name} checksum={checksum}", file=sys.stderr)


def bench_pairwise(
    name: str,
    first_strings: Sequence,
    second_strings: Sequence,
    scalar_function: Callable,
    cells_per_pair: np.ndarray,
    bytes_per_pair: np.ndarray,
    time_limit_seconds: int = 10,
):
    """Benchmark a scalar function called once per pair.

    A one-at-a-time benchmark expressed as a reduction: reduce_in_windows sums the
    distances over a window of pairs at a time, so the per-pair calls dispatch in C
    with no Python bytecode in the inner loop.
    """
    count_pairs = len(first_strings)
    deadline_nanoseconds = now_nanoseconds() + int(time_limit_seconds * 1e9)
    start_nanoseconds = now_nanoseconds()
    bar = tqdm(total=count_pairs, desc=name, unit="pairs", leave=False)
    try:
        checksum, processed_pairs = reduce_in_windows(
            scalar_function,
            first_strings,
            second_strings,
            deadline_nanoseconds=deadline_nanoseconds,
            progress=bar,
        )
    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")
        return
    finally:
        bar.close()
    _report(name, start_nanoseconds, processed_pairs, int(checksum), cells_per_pair, bytes_per_pair)


def bench_batched(
    name: str,
    first_strings: Sequence,
    second_strings: Sequence,
    array_kernel: Callable,
    cells_per_pair: np.ndarray,
    bytes_per_pair: np.ndarray,
    batch_size: int,
    time_limit_seconds: int = 10,
):
    """Benchmark an array kernel that scores a whole batch of pairs per call.

    A batched benchmark: the slice is the kernel's real input (a StringZilla Strs or a
    cuDF Series), not an artifact of the harness, so clamped_subranges hands it whole
    batches and the clock is read once per call. A batch size of 1 measures the
    single-pair latency of an array engine.
    """
    count_pairs = len(first_strings)
    deadline_nanoseconds = now_nanoseconds() + int(time_limit_seconds * 1e9)
    checksum = 0
    processed_pairs = 0
    start_nanoseconds = now_nanoseconds()
    bar = tqdm(total=count_pairs, desc=name, unit="pairs", leave=False)
    try:
        for low, high in clamped_subranges(count_pairs, batch_size):
            if now_nanoseconds() > deadline_nanoseconds:
                break
            results = array_kernel(first_strings[low:high], second_strings[low:high])
            checksum += _checksum_from_results(results)
            processed_pairs = high
            bar.update(high - low)
    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")
        return
    finally:
        bar.close()
    _report(name, start_nanoseconds, processed_pairs, checksum, cells_per_pair, bytes_per_pair)


def _checksum_from_results(results: Any) -> int:
    """Convert various result types into an integer checksum cleanly.

    Accepts numpy arrays/scalars, Python scalars, or iterables of numerics.
    """
    # Numpy array: use vectorized sum
    if isinstance(results, np.ndarray):
        return int(results.sum())

    # Numpy scalar
    if isinstance(results, np.generic):
        return int(results)

    # Plain Python scalar
    if isinstance(results, (int, float)):
        return int(results)

    # Iterable of numerics (but not string/bytes)
    if hasattr(results, "__iter__") and not isinstance(results, (str, bytes)):
        return int(sum(int(value) for value in results))

    # Fallback: try to coerce directly
    try:
        return int(results)
    except Exception:
        return 0


def benchmark_third_party_edit_distances(
    string_pairs: tuple[list[str], list[str]],
    bytes_per_pair: np.ndarray,
    batch_size: int | None = None,
    time_limit_seconds: int = 10,
    filter_pattern: re.Pattern | None = None,
    cells_per_pair_binary: np.ndarray | None = None,
    cells_per_pair_utf8: np.ndarray | None = None,
):
    """Benchmark StringZilla NeedlemanWunsch/SmithWaterman score engines across CPU and GPU variants."""

    first_strings, second_strings = string_pairs

    # RapidFuzz
    if should_run("levenshtein/rapidfuzz.Levenshtein.distance", filter_pattern):
        bench_pairwise(
            "rapidfuzz.Levenshtein.distance",
            first_strings,
            second_strings,
            rf.distance,
            cells_per_pair_utf8,  # UTF-8 codepoints
            bytes_per_pair,
            time_limit_seconds=time_limit_seconds,
        )

    # python-Levenshtein
    if should_run("levenshtein/Levenshtein.distance", filter_pattern):
        bench_pairwise(
            "Levenshtein.distance",
            first_strings,
            second_strings,
            le.distance,
            cells_per_pair_utf8,  # UTF-8 codepoints
            bytes_per_pair,
            time_limit_seconds=time_limit_seconds,
        )

    # Jellyfish
    if should_run("levenshtein/jellyfish.levenshtein_distance", filter_pattern):
        bench_pairwise(
            "jellyfish.levenshtein_distance",
            first_strings,
            second_strings,
            jf.levenshtein_distance,
            cells_per_pair_utf8,  # UTF-8 codepoints
            bytes_per_pair,
            time_limit_seconds=time_limit_seconds,
        )

    # EditDistance
    if should_run("levenshtein/editdistance.eval", filter_pattern):
        bench_pairwise(
            "editdistance.eval",
            first_strings,
            second_strings,
            ed.eval,
            cells_per_pair_utf8,  # UTF-8 codepoints
            bytes_per_pair,
            time_limit_seconds=time_limit_seconds,
        )

    # NLTK
    if should_run("levenshtein/nltk.edit_distance", filter_pattern):
        bench_pairwise(
            "nltk.edit_distance",
            first_strings,
            second_strings,
            nltk_ed,
            cells_per_pair_utf8,  # UTF-8 codepoints
            bytes_per_pair,
            time_limit_seconds=time_limit_seconds,
        )

    # Edlib
    if should_run("levenshtein/edlib.align", filter_pattern):

        def edlib_distance(first_string: str, second_string: str) -> int:
            return edlib.align(first_string, second_string, mode="NW", task="distance")["editDistance"]

        bench_pairwise(
            "edlib.align",
            first_strings,
            second_strings,
            edlib_distance,
            cells_per_pair_binary,  # Binary/bytes
            bytes_per_pair,
            time_limit_seconds=time_limit_seconds,
        )

    # Polyleven (if available)
    if should_run("levenshtein/polyleven.levenshtein", filter_pattern) and POLYLEVEN_AVAILABLE:
        bench_pairwise(
            "polyleven.levenshtein",
            first_strings,
            second_strings,
            polyleven.levenshtein,
            cells_per_pair_binary,  # Binary/bytes
            bytes_per_pair,
            time_limit_seconds=time_limit_seconds,
        )

    # cuDF edit_distance
    gpu_batch_size = auto_batch_size(
        gpu_multiprocessor_count(0) or 64, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE
    )
    if should_run(f"levenshtein/cudf.edit_distance<1gpu,batch={gpu_batch_size}>", filter_pattern) and CUDF_AVAILABLE:

        def cudf_kernel(first_slice: Sequence, second_slice: Sequence) -> list[int]:
            results = first_slice.str.edit_distance(second_slice)
            return results.to_arrow().to_numpy()

        bench_batched(
            f"cudf.edit_distance<1gpu,batch={gpu_batch_size}>",
            cudf.Series(first_strings),
            cudf.Series(second_strings),
            cudf_kernel,
            cells_per_pair_utf8,  # UTF-8 codepoints
            bytes_per_pair,
            gpu_batch_size,
            time_limit_seconds=time_limit_seconds,
        )


def benchmark_stringzillas_edit_distances(
    string_pairs: tuple[list[str], list[str]],
    cells_per_pair: np.ndarray,
    bytes_per_pair: np.ndarray,
    is_utf8: bool,
    time_limit_seconds: int = 10,
    batch_size: int | None = None,
    filter_pattern: re.Pattern | None = None,
    szs_class: Any = szs.LevenshteinDistances,
    szs_name: str = "stringzillas.LevenshteinDistances",
):
    """Benchmark StringZilla NeedlemanWunsch/SmithWaterman score engines across CPU and GPU variants."""

    cpu_cores = os.cpu_count()
    default_scope = szs.DeviceScope()
    cpu_scope = szs.DeviceScope(cpu_cores=cpu_cores)
    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:
        gpu_scope = None

    first_strings = sz.Strs(string_pairs[0])
    second_strings = sz.Strs(string_pairs[1])

    one_cpu_batch_size = auto_batch_size(1, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    all_cpu_batch_size = auto_batch_size(cpu_cores, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    gpu_batch_size = auto_batch_size(
        gpu_multiprocessor_count(0) or 64, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE
    )

    def run_variant(label_suffix: str, scope, variant_batch_size: int):
        engine = szs_class(capabilities=scope)

        def kernel(first_slice: Sequence, second_slice: Sequence) -> list[int]:
            return engine(first_slice, second_slice, scope)

        bench_batched(
            f"{szs_name}{label_suffix}",
            first_strings,
            second_strings,
            kernel,
            cells_per_pair,
            bytes_per_pair,
            variant_batch_size,
            time_limit_seconds=time_limit_seconds,
        )

    # Single-pair latency: one native call per pair (batch size 1).
    if should_run(f"levenshtein/{szs_name}<1cpu>", filter_pattern):
        run_variant("<1cpu>", default_scope, 1)
    if should_run(f"levenshtein/{szs_name}<{cpu_cores}cpu>", filter_pattern):
        run_variant(f"<{cpu_cores}cpu>", cpu_scope, 1)
    if should_run(f"levenshtein/{szs_name}<1gpu>", filter_pattern) and not is_utf8 and gpu_scope is not None:
        run_variant("<1gpu>", gpu_scope, 1)

    # Batch throughput: many pairs per native call.
    if should_run(f"levenshtein/{szs_name}<1cpu,batch={one_cpu_batch_size}>", filter_pattern):
        run_variant(f"<1cpu,batch={one_cpu_batch_size}>", default_scope, one_cpu_batch_size)
    if should_run(f"levenshtein/{szs_name}<{cpu_cores}cpu,batch={all_cpu_batch_size}>", filter_pattern):
        run_variant(f"<{cpu_cores}cpu,batch={all_cpu_batch_size}>", cpu_scope, all_cpu_batch_size)
    if (
        should_run(f"levenshtein/{szs_name}<1gpu,batch={gpu_batch_size}>", filter_pattern)
        and not is_utf8
        and gpu_scope is not None
    ):
        run_variant(f"<1gpu,batch={gpu_batch_size}>", gpu_scope, gpu_batch_size)


def benchmark_third_party_similarity_scores(
    string_pairs: tuple[list[str], list[str]],
    bytes_per_pair: np.ndarray,
    time_limit_seconds: int = 10,
    filter_pattern: re.Pattern | None = None,
    gap_open: int = -10,
    gap_extend: int = -2,
    cells_per_pair: np.ndarray | None = None,
):
    """Benchmark various similarity scoring implementations."""

    # BioPython
    if should_run("needleman-wunsch/biopython.PairwiseAligner.score", filter_pattern) and BIOPYTHON_AVAILABLE:
        aligner = Align.PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_extend

        bench_pairwise(
            "biopython.PairwiseAligner.score",
            string_pairs[0],
            string_pairs[1],
            aligner.score,
            cells_per_pair,
            bytes_per_pair,
            time_limit_seconds=time_limit_seconds,
        )


def benchmark_stringzillas_similarity_scores(
    string_pairs: tuple[list[str], list[str]],
    bytes_per_pair: np.ndarray,
    time_limit_seconds: int = 10,
    batch_size: int | None = None,
    filter_pattern: re.Pattern | None = None,
    szs_class: Any = szs.NeedlemanWunschScores,
    szs_name: str = "stringzillas.NeedlemanWunschScores",
    category: str = "needleman-wunsch",
    gap_open: int = -10,
    gap_extend: int = -2,
    cells_per_pair: np.ndarray | None = None,
):
    """Benchmark StringZilla NeedlemanWunsch/SmithWaterman score engines across CPU and GPU variants."""

    cpu_cores = os.cpu_count()
    default_scope = szs.DeviceScope()
    cpu_scope = szs.DeviceScope(cpu_cores=cpu_cores)
    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:
        gpu_scope = None

    # StringZilla v5 scores alignments from a 256-byte -> class map plus a 32x32 cost matrix
    # between classes (max 32 classes). With BioPython we mirror BLOSUM62 — each amino-acid
    # letter is its own class, every other byte falls into a shared "other" class — so the
    # comparison against `biopython.PairwiseAligner` stays apples-to-apples. Without BioPython
    # we fall back to unary match=+2 / mismatch=-1 scoring (matching the Rust bench).
    if BIOPYTHON_AVAILABLE:
        aligner = Align.PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        subs_packed = np.array(aligner.substitution_matrix).astype(np.int8)
        alphabet = aligner.substitution_matrix.alphabet
        num_classes = len(alphabet)
        byte_to_class = np.full(256, num_classes, dtype=np.uint8)
        for class_index, letter in enumerate(alphabet):
            byte_to_class[ord(letter)] = class_index
        class_substitution_costs = np.full((32, 32), 127, dtype=np.int8)
        class_substitution_costs[:num_classes, :num_classes] = subs_packed[:num_classes, :num_classes]
    else:
        byte_to_class = np.array([byte % 32 for byte in range(256)], dtype=np.uint8)
        class_substitution_costs = np.full((32, 32), -1, dtype=np.int8)
        np.fill_diagonal(class_substitution_costs, 2)

    first_strings = sz.Strs(string_pairs[0])
    second_strings = sz.Strs(string_pairs[1])

    one_cpu_batch_size = auto_batch_size(1, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    all_cpu_batch_size = auto_batch_size(cpu_cores, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    gpu_batch_size = auto_batch_size(
        gpu_multiprocessor_count(0) or 64, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE
    )

    def run_variant(label_suffix: str, scope, variant_batch_size: int):
        engine = szs_class(
            byte_to_class,
            class_substitution_costs,
            open=gap_open,
            extend=gap_extend,
            capabilities=scope,
        )

        def kernel(first_slice: Sequence, second_slice: Sequence) -> list[int]:
            return engine(first_slice, second_slice, scope)

        bench_batched(
            f"{szs_name}{label_suffix}",
            first_strings,
            second_strings,
            kernel,
            cells_per_pair,
            bytes_per_pair,
            variant_batch_size,
            time_limit_seconds=time_limit_seconds,
        )

    # Single-pair latency: one native call per pair (batch size 1).
    if should_run(f"{category}/{szs_name}<1cpu>", filter_pattern):
        run_variant("<1cpu>", default_scope, 1)
    if should_run(f"{category}/{szs_name}<{cpu_cores}cpu>", filter_pattern):
        run_variant(f"<{cpu_cores}cpu>", cpu_scope, 1)
    if should_run(f"{category}/{szs_name}<1gpu>", filter_pattern) and gpu_scope is not None:
        run_variant("<1gpu>", gpu_scope, 1)

    # Batch throughput: many pairs per native call.
    if should_run(f"{category}/{szs_name}<1cpu,batch={one_cpu_batch_size}>", filter_pattern):
        run_variant(f"<1cpu,batch={one_cpu_batch_size}>", default_scope, one_cpu_batch_size)
    if should_run(f"{category}/{szs_name}<{cpu_cores}cpu,batch={all_cpu_batch_size}>", filter_pattern):
        run_variant(f"<{cpu_cores}cpu,batch={all_cpu_batch_size}>", cpu_scope, all_cpu_batch_size)
    if should_run(f"{category}/{szs_name}<1gpu,batch={gpu_batch_size}>", filter_pattern) and gpu_scope is not None:
        run_variant(f"<1gpu,batch={gpu_batch_size}>", gpu_scope, gpu_batch_size)


def generate_random_pairs(strings: Sequence, num_pairs: int) -> tuple[list[str], list[str]]:
    """Generate random string pairs from a list of strings."""
    return [(random.choice(strings), random.choice(strings)) for _ in range(num_pairs)]


_main_epilog = """
Examples:

  # Benchmark with a file
  %(prog)s --dataset leipzig1M.txt

  # Benchmark with limited pairs
  %(prog)s --dataset leipzig1M.txt --max-pairs 1000

  # Benchmark protein sequences with BioPython
  %(prog)s --bio --dataset acgt_1k.txt

  # Custom time limit
  %(prog)s --dataset leipzig1M.txt --time-limit 30
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla similarity operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)
    parser.add_argument(
        "-n",
        "--max-pairs",
        type=int,
        help="Maximum number of string pairs to process",
    )
    parser.add_argument(
        "--bio",
        action="store_true",
        help="Include BioPython + SW/NW alignment benchmarks (protein alphabet)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Items processed per core (overrides STRINGWARS_BATCH_PER_CORE, default: 256)",
    )

    args = parser.parse_args()

    if not args.dataset and not os.environ.get("STRINGWARS_DATASET"):
        parser.error("Dataset is required (use --dataset or STRINGWARS_DATASET env var)")

    # Compile filter pattern
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Load dataset and generate pairs
    dataset = load_dataset(args.dataset, size_limit=args.dataset_limit)
    strings = tokenize_dataset(dataset, tokens_mode=resolve_tokens(args.tokens, "words"))

    # Generate random pairs
    num_pairs = args.max_pairs or min(100_000, len(strings))
    first_strings = [random.choice(strings) for _ in range(num_pairs)]
    second_strings = [random.choice(strings) for _ in range(num_pairs)]

    total_chars = sum(len(first) + len(second) for first, second in zip(first_strings, second_strings, strict=True))
    avg_length = total_chars / (2 * num_pairs)
    # Cells per pair = product of the two lengths. Encode each string exactly once
    # (rather than twice per pair) to get the byte lengths for the binary algorithms.
    first_codepoints = np.fromiter((len(s) for s in first_strings), dtype=np.int64, count=num_pairs)
    second_codepoints = np.fromiter((len(s) for s in second_strings), dtype=np.int64, count=num_pairs)
    first_bytes = np.fromiter((len(s.encode("utf-8")) for s in first_strings), dtype=np.int64, count=num_pairs)
    second_bytes = np.fromiter((len(s.encode("utf-8")) for s in second_strings), dtype=np.int64, count=num_pairs)
    cells_per_pair_binary = first_bytes * second_bytes
    cells_per_pair_utf8 = first_codepoints * second_codepoints
    # Bytes spanned by each pair: the data fed to the kernel for that pair.
    bytes_per_pair = first_bytes + second_bytes

    print(f"Prepared {num_pairs:,} string pairs from {len(strings):,} unique strings")
    print(f"Average string length: {avg_length:.1f} chars")
    print(f"Total characters: {total_chars:,}")
    print(f"Time limit per benchmark: {args.time_limit}s")
    print()

    print("\nUniform Gap Costs")
    benchmark_third_party_edit_distances(
        (first_strings, second_strings),
        bytes_per_pair,
        time_limit_seconds=args.time_limit,
        filter_pattern=filter_pattern,
        batch_size=args.batch_size,
        cells_per_pair_binary=cells_per_pair_binary,
        cells_per_pair_utf8=cells_per_pair_utf8,
    )

    benchmark_stringzillas_edit_distances(
        (first_strings, second_strings),
        cells_per_pair_binary,
        bytes_per_pair,
        is_utf8=False,
        time_limit_seconds=args.time_limit,
        batch_size=args.batch_size,
        filter_pattern=filter_pattern,
        szs_class=szs.LevenshteinDistances,
        szs_name="stringzillas.LevenshteinDistances",
    )
    benchmark_stringzillas_edit_distances(
        (first_strings, second_strings),
        cells_per_pair_utf8,
        bytes_per_pair,
        is_utf8=True,
        time_limit_seconds=args.time_limit,
        batch_size=args.batch_size,
        filter_pattern=filter_pattern,
        szs_class=szs.LevenshteinDistancesUTF8,
        szs_name="stringzillas.LevenshteinDistancesUTF8",
    )

    if args.bio:
        # Linear gap costs (open == extend)
        print("\nLinear Gap Costs")
        benchmark_third_party_similarity_scores(
            (first_strings, second_strings),
            bytes_per_pair,
            time_limit_seconds=args.time_limit,
            filter_pattern=filter_pattern,
            gap_open=-2,
            gap_extend=-2,
            cells_per_pair=cells_per_pair_binary,
        )
        benchmark_stringzillas_similarity_scores(
            (first_strings, second_strings),
            bytes_per_pair,
            time_limit_seconds=args.time_limit,
            batch_size=args.batch_size,
            filter_pattern=filter_pattern,
            szs_class=szs.NeedlemanWunschScores,
            szs_name="stringzillas.NeedlemanWunschScores",
            category="needleman-wunsch",
            gap_open=-2,
            gap_extend=-2,
            cells_per_pair=cells_per_pair_binary,
        )
        benchmark_stringzillas_similarity_scores(
            (first_strings, second_strings),
            bytes_per_pair,
            time_limit_seconds=args.time_limit,
            batch_size=args.batch_size,
            filter_pattern=filter_pattern,
            szs_class=szs.SmithWatermanScores,
            szs_name="stringzillas.SmithWatermanScores",
            category="smith-waterman",
            gap_open=-2,
            gap_extend=-2,
            cells_per_pair=cells_per_pair_binary,
        )

        # Affine gap costs (open != extend)
        print("\nAffine Gap Costs")
        benchmark_third_party_similarity_scores(
            (first_strings, second_strings),
            bytes_per_pair,
            time_limit_seconds=args.time_limit,
            filter_pattern=filter_pattern,
            gap_open=-10,
            gap_extend=-2,
            cells_per_pair=cells_per_pair_binary,
        )
        benchmark_stringzillas_similarity_scores(
            (first_strings, second_strings),
            bytes_per_pair,
            time_limit_seconds=args.time_limit,
            batch_size=args.batch_size,
            filter_pattern=filter_pattern,
            szs_class=szs.NeedlemanWunschScores,
            szs_name="stringzillas.NeedlemanWunschScores",
            category="needleman-wunsch",
            gap_open=-10,
            gap_extend=-2,
            cells_per_pair=cells_per_pair_binary,
        )
        benchmark_stringzillas_similarity_scores(
            (first_strings, second_strings),
            bytes_per_pair,
            time_limit_seconds=args.time_limit,
            batch_size=args.batch_size,
            filter_pattern=filter_pattern,
            szs_class=szs.SmithWatermanScores,
            szs_name="stringzillas.SmithWatermanScores",
            category="smith-waterman",
            gap_open=-10,
            gap_extend=-2,
            cells_per_pair=cells_per_pair_binary,
        )

    return 0


if __name__ == "__main__":
    exit(main())
