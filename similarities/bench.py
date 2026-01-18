# /// script
# dependencies = [
#   "stringzilla",
#   "stringzillas-cpus",
#   "rapidfuzz",
#   "python-Levenshtein",
#   "levenshtein",
#   "jellyfish",
#   "editdistance",
#   "distance",
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

Examples:
  uv run --with stringzillas-cpus similarities/bench.py --dataset README.md --max-pairs 1000
  uv run --with stringzillas-cpus similarities/bench.py --dataset xlsum.csv --bio -k "biopython"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines uv run --with stringzillas-cpus similarities/bench.py
"""

import os
import random
import argparse
import re
from typing import List, Callable, Tuple, Optional, Any, Sequence

from tqdm import tqdm
import numpy as np

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns, should_run

# String similarity libraries
import stringzilla as sz
import stringzillas as szs
import jellyfish as jf
import Levenshtein as le
import editdistance as ed
from rapidfuzz.distance import Levenshtein as rf
from nltk.metrics.distance import edit_distance as nltk_ed
import edlib

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


def cells_in_pair(a: str, b: str, is_utf8: bool) -> int:
    # Use codepoints for UTF-8 aware algos; bytes for binary algos
    if is_utf8:
        return len(a) * len(b)
    else:
        return len(a.encode("utf-8")) * len(b.encode("utf-8"))


def log_similarity_operation(
    name: str,
    string_pairs: Tuple[List[str], List[str]],
    similarity_func: Callable,
    cells_per_pair: np.ndarray,
    timeout_seconds: int = 10,
    batch_size: Optional[int] = None,
):
    """Benchmark a similarity operation with timeout and progress tracking.

    Supports batch processing by attempting `similarity_func(list_a, list_b)` when
    `batch_size >= 1`. For single-item batches, a single-item slice is passed, as
    opposed to an individual string.
    """
    # Normalize inputs

    processed_pairs = 0
    processed_cells = 0
    checksum = 0
    start_ns = now_ns()

    def timed_out() -> bool:
        return (now_ns() - start_ns) > int(timeout_seconds * 1e9)

    count_pairs = len(string_pairs[0])
    bar = tqdm(total=count_pairs, desc=name, unit="pairs", leave=False)

    try:
        for first_offset in range(0, count_pairs, batch_size) if batch_size else range(count_pairs):
            if timed_out():
                break

            if batch_size is None:
                try:
                    a, b = string_pairs[0][first_offset], string_pairs[1][first_offset]
                    result = similarity_func(a, b)
                    results = [result]
                    batch_pairs = 1
                    batch_cells = cells_per_pair[first_offset]
                except Exception as e:
                    print(f"\nError at offset {first_offset} (single item):")
                    print(f"  String A (len={len(string_pairs[0][first_offset])}): {string_pairs[0][first_offset]}...")
                    print(f"  String B (len={len(string_pairs[1][first_offset])}): {string_pairs[1][first_offset]}...")
                    print(f"  Error: {type(e).__name__}: {e}")
                    raise
            else:
                batch_pairs = batch_size if (first_offset + batch_size) <= count_pairs else (count_pairs - first_offset)
                try:
                    a_array = string_pairs[0][first_offset : first_offset + batch_pairs]
                    b_array = string_pairs[1][first_offset : first_offset + batch_pairs]
                    results = similarity_func(a_array, b_array)
                    batch_cells = cells_per_pair[first_offset : first_offset + batch_pairs].sum()
                except Exception as e:
                    print(f"\nError at offset {first_offset} (batch_size={batch_size}):")
                    print(f"  Batch range: [{first_offset}:{first_offset + batch_pairs}]")
                    print(f"  First pair: A={string_pairs[0][first_offset]}..., B={string_pairs[1][first_offset]}...")
                    print(f"  Error: {type(e).__name__}: {e}")
                    raise

            # To validate the results, compute a checksum (handles numpy/list/scalar)
            batch_checksum = _checksum_from_results(results)

            checksum += batch_checksum
            processed_pairs += batch_pairs
            processed_cells += batch_cells

            elapsed_s = (now_ns() - start_ns) / 1e9
            if elapsed_s > 0:
                pairs_per_sec = processed_pairs / elapsed_s
                cells_per_sec = processed_cells / elapsed_s
                bar.set_postfix(
                    {
                        "pairs/s": f"{pairs_per_sec:.0f}",
                        "CUPS": f"{cells_per_sec:,.1f}",
                        "checksum": f"{checksum}",
                    }
                )
            bar.update(batch_pairs)
    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")
        return
    finally:
        bar.close()

    total_time_ns = now_ns() - start_ns
    total_time_s = total_time_ns / 1e9
    if processed_pairs > 0:
        pairs_per_sec = processed_pairs / total_time_s
        cells_per_sec = processed_cells / total_time_s
        mcups = cells_per_sec / 1e6  # Convert to MCUPS (Mega Cell Updates Per Second)
        print(
            f"{name:35s}: {total_time_s:8.3f}s ~ {mcups:10,.1f} MCUPS ~ {pairs_per_sec:8,.0f} pairs/s ~ checksum={checksum}"
        )
    else:
        print(f"{name}: No pairs processed")


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
        return int(sum(int(x) for x in results))

    # Fallback: try to coerce directly
    try:
        return int(results)
    except Exception:
        return 0


def benchmark_third_party_edit_distances(
    string_pairs: Tuple[List[str], List[str]],
    timeout_seconds: int = 10,
    filter_pattern: Optional[re.Pattern] = None,
    batch_size: int = 2048,
    cells_per_pair_binary: Optional[np.ndarray] = None,
    cells_per_pair_utf8: Optional[np.ndarray] = None,
):
    """Benchmark various edit distance implementations."""

    # RapidFuzz
    if should_run("levenshtein/rapidfuzz.Levenshtein.distance()", filter_pattern):
        log_similarity_operation(
            "rapidfuzz.Levenshtein.distance()",
            string_pairs,
            rf.distance,
            timeout_seconds=timeout_seconds,
            batch_size=None,  # pass individual strings
            cells_per_pair=cells_per_pair_utf8,  # UTF-8 codepoints
        )

    # RapidFuzz batch API
    # if should_run(f"levenshtein/rapidfuzz.Levenshtein.cpdist(batch={batch_size})", filter_pattern):
    #     log_similarity_operation(
    #         f"rapidfuzz.Levenshtein.cpdist(batch={batch_size})",
    #         string_pairs,
    #         rf.cpdist,
    #     )

    # python-Levenshtein
    if should_run("levenshtein/Levenshtein.distance()", filter_pattern):
        log_similarity_operation(
            "Levenshtein.distance()",
            string_pairs,
            le.distance,
            timeout_seconds=timeout_seconds,
            batch_size=None,  # pass individual strings
            cells_per_pair=cells_per_pair_utf8,  # UTF-8 codepoints
        )

    # Jellyfish
    if should_run("levenshtein/jellyfish.levenshtein_distance()", filter_pattern):
        log_similarity_operation(
            "jellyfish.levenshtein_distance()",
            string_pairs,
            jf.levenshtein_distance,
            timeout_seconds=timeout_seconds,
            batch_size=None,  # pass individual strings
            cells_per_pair=cells_per_pair_utf8,  # UTF-8 codepoints
        )

    # EditDistance
    if should_run("levenshtein/editdistance.eval()", filter_pattern):
        log_similarity_operation(
            "editdistance.eval()",
            string_pairs,
            ed.eval,
            timeout_seconds=timeout_seconds,
            batch_size=None,  # pass individual strings
            cells_per_pair=cells_per_pair_utf8,  # UTF-8 codepoints
        )

    # NLTK
    if should_run("levenshtein/nltk.edit_distance()", filter_pattern):
        log_similarity_operation(
            "nltk.edit_distance()",
            string_pairs,
            nltk_ed,
            timeout_seconds=timeout_seconds,
            batch_size=None,  # pass individual strings
            cells_per_pair=cells_per_pair_utf8,  # UTF-8 codepoints
        )

    # Edlib
    if should_run("levenshtein/edlib.align()", filter_pattern):

        def kernel(a: str, b: str) -> int:
            return edlib.align(a, b, mode="NW", task="distance")["editDistance"]

        log_similarity_operation(
            "edlib.align()",
            string_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=None,  # pass individual strings
            cells_per_pair=cells_per_pair_binary,  # Binary/bytes
        )

    # Polyleven (if available)
    if should_run("levenshtein/polyleven.levenshtein()", filter_pattern) and POLYLEVEN_AVAILABLE:
        log_similarity_operation(
            "polyleven.levenshtein()",
            string_pairs,
            polyleven.levenshtein,
            timeout_seconds=timeout_seconds,
            batch_size=None,  # pass individual strings
            cells_per_pair=cells_per_pair_binary,  # Binary/bytes
        )

    # cuDF edit_distance
    if should_run(f"levenshtein/cudf.edit_distance(1xGPU,batch={batch_size})", filter_pattern) and CUDF_AVAILABLE:

        def batch_kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            results = a_array.str.edit_distance(b_array)
            return results.to_arrow().to_numpy()

        moved_pairs = (cudf.Series(string_pairs[0]), cudf.Series(string_pairs[1]))
        log_similarity_operation(
            f"cudf.edit_distance(1xGPU,batch={batch_size})",
            moved_pairs,
            batch_kernel,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            cells_per_pair=cells_per_pair_utf8,  # UTF-8 codepoints
        )


def benchmark_stringzillas_edit_distances(
    string_pairs: Tuple[List[str], List[str]],
    cells_per_pair: np.ndarray,
    is_utf8: bool,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    filter_pattern: Optional[re.Pattern] = None,
    szs_class: Any = szs.LevenshteinDistances,
    szs_name: str = "stringzillas.LevenshteinDistances",
):
    """Benchmark various edit distance implementations."""

    cpu_cores = os.cpu_count()
    default_scope = szs.DeviceScope()
    cpu_scope = szs.DeviceScope(cpu_cores=cpu_cores)
    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:
        gpu_scope = None

    moved_pairs = (sz.Strs(string_pairs[0]), sz.Strs(string_pairs[1]))

    # Single-input variants on 1 CPU core
    if should_run(f"levenshtein/{szs_name}(1xCPU)", filter_pattern):

        engine = szs_class(capabilities=default_scope)

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, default_scope)

        log_similarity_operation(
            f"{szs_name}(1xCPU)",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            cells_per_pair=cells_per_pair,
        )

    # Single-input variants on all CPU cores
    if should_run(f"levenshtein/{szs_name}({cpu_cores}xCPU)", filter_pattern):

        engine = szs_class(capabilities=cpu_scope)

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, cpu_scope)

        log_similarity_operation(
            f"{szs_name}({cpu_cores}xCPU)",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            cells_per_pair=cells_per_pair,
        )

    # Single-input variants on GPU
    if should_run(f"levenshtein/{szs_name}(1xGPU)", filter_pattern) and not is_utf8 and gpu_scope is not None:

        engine = szs_class(capabilities=gpu_scope)

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, gpu_scope)

        log_similarity_operation(
            f"{szs_name}(1xGPU)",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            cells_per_pair=cells_per_pair,
        )

    # Batch-input variants on 1 CPU core
    if should_run(f"levenshtein/{szs_name}(1xCPU,batch={batch_size})", filter_pattern):

        engine = szs_class(capabilities=default_scope)

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, default_scope)

        log_similarity_operation(
            f"{szs_name}(1xCPU,batch={batch_size})",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            cells_per_pair=cells_per_pair,
        )

    # Batch-input variants on all CPU cores
    if should_run(f"levenshtein/{szs_name}({cpu_cores}xCPU,batch={batch_size})", filter_pattern):

        engine = szs_class(capabilities=cpu_scope)

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, cpu_scope)

        log_similarity_operation(
            f"{szs_name}({cpu_cores}xCPU,batch={batch_size})",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            cells_per_pair=cells_per_pair,
        )

    # Batch-input variants on GPU
    if should_run(f"levenshtein/{szs_name}(1xGPU,batch={batch_size})", filter_pattern) and not is_utf8 and gpu_scope is not None:

        engine = szs_class(capabilities=gpu_scope)

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, gpu_scope)

        log_similarity_operation(
            f"{szs_name}(1xGPU,batch={batch_size})",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            cells_per_pair=cells_per_pair,
        )


def benchmark_third_party_similarity_scores(
    string_pairs: Tuple[List[str], List[str]],
    timeout_seconds: int = 10,
    filter_pattern: Optional[re.Pattern] = None,
    gap_open: int = -10,
    gap_extend: int = -2,
    cells_per_pair: Optional[np.ndarray] = None,
):
    """Benchmark various similarity scoring implementations."""

    # BioPython
    if should_run("needleman-wunsch/biopython.PairwiseAligner.score()", filter_pattern) and BIOPYTHON_AVAILABLE:
        aligner = Align.PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_extend

        log_similarity_operation(
            "biopython.PairwiseAligner.score()",
            string_pairs,
            aligner.score,
            timeout_seconds=timeout_seconds,
            batch_size=None,  # pass individual strings
            cells_per_pair=cells_per_pair,
        )


def benchmark_stringzillas_similarity_scores(
    string_pairs: Tuple[List[str], List[str]],
    timeout_seconds: int = 10,
    batch_size: int = 2048,
    filter_pattern: Optional[re.Pattern] = None,
    szs_class: Any = szs.NeedlemanWunschScores,
    szs_name: str = "stringzillas.NeedlemanWunschScores",
    category: str = "needleman-wunsch",
    gap_open: int = -10,
    gap_extend: int = -2,
    cells_per_pair: Optional[np.ndarray] = None,
):
    """Benchmark various edit distance implementations."""

    cpu_cores = os.cpu_count()
    default_scope = szs.DeviceScope()
    cpu_scope = szs.DeviceScope(cpu_cores=cpu_cores)
    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:
        gpu_scope = None

    # Build BLOSUM matrix if BioPython is available
    blosum = None
    if BIOPYTHON_AVAILABLE:
        aligner = Align.PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

        subs_packed = np.array(aligner.substitution_matrix).astype(np.int8)
        blosum = np.zeros((256, 256), dtype=np.int8)
        blosum.fill(127)  # Large penalty for invalid characters

        for packed_row, packed_row_aminoacid in enumerate(aligner.substitution_matrix.alphabet):
            for packed_column, packed_column_aminoacid in enumerate(aligner.substitution_matrix.alphabet):
                reconstructed_row = ord(packed_row_aminoacid)
                reconstructed_column = ord(packed_column_aminoacid)
                blosum[reconstructed_row, reconstructed_column] = subs_packed[packed_row, packed_column]

    moved_pairs = (sz.Strs(string_pairs[0]), sz.Strs(string_pairs[1]))

    # Single-input variants on 1 CPU core
    if should_run(f"{category}/{szs_name}(1xCPU)", filter_pattern):

        engine = szs_class(
            capabilities=default_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, default_scope)

        log_similarity_operation(
            f"{szs_name}(1xCPU)",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            cells_per_pair=cells_per_pair,
        )

    # Single-input variants on all CPU cores
    if should_run(f"{category}/{szs_name}({cpu_cores}xCPU)", filter_pattern):

        engine = szs_class(
            capabilities=cpu_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, cpu_scope)

        log_similarity_operation(
            f"{szs_name}({cpu_cores}xCPU)",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            cells_per_pair=cells_per_pair,
        )

    # Single-input variants on GPU
    if should_run(f"{category}/{szs_name}(1xGPU)", filter_pattern) and gpu_scope is not None:

        engine = szs_class(
            capabilities=gpu_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, gpu_scope)

        log_similarity_operation(
            f"{szs_name}(1xGPU)",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            cells_per_pair=cells_per_pair,
        )

    # Batch-input variants on 1 CPU core
    if should_run(f"{category}/{szs_name}(1xCPU,batch={batch_size})", filter_pattern):

        engine = szs_class(
            capabilities=default_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, default_scope)

        log_similarity_operation(
            f"{szs_name}(1xCPU,batch={batch_size})",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            cells_per_pair=cells_per_pair,
        )

    # Batch-input variants on all CPU cores
    if should_run(f"{category}/{szs_name}({cpu_cores}xCPU,batch={batch_size})", filter_pattern):

        engine = szs_class(
            capabilities=cpu_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, cpu_scope)

        log_similarity_operation(
            f"{szs_name}({cpu_cores}xCPU,batch={batch_size})",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            cells_per_pair=cells_per_pair,
        )

    # Batch-input variants on GPU
    if should_run(f"{category}/{szs_name}(1xGPU,batch={batch_size})", filter_pattern) and gpu_scope is not None:

        engine = szs_class(
            capabilities=gpu_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a_array: Sequence, b_array: Sequence) -> List[int]:
            return engine(a_array, b_array, gpu_scope)

        log_similarity_operation(
            f"{szs_name}(1xGPU,batch={batch_size})",
            moved_pairs,
            kernel,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            cells_per_pair=cells_per_pair,
        )


def generate_random_pairs(strings: Sequence, num_pairs: int) -> Tuple[List[str], List[str]]:
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
        default=2048,
        help="Number of pairs to process per call in batch-capable APIs (default: 2048)",
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
    strings = tokenize_dataset(dataset, tokens_mode=args.tokens)

    # Generate random pairs
    num_pairs = args.max_pairs or min(100_000, len(strings))
    first_strs = [random.choice(strings) for _ in range(num_pairs)]
    second_strs = [random.choice(strings) for _ in range(num_pairs)]

    total_chars = sum(len(a) + len(b) for a, b in zip(first_strs, second_strs))
    avg_length = total_chars / (2 * num_pairs)
    cells_per_pair_binary = np.array([cells_in_pair(a, b, is_utf8=False) for a, b in zip(first_strs, second_strs)])
    cells_per_pair_utf8 = np.array([cells_in_pair(a, b, is_utf8=True) for a, b in zip(first_strs, second_strs)])

    print(f"Prepared {num_pairs:,} string pairs from {len(strings):,} unique strings")
    print(f"Average string length: {avg_length:.1f} chars")
    print(f"Total characters: {total_chars:,}")
    print(f"Timeout per benchmark: {args.time_limit}s")
    print()

    print("\n=== Uniform Gap Costs ===")
    benchmark_third_party_edit_distances(
        (first_strs, second_strs),
        timeout_seconds=args.time_limit,
        filter_pattern=filter_pattern,
        batch_size=args.batch_size,
        cells_per_pair_binary=cells_per_pair_binary,
        cells_per_pair_utf8=cells_per_pair_utf8,
    )

    benchmark_stringzillas_edit_distances(
        (first_strs, second_strs),
        timeout_seconds=args.time_limit,
        batch_size=args.batch_size,
        filter_pattern=filter_pattern,
        szs_class=szs.LevenshteinDistances,
        szs_name="stringzillas.LevenshteinDistances",
        cells_per_pair=cells_per_pair_binary,
        is_utf8=False,
    )
    benchmark_stringzillas_edit_distances(
        (first_strs, second_strs),
        timeout_seconds=args.time_limit,
        batch_size=args.batch_size,
        filter_pattern=filter_pattern,
        szs_class=szs.LevenshteinDistancesUTF8,
        szs_name="stringzillas.LevenshteinDistancesUTF8",
        cells_per_pair=cells_per_pair_utf8,
        is_utf8=True,
    )

    if args.bio:
        # Linear gap costs (open == extend)
        print("\n=== Linear Gap Costs ===")
        benchmark_third_party_similarity_scores(
            (first_strs, second_strs),
            timeout_seconds=args.time_limit,
            filter_pattern=filter_pattern,
            gap_open=-2,
            gap_extend=-2,
            cells_per_pair=cells_per_pair_binary,
        )
        benchmark_stringzillas_similarity_scores(
            (first_strs, second_strs),
            timeout_seconds=args.time_limit,
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
            (first_strs, second_strs),
            timeout_seconds=args.time_limit,
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
        print("\n=== Affine Gap Costs ===")
        benchmark_third_party_similarity_scores(
            (first_strs, second_strs),
            timeout_seconds=args.time_limit,
            filter_pattern=filter_pattern,
            gap_open=-10,
            gap_extend=-2,
            cells_per_pair=cells_per_pair_binary,
        )
        benchmark_stringzillas_similarity_scores(
            (first_strs, second_strs),
            timeout_seconds=args.time_limit,
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
            (first_strs, second_strs),
            timeout_seconds=args.time_limit,
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
