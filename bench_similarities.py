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
  python bench_similarities.py --dataset README.md --max-pairs 1000
  python bench_similarities.py --dataset xlsum.csv --bio -k "biopython"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines python bench_similarities.py
"""

import os
import random
import argparse
import itertools
import re
from typing import List, Callable, Tuple, Optional, Any

from tqdm import tqdm
import numpy as np

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns, name_matches

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


def log_similarity_operation(
    name: str,
    string_pairs: List[Tuple[str, str]],
    similarity_func: Callable,
    timeout_seconds: int = 10,
    batch_size: int = 2048,
    is_utf8: bool = False,
):
    """Benchmark a similarity operation with timeout and progress tracking.

    Supports batch processing by attempting `similarity_func(list_a, list_b)` when
    `batch_size > 1`. If the function does not accept batched inputs, falls back
    to per-pair calls transparently.
    """
    # Normalize inputs
    batch_size = max(1, int(batch_size or 1))

    processed_pairs = 0
    processed_cells = 0
    checksum = 0
    start_ns = now_ns()

    def timed_out() -> bool:
        return (now_ns() - start_ns) > int(timeout_seconds * 1e9)

    def cells_in_pair(a: str, b: str) -> int:
        # Use codepoints for UTF-8 aware algos; bytes for binary algos
        if is_utf8:
            return len(a) * len(b)
        else:
            return len(a.encode("utf-8")) * len(b.encode("utf-8"))

    bar = tqdm(total=len(string_pairs), desc=name, unit="pairs", leave=False)

    try:
        for pairs_batch in itertools.batched(string_pairs, batch_size):
            if timed_out():
                break

            if batch_size == 1:
                a, b = pairs_batch[0]
                result = similarity_func(a, b)
                results = [result]
                batch_pairs = 1
                batch_cells = cells_in_pair(a, b)
            else:
                a_array = np.array([a for a, _ in pairs_batch])
                b_array = np.array([b for _, b in pairs_batch])
                results = similarity_func(a_array, b_array)
                batch_pairs = len(pairs_batch)
                batch_cells = sum(cells_in_pair(a, b) for a, b in pairs_batch)

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
    string_pairs: List[Tuple[str, str]],
    timeout_seconds: int = 10,
    filter_pattern: Optional[re.Pattern] = None,
    batch_size: int = 2048,
):
    """Benchmark various edit distance implementations."""

    # RapidFuzz
    if name_matches("rapidfuzz.Levenshtein.distance", filter_pattern):
        log_similarity_operation(
            "rapidfuzz.Levenshtein.distance",
            string_pairs,
            rf.distance,
            timeout_seconds,
            batch_size=1,  # ? Batch API is different
            is_utf8=True,  # UTF-8 codepoints
        )

    # RapidFuzz batch API
    # if name_matches(f"rapidfuzz.Levenshtein.cpdist(batch={batch_size})", filter_pattern):
    #     log_similarity_operation(
    #         f"rapidfuzz.Levenshtein.cpdist(batch={batch_size})",
    #         string_pairs,
    #         rf.cpdist,
    #         batch_size=timeout_seconds,
    #         is_utf8=batch_size,
    #     )

    # python-Levenshtein
    if name_matches("Levenshtein.distance", filter_pattern):
        log_similarity_operation(
            "Levenshtein.distance",
            string_pairs,
            le.distance,
            timeout_seconds,
            batch_size=1,
            is_utf8=True,  # UTF-8 codepoints
        )

    # Jellyfish
    if name_matches("jellyfish.levenshtein_distance", filter_pattern):
        log_similarity_operation(
            "jellyfish.levenshtein_distance",
            string_pairs,
            jf.levenshtein_distance,
            timeout_seconds,
            batch_size=1,
            is_utf8=True,  # UTF-8 codepoints
        )

    # EditDistance
    if name_matches("editdistance.eval", filter_pattern):
        log_similarity_operation(
            "editdistance.eval",
            string_pairs,
            ed.eval,
            timeout_seconds,
            batch_size=1,
            is_utf8=True,  # UTF-8 codepoints
        )

    # NLTK
    if name_matches("nltk.edit_distance", filter_pattern):
        log_similarity_operation(
            "nltk.edit_distance",
            string_pairs,
            nltk_ed,
            timeout_seconds,
            batch_size=1,
            is_utf8=True,  # UTF-8 codepoints
        )

    # Edlib
    if name_matches("edlib.align", filter_pattern):

        def kernel(a: str, b: str) -> int:
            return edlib.align(a, b, mode="NW", task="distance")["editDistance"]

        log_similarity_operation(
            "edlib.align",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=1,
            is_utf8=False,  # Binary/bytes
        )

    # Polyleven (if available)
    if name_matches("polyleven.levenshtein", filter_pattern) and POLYLEVEN_AVAILABLE:
        log_similarity_operation(
            "polyleven.levenshtein",
            string_pairs,
            polyleven.levenshtein,
            timeout_seconds,
            batch_size=1,
            is_utf8=False,  # Binary/bytes
        )

    # cuDF edit_distance
    if name_matches(f"cudf.edit_distance(batch={batch_size})", filter_pattern) and CUDF_AVAILABLE:

        def batch_kernel(a_list: List[str], b_list: List[str]) -> List[int]:
            # Create cuDF Series from string lists
            s1 = cudf.Series(a_list)
            s2 = cudf.Series(b_list)
            # Compute edit distances and return as list
            results = s1.str.edit_distance(s2)
            return results.to_arrow().to_numpy()

        log_similarity_operation(
            f"cudf.edit_distance(batch={batch_size})",
            string_pairs,
            batch_kernel,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            is_utf8=True,  # UTF-8 codepoints
        )


def benchmark_stringzillas_edit_distances(
    string_pairs: List[Tuple[str, str]],
    timeout_seconds: int = 10,
    batch_size: int = 1,
    filter_pattern: Optional[re.Pattern] = None,
    szs_class: Any = szs.LevenshteinDistances,
    szs_name: str = "stringzillas.LevenshteinDistances",
    is_utf8: bool = False,
):
    """Benchmark various edit distance implementations."""

    cpu_cores = os.cpu_count()
    default_scope = szs.DeviceScope()
    cpu_scope = szs.DeviceScope(cpu_cores=cpu_cores)
    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:
        gpu_scope = None

    # Single-input variants on 1 CPU core
    if name_matches(f"{szs_name}(1xCPU)", filter_pattern):

        engine = szs_class(capabilities=default_scope)

        def kernel(a: str, b: str) -> int:
            a_array = sz.Strs([a])
            b_array = sz.Strs([b])
            return engine(a_array, b_array, default_scope)[0]

        log_similarity_operation(
            f"{szs_name}(1xCPU)",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=1,
            is_utf8=is_utf8,
        )

    # Single-input variants on all CPU cores
    if name_matches(f"{szs_name}({cpu_cores}xCPU)", filter_pattern):

        engine = szs_class(capabilities=cpu_scope)

        def kernel(a: str, b: str) -> int:
            a_array = sz.Strs([a])
            b_array = sz.Strs([b])
            return engine(a_array, b_array, cpu_scope)[0]

        log_similarity_operation(
            f"{szs_name}({cpu_cores}xCPU)",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=1,
            is_utf8=is_utf8,
        )

    # Single-input variants on GPU
    if name_matches(f"{szs_name}(1xGPU)", filter_pattern) and not is_utf8 and gpu_scope is not None:

        engine = szs_class(capabilities=gpu_scope)

        def kernel(a: str, b: str) -> int:
            a_array = sz.Strs([a])
            b_array = sz.Strs([b])
            return engine(a_array, b_array, gpu_scope)[0]

        log_similarity_operation(
            f"{szs_name}(1xGPU)",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=1,
            is_utf8=is_utf8,
        )

    # Batch-input variants on 1 CPU core
    if name_matches(f"{szs_name}(1xCPU,batch={batch_size})", filter_pattern):

        engine = szs_class(capabilities=default_scope)

        def kernel(a_list: List[str], b_list: List[str]) -> List[int]:
            a_array = sz.Strs(a_list)
            b_array = sz.Strs(b_list)
            return engine(a_array, b_array, default_scope)

        log_similarity_operation(
            f"{szs_name}(1xCPU,batch={batch_size})",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=batch_size,
            is_utf8=is_utf8,
        )

    # Batch-input variants on all CPU cores
    if name_matches(f"{szs_name}({cpu_cores}xCPU,batch={batch_size})", filter_pattern):

        engine = szs_class(capabilities=cpu_scope)

        def kernel(a_list: List[str], b_list: List[str]) -> List[int]:
            a_array = sz.Strs(a_list)
            b_array = sz.Strs(b_list)
            return engine(a_array, b_array, cpu_scope)

        log_similarity_operation(
            f"{szs_name}({cpu_cores}xCPU,batch={batch_size})",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=batch_size,
            is_utf8=is_utf8,
        )

    # Batch-input variants on GPU
    if name_matches(f"{szs_name}(1xGPU,batch={batch_size})", filter_pattern) and not is_utf8 and gpu_scope is not None:

        engine = szs_class(capabilities=gpu_scope)

        def kernel(a_list: List[str], b_list: List[str]) -> List[int]:
            a_array = sz.Strs(a_list)
            b_array = sz.Strs(b_list)
            return engine(a_array, b_array, gpu_scope)

        log_similarity_operation(
            f"{szs_name}(1xGPU,batch={batch_size})",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=batch_size,
            is_utf8=is_utf8,
        )


def benchmark_third_party_similarity_scores(
    string_pairs: List[Tuple[str, str]],
    timeout_seconds: int = 10,
    filter_pattern: Optional[re.Pattern] = None,
    gap_open: int = -10,
    gap_extend: int = -2,
):
    """Benchmark various similarity scoring implementations."""

    # BioPython
    if name_matches("biopython.PairwiseAligner.score", filter_pattern) and BIOPYTHON_AVAILABLE:
        aligner = Align.PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_extend

        log_similarity_operation(
            "biopython.PairwiseAligner.score",
            string_pairs,
            aligner.score,
            timeout_seconds,
            1,
            False,
        )


def benchmark_stringzillas_similarity_scores(
    string_pairs: List[Tuple[str, str]],
    timeout_seconds: int = 10,
    batch_size: int = 2048,
    filter_pattern: Optional[re.Pattern] = None,
    szs_class: Any = szs.NeedlemanWunschScores,
    szs_name: str = "stringzillas.NeedlemanWunschScores",
    gap_open: int = -10,
    gap_extend: int = -2,
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

    # Single-input variants on 1 CPU core
    if name_matches(f"{szs_name}(1xCPU)", filter_pattern):

        engine = szs_class(
            capabilities=default_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a: str, b: str) -> int:
            a_array = sz.Strs([a])
            b_array = sz.Strs([b])
            return engine(a_array, b_array, default_scope)[0]

        log_similarity_operation(
            f"{szs_name}(1xCPU)",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=1,
            is_utf8=False,
        )

    # Single-input variants on all CPU cores
    if name_matches(f"{szs_name}({cpu_cores}xCPU)", filter_pattern):

        engine = szs_class(
            capabilities=cpu_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a: str, b: str) -> int:
            a_array = sz.Strs([a])
            b_array = sz.Strs([b])
            return engine(a_array, b_array, cpu_scope)[0]

        log_similarity_operation(
            f"{szs_name}({cpu_cores}xCPU)",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=1,
            is_utf8=False,
        )

    # Single-input variants on GPU
    if name_matches(f"{szs_name}(1xGPU)", filter_pattern) and gpu_scope is not None:

        engine = szs_class(
            capabilities=gpu_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a: str, b: str) -> int:
            a_array = sz.Strs([a])
            b_array = sz.Strs([b])
            return engine(a_array, b_array, gpu_scope)[0]

        log_similarity_operation(
            f"{szs_name}(1xGPU)",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=1,
            is_utf8=False,
        )

    # Batch-input variants on 1 CPU core
    if name_matches(f"{szs_name}(1xCPU,batch={batch_size})", filter_pattern):

        engine = szs_class(
            capabilities=default_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a_list: List[str], b_list: List[str]) -> List[int]:
            a_array = sz.Strs(a_list)
            b_array = sz.Strs(b_list)
            return engine(a_array, b_array, default_scope)

        log_similarity_operation(
            f"{szs_name}(1xCPU,batch={batch_size})",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=batch_size,
            is_utf8=False,
        )

    # Batch-input variants on all CPU cores
    if name_matches(f"{szs_name}({cpu_cores}xCPU,batch={batch_size})", filter_pattern):

        engine = szs_class(
            capabilities=cpu_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a_list: List[str], b_list: List[str]) -> List[int]:
            a_array = sz.Strs(a_list)
            b_array = sz.Strs(b_list)
            return engine(a_array, b_array, cpu_scope)

        log_similarity_operation(
            f"{szs_name}({cpu_cores}xCPU,batch={batch_size})",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=batch_size,
            is_utf8=False,
        )

    # Batch-input variants on GPU
    if name_matches(f"{szs_name}(1xGPU,batch={batch_size})", filter_pattern) and gpu_scope is not None:

        engine = szs_class(
            capabilities=gpu_scope,
            substitution_matrix=blosum,
            open=gap_open,
            extend=gap_extend,
        )

        def kernel(a_list: List[str], b_list: List[str]) -> List[int]:
            a_array = sz.Strs(a_list)
            b_array = sz.Strs(b_list)
            return engine(a_array, b_array, gpu_scope)

        log_similarity_operation(
            f"{szs_name}(1xGPU,batch={batch_size})",
            string_pairs,
            kernel,
            timeout_seconds,
            batch_size=batch_size,
            is_utf8=False,
        )


def generate_random_pairs(strings: List[str], num_pairs: int) -> List[Tuple[str, str]]:
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
    strings = [line.strip() for line in dataset.split("\n") if line.strip()]

    # Generate random pairs
    num_pairs = args.max_pairs or min(100_000, len(strings) * 10)
    pairs = generate_random_pairs(strings, num_pairs)

    total_chars = sum(len(a) + len(b) for a, b in pairs)
    avg_length = total_chars / (2 * len(pairs))

    print(f"Prepared {len(pairs):,} string pairs from {len(strings):,} unique strings")
    print(f"Average string length: {avg_length:.1f} chars")
    print(f"Total characters: {total_chars:,}")
    print(f"Timeout per benchmark: {args.time_limit}s")
    print()

    print("\n=== Uniform Gap Costs ===")
    benchmark_third_party_edit_distances(
        pairs,
        timeout_seconds=args.time_limit,
        filter_pattern=filter_pattern,
        batch_size=args.batch_size,
    )

    benchmark_stringzillas_edit_distances(
        pairs,
        timeout_seconds=args.time_limit,
        batch_size=args.batch_size,
        filter_pattern=filter_pattern,
        szs_class=szs.LevenshteinDistances,
        szs_name="stringzillas.LevenshteinDistances",
        is_utf8=False,
    )
    benchmark_stringzillas_edit_distances(
        pairs,
        timeout_seconds=args.time_limit,
        batch_size=args.batch_size,
        filter_pattern=filter_pattern,
        szs_class=szs.LevenshteinDistancesUTF8,
        szs_name="stringzillas.LevenshteinDistancesUTF8",
        is_utf8=True,
    )

    if args.bio:
        # Linear gap costs (open == extend)
        print("\n=== Linear Gap Costs ===")
        benchmark_third_party_similarity_scores(
            pairs,
            timeout_seconds=args.time_limit,
            filter_pattern=filter_pattern,
            gap_open=-2,
            gap_extend=-2,
        )
        benchmark_stringzillas_similarity_scores(
            pairs,
            timeout_seconds=args.time_limit,
            batch_size=args.batch_size,
            filter_pattern=filter_pattern,
            szs_class=szs.NeedlemanWunschScores,
            szs_name="stringzillas.NeedlemanWunschScores",
            gap_open=-2,
            gap_extend=-2,
        )
        benchmark_stringzillas_similarity_scores(
            pairs,
            timeout_seconds=args.time_limit,
            batch_size=args.batch_size,
            filter_pattern=filter_pattern,
            szs_class=szs.SmithWatermanScores,
            szs_name="stringzillas.SmithWatermanScores",
            gap_open=-2,
            gap_extend=-2,
        )

        # Affine gap costs (open != extend)
        print("\n=== Affine Gap Costs ===")
        benchmark_third_party_similarity_scores(
            pairs,
            timeout_seconds=args.time_limit,
            filter_pattern=filter_pattern,
            gap_open=-10,
            gap_extend=-2,
        )
        benchmark_stringzillas_similarity_scores(
            pairs,
            timeout_seconds=args.time_limit,
            batch_size=args.batch_size,
            filter_pattern=filter_pattern,
            szs_class=szs.NeedlemanWunschScores,
            szs_name="stringzillas.NeedlemanWunschScores",
            gap_open=-10,
            gap_extend=-2,
        )
        benchmark_stringzillas_similarity_scores(
            pairs,
            timeout_seconds=args.time_limit,
            batch_size=args.batch_size,
            filter_pattern=filter_pattern,
            szs_class=szs.SmithWatermanScores,
            szs_name="stringzillas.SmithWatermanScores",
            gap_open=-10,
            gap_extend=-2,
        )

    return 0


if __name__ == "__main__":
    exit(main())
