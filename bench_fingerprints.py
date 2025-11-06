# /// script
# dependencies = [
#   "stringzilla",
#   "stringzillas-cpus",
#   "datasketch",
#   "numpy",
#   "tqdm",
# ]
# ///
"""
Fingerprinting benchmarks in Python: docs/s for MinHash operations.

- MinHash: datasketch.MinHash, stringzillas.Fingerprints, cudf.minhash_ngrams,
  and cudf.minhash64_ngrams (if RAPIDS cuDF is installed).

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  uv run --with stringzillas-cpus bench_fingerprints.py --dataset README.md --tokens lines
  uv run --with stringzillas-cpus bench_fingerprints.py --dataset xlsum.csv --tokens words -k "datasketch"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines uv run --with stringzillas-cpus bench_fingerprints.py
"""

import os
import argparse
import itertools
import re
import sys
from typing import Callable, Any, List, Optional, Iterable

from tqdm import tqdm
import numpy as np

from datasketch import MinHash
import stringzillas as szs
import stringzilla as sz

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns, should_run

# For RAPIDS cuDF GPU-accelerated MinHash
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

# Fixed n-gram widths for multi-scale fingerprinting (matching Rust benchmark)
NGRAM_WIDTHS = [5, 9, 17, 33]
NGRAM_WIDTHS_ARRAY = np.array(NGRAM_WIDTHS, dtype=np.uint64)


def log_system_info():
    """Log Python version and fingerprinting library versions."""

    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- DataSketch: {MinHash.__module__.split('.')[0]} (available)")
    if CUDF_AVAILABLE:
        print(f"- CuDF: {cudf.__version__}")
    print()  # Add blank line


_main_epilog = """
Examples:

  # Benchmark all algorithms with default settings
  %(prog)s --dataset leipzig1M.txt

  # Benchmark with limited docs and specific dimensions
  %(prog)s --dataset leipzig1M.txt --max-docs 1000 --dimensions 128

  # Test only specific algorithms
  %(prog)s --dataset leipzig1M.txt -k "(datasketch|szs.Fingerprints)"

  # GPU-only benchmarks
  %(prog)s --dataset leipzig1M.txt -k "(cudf|GPU)"

  # High-throughput batch processing
  %(prog)s --dataset leipzig1M.txt --batch-size 1024
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla fingerprinting operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)
    parser.add_argument("-n", "--max-docs", type=int, help="Maximum number of docs to process")
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        default=256,
        help="Number of hash functions for MinHash (default: 256)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for batch-capable APIs (default: 1)",
    )

    args = parser.parse_args()

    # Compile filter pattern
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Load and tokenize dataset
    dataset = load_dataset(args.dataset, size_limit=args.dataset_limit)
    tokens = tokenize_dataset(dataset, args.tokens)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    # Limit number of documents if specified
    if args.max_docs is not None:
        tokens = tokens[: args.max_docs]

    docs_sizes = [len(doc.encode("utf-8")) for doc in tokens]
    total_bytes = sum(docs_sizes)
    avg_doc_length = total_bytes / len(tokens) if tokens else 0

    print(f"Dataset: {len(tokens):,} docs, {total_bytes:,} bytes, {avg_doc_length:.1f} avg doc length")
    log_system_info()

    # Pre-allocated matrices for quality analysis: N_docs × N_dims (engine-specific types)
    total_documents = len(tokens)
    datasketch_matrix = np.zeros((total_documents, args.dimensions), dtype=np.uint32)
    cudf_matrix = np.zeros((total_documents, args.dimensions), dtype=np.uint32)
    szs_matrix = np.zeros((total_documents, args.dimensions), dtype=np.uint32)
    return 1


if __name__ == "__main__":
    exit(main())
