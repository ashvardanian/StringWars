#!/usr/bin/env python3
# /// script
# dependencies = [
#   "stringzilla",
#   "pandas",
#   "pyarrow",
#   "polars",
# ]
# ///
"""
Sort benchmarks in Python: cmp/s for sort operations.

- Sort: list.sort, stringzilla.Strs.sort, pandas, pyarrow, polars, cuDF (optional)

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  uv run bench_sequence.py --dataset README.md --tokens lines
  uv run bench_sequence.py --dataset xlsum.csv --tokens words -k "list.sort"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines uv run bench_sequence.py
"""
import os


import argparse
import math
import re
import sys
from typing import List, Optional


# Assume core deps are present; only cuDF is optional
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
import stringzilla as sz

try:
    import cudf

    CUDF_AVAILABLE = True
except Exception:
    cudf = None  # type: ignore
    CUDF_AVAILABLE = False
else:
    # cuDF sorts run on GPU; nothing to set for CPU threads here
    pass

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns, should_run


def log_system_info():
    """Log Python version and sequence library versions."""

    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- Pandas: {pd.__version__}")
    print(f"- PyArrow: {pa.__version__}")
    print(f"- Polars: {pl.__version__}")
    if CUDF_AVAILABLE:
        print(f"- CuDF: {cudf.__version__}")
    print()  # Add blank line


def bench_sort_operation(name: str, operation: callable, n_items: int):
    """Timing wrapper for sorting operations"""
    start = now_ns()
    result = operation()
    end = now_ns()

    secs = (end - start) / 1e9
    # For sorting operations: estimate comparisons as n*log2(n)
    comparisons = n_items * math.log2(max(n_items, 2))
    cmp_per_sec = comparisons / secs if secs > 0 else 0.0
    gb_per_sec = (n_items * 10) / (1e9 * secs) if secs > 0 else 0.0

    print(f"{name:35s}: {secs:8.3f}s ~ {gb_per_sec:8.3f} GB/s ~ {cmp_per_sec:10,.0f} cmp/s")
    return result


_main_epilog = """
Examples:

  # Benchmark all sorting operations with default settings
  %(prog)s --dataset README.md --tokens lines

  # Test only Python list.sort
  %(prog)s --dataset data.txt --tokens lines -k "list.sort"

  # Compare StringZilla vs other libraries
  %(prog)s --dataset large.txt --tokens words -k "stringzilla.Strs|pandas|polars"

  # GPU-accelerated sorting (if cuDF available)
  %(prog)s --dataset text.txt --tokens lines -k "cudf"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark string sorting operations with various libraries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)

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

    total_chars = sum(len(token) for token in tokens)
    avg_token_length = total_chars / len(tokens)
    print(f"Dataset: {len(tokens):,} tokens, {total_chars:,} chars, {avg_token_length:.1f} avg token length")
    log_system_info()

    print("\n=== Sort Benchmarks ===")

    # Python list.sort
    if should_run("list.sort", filter_pattern):
        py_list = list(tokens)
        bench_sort_operation("list.sort", lambda: py_list.sort(), len(tokens))

    # StringZilla
    if should_run("stringzilla.Strs.sorted", filter_pattern):
        sz_strs = sz.Strs(tokens)
        bench_sort_operation("stringzilla.Strs.sorted", lambda: sz_strs.sorted(), len(tokens))

    # Pandas
    if should_run("pandas.Series.sort_values", filter_pattern):
        s = pd.Series(tokens)
        bench_sort_operation("pandas.Series.sort_values", lambda: s.sort_values(ignore_index=True), len(tokens))

    # PyArrow
    if should_run("pyarrow.compute.sort_indices", filter_pattern):
        # Choose Arrow string type without timing the conversion
        INT32_MAX = 2_147_483_647
        total_bytes = 0
        for s_ in tokens:
            total_bytes += len(s_.encode("utf-8", errors="ignore"))
            if total_bytes > INT32_MAX:
                break
        use_large = total_bytes > INT32_MAX
        arr = pa.array(tokens, type=pa.large_string() if use_large else pa.string())

        def _pa_sort_call():
            idx = pc.sort_indices(arr)
            _ = pc.take(arr, idx)

        bench_sort_operation("pyarrow.compute.sort_indices", _pa_sort_call, len(tokens))

    # Polars
    if should_run("polars.Series.sort", filter_pattern):
        ps = pl.Series(tokens)
        bench_sort_operation("polars.Series.sort", lambda: ps.sort(), len(tokens))

    # cuDF GPU (if available)
    if CUDF_AVAILABLE and should_run("cudf.Series.sort_values", filter_pattern):
        cs = cudf.Series(tokens)
        bench_sort_operation("cudf.Series.sort_values", lambda: cs.sort_values(ignore_index=True), len(tokens))

    return 0


if __name__ == "__main__":
    exit(main())
