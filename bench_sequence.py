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
  python bench_sequence.py --dataset README.md --tokens lines
  python bench_sequence.py --dataset xlsum.csv --tokens words -k "list.sort"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines python bench_sequence.py
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

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns, name_matches
import stringzilla as sz


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


def run_benches(
    tokens: List[str],
    filter_pattern: Optional[re.Pattern],
):
    print("\n=== Sort Benchmarks ===")

    # Python list.sort
    if name_matches("list.sort", filter_pattern):
        py_list = list(tokens)
        bench_sort_operation("list.sort", lambda: py_list.sort(), len(tokens))

    # StringZilla
    if name_matches("sz.Strs.sorted", filter_pattern):
        sz_strs = sz.Strs(tokens)
        bench_sort_operation("sz.Strs.sorted", lambda: sz_strs.sorted(), len(tokens))

    # Pandas
    if name_matches("pandas.Series.sort_values", filter_pattern):
        s = pd.Series(tokens)
        bench_sort_operation("pandas.Series.sort_values", lambda: s.sort_values(ignore_index=True), len(tokens))

    # PyArrow
    if name_matches("pyarrow.compute.sort_indices", filter_pattern):
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
    if name_matches("polars.Series.sort", filter_pattern):
        ps = pl.Series(tokens)
        bench_sort_operation("polars.Series.sort", lambda: ps.sort(), len(tokens))

    # cuDF GPU (if available)
    if CUDF_AVAILABLE and name_matches("cudf.Series.sort_values", filter_pattern):
        cs = cudf.Series(tokens)
        bench_sort_operation("cudf.Series.sort_values", lambda: cs.sort_values(ignore_index=True), len(tokens))


def bench(
    dataset_path: Optional[str] = None,
    tokens_mode: Optional[str] = None,
    filter_pattern: Optional[re.Pattern] = None,
    dataset_limit: Optional[str] = None,
):
    """Run string sorting benchmarks."""
    dataset = load_dataset(dataset_path, size_limit=dataset_limit)
    tokens = tokenize_dataset(dataset, tokens_mode)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    total_chars = sum(len(token) for token in tokens)
    avg_token_length = total_chars / len(tokens)
    print(f"Dataset: {len(tokens):,} tokens, {total_chars:,} chars, {avg_token_length:.1f} avg token length")
    log_system_info()

    run_benches(tokens, filter_pattern)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark string sorting operations with various libraries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Compile regex pattern
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Run benchmark
    bench(args.dataset, args.tokens, filter_pattern, args.dataset_limit)


if __name__ == "__main__":
    exit(main())
