#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla",
#   "numpy",
#   "pandas",
#   "pyarrow",
#   "polars",
# ]
# ///
"""
Sort benchmarks in Python: cmp/s for sort operations.

- Sort: list.sort, stringzilla.Strs.sort, numpy, pandas, pyarrow, polars, cuDF (optional)

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  uv run sequence/bench.py --dataset README.md --tokens lines
  uv run sequence/bench.py --dataset xlsum.csv --tokens words -k "list.sort"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines uv run sequence/bench.py
"""

import argparse
import math
import re
import sys
from collections.abc import Callable

# Assume core deps are present; only cuDF is optional
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
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

from utils import (
    add_common_args,
    load_dataset,
    now_nanoseconds,
    report_stats,
    resolve_tokens,
    should_run,
    tokenize_dataset,
)


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


def bench_sort_operation(
    name: str,
    build_input: Callable[[], object],
    sort_input: Callable[[object], object],
    n_items: int,
    time_limit_seconds: float,
):
    """Repeatedly sort the same unsorted data under a wall-clock time budget.

    Each pass rebuilds the input from the original unsorted tokens via `build_input`
    (so every pass sorts identical data and only the sort itself is timed), runs
    `sort_input`, and accumulates the elapsed time and the comparison-count estimate.
    The loop stops once the per-benchmark deadline passes.
    """
    deadline_nanoseconds = now_nanoseconds() + int(time_limit_seconds * 1e9)

    elapsed_nanoseconds = 0
    comparisons = 0.0
    # Per-pass comparison estimate for an n*log2(n) sort, preserved exactly.
    comparisons_per_pass = n_items * math.log2(max(n_items, 2))
    result = None

    while True:
        unsorted_input = build_input()
        start = now_nanoseconds()
        result = sort_input(unsorted_input)
        end = now_nanoseconds()
        elapsed_nanoseconds += end - start
        comparisons += comparisons_per_pass
        if end >= deadline_nanoseconds:
            break

    seconds = elapsed_nanoseconds / 1e9
    report_stats(name, "comparisons", seconds, int(comparisons), 0)
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
    tokens_mode = resolve_tokens(args.tokens, "words")
    tokens = tokenize_dataset(dataset, tokens_mode)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    total_chars = sum(len(token) for token in tokens)
    avg_token_length = total_chars / len(tokens)
    print(f"Dataset: {len(tokens):,} tokens, {total_chars:,} chars, {avg_token_length:.1f} avg token length")
    log_system_info()

    print("\nSort Benchmarks")

    n_items = len(tokens)

    # Python list.sort — mutates in place, so rebuild a fresh copy of the unsorted
    # tokens before every timed pass.
    if should_run("argsort/std.list.sort()", filter_pattern):

        def std_sort(py_list):
            py_list.sort()
            return py_list

        bench_sort_operation("std.list.sort()", lambda: list(tokens), std_sort, n_items, args.time_limit)

    # StringZilla — rebuild the Strs view each pass so every pass sorts identical data.
    if should_run("argsort/stringzilla.Strs.sorted()", filter_pattern):
        bench_sort_operation(
            "stringzilla.Strs.sorted()", lambda: sz.Strs(tokens), lambda strs: strs.sorted(), n_items, args.time_limit
        )

    # NumPy (object-dtype array; the most familiar Python baseline). argsort is
    # non-mutating, so the prebuilt array is the same unsorted input every pass.
    if should_run("argsort/numpy.argsort()", filter_pattern):
        np_array = np.array(tokens, dtype=object)
        bench_sort_operation(
            "numpy.argsort()",
            lambda: np_array,
            lambda array: np.argsort(array, kind="stable"),
            n_items,
            args.time_limit,
        )

    # Pandas (sort_values returns a new Series; the source stays unsorted).
    if should_run("argsort/pandas.Series.sort_values()", filter_pattern):
        s = pd.Series(tokens)
        bench_sort_operation(
            "pandas.Series.sort_values()",
            lambda: s,
            lambda series: series.sort_values(ignore_index=True),
            n_items,
            args.time_limit,
        )

    # PyArrow
    if should_run("argsort/pyarrow.compute.sort_indices()", filter_pattern):
        # Choose Arrow string type without timing the conversion
        INT32_MAX = 2_147_483_647
        total_bytes = 0
        for s_ in tokens:
            total_bytes += len(s_.encode("utf-8", errors="ignore"))
            if total_bytes > INT32_MAX:
                break
        use_large = total_bytes > INT32_MAX
        arr = pa.array(tokens, type=pa.large_string() if use_large else pa.string())

        def pyarrow_sort(array):
            indices = pc.sort_indices(array)
            return pc.take(array, indices)

        bench_sort_operation("pyarrow.compute.sort_indices()", lambda: arr, pyarrow_sort, n_items, args.time_limit)

    # Polars (sort returns a new Series; the source stays unsorted).
    if should_run("argsort/polars.Series.sort()", filter_pattern):
        ps = pl.Series(tokens)
        bench_sort_operation("polars.Series.sort()", lambda: ps, lambda series: series.sort(), n_items, args.time_limit)

    # cuDF GPU (if available; sort_values returns a new Series).
    if CUDF_AVAILABLE and should_run("argsort/cudf.Series.sort_values(1xGPU)", filter_pattern):
        cs = cudf.Series(tokens)
        bench_sort_operation(
            "cudf.Series.sort_values(1xGPU)",
            lambda: cs,
            lambda series: series.sort_values(ignore_index=True),
            n_items,
            args.time_limit,
        )

    return 0


if __name__ == "__main__":
    exit(main())
