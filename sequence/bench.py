"""Sorting benchmarks in Python, reported in comparisons/s. Mirrors `sequence/bench.rs`.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group sequence sequence/bench.py
"""

import argparse
import functools
import math
import os
from collections.abc import Callable

# Must precede `import polars`: Polars sizes its thread pool once, at import. Every other
# engine in this suite sorts on one core, so leaving Polars on all 18 made its rows 5-10x
# faster than the contenders they sit beside.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

# Assume core deps are present; only cuDF is optional
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa
import pyarrow.compute as pc
import stringzilla as sz

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

from stringwars import (
    Bytes,
    MeasureSpec,
    Settings,
    finish,
    log_dataset,
    log_timing_overhead,
    measure_with_setup,
    note_unavailable,
    print_machine,
    print_settings,
    read_settings,
    resolve_dataset,
)


def bench_sort_operation[T](
    settings: Settings,
    name: str,
    build_input: Callable[[], T],
    sort_input: Callable[[T], object],
    token_count: int,
    token_bytes: Bytes,
) -> None:
    """One pass sorts a freshly built copy of the unsorted tokens; the rebuild is untimed."""
    comparisons_per_pass = token_count * math.log2(max(token_count, 2))
    work = MeasureSpec(unit="comparisons", elements=int(comparisons_per_pass), total_bytes=token_bytes)
    measure_with_setup(settings, name, work, build_input, sort_input)


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    libraries = {
        "StringZilla": f"{sz.__version__} with {sz.__capabilities_str__}",
        "Pandas": pd.__version__,
        "PyArrow": pa.__version__,
        "Polars": f"{pl.__version__} on {pl.thread_pool_size()} threads",
    }
    if CUDF_AVAILABLE:
        libraries["cuDF"] = cudf.__version__
    print_machine(libraries)
    settings = read_settings("sequence")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    log_dataset(dataset)
    log_timing_overhead(settings)
    tokens = dataset.text_tokens()

    print("\nSort Benchmarks")

    token_count = len(tokens)
    token_bytes = dataset.token_bytes

    # Python list.sort — mutates in place, so rebuild a fresh copy of the unsorted
    # tokens before every timed pass.

    def std_sort(py_list: list[str]) -> list[str]:
        py_list.sort()
        return py_list

    bench_sort_operation(settings, "argsort/list.sort", lambda: list(tokens), std_sort, token_count, token_bytes)
    # Case-insensitive list.sort driven by StringZilla's own pairwise Unicode case-folding
    # comparator, `sz.utf8_uncased_order`, adapted to a sort key via `functools.cmp_to_key`.
    # CPython's sort takes only a key (no `cmp=`), so the comparator is wrapped per element;
    # holding the folding identical to `stringzilla.Strs.sorted(uncased=True)` isolates the sort
    # algorithm itself (CPython's Timsort vs StringZilla's radix sort).
    uncased_key = functools.cmp_to_key(sz.utf8_uncased_order)

    def std_sort_uncased(py_list: list[str]) -> list[str]:
        py_list.sort(key=uncased_key)
        return py_list

    bench_sort_operation(
        settings, "argsort/list.sort<uncased>", lambda: list(tokens), std_sort_uncased, token_count, token_bytes
    )
    # StringZilla — rebuild the Strs view each pass so every pass sorts identical data.
    bench_sort_operation(
        settings,
        "argsort/stringzilla.Strs.sorted",
        lambda: sz.Strs(tokens),
        lambda strs: strs.sorted(),
        token_count,
        token_bytes,
    )

    # StringZilla case-insensitive sort: orders by Unicode case-folding natively.
    bench_sort_operation(
        settings,
        "argsort/stringzilla.Strs.sorted<uncased>",
        lambda: sz.Strs(tokens),
        lambda strs: strs.sorted(uncased=True),
        token_count,
        token_bytes,
    )

    # StringZilla argsort: writes the index permutation into a caller-owned NumPy buffer
    # (`out=`), so no per-pass allocation — the same zero-copy reuse the other argsort engines
    # get. The buffer holds `sz_sorted_idx_t` indices (pointer-sized unsigned), i.e. `np.uintp`.
    argsort_out = np.empty(token_count, dtype=np.uintp)
    bench_sort_operation(
        settings,
        "argsort/stringzilla.Strs.argsort",
        lambda: sz.Strs(tokens),
        lambda strs: strs.argsort(out=argsort_out),
        token_count,
        token_bytes,
    )
    # StringZilla case-insensitive argsort: index permutation under Unicode case-folding, same
    # caller-owned `out=` buffer so the only difference from the cased row is the comparator.
    argsort_out_uncased = np.empty(token_count, dtype=np.uintp)
    bench_sort_operation(
        settings,
        "argsort/stringzilla.Strs.argsort<uncased>",
        lambda: sz.Strs(tokens),
        lambda strs: strs.argsort(uncased=True, out=argsort_out_uncased),
        token_count,
        token_bytes,
    )
    # NumPy (object-dtype array; the most familiar Python baseline). argsort is
    # non-mutating, so the prebuilt array is the same unsorted input every pass.
    np_array = np.array(tokens, dtype=object)
    bench_sort_operation(
        settings,
        "argsort/numpy.argsort",
        lambda: np_array,
        lambda array: np.argsort(array, kind="stable"),
        token_count,
        token_bytes,
    )
    # Pandas (sort_values returns a new Series; the source stays unsorted). Force a stable sort
    # (`kind="stable"`); the default `quicksort` is unstable, and StringZilla's sort is always
    # stable, so a stable comparator keeps the head-to-head honest.
    s = pd.Series(tokens)
    bench_sort_operation(
        settings,
        "argsort/pandas.Series.sort_values",
        lambda: s,
        lambda series: series.sort_values(ignore_index=True, kind="stable"),
        token_count,
        token_bytes,
    )
    # PyArrow. `string` carries 32-bit offsets, so a tape past that needs `large_string`.
    INT32_MAX = 2_147_483_647
    use_large = dataset.token_bytes > INT32_MAX
    arr = pa.array(tokens, type=pa.large_string() if use_large else pa.string())

    bench_sort_operation(
        settings,
        "argsort/pyarrow.compute.sort_indices",
        lambda: arr,
        lambda array: pc.sort_indices(array),
        token_count,
        token_bytes,
    )
    # Polars argsort: returns an index Series (no materialization).
    ps = pl.Series(tokens)
    bench_sort_operation(
        settings,
        "argsort/polars.Series.arg_sort",
        lambda: ps,
        lambda series: series.arg_sort(),
        token_count,
        token_bytes,
    )
    # Polars full sort (returns a new materialized Series; the source stays unsorted).
    ps = pl.Series(tokens)
    bench_sort_operation(
        settings, "argsort/polars.Series.sort", lambda: ps, lambda series: series.sort(), token_count, token_bytes
    )
    # cuDF GPU (if available; sort_values returns a new Series).
    if not CUDF_AVAILABLE:
        note_unavailable("argsort/cudf.Series.sort_values<1gpu>", "cudf not installed")
    elif settings.selects("argsort/cudf.Series.sort_values<1gpu>"):
        cs = cudf.Series(tokens)
        bench_sort_operation(
            settings,
            "argsort/cudf.Series.sort_values<1gpu>",
            lambda: cs,
            lambda series: series.sort_values(ignore_index=True),
            token_count,
            token_bytes,
        )

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
