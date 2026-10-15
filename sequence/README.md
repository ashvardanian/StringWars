# Sequence Operations Benchmarks

Benchmarks for string collection operations like sorting across Rust and Python implementations.

## Overview

Rust has several Dataframe libraries, DBMS and Search engines that heavily rely on string sorting and intersections.
Those operations mostly are implemented using conventional algorithms:

- Comparison-based Quicksort or Mergesort for sorting.
- Hash-based or Tree-based algorithms for intersections.

Assuming the compares can be accelerated with SIMD and so can be the hash functions, StringZilla could already provide a performance boost in such applications, but starting with v4 it also provides specialized algorithms for sorting and intersections.
Those are directly compatible with arbitrary string-comparable collection types with a support of an indexed access to the elements.

The `argsort` and `sorted` rows sort short whitespace-delimited words from `xlsum.csv` on a single core, in plain _byte order_ and in _Unicode case-folded_ order (where `ß` sorts as `ss`).
The tables fuse both index sorts and full sorts: `argsort` / `arg_sort` / `lexsort_to_indices` / `sort_indices` return only the index permutation, while `sort` / `sorted` / `sort_values` return the reordered sequence.
StringZilla's `argsort` writes that permutation into a caller-owned buffer — in Python a NumPy `out=` array — so it allocates nothing per call, and `Strs.sorted` hands back a reordered _view_ over the shared bytes instead of copying the strings.
Every engine builds its container outside the timed region and reuses its output buffer where the API allows.
StringZilla's sort is always stable, so each competitor is configured for a stable sort where it exposes one: NumPy and pandas use `kind="stable"`, Rust's `std` uses the stable `sort_by_key` / `sort_by`, and Polars keeps `maintain_order` on in Rust — its Python `Series.sort` has no stability flag, so that row runs Polars' default order.
For the case-folded column every folding sort shares one comparator, StringZilla's `sz_utf8_uncased_order`, so the gap measures the sort algorithm rather than differing Unicode tables; in Python that comparator is reached through `functools.cmp_to_key`, whose per-element proxy dispatch dominates the `list.sort` row.
Among the rivals, only `std::sort_by_key` and `list.sort` sort in case-folded order, and `cudf.Series.sort_values` sorts on the GPU.

Every table reports single-core throughput in millions of string compares per second.
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `polars::DataFrame::sort`, `arrow::lexsort_to_indices`, `polars::Series::arg_sort`.

| StringZilla                                 |            Speed | Speedup | Alternative            |            Speed |
| :------------------------------------------ | ---------------: | ------: | :--------------------- | ---------------: |
| `stringzilla::argsort`, byte order          | 350 M compares/s |   1.49× | `polars::Series::sort` | 235 M compares/s |
| `stringzilla::argsort`, Unicode case-folded | 200 M compares/s |   5.56× | `std::sort_by_key`     |  36 M compares/s |

> Measured October 14, 2026.

### Python

Also benchmarked: `pyarrow.compute.sort_indices`, `pandas.Series.sort_values`, `polars.Series.arg_sort`, `numpy.argsort`.

| StringZilla                                     |             Speed | Speedup | Alternative          |             Speed |
| :---------------------------------------------- | ----------------: | ------: | :------------------- | ----------------: |
| `stringzilla.Strs.argsort`, byte order          |  340 M compares/s |   1.30× | `polars.Series.sort` |  262 M compares/s |
| `stringzilla.Strs.argsort`, Unicode case-folded |  180 M compares/s |   11.1× | `list.sort`          | 16.2 M compares/s |
| `stringzilla.Strs.sorted`, byte order           |  280 M compares/s |   1.07× | `polars.Series.sort` |  262 M compares/s |
| `stringzilla.Strs.sorted`, Unicode case-folded  |  162 M compares/s |   10.0× | `list.sort`          | 16.2 M compares/s |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `polars::Series::sort`, `polars::DataFrame::sort`, `polars::Series::arg_sort`.

| StringZilla                                 |              Speed | Speedup | Alternative                 |              Speed |
| :------------------------------------------ | -----------------: | ------: | :-------------------------- | -----------------: |
| `stringzilla::argsort`, byte order          | 540.9 M compares/s |   1.65× | `arrow::lexsort_to_indices` | 328.6 M compares/s |
| `stringzilla::argsort`, Unicode case-folded | 232.7 M compares/s |   4.41× | `std::sort_by_key`          | 52.72 M compares/s |

> Measured July 29, 2026.

### Python

Also benchmarked: `pyarrow.compute.sort_indices`, `pandas.Series.sort_values`, `polars.Series.arg_sort`, `numpy.argsort`.

| StringZilla                                     |              Speed | Speedup | Alternative          |              Speed |
| :---------------------------------------------- | -----------------: | ------: | :------------------- | -----------------: |
| `stringzilla.Strs.argsort`, byte order          | 577.8 M compares/s |   2.05× | `polars.Series.sort` | 281.4 M compares/s |
| `stringzilla.Strs.argsort`, Unicode case-folded | 303.4 M compares/s |   13.6× | `list.sort`          | 22.25 M compares/s |
| `stringzilla.Strs.sorted`, byte order           | 537.4 M compares/s |   1.91× | `polars.Series.sort` | 281.4 M compares/s |
| `stringzilla.Strs.sorted`, Unicode case-folded  | 288.6 M compares/s |   13.0× | `list.sort`          | 22.25 M compares/s |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
