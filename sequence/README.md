# Sequence Operations Benchmarks

Benchmarks for string collection operations like sorting across Rust and Python implementations.

## Overview

Rust has several Dataframe libraries, DBMS and Search engines that heavily rely on string sorting and intersections.
Those operations mostly are implemented using conventional algorithms:

- Comparison-based Quicksort or Mergesort for sorting.
- Hash-based or Tree-based algorithms for intersections.

Assuming the compares can be accelerated with SIMD and so can be the hash functions, StringZilla could already provide a performance boost in such applications, but starting with v4 it also provides specialized algorithms for sorting and intersections.
Those are directly compatible with arbitrary string-comparable collection types with a support of an indexed access to the elements.

## String Sorting

| Library                                     |               Short Words |              Long Lines |
| ------------------------------------------- | ------------------------: | ----------------------: |
| Rust                                        |                           |                         |
| `std::sort_unstable_by_key`                 |        71.30 M compares/s |     106.49 M compares/s |
| `rayon::par_sort_unstable_by_key` on 1x SPR |       392.73 M compares/s |     254.81 M compares/s |
| `polars::Series::sort`                      |   __711.06 M compares/s__ | __264.07 M compares/s__ |
| `polars::Series::arg_sort`                  |       223.34 M compares/s |     168.70 M compares/s |
| `arrow::lexsort_to_indices`                 |       112.81 M compares/s |     145.98 M compares/s |
| `stringzilla::argsort_permutation`          |       204.64 M compares/s |     136.18 M compares/s |
|                                             |                           |                         |
| Python                                      |                           |                         |
| `list.sort` on 1x SPR                       |        50.69 M compares/s |      30.27 M compares/s |
| `pandas.Series.sort_values` on 1x SPR       |        58.72 M compares/s |      12.62 M compares/s |
| `pyarrow.compute.sort_indices` on 1x SPR    |        63.04 M compares/s |      13.17 M compares/s |
| `polars.Series.sort` on 1x SPR              |       998.24 M compares/s | __300.14 M compares/s__ |
| `cudf.Series.sort_values` on H100           | __9'463.59 M compares/s__ |      66.44 M compares/s |
| `stringzilla.Strs.sorted` on 1x SPR         |       191.79 M compares/s |      41.33 M compares/s |

> Measured 2026-06-17 on an Intel Xeon Platinum 8468 (Sapphire Rapids).

---

See [README.md](README.md) for dataset information and replication instructions.
