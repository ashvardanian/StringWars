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
| `std::sort_unstable_by_key`                 |        54.35 M compares/s |      57.70 M compares/s |
| `rayon::par_sort_unstable_by_key` on 1x SPR |        47.08 M compares/s |      50.35 M compares/s |
| `polars::Series::sort`                      |       200.34 M compares/s |      65.44 M compares/s |
| `polars::Series::arg_sort`                  |        25.01 M compares/s |      14.05 M compares/s |
| `arrow::lexsort_to_indices`                 |       122.20 M compares/s |  __84.73 M compares/s__ |
| `stringzilla::argsort_permutation`          |   __213.73 M compares/s__ |      74.64 M compares/s |
|                                             |                           |                         |
| Python                                      |                           |                         |
| `list.sort` on 1x SPR                       |        47.06 M compares/s |      22.36 M compares/s |
| `pandas.Series.sort_values` on 1x SPR       |         9.39 M compares/s |      11.93 M compares/s |
| `pyarrow.compute.sort_indices` on 1x SPR    |        62.17 M compares/s |       5.53 M compares/s |
| `polars.Series.sort` on 1x SPR              |       223.38 M compares/s | __181.60 M compares/s__ |
| `cudf.Series.sort_values` on H100           | __9'463.59 M compares/s__ |      66.44 M compares/s |
| `stringzilla.Strs.sorted` on 1x SPR         |       171.13 M compares/s |      77.88 M compares/s |

---

See [README.md](README.md) for dataset information and replication instructions.
