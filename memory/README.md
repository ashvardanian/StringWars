# Memory Benchmarks

Benchmarks for random byte generation, lookup-table transforms, and the fill, copy and move primitives, across Rust and Python implementations.

## Overview

Some of the most common operations in data processing are random generation and lookup tables.
That's true not only for strings but for any data type, and StringZilla has been extensively used in Image Processing and Bioinformatics for those purposes.

The `fill_random` and `random` rows generate random bytes.

The `lookup_inplace`, `lookup_in` and `translate` rows perform lookups in a precomputed table of 256 bytes.
The `StringsView::lookup_in`, `Strings.translate` and Arrow `take` rows map a whole column of tokens in one call, so they pay no per-token dispatch.

The `Strings::copy_in` and `stringzilla.Strings` rows copy every token into a matching slice of a second, non-overlapping arena — the `memcpy` pattern.
Together with the `BinaryArray` and `pyarrow.array` rows they gather a whole column into one tape of bytes and offsets in one call.

The `fill`, `copy` and `move_` rivals (`memset`, `memcpy` and `memmove` patterns, plus `zeroize::zeroize`) are benchmarked as listed, with no StringZilla row yet.

Every table reports single-core throughput in bytes per second, over whitespace-delimited words (_short words_) and newline-delimited lines (_long lines_).
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.
The `serial code` alternative is a plain loop over the same bytes.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `getrandom::fill`, `rand_chacha::ChaCha20Rng`, `std::ptr::write_bytes`, `slice::fill`, `zeroize::zeroize`, `std::ptr::copy_nonoverlapping`, `slice::copy_from_slice`, `std::ptr::copy`, `slice::copy_within`.

| StringZilla                                        |     Speed | Speedup | Alternative                                   |     Speed |
| :------------------------------------------------- | --------: | ------: | :-------------------------------------------- | --------: |
| `stringzilla::fill_random`, short words            | 1.47 GB/s |   2.27× | `rand_xoshiro::Xoshiro128Plus`                |  648 MB/s |
| `stringzilla::fill_random`, long lines             |   10 GB/s |   2.22× | `rand_xoshiro::Xoshiro128Plus`                | 4.51 GB/s |
| `stringzilla::lookup_inplace`, short words         |  356 MB/s |   0.47× | serial code                                   |  751 MB/s |
| `stringzilla::lookup_inplace`, long lines          |   10 GB/s |   5.56× | serial code                                   |  1.8 GB/s |
| `stringzilla::StringsView::lookup_in`, short words |  380 MB/s |   1.00× | `arrow::compute::take`                        |  380 MB/s |
| `stringzilla::StringsView::lookup_in`, long lines  | 2.11 GB/s |   5.49× | `arrow::compute::take`                        |  384 MB/s |
| `stringzilla::Strings::copy_in`, short words       |  510 MB/s |   1.04× | `arrow::array::BinaryArray::from_iter_values` |  490 MB/s |
| `stringzilla::Strings::copy_in`, long lines        | 1.86 GB/s |   2.81× | `arrow::array::BinaryArray::from_iter_values` |  663 MB/s |

> Measured October 14, 2026.

### Python

Also benchmarked: `numpy.Philox`, `pycryptodome.AES-CTR`, `numpy.indexing<new>`, `numpy.indexing<inplace>`, `numpy.take<new>`, `numpy.take<inplace>`, `opencv.LUT<new>`, `opencv.LUT<inplace>`.

| StringZilla                                   |     Speed | Speedup | Alternative            |      Speed |
| :-------------------------------------------- | --------: | ------: | :--------------------- | ---------: |
| `stringzilla.fill_random`, short words        | 98.3 MB/s |   7.56× | `numpy.PCG64`          |    13 MB/s |
| `stringzilla.fill_random`, long lines         |   20 GB/s |   7.69× | `numpy.PCG64`          |   2.6 GB/s |
| `stringzilla.random`, short words             |  376 MB/s |   28.9× | `numpy.PCG64`          |    13 MB/s |
| `stringzilla.random`, long lines              |   36 GB/s |   13.8× | `numpy.PCG64`          |   2.6 GB/s |
| `stringzilla.translate<new>`, short words     | 85.6 MB/s |   0.56× | `bytes.translate<new>` | 153.5 MB/s |
| `stringzilla.translate<new>`, long lines      |  5.5 GB/s |   3.24× | `bytes.translate<new>` |   1.7 GB/s |
| `stringzilla.translate<inplace>`, short words | 75.5 MB/s |   0.49× | `bytes.translate<new>` | 153.5 MB/s |
| `stringzilla.translate<inplace>`, long lines  |  5.9 GB/s |   3.47× | `bytes.translate<new>` |   1.7 GB/s |
| `stringzilla.Strings.translate`, short words  |  1.2 GB/s |   0.35× | `pyarrow.compute.take` |   3.4 GB/s |
| `stringzilla.Strings.translate`, long lines   |  1.7 GB/s |   0.50× | `pyarrow.compute.take` |   3.4 GB/s |
| `stringzilla.Strings`, short words            |  203 MB/s |   0.65× | `pyarrow.array`        |   310 MB/s |
| `stringzilla.Strings`, long lines             | 1.87 GB/s |   0.62× | `pyarrow.array`        |     3 GB/s |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `getrandom::fill`, `rand_chacha::ChaCha20Rng`, `std::ptr::write_bytes`, `slice::fill`, `zeroize::zeroize`, `std::ptr::copy_nonoverlapping`, `slice::copy_from_slice`, `std::ptr::copy`, `slice::copy_within`.

| StringZilla                                |      Speed | Speedup | Alternative                    |      Speed |
| :----------------------------------------- | ---------: | ------: | :----------------------------- | ---------: |
| `stringzilla::fill_random`, short words    |  1.15 GB/s |   1.12× | `rand_xoshiro::Xoshiro128Plus` |  1.03 GB/s |
| `stringzilla::fill_random`, long lines     | 34.54 GB/s |   6.08× | `rand_xoshiro::Xoshiro128Plus` |  5.68 GB/s |
| `stringzilla::lookup_inplace`, short words |  0.79 GB/s |   0.73× | serial code                    |  1.08 GB/s |
| `stringzilla::lookup_inplace`, long lines  | 13.75 GB/s |   2.76× | serial code                    |  4.98 GB/s |

> Measured July 29, 2026.

### Python

Also benchmarked: `pycryptodome.AES-CTR`, `numpy.indexing<new>`, `numpy.indexing<inplace>`, `numpy.take<new>`, `numpy.take<inplace>`, `opencv.LUT<new>`, `opencv.LUT<inplace>`.

| StringZilla                                   |      Speed | Speedup | Alternative            |      Speed |
| :-------------------------------------------- | ---------: | ------: | :--------------------- | ---------: |
| `stringzilla.fill_random`, short words        |  0.13 GB/s |   4.33× | `numpy.PCG64`          |  0.03 GB/s |
| `stringzilla.fill_random`, long lines         | 24.09 GB/s |   8.42× | `numpy.Philox`         |  2.86 GB/s |
| `stringzilla.random`, short words             |  0.42 GB/s |   14.0× | `numpy.PCG64`          |  0.03 GB/s |
| `stringzilla.random`, long lines              | 43.71 GB/s |   15.3× | `numpy.Philox`         |  2.86 GB/s |
| `stringzilla.translate<new>`, short words     |  0.18 GB/s |   0.86× | `bytes.translate<new>` |  0.21 GB/s |
| `stringzilla.translate<new>`, long lines      | 10.58 GB/s |   2.40× | `bytes.translate<new>` |  4.41 GB/s |
| `stringzilla.translate<inplace>`, short words |  0.14 GB/s |   0.67× | `bytes.translate<new>` |  0.21 GB/s |
| `stringzilla.translate<inplace>`, long lines  | 11.25 GB/s |   2.55× | `bytes.translate<new>` |  4.41 GB/s |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
