# Memory Benchmarks

Benchmarks for random byte generation, lookup-table transforms, and the fill, copy and move primitives, across Rust and Python implementations.

## Overview

Some of the most common operations in data processing are random generation and lookup tables.
That's true not only for strings but for any data type, and StringZilla has been extensively used in Image Processing and Bioinformatics for those purposes.

## Random Byte Generation

### Intel Xeon4 Sapphire Rapids

| Library                        |    Short Words |     Long Lines |
| ------------------------------ | -------------: | -------------: |
| Rust                           |                |                |
| `getrandom::fill`              |      0.03 GB/s |      0.43 GB/s |
| `rand_chacha::ChaCha20Rng`     |      0.06 GB/s |      1.86 GB/s |
| `rand_xoshiro::Xoshiro128Plus` |      0.37 GB/s |      3.75 GB/s |
| `stringzilla::fill_random`     | __0.941 GB/s__ |  __7.99 GB/s__ |
|                                |                |                |
| Python                         |                |                |
| `numpy.PCG64`                  |     0.009 GB/s |      1.74 GB/s |
| `numpy.Philox`                 |     0.009 GB/s |      1.35 GB/s |
| `pycryptodome.AES-CTR`         |     0.009 GB/s |      0.34 GB/s |
| `stringzilla.fill_random`      |              — |              — |
| `stringzilla.random`           |  __0.10 GB/s__ | __17.19 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                        |   Short Words |     Long Lines |
| ------------------------------ | ------------: | -------------: |
| Rust                           |               |                |
| `getrandom::fill`              |             — |      0.22 GB/s |
| `rand_chacha::ChaCha20Rng`     |     0.38 GB/s |      0.83 GB/s |
| `rand_xoshiro::Xoshiro128Plus` |     1.03 GB/s |      5.68 GB/s |
| `stringzilla::fill_random`     | __1.15 GB/s__ | __34.54 GB/s__ |
|                                |               |                |
| Python                         |               |                |
| `numpy.PCG64`                  |     0.03 GB/s |      2.18 GB/s |
| `numpy.Philox`                 |     0.03 GB/s |      2.86 GB/s |
| `pycryptodome.AES-CTR`         |     0.02 GB/s |      0.43 GB/s |
| `stringzilla.fill_random`      |     0.13 GB/s |     24.09 GB/s |
| `stringzilla.random`           | __0.42 GB/s__ | __43.71 GB/s__ |

> Measured July 29, 2026.

## Lookup Tables

Performing in-place lookups in a precomputed table of 256 bytes:

### Intel Xeon4 Sapphire Rapids

| Library                          |   Short Words |     Long Lines |
| -------------------------------- | ------------: | -------------: |
| Rust                             |               |                |
| serial code                      | __0.44 GB/s__ |      3.78 GB/s |
| `stringzilla::lookup_inplace`    |     0.39 GB/s | __9.518 GB/s__ |
|                                  |               |                |
| Python                           |               |                |
| `bytes.translate<new>`           | __0.11 GB/s__ |      2.50 GB/s |
| `numpy.indexing<new>`            |             — |              — |
| `numpy.indexing<inplace>`        |             — |              — |
| `numpy.take<new>`                |    0.009 GB/s |      0.80 GB/s |
| `numpy.take<inplace>`            |             — |              — |
| `opencv.LUT<new>`                |    0.009 GB/s |      1.86 GB/s |
| `opencv.LUT<inplace>`            |    0.009 GB/s |      2.01 GB/s |
| `stringzilla.translate<new>`     |     0.08 GB/s |      7.39 GB/s |
| `stringzilla.translate<inplace>` |     0.07 GB/s |  __7.47 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                          |   Short Words |     Long Lines |
| -------------------------------- | ------------: | -------------: |
| Rust                             |               |                |
| serial code                      | __1.08 GB/s__ |      4.98 GB/s |
| `stringzilla::lookup_inplace`    |     0.79 GB/s | __13.75 GB/s__ |
|                                  |               |                |
| Python                           |               |                |
| `bytes.translate<new>`           | __0.21 GB/s__ |      4.41 GB/s |
| `numpy.indexing<new>`            |     0.03 GB/s |      1.10 GB/s |
| `numpy.indexing<inplace>`        |     0.02 GB/s |      1.05 GB/s |
| `numpy.take<new>`                |     0.02 GB/s |      0.71 GB/s |
| `numpy.take<inplace>`            |     0.02 GB/s |      0.68 GB/s |
| `opencv.LUT<new>`                |     0.02 GB/s |      2.79 GB/s |
| `opencv.LUT<inplace>`            |     0.03 GB/s |      2.93 GB/s |
| `stringzilla.translate<new>`     |     0.18 GB/s |     10.58 GB/s |
| `stringzilla.translate<inplace>` |     0.14 GB/s | __11.25 GB/s__ |

> Measured July 29, 2026.

## Memory Fills

Overwriting every token in place with one constant byte — the `memset` pattern.
`zeroize::zeroize` belongs here rather than with the random generators: it writes zeros, and it does so through volatile stores that the compiler may not widen or elide.

### Apple M5 Pro

| Library                 |   Short Words |     Long Lines |
| ----------------------- | ------------: | -------------: |
| Rust                    |               |                |
| `stringzilla::fill`     |     1.04 GB/s |     38.69 GB/s |
| `std::ptr::write_bytes` |     1.16 GB/s | __48.96 GB/s__ |
| `slice::fill`           |     1.17 GB/s |     48.89 GB/s |
| `zeroize::zeroize`      | __1.18 GB/s__ |      3.97 GB/s |

> Measured July 29, 2026.

## Memory Copies

Copying every token into a matching slice of a second, non-overlapping arena — the `memcpy` pattern.

### Apple M5 Pro

| Library                         |   Short Words |     Long Lines |
| ------------------------------- | ------------: | -------------: |
| Rust                            |               |                |
| `stringzilla::copy`             |     1.07 GB/s |     35.19 GB/s |
| `slice::copy_from_slice`        | __1.16 GB/s__ | __39.36 GB/s__ |
| `std::ptr::copy_nonoverlapping` |    0.997 GB/s |     39.25 GB/s |

> Measured July 29, 2026.

## Memory Moves

Shifting every token 8 bytes forward inside its own buffer, so source and destination overlap — the `memmove` pattern.

### Apple M5 Pro

| Library              |   Short Words |     Long Lines |
| -------------------- | ------------: | -------------: |
| Rust                 |               |                |
| `stringzilla::move_` |     1.02 GB/s |     33.75 GB/s |
| `std::ptr::copy`     |     1.05 GB/s | __40.00 GB/s__ |
| `slice::copy_within` | __1.07 GB/s__ |     39.94 GB/s |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
