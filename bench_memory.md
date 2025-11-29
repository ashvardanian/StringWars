# Random Generation and Lookup Table Benchmarks

Benchmarks for random byte generation and lookup table operations across Rust and Python implementations.

## Overview

Some of the most common operations in data processing are random generation and lookup tables.
That's true not only for strings but for any data type, and StringZilla has been extensively used in Image Processing and Bioinformatics for those purposes.

## Random Byte Generation

| Library                        |    Short Words |      Long Lines |
| ------------------------------ | -------------: | --------------: |
| Rust                           |                |                 |
| `getrandom::fill`              |     0.18 GiB/s |      0.45 GiB/s |
| `rand_chacha::ChaCha20Rng`     |     0.62 GiB/s |      1.85 GiB/s |
| `rand_xoshiro::Xoshiro128Plus` |     0.83 GiB/s |      3.85 GiB/s |
| `zeroize::zeroize`             |     0.66 GiB/s |      4.73 GiB/s |
| `stringzilla::fill_random`     | __2.47 GiB/s__ | __10.57 GiB/s__ |
|                                |                |                 |
| Python                         |                |                 |
| `numpy.PCG64`                  |     0.01 GiB/s |      1.28 GiB/s |
| `numpy.Philox`                 |     0.01 GiB/s |      1.59 GiB/s |
| `pycryptodome.AES-CTR`         |     0.01 GiB/s |     13.16 GiB/s |
| `stringzilla.random`           | __0.11 GiB/s__ | __20.37 GiB/s__ |

## Lookup Tables

Performing in-place lookups in a precomputed table of 256 bytes:

| Library                         |    Short Words |     Long Lines |
| ------------------------------- | -------------: | -------------: |
| Rust                            |                |                |
| serial code                     | __0.61 GiB/s__ |     1.49 GiB/s |
| `stringzilla::lookup_inplace`   |     0.54 GiB/s | __9.90 GiB/s__ |
|                                 |                |                |
| Python                          |                |                |
| `bytes.translate`               |     0.05 GiB/s |     1.92 GiB/s |
| `numpy.take`                    |     0.01 GiB/s |     0.85 GiB/s |
| `opencv.LUT`                    |     0.01 GiB/s |     1.95 GiB/s |
| `opencv.LUT` inplace            |     0.01 GiB/s |     2.16 GiB/s |
| `stringzilla.translate`         |     0.07 GiB/s |     7.92 GiB/s |
| `stringzilla.translate` inplace | __0.06 GiB/s__ | __8.14 GiB/s__ |

---

See [README.md](README.md) for dataset information and replication instructions.
