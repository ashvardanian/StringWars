# Random Generation and Lookup Table Benchmarks

Benchmarks for random byte generation and lookup table operations across Rust and Python implementations.

## Overview

Some of the most common operations in data processing are random generation and lookup tables.
That's true not only for strings but for any data type, and StringZilla has been extensively used in Image Processing and Bioinformatics for those purposes.

## Random Byte Generation

| Library                        |   Short Words |     Long Lines |
| ------------------------------ | ------------: | -------------: |
| Rust                           |               |                |
| `getrandom::fill`              |     0.18 GB/s |      0.45 GB/s |
| `rand_chacha::ChaCha20Rng`     |     0.62 GB/s |      1.85 GB/s |
| `rand_xoshiro::Xoshiro128Plus` |     0.83 GB/s |      3.85 GB/s |
| `zeroize::zeroize`             |     0.66 GB/s |      4.73 GB/s |
| `stringzilla::fill_random`     | **2.47 GB/s** | **10.57 GB/s** |
|                                |               |                |
| Python                         |               |                |
| `numpy.PCG64`                  |     0.01 GB/s |      1.28 GB/s |
| `numpy.Philox`                 |     0.01 GB/s |      1.59 GB/s |
| `pycryptodome.AES-CTR`         |     0.01 GB/s |     13.16 GB/s |
| `stringzilla.random`           | **0.11 GB/s** | **20.37 GB/s** |

## Lookup Tables

Performing in-place lookups in a precomputed table of 256 bytes:

| Library                         |   Short Words |    Long Lines |
| ------------------------------- | ------------: | ------------: |
| Rust                            |               |               |
| serial code                     | **0.61 GB/s** |     1.49 GB/s |
| `stringzilla::lookup_inplace`   |     0.54 GB/s | **9.90 GB/s** |
|                                 |               |               |
| Python                          |               |               |
| `bytes.translate`               |     0.05 GB/s |     1.92 GB/s |
| `numpy.take`                    |     0.01 GB/s |     0.85 GB/s |
| `opencv.LUT`                    |     0.01 GB/s |     1.95 GB/s |
| `opencv.LUT` inplace            |     0.01 GB/s |     2.16 GB/s |
| `stringzilla.translate`         |     0.07 GB/s |     7.92 GB/s |
| `stringzilla.translate` inplace | **0.06 GB/s** | **8.14 GB/s** |

---

See [README.md](README.md) for dataset information and replication instructions.
