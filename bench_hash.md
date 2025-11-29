# Hash Benchmarks

Benchmarks for hashing functions across Rust and Python implementations.

Many great hashing libraries exist in Rust, C, and C++.
Typical top choices are `aHash`, `xxHash`, `blake3`, `gxhash`, `CityHash`, `MurmurHash`, `crc32fast`, or the native `std::hash`.
Many of them have similar pitfalls:

- They are not always documented to have a certain reproducible output and are recommended for use only for local in-memory construction of hash tables, not for serialization or network communication.
- They don't always support streaming and require the whole input to be available in memory at once.
- They don't always pass the SMHasher test suite, especially with `--extra` checks enabled.
- They generally don't have a dynamic dispatch mechanism to simplify shipping of precompiled software.
- They are rarely available for multiple programming languages.

StringZilla addresses those issues and seems to provide competitive performance.

## Single Hash

On Intel Sapphire Rapids CPU, on `xlsum.csv` dataset, the following numbers can be expected for hashing individual whitespace-delimited words and newline-delimited lines:

| Library               | Bits  | Ports | Arm |   Short Words |     Long Lines |
| --------------------- | :---: | :---: | :-: | ------------: | -------------: |
| Rust                  |       |       |     |               |                |
| `std::hash`           |  64   |   -   |  +  |     0.43 GB/s |      3.74 GB/s |
| `crc32fast::hash`     |  32   |   +   |  +  |     0.49 GB/s |      8.45 GB/s |
| `xxh3::xxh3_64`       |  64   |   +   |  +  |     1.08 GB/s |      9.48 GB/s |
| `aHash::hash_one`     |  64   |   -   |  +  |     1.23 GB/s |      8.61 GB/s |
| `foldhash::hash_one`  |  64   |   -   |  +  |     1.02 GB/s |      8.24 GB/s |
| `gxhash::gxhash64`    |  64   |   -   |  -  |     2.68 GB/s |      9.19 GB/s |
| `stringzilla::hash`   |  64   |   +   |  +  | **1.84 GB/s** | **11.38 GB/s** |
|                       |       |       |     |               |                |
| Python                |       |       |     |               |                |
| `hash`                | 32/64 |   -   |  +  |     0.13 GB/s |      4.27 GB/s |
| `xxhash.xxh3_64`      |  64   |   +   |  +  |     0.04 GB/s |      6.38 GB/s |
| `google_crc32c.value` |  32   |   +   |  +  |     0.04 GB/s |      5.96 GB/s |
| `mmh3.hash32`         |  32   |   +   |  +  |     0.05 GB/s |      2.65 GB/s |
| `mmh3.hash64`         |  64   |   +   |  +  |     0.03 GB/s |      4.45 GB/s |
| `cityhash.CityHash64` |  64   |   +   |  -  |     0.06 GB/s |      4.87 GB/s |
| `stringzilla.hash`    |  64   |   +   |  +  | **0.14 GB/s** |  **9.19 GB/s** |

> **Ports** means availability in multiple other programming languages, like C, C++, Python, Java, Go, JavaScript, etc.
> **Arm** indicates support for Arm architecture. Most hash functions work on both x86 and Arm, but gxHash and many MurMurHash and CityHash implementations don't.

## Streaming Hash

In larger systems, we often need the ability to incrementally hash the data.
This is especially important in distributed systems, where the data is too large to fit into memory at once.

| Library                    | Bits | Ports |   Short Words |     Long Lines |
| -------------------------- | :--: | :---: | ------------: | -------------: |
| Rust                       |      |       |               |                |
| `std::hash::DefaultHasher` |  64  |   -   |     0.51 GB/s |      3.92 GB/s |
| `aHash::AHasher`           |  64  |   -   | **1.30 GB/s** |      8.56 GB/s |
| `foldhash::FoldHasher`     |  64  |   -   |     1.27 GB/s |      8.18 GB/s |
| `crc32fast::Hasher`        |  32  |   +   |     0.37 GB/s |      8.39 GB/s |
| `stringzilla::Hasher`      |  64  |   +   |     0.89 GB/s | **11.03 GB/s** |
|                            |      |       |               |                |
| Python                     |      |       |               |                |
| `xxhash.xxh3_64`           |  64  |   +   |     0.09 GB/s |      7.09 GB/s |
| `google_crc32c.Checksum`   |  32  |   +   |     0.04 GB/s |      5.96 GB/s |
| `stringzilla.Hasher`       |  64  |   +   | **0.35 GB/s** |  **6.04 GB/s** |

## Checksum and Cryptographic Hashing

For reference, one may want to put those numbers next to check-sum calculation speeds on one end of complexity and cryptographic hashing speeds on the other end.

| Library                | Bits | Ports | Short Words | Long Lines |
| ---------------------- | :--: | :---: | ----------: | ---------: |
| Rust                   |      |       |             |            |
| `stringzilla::bytesum` |  64  |   +   |   2.16 GB/s | 11.65 GB/s |
| `blake3::hash`         | 256  |   +   |   0.10 GB/s |  1.97 GB/s |
|                        |      |       |             |            |
| Python                 |      |       |             |            |
| `stringzilla.bytesum`  |  64  |   +   |   0.16 GB/s |  8.62 GB/s |
| `blake3.digest`        | 256  |   +   |   0.02 GB/s |  1.82 GB/s |

---

See [README.md](README.md) for dataset information and replication instructions.
