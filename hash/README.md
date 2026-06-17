# Hash Benchmarks

Benchmarks for hashing functions across Rust and Python implementations.

Many great hashing libraries exist in Rust, C, and C++.
Typical top choices are `aHash`, `xxHash`, `blake3`, `CityHash`, `MurmurHash`, `crc32fast`, or the native `std::hash`.
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
| `std::hash`           |  64   |   -   |  +  |     0.20 GB/s |      3.51 GB/s |
| `crc32fast::hash`     |  32   |   +   |  +  |     0.28 GB/s |      8.91 GB/s |
| `xxh3::xxh3_64`       |  64   |   +   |  +  |     0.52 GB/s |      9.00 GB/s |
| `aHash::hash_one`     |  64   |   -   |  +  |     0.54 GB/s |      8.69 GB/s |
| `foldhash::hash_one`  |  64   |   -   |  +  |     0.49 GB/s |      8.31 GB/s |
| `stringzilla::hash`   |  64   |   +   |  +  | __0.70 GB/s__ | __10.57 GB/s__ |
|                       |       |       |     |               |                |
| Python                |       |       |     |               |                |
| `hash`                | 32/64 |   -   |  +  |     0.05 GB/s |      3.01 GB/s |
| `xxhash.xxh3_64`      |  64   |   +   |  +  |     0.03 GB/s |      4.25 GB/s |
| `google_crc32c.value` |  32   |   +   |  +  |     0.05 GB/s |      5.02 GB/s |
| `mmh3.hash32`         |  32   |   +   |  +  |     0.05 GB/s |      2.49 GB/s |
| `mmh3.hash64`         |  64   |   +   |  +  |     0.04 GB/s |      3.74 GB/s |
| `cityhash.CityHash64` |  64   |   +   |  -  |     0.07 GB/s |      4.54 GB/s |
| `stringzilla.hash`    |  64   |   +   |  +  | __0.07 GB/s__ |  __6.21 GB/s__ |

> Measured 2026-06-17 on an Intel Xeon Platinum 8468 (Sapphire Rapids).

> __Ports__ means availability in multiple other programming languages, like C, C++, Python, Java, Go, JavaScript, etc.
> __Arm__ indicates support for Arm architecture. Most hash functions work on both x86 and Arm, but gxHash and many MurMurHash and CityHash implementations don't.

## Streaming Hash

In larger systems, we often need the ability to incrementally hash the data.
This is especially important in distributed systems, where the data is too large to fit into memory at once.

| Library                    | Bits | Ports |   Short Words |     Long Lines |
| -------------------------- | :--: | :---: | ------------: | -------------: |
| Rust                       |      |       |               |                |
| `std::hash::DefaultHasher` |  64  |   -   |     0.36 GB/s |      3.54 GB/s |
| `aHash::AHasher`           |  64  |   -   | __1.00 GB/s__ |      7.83 GB/s |
| `foldhash::FoldHasher`     |  64  |   -   |     0.78 GB/s |      8.11 GB/s |
| `crc32fast::Hasher`        |  32  |   +   |     0.32 GB/s | __8.79 GB/s__  |
| `stringzilla::Hasher`      |  64  |   +   |     0.33 GB/s |      8.17 GB/s |
|                            |      |       |               |                |
| Python                     |      |       |               |                |
| `xxhash.xxh3_64`           |  64  |   +   |     0.06 GB/s |      4.73 GB/s |
| `google_crc32c.Checksum`   |  32  |   +   |     0.05 GB/s |      4.81 GB/s |
| `stringzilla.Hasher`       |  64  |   +   | __0.07 GB/s__ |  __5.81 GB/s__ |

> Measured 2026-06-17 on an Intel Xeon Platinum 8468 (Sapphire Rapids).

## Checksum and Cryptographic Hashing

For reference, one may want to put those numbers next to check-sum calculation speeds on one end of complexity and cryptographic hashing speeds on the other end.

| Library                | Bits | Ports | Short Words | Long Lines |
| ---------------------- | :--: | :---: | ----------: | ---------: |
| Rust                   |      |       |             |            |
| `stringzilla::bytesum` |  64  |   +   |   0.78 GB/s | 11.45 GB/s |
| `blake3::hash`         | 256  |   +   |   0.08 GB/s |  1.92 GB/s |
|                        |      |       |             |            |
| Python                 |      |       |             |            |
| `stringzilla.bytesum`  |  64  |   +   |   0.07 GB/s |  6.14 GB/s |
| `blake3.digest`        | 256  |   +   |   0.02 GB/s |  1.33 GB/s |

> Measured 2026-06-17 on an Intel Xeon Platinum 8468 (Sapphire Rapids).

---

See [README.md](README.md) for dataset information and replication instructions.
