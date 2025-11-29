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

| Library               | Bits  | Ports | Arm |    Short Words |      Long Lines |
| --------------------- | :---: | :---: | :-: | -------------: | --------------: |
| Rust                  |       |       |     |                |                 |
| `std::hash`           |  64   |   -   |  +  |     0.43 GiB/s |      3.74 GiB/s |
| `crc32fast::hash`     |  32   |   +   |  +  |     0.49 GiB/s |      8.45 GiB/s |
| `xxh3::xxh3_64`       |  64   |   +   |  +  |     1.08 GiB/s |      9.48 GiB/s |
| `aHash::hash_one`     |  64   |   -   |  +  |     1.23 GiB/s |      8.61 GiB/s |
| `foldhash::hash_one`  |  64   |   -   |  +  |     1.02 GiB/s |      8.24 GiB/s |
| `gxhash::gxhash64`    |  64   |   -   |  -  |     2.68 GiB/s |      9.19 GiB/s |
| `stringzilla::hash`   |  64   |   +   |  +  | __1.84 GiB/s__ | __11.38 GiB/s__ |
|                       |       |       |     |                |                 |
| Python                |       |       |     |                |                 |
| `hash`                | 32/64 |   -   |  +  |     0.13 GiB/s |      4.27 GiB/s |
| `xxhash.xxh3_64`      |  64   |   +   |  +  |     0.04 GiB/s |      6.38 GiB/s |
| `google_crc32c.value` |  32   |   +   |  +  |     0.04 GiB/s |      5.96 GiB/s |
| `mmh3.hash32`         |  32   |   +   |  +  |     0.05 GiB/s |      2.65 GiB/s |
| `mmh3.hash64`         |  64   |   +   |  +  |     0.03 GiB/s |      4.45 GiB/s |
| `cityhash.CityHash64` |  64   |   +   |  -  |     0.06 GiB/s |      4.87 GiB/s |
| `stringzilla.hash`    |  64   |   +   |  +  | __0.14 GiB/s__ |  __9.19 GiB/s__ |

> __Ports__ means availability in multiple other programming languages, like C, C++, Python, Java, Go, JavaScript, etc.
> __Arm__ indicates support for Arm architecture. Most hash functions work on both x86 and Arm, but gxHash and many MurMurHash and CityHash implementations don't.

## Streaming Hash

In larger systems, we often need the ability to incrementally hash the data.
This is especially important in distributed systems, where the data is too large to fit into memory at once.

| Library                    | Bits | Ports |    Short Words |      Long Lines |
| -------------------------- | :--: | :---: | -------------: | --------------: |
| Rust                       |      |       |                |                 |
| `std::hash::DefaultHasher` |  64  |   -   |     0.51 GiB/s |      3.92 GiB/s |
| `aHash::AHasher`           |  64  |   -   | __1.30 GiB/s__ |      8.56 GiB/s |
| `foldhash::FoldHasher`     |  64  |   -   |     1.27 GiB/s |      8.18 GiB/s |
| `crc32fast::Hasher`        |  32  |   +   |     0.37 GiB/s |      8.39 GiB/s |
| `stringzilla::Hasher`      |  64  |   +   |     0.89 GiB/s | __11.03 GiB/s__ |
|                            |      |       |                |                 |
| Python                     |      |       |                |                 |
| `xxhash.xxh3_64`           |  64  |   +   |     0.09 GiB/s |       7.09 GB/s |
| `google_crc32c.Checksum`   |  32  |   +   |     0.04 GiB/s |      5.96 GiB/s |
| `stringzilla.Hasher`       |  64  |   +   | __0.35 GiB/s__ |   __6.04 GB/s__ |

## Checksum and Cryptographic Hashing

For reference, one may want to put those numbers next to check-sum calculation speeds on one end of complexity and cryptographic hashing speeds on the other end.

| Library                | Bits | Ports | Short Words |  Long Lines |
| ---------------------- | :--: | :---: | ----------: | ----------: |
| Rust                   |      |       |             |             |
| `stringzilla::bytesum` |  64  |   +   |  2.16 GiB/s | 11.65 GiB/s |
| `blake3::hash`         | 256  |   +   |  0.10 GiB/s |  1.97 GiB/s |
|                        |      |       |             |             |
| Python                 |      |       |             |             |
| `stringzilla.bytesum`  |  64  |   +   |  0.16 GiB/s |  8.62 GiB/s |
| `blake3.digest`        | 256  |   +   |  0.02 GiB/s |  1.82 GiB/s |

---

See [README.md](README.md) for dataset information and replication instructions.
