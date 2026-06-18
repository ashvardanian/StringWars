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

On Intel Sapphire Rapids CPU, on `xlsum.csv` dataset, the following numbers can be expected for hashing individual whitespace-delimited words and newline-delimited lines.
The __Ports__ column marks availability in other languages, like C, C++, Python, Java, Go, JavaScript.
The __Arm__ column marks Arm support — most hash functions run on both x86 and Arm, but gxHash and many MurmurHash and CityHash implementations don't.

### Intel Xeon4 Sapphire Rapids

| Library               | Bits  | Ports |  Arm  |   Short Words |     Long Lines |
| --------------------- | :---: | :---: | :---: | ------------: | -------------: |
| Rust                  |       |       |       |               |                |
| `std::hash`           |  64   |   -   |   +   |     0.32 GB/s |      3.98 GB/s |
| `crc32fast::hash`     |  32   |   +   |   +   |     0.39 GB/s |      9.49 GB/s |
| `xxh3::xxh3_64`       |  64   |   +   |   +   |     0.66 GB/s |     10.00 GB/s |
| `aHash::hash_one`     |  64   |   -   |   +   |     0.74 GB/s |      9.05 GB/s |
| `foldhash::hash_one`  |  64   |   -   |   +   |     0.72 GB/s |      8.71 GB/s |
| `stringzilla::hash`   |  64   |   +   |   +   | __0.95 GB/s__ | __12.22 GB/s__ |
|                       |       |       |       |               |                |
| Python                |       |       |       |               |                |
| `hash`                | 32/64 |   -   |   +   |     0.06 GB/s |      3.72 GB/s |
| `xxhash.xxh3_64`      |  64   |   +   |   +   |     0.03 GB/s |      6.16 GB/s |
| `google_crc32c.value` |  32   |   +   |   +   |     0.06 GB/s |      7.10 GB/s |
| `mmh3.hash32`         |  32   |   +   |   +   |     0.06 GB/s |      2.76 GB/s |
| `mmh3.hash64`         |  64   |   +   |   +   |     0.05 GB/s |      4.82 GB/s |
| `cityhash.CityHash64` |  64   |   +   |   -   |     0.07 GB/s |      5.71 GB/s |
| `stringzilla.hash`    |  64   |   +   |   +   | __0.07 GB/s__ |  __7.99 GB/s__ |

> Measured June 17, 2026 on an Intel Xeon4 Sapphire Rapids.

## Streaming Hash

In larger systems, we often need the ability to incrementally hash the data.
This is especially important in distributed systems, where the data is too large to fit into memory at once.

### Intel Xeon4 Sapphire Rapids

| Library                    | Bits  | Ports |   Short Words |    Long Lines |
| -------------------------- | :---: | :---: | ------------: | ------------: |
| Rust                       |       |       |               |               |
| `std::hash::DefaultHasher` |  64   |   -   |     0.49 GB/s |     4.05 GB/s |
| `aHash::AHasher`           |  64   |   -   | __1.29 GB/s__ |     8.59 GB/s |
| `foldhash::FoldHasher`     |  64   |   -   |     1.10 GB/s |     8.85 GB/s |
| `crc32fast::Hasher`        |  32   |   +   |     0.39 GB/s |     9.47 GB/s |
| `stringzilla::Hasher`      |  64   |   +   |     0.44 GB/s | __9.84 GB/s__ |
|                            |       |       |               |               |
| Python                     |       |       |               |               |
| `xxhash.xxh3_64`           |  64   |   +   |     0.06 GB/s |     6.93 GB/s |
| `google_crc32c.Checksum`   |  32   |   +   |     0.05 GB/s |     7.19 GB/s |
| `stringzilla.Hasher`       |  64   |   +   | __0.08 GB/s__ | __8.29 GB/s__ |

> Measured June 17, 2026 on an Intel Xeon4 Sapphire Rapids.

## Checksum and Cryptographic Hashing

For reference, one may want to put those numbers next to check-sum calculation speeds on one end of complexity and cryptographic hashing speeds on the other end.

### Intel Xeon4 Sapphire Rapids

| Library                | Bits  | Ports |   Short Words |     Long Lines |
| ---------------------- | :---: | :---: | ------------: | -------------: |
| Rust                   |       |       |               |                |
| `stringzilla::bytesum` |  64   |   +   | __0.98 GB/s__ | __12.62 GB/s__ |
| `blake3::hash`         |  256  |   +   |     0.11 GB/s |      1.77 GB/s |
|                        |       |       |               |                |
| Python                 |       |       |               |                |
| `stringzilla.bytesum`  |  64   |   +   | __0.08 GB/s__ |  __8.37 GB/s__ |
| `blake3.digest`        |  256  |   +   |     0.02 GB/s |      1.68 GB/s |

> Measured June 17, 2026 on an Intel Xeon4 Sapphire Rapids.

---

See [README.md](README.md) for dataset information and replication instructions.
