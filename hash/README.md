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

| Library                     | Bits | Ports | Arm |   Short Words |     Long Lines |
| --------------------------- | :--: | :---: | :-: | ------------: | -------------: |
| Rust                        |      |       |     |               |                |
| `std::hash`                 |  64  |   -   |  +  |     0.30 GB/s |      3.71 GB/s |
| `crc32fast::hash`           |  32  |   +   |  +  |     0.36 GB/s |      8.84 GB/s |
| `xxh3::xxh3_64`             |  64  |   +   |  +  |     0.61 GB/s |     9.313 GB/s |
| `aHash::hash_one`           |  64  |   -   |  +  |     0.69 GB/s |      8.43 GB/s |
| `foldhash::hash_one`        |  64  |   -   |  +  |     0.67 GB/s |      8.11 GB/s |
| `wyhash::wyhash`            |  64  |   +   |  +  |             — |              — |
| `murmurhash32::murmurhash3` |  32  |   +   |  +  |             — |              — |
| `stringzilla::hash`         |  64  |   +   |  +  | __0.88 GB/s__ | __11.38 GB/s__ |
|                             |      |       |     |               |                |
| Python                      |      |       |     |               |                |
| `xxhash.xxh3_64`            |  64  |   +   |  +  |     0.03 GB/s |      5.74 GB/s |
| `google_crc32c.value`       |  32  |   +   |  +  |     0.06 GB/s |      6.61 GB/s |
| `mmh3.hash32`               |  32  |   +   |  +  |     0.06 GB/s |      2.57 GB/s |
| `mmh3.hash64`               |  64  |   +   |  +  |     0.05 GB/s |      4.49 GB/s |
| `mmh3.hash128`              | 128  |   +   |  +  |             — |              — |
| `cityhash.CityHash64`       |  64  |   +   |  -  |     0.07 GB/s |      5.32 GB/s |
| `cityhash.CityHash128`      | 128  |   +   |  -  |             — |              — |
| `stringzilla.hash`          |  64  |   +   |  +  | __0.07 GB/s__ |  __7.44 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                     | Bits | Ports | Arm |   Short Words |     Long Lines |
| --------------------------- | :--: | :---: | :-: | ------------: | -------------: |
| Rust                        |      |       |     |               |                |
| `std::hash`                 |  64  |   -   |  +  |     0.77 GB/s |      5.36 GB/s |
| `crc32fast::hash`           |  32  |   +   |  +  |     0.68 GB/s |     10.77 GB/s |
| `xxh3::xxh3_64`             |  64  |   +   |  +  | __1.97 GB/s__ |     43.77 GB/s |
| `aHash::hash_one`           |  64  |   -   |  +  |     1.88 GB/s |     20.19 GB/s |
| `foldhash::hash_one`        |  64  |   -   |  +  |     1.87 GB/s | __55.06 GB/s__ |
| `wyhash::wyhash`            |  64  |   +   |  +  |    0.941 GB/s |     25.16 GB/s |
| `murmurhash32::murmurhash3` |  32  |   +   |  +  |     0.92 GB/s |      3.23 GB/s |
| `stringzilla::hash`         |  64  |   +   |  +  |     1.02 GB/s |     31.43 GB/s |
|                             |      |       |     |               |                |
| Python                      |      |       |     |               |                |
| `xxhash.xxh3_64`            |  64  |   +   |  +  |     0.47 GB/s | __32.99 GB/s__ |
| `google_crc32c.value`       |  32  |   +   |  +  |     0.24 GB/s |      5.55 GB/s |
| `mmh3.hash32`               |  32  |   +   |  +  |     0.19 GB/s |      3.05 GB/s |
| `mmh3.hash64`               |  64  |   +   |  +  |     0.13 GB/s |      6.96 GB/s |
| `mmh3.hash128`              | 128  |   +   |  +  |     0.16 GB/s |      7.22 GB/s |
| `cityhash.CityHash64`       |  64  |   +   |  -  | __0.51 GB/s__ |     17.13 GB/s |
| `cityhash.CityHash128`      | 128  |   +   |  -  |     0.16 GB/s |     16.94 GB/s |
| `stringzilla.hash`          |  64  |   +   |  +  |     0.34 GB/s |     30.53 GB/s |

> Measured July 29, 2026.

## Streaming Hash

In larger systems, we often need the ability to incrementally hash the data.
This is especially important in distributed systems, where the data is too large to fit into memory at once.

### Intel Xeon4 Sapphire Rapids

| Library                    | Bits | Ports |   Short Words |    Long Lines |
| -------------------------- | :--: | :---: | ------------: | ------------: |
| Rust                       |      |       |               |               |
| `std::hash::DefaultHasher` |  64  |   -   |     0.46 GB/s |     3.77 GB/s |
| `aHash::AHasher`           |  64  |   -   | __1.20 GB/s__ |     8.00 GB/s |
| `foldhash::FoldHasher`     |  64  |   -   |     1.02 GB/s |     8.24 GB/s |
| `crc32fast::Hasher`        |  32  |   +   |     0.36 GB/s |     8.82 GB/s |
| `stringzilla::Hasher`      |  64  |   +   |     0.41 GB/s | __9.16 GB/s__ |
|                            |      |       |               |               |
| Python                     |      |       |               |               |
| `xxhash.xxh3_64`           |  64  |   +   |     0.06 GB/s |     6.45 GB/s |
| `google_crc32c.Checksum`   |  32  |   +   |     0.05 GB/s |     6.70 GB/s |
| `stringzilla.Hasher`       |  64  |   +   | __0.07 GB/s__ | __7.72 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                    | Bits | Ports |   Short Words |     Long Lines |
| -------------------------- | :--: | :---: | ------------: | -------------: |
| Rust                       |      |       |               |                |
| `std::hash::DefaultHasher` |  64  |   -   |    0.941 GB/s |      5.46 GB/s |
| `aHash::AHasher`           |  64  |   -   | __2.27 GB/s__ |     20.11 GB/s |
| `foldhash::FoldHasher`     |  64  |   -   |     2.07 GB/s | __54.71 GB/s__ |
| `crc32fast::Hasher`        |  32  |   +   |     0.74 GB/s |     10.55 GB/s |
| `stringzilla::Hasher`      |  64  |   +   |     0.67 GB/s |     17.02 GB/s |
|                            |      |       |               |                |
| Python                     |      |       |               |                |
| `xxhash.xxh3_64`           |  64  |   +   |     0.20 GB/s | __14.84 GB/s__ |
| `google_crc32c.Checksum`   |  32  |   +   |     0.14 GB/s |      5.36 GB/s |
| `stringzilla.Hasher`       |  64  |   +   | __0.43 GB/s__ |     13.17 GB/s |

> Measured July 29, 2026.

## Checksum and Cryptographic Hashing

For reference, one may want to put those numbers next to check-sum calculation speeds on one end of complexity and cryptographic hashing speeds on the other end.

### Intel Xeon4 Sapphire Rapids

| Library                | Bits | Ports |   Short Words |     Long Lines |
| ---------------------- | :--: | :---: | ------------: | -------------: |
| Rust                   |      |       |               |                |
| `stringzilla::bytesum` |  64  |   +   | __0.91 GB/s__ | __11.75 GB/s__ |
| `blake3::hash`         | 256  |   +   |     0.10 GB/s |      1.65 GB/s |
| `sha2::Sha256`         | 256  |   +   |             — |              — |
| `ring::SHA256`         | 256  |   +   |             — |              — |
| `stringzilla::Sha256`  | 256  |   +   |             — |              — |
|                        |      |       |               |                |
| Python                 |      |       |               |                |
| `stringzilla.bytesum`  |  64  |   +   | __0.07 GB/s__ |  __7.80 GB/s__ |
| `blake3.blake3`        | 256  |   +   |     0.02 GB/s |      1.56 GB/s |
| `hashlib.sha256`       | 256  |   +   |             — |              — |
| `stringzilla.Sha256`   | 256  |   +   |             — |              — |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                | Bits | Ports |    Short Words |     Long Lines |
| ---------------------- | :--: | :---: | -------------: | -------------: |
| Rust                   |      |       |                |                |
| `stringzilla::bytesum` |  64  |   +   | __0.997 GB/s__ | __25.98 GB/s__ |
| `blake3::hash`         | 256  |   +   |      0.15 GB/s |      1.66 GB/s |
| `sha2::Sha256`         | 256  |   +   |      0.35 GB/s |      3.07 GB/s |
| `ring::SHA256`         | 256  |   +   |      0.21 GB/s |      3.06 GB/s |
| `stringzilla::Sha256`  | 256  |   +   |      0.27 GB/s |      3.07 GB/s |
|                        |      |       |                |                |
| Python                 |      |       |                |                |
| `stringzilla.bytesum`  |  64  |   +   |  __0.40 GB/s__ | __19.31 GB/s__ |
| `blake3.blake3`        | 256  |   +   |      0.03 GB/s |      1.50 GB/s |
| `hashlib.sha256`       | 256  |   +   |      0.05 GB/s |      2.70 GB/s |
| `stringzilla.Sha256`   | 256  |   +   |      0.11 GB/s |      2.94 GB/s |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
