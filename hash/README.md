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

Every table reports single-core throughput in bytes per second, on the `xlsum.csv` dataset, for individual whitespace-delimited words (_short words_) and newline-delimited lines (_long lines_).
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.
Among the rivals, `cityhash` has no Arm port, and `std::hash`, `aHash` and `foldhash` exist only in Rust.

The `hash` rows are the single-shot hash of each token.
The `Hasher` rows are the streaming hash: in larger systems, we often need the ability to incrementally hash the data, which is especially important in distributed systems, where the data is too large to fit into memory at once.
The `bytesum`, `Sha256`, `Sha256s` and `Sha256s<sorted>` rows put the hashes next to check-sum calculation speeds on one end of complexity and cryptographic hashing speeds on the other end, against the fastest other library in the same group.
SHA256 runs across four backends, including the multi-state `Sha256s` that advances sixteen messages at once, one per lane.
A batch retires when its longest member does, so `Sha256s<sorted>` draws each batch from a length-ordered copy of the same tokens, and the distance between the two rows is what length skew costs.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `std::hash`, `foldhash::hash_one`, `wyhash::wyhash`, `murmurhash32::murmurhash3`, `cityhash::city_hash_64`, `std::hash::DefaultHasher`, `foldhash::FoldHasher`, `crc32fast::Hasher`, `blake3::hash`, `ring::SHA256`.

| StringZilla                                 |     Speed | Speedup | Alternative       |     Speed |
| :------------------------------------------ | --------: | ------: | :---------------- | --------: |
| `stringzilla::hash`, short words            |  750 MB/s |   0.63× | `aHash::hash_one` |  1.2 GB/s |
| `stringzilla::hash`, long lines             |   10 GB/s |   1.00× | `xxh3::xxh3_64`   |   10 GB/s |
| `stringzilla::Hasher`, short words          |  600 MB/s |   0.50× | `aHash::AHasher`  |  1.2 GB/s |
| `stringzilla::Hasher`, long lines           |   13 GB/s |   1.30× | `aHash::AHasher`  |   10 GB/s |
| `stringzilla::bytesum`, short words         |  740 MB/s |   1.30× | `crc32fast::hash` |  570 MB/s |
| `stringzilla::bytesum`, long lines          |   20 GB/s |   2.27× | `crc32fast::hash` |  8.8 GB/s |
| `stringzilla::Sha256`, short words          |   84 MB/s |   0.65× | `sha2::Sha256`    |  130 MB/s |
| `stringzilla::Sha256`, long lines           |  1.6 GB/s |   0.94× | `sha2::Sha256`    | 1.71 GB/s |
| `stringzilla::Sha256s`, short words         |  180 MB/s |   1.38× | `sha2::Sha256`    |  130 MB/s |
| `stringzilla::Sha256s`, long lines          |  1.1 GB/s |   0.64× | `sha2::Sha256`    | 1.71 GB/s |
| `stringzilla::Sha256s<sorted>`, short words |  180 MB/s |   1.38× | `sha2::Sha256`    |  130 MB/s |
| `stringzilla::Sha256s<sorted>`, long lines  | 3.06 GB/s |   1.79× | `sha2::Sha256`    | 1.71 GB/s |

> Measured October 14, 2026.

### Python

Also benchmarked: `mmh3.hash32`, `mmh3.hash64`, `mmh3.hash128`, `cityhash.CityHash128`, `google_crc32c.Checksum`, `blake3.blake3`.

| StringZilla                                |    Speed | Speedup | Alternative           |    Speed |
| :----------------------------------------- | -------: | ------: | :-------------------- | -------: |
| `stringzilla.hash`, short words            | 287 MB/s |   0.98× | `cityhash.CityHash64` | 293 MB/s |
| `stringzilla.hash`, long lines             |  20 GB/s |   2.00× | `xxhash.xxh3_64`      |  10 GB/s |
| `stringzilla.Hasher`, short words          | 521 MB/s |   1.56× | `xxhash.xxh3_64`      | 333 MB/s |
| `stringzilla.Hasher`, long lines           |  10 GB/s |   1.10× | `xxhash.xxh3_64`      | 9.1 GB/s |
| `stringzilla.bytesum`, short words         | 300 MB/s |   1.81× | `google_crc32c.value` | 166 MB/s |
| `stringzilla.bytesum`, long lines          |  20 GB/s |   2.22× | `google_crc32c.value` |   9 GB/s |
| `stringzilla.Sha256`, short words          |  32 MB/s |   1.88× | `hashlib.sha256`      |  17 MB/s |
| `stringzilla.Sha256`, long lines           | 1.5 GB/s |   0.94× | `hashlib.sha256`      | 1.6 GB/s |
| `stringzilla.Sha256s`, short words         | 140 MB/s |   8.24× | `hashlib.sha256`      |  17 MB/s |
| `stringzilla.Sha256s`, long lines          | 1.0 GB/s |   0.63× | `hashlib.sha256`      | 1.6 GB/s |
| `stringzilla.Sha256s<sorted>`, short words | 120 MB/s |   7.06× | `hashlib.sha256`      |  17 MB/s |
| `stringzilla.Sha256s<sorted>`, long lines  | 2.9 GB/s |   1.81× | `hashlib.sha256`      | 1.6 GB/s |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `std::hash`, `aHash::hash_one`, `wyhash::wyhash`, `murmurhash32::murmurhash3`, `std::hash::DefaultHasher`, `crc32fast::Hasher`, `blake3::hash`, `ring::SHA256`.

| StringZilla                         |      Speed | Speedup | Alternative            |      Speed |
| :---------------------------------- | ---------: | ------: | :--------------------- | ---------: |
| `stringzilla::hash`, short words    |  1.02 GB/s |   0.52× | `xxh3::xxh3_64`        |  1.97 GB/s |
| `stringzilla::hash`, long lines     | 31.43 GB/s |   0.57× | `foldhash::hash_one`   | 55.06 GB/s |
| `stringzilla::Hasher`, short words  |  0.67 GB/s |   0.30× | `aHash::AHasher`       |  2.27 GB/s |
| `stringzilla::Hasher`, long lines   | 17.02 GB/s |   0.31× | `foldhash::FoldHasher` | 54.71 GB/s |
| `stringzilla::bytesum`, short words | 0.997 GB/s |   1.47× | `crc32fast::hash`      |  0.68 GB/s |
| `stringzilla::bytesum`, long lines  | 25.98 GB/s |   2.41× | `crc32fast::hash`      | 10.77 GB/s |
| `stringzilla::Sha256`, short words  |  0.27 GB/s |   0.77× | `sha2::Sha256`         |  0.35 GB/s |
| `stringzilla::Sha256`, long lines   |  3.07 GB/s |   1.00× | `sha2::Sha256`         |  3.07 GB/s |

> Measured July 29, 2026.

### Python

Also benchmarked: `mmh3.hash32`, `mmh3.hash64`, `mmh3.hash128`, `cityhash.CityHash128`, `google_crc32c.Checksum`, `blake3.blake3`.

| StringZilla                        |      Speed | Speedup | Alternative           |      Speed |
| :--------------------------------- | ---------: | ------: | :-------------------- | ---------: |
| `stringzilla.hash`, short words    |  0.34 GB/s |   0.67× | `cityhash.CityHash64` |  0.51 GB/s |
| `stringzilla.hash`, long lines     | 30.53 GB/s |   0.93× | `xxhash.xxh3_64`      | 32.99 GB/s |
| `stringzilla.Hasher`, short words  |  0.43 GB/s |   2.15× | `xxhash.xxh3_64`      |  0.20 GB/s |
| `stringzilla.Hasher`, long lines   | 13.17 GB/s |   0.89× | `xxhash.xxh3_64`      | 14.84 GB/s |
| `stringzilla.bytesum`, short words |  0.40 GB/s |   1.67× | `google_crc32c.value` |  0.24 GB/s |
| `stringzilla.bytesum`, long lines  | 19.31 GB/s |   3.48× | `google_crc32c.value` |  5.55 GB/s |
| `stringzilla.Sha256`, short words  |  0.11 GB/s |   2.20× | `hashlib.sha256`      |  0.05 GB/s |
| `stringzilla.Sha256`, long lines   |  2.94 GB/s |   1.09× | `hashlib.sha256`      |  2.70 GB/s |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
