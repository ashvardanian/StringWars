# Substring and Byte-Set Search Benchmarks

Benchmarks for substring search and character-set matching across Rust and Python implementations.

## Substring Search

Substring search is one of the most common operations in text processing, and one of the slowest.
Most of the time, programmers don't think about replacing the `str::find` method, as it's already expected to be optimized.
In many languages it's offloaded to the C standard library [`memmem`](https://man7.org/linux/man-pages/man3/memmem.3.html) or [`strstr`](https://en.cppreference.com/w/c/string/byte/strstr) for `NULL`-terminated strings.
The C standard library is, however, also implemented by humans, and a better solution can be created.

### Forward Search

### Intel Xeon4 Sapphire Rapids

| Library                | Short Word Queries | Long Line Queries |
| ---------------------- | -----------------: | ----------------: |
| Rust                   |                    |                   |
| `std::str::find`       |          8.70 GB/s |        10.66 GB/s |
| `memmem::find`         |          8.87 GB/s |        10.51 GB/s |
| `memmem::Finder`       |          9.30 GB/s |        10.55 GB/s |
| `stringzilla::find`    |     __10.63 GB/s__ |    __10.73 GB/s__ |
|                        |                    |                   |
| Python                 |                    |                   |
| `str.find`             |          0.68 GB/s |         1.06 GB/s |
| `pyahocorasick.iter`   |                  — |                 — |
| `stringzilla.Str.find` |      __3.14 GB/s__ |    __10.84 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                | Short Word Queries | Long Line Queries |
| ---------------------- | -----------------: | ----------------: |
| Rust                   |                    |                   |
| `std::str::find`       |         33.25 GB/s |        47.17 GB/s |
| `memmem::find`         |         33.23 GB/s |        47.09 GB/s |
| `memmem::Finder`       |     __35.25 GB/s__ |    __47.23 GB/s__ |
| `stringzilla::find`    |         27.98 GB/s |        30.91 GB/s |
|                        |                    |                   |
| Python                 |                    |                   |
| `str.find`             |          2.83 GB/s |        16.79 GB/s |
| `pyahocorasick.iter`   |          1.23 GB/s |         1.15 GB/s |
| `stringzilla.Str.find` |     __24.96 GB/s__ |    __35.46 GB/s__ |

> Measured July 29, 2026.

### Reverse Search

Interestingly, the reverse order search is almost never implemented in SIMD, assuming fewer people ever need it.
Still, those are provided by StringZilla mostly for parsing tasks and feature parity.

### Intel Xeon4 Sapphire Rapids

| Library                 | Short Word Queries | Long Line Queries |
| ----------------------- | -----------------: | ----------------: |
| Rust                    |                    |                   |
| `std::str::rfind`       |          2.74 GB/s |         4.85 GB/s |
| `memmem::rfind`         |          2.73 GB/s |         4.68 GB/s |
| `memmem::FinderRev`     |          2.76 GB/s |         4.68 GB/s |
| `stringzilla::rfind`    |     __10.05 GB/s__ |    __10.66 GB/s__ |
|                         |                    |                   |
| Python                  |                    |                   |
| `str.rfind`             |          1.29 GB/s |         3.54 GB/s |
| `stringzilla.Str.rfind` |      __7.23 GB/s__ |    __10.83 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                 | Short Word Queries | Long Line Queries |
| ----------------------- | -----------------: | ----------------: |
| Rust                    |                    |                   |
| `std::str::rfind`       |          5.66 GB/s |         4.15 GB/s |
| `memmem::rfind`         |          5.65 GB/s |         4.16 GB/s |
| `memmem::FinderRev`     |          5.70 GB/s |         4.14 GB/s |
| `stringzilla::rfind`    |     __26.93 GB/s__ |    __30.59 GB/s__ |
|                         |                    |                   |
| Python                  |                    |                   |
| `str.rfind`             |          2.64 GB/s |         3.85 GB/s |
| `stringzilla.Str.rfind` |     __21.01 GB/s__ |    __30.46 GB/s__ |

> Measured July 29, 2026.

## Byte-Set Search

StringWars takes a few representative examples of various character sets that appear in real parsing or string validation tasks:

- tabulation characters, like `\n\r\v\f`;
- HTML and XML markup characters, like `</>&'\"=[]`;
- numeric characters, like `0123456789`.

It's common in such cases, to pre-construct some library-specific filter-object or Finite State Machine (FSM) to search for a set of characters.
Once that object is constructed, all of its inclusions in each token (word or line) are counted.

### Intel Xeon4 Sapphire Rapids

| Library                         |   Short Words |    Long Lines |
| ------------------------------- | ------------: | ------------: |
| Rust                            |               |               |
| `bstr::find_byteset`            |             — |             — |
| `regex::find_iter`              |     0.19 GB/s |     4.72 GB/s |
| `aho_corasick::find_iter`       |     0.32 GB/s |     0.47 GB/s |
| `stringzilla::find_byteset`     | __1.12 GB/s__ | __7.77 GB/s__ |
|                                 |               |               |
| Python                          |               |               |
| `re.finditer`                   |     0.05 GB/s |     0.20 GB/s |
| `stringzilla.Str.find_first_of` | __0.11 GB/s__ | __8.71 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                         |   Short Words |     Long Lines |
| ------------------------------- | ------------: | -------------: |
| Rust                            |               |                |
| `bstr::find_byteset`            |     1.37 GB/s |      3.42 GB/s |
| `regex::find_iter`              |   390.20 MB/s |      8.76 GB/s |
| `aho_corasick::find_iter`       |   694.71 MB/s |    940.07 MB/s |
| `stringzilla::find_byteset`     | __1.43 GB/s__ | __12.20 GB/s__ |
|                                 |               |                |
| Python                          |               |                |
| `re.finditer`                   |   717.01 MB/s |    588.93 MB/s |
| `stringzilla.Str.find_first_of` | __3.73 GB/s__ |  __4.07 GB/s__ |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
