# Substring and Byte-Set Search Benchmarks

Benchmarks for substring search and character-set matching across Rust and Python implementations.

## Substring Search

Substring search is one of the most common operations in text processing, and one of the slowest.
Most of the time, programmers don't think about replacing the `str::find` method, as it's already expected to be optimized.
In many languages it's offloaded to the C standard library [`memmem`](https://man7.org/linux/man-pages/man3/memmem.3.html) or [`strstr`](https://en.cppreference.com/w/c/string/byte/strstr) for `NULL`-terminated strings.
The C standard library is, however, also implemented by humans, and a better solution can be created.

### Forward Search (find)

| Library             | Short Word Queries | Long Line Queries |
| ------------------- | -----------------: | ----------------: |
| Rust                |                    |                   |
| `std::str::find`    |          8.90 GB/s |        11.27 GB/s |
| `memmem::find`      |          8.70 GB/s |        11.12 GB/s |
| `memmem::Finder`    |          9.31 GB/s |    __11.33 GB/s__ |
| `stringzilla::find` |     __10.32 GB/s__ |        11.24 GB/s |
|                     |                    |                   |
| Python              |                    |                   |
| `str.find`          |          0.93 GB/s |         1.14 GB/s |
| `stringzilla.find`  |      __2.32 GB/s__ |    __11.82 GB/s__ |

> Measured 2026-06-17 on an Intel Xeon Platinum 8468 (Sapphire Rapids).

### Reverse Search (rfind)

Interestingly, the reverse order search is almost never implemented in SIMD, assuming fewer people ever need it.
Still, those are provided by StringZilla mostly for parsing tasks and feature parity.

| Library              | Short Word Queries | Long Line Queries |
| -------------------- | -----------------: | ----------------: |
| Rust                 |                    |                   |
| `std::str::rfind`    |          3.19 GB/s |         5.43 GB/s |
| `memmem::rfind`      |          3.20 GB/s |         5.52 GB/s |
| `memmem::FinderRev`  |          3.05 GB/s |         5.54 GB/s |
| `stringzilla::rfind` |      __9.86 GB/s__ |    __10.87 GB/s__ |
|                      |                    |                   |
| Python               |                    |                   |
| `str.rfind`          |          1.33 GB/s |         4.55 GB/s |
| `stringzilla.rfind`  |      __8.64 GB/s__ |    __11.49 GB/s__ |

> Measured 2026-06-17 on an Intel Xeon Platinum 8468 (Sapphire Rapids).

## Byte-Set Search

StringWars takes a few representative examples of various character sets that appear in real parsing or string validation tasks:

- tabulation characters, like `\n\r\v\f`;
- HTML and XML markup characters, like `</>&'\"=[]`;
- numeric characters, like `0123456789`.

It's common in such cases, to pre-construct some library-specific filter-object or Finite State Machine (FSM) to search for a set of characters.
Once that object is constructed, all of its inclusions in each token (word or line) are counted.

| Library                         |   Short Words |    Long Lines |
| ------------------------------- | ------------: | ------------: |
| Rust                            |               |               |
| `bstr::iter`                    |     0.24 GB/s |     0.26 GB/s |
| `regex::find_iter`              |     0.20 GB/s |     5.28 GB/s |
| `aho_corasick::find_iter`       |     0.35 GB/s |     0.53 GB/s |
| `stringzilla::find_byteset`     | __1.22 GB/s__ | __8.44 GB/s__ |
|                                 |               |               |
| Python                          |               |               |
| `re.finditer`                   |     0.05 GB/s |     0.20 GB/s |
| `stringzilla.Str.find_first_of` | __0.13 GB/s__ | __8.96 GB/s__ |

> Measured 2026-06-17 on an Intel Xeon Platinum 8468 (Sapphire Rapids).

---

See [README.md](README.md) for dataset information and replication instructions.
