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
| `std::str::find`    |          9.45 GB/s |        10.88 GB/s |
| `memmem::find`      |          9.48 GB/s |        10.83 GB/s |
| `memmem::Finder`    |          9.51 GB/s |    **10.99 GB/s** |
| `stringzilla::find` |     **10.51 GB/s** |        10.82 GB/s |
|                     |                    |                   |
| Python              |                    |                   |
| `str.find`          |          1.05 GB/s |         1.23 GB/s |
| `stringzilla.find`  |     **10.82 GB/s** |    **11.79 GB/s** |

### Reverse Search (rfind)

Interestingly, the reverse order search is almost never implemented in SIMD, assuming fewer people ever need it.
Still, those are provided by StringZilla mostly for parsing tasks and feature parity.

| Library              | Short Word Queries | Long Line Queries |
| -------------------- | -----------------: | ----------------: |
| Rust                 |                    |                   |
| `std::str::rfind`    |          2.72 GB/s |         5.94 GB/s |
| `memmem::rfind`      |          2.70 GB/s |         5.90 GB/s |
| `memmem::FinderRev`  |          2.79 GB/s |         5.81 GB/s |
| `stringzilla::rfind` |     **10.34 GB/s** |    **10.66 GB/s** |
|                      |                    |                   |
| Python               |                    |                   |
| `str.rfind`          |          1.54 GB/s |         3.84 GB/s |
| `stringzilla.rfind`  |      **7.15 GB/s** |    **11.56 GB/s** |

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
| `bstr::iter`                    |     0.26 GB/s |     0.25 GB/s |
| `regex::find_iter`              |     0.23 GB/s |     5.22 GB/s |
| `aho_corasick::find_iter`       |     0.41 GB/s |     0.50 GB/s |
| `stringzilla::find_byteset`     | **1.61 GB/s** | **8.17 GB/s** |
|                                 |               |               |
| Python                          |               |               |
| `re.finditer`                   |     0.04 GB/s |     0.19 GB/s |
| `stringzilla.Str.find_first_of` | **0.11 GB/s** | **8.79 GB/s** |

---

See [README.md](README.md) for dataset information and replication instructions.
