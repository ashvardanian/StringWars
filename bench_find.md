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
| `std::str::find`    |         9.45 GiB/s |       10.88 GiB/s |
| `memmem::find`      |         9.48 GiB/s |       10.83 GiB/s |
| `memmem::Finder`    |         9.51 GiB/s |   __10.99 GiB/s__ |
| `stringzilla::find` |    __10.51 GiB/s__ |       10.82 GiB/s |
|                     |                    |                   |
| Python              |                    |                   |
| `str.find`          |         1.05 GiB/s |        1.23 GiB/s |
| `stringzilla.find`  |    __10.82 GiB/s__ |   __11.79 GiB/s__ |

### Reverse Search (rfind)

Interestingly, the reverse order search is almost never implemented in SIMD, assuming fewer people ever need it.
Still, those are provided by StringZilla mostly for parsing tasks and feature parity.

| Library              | Short Word Queries | Long Line Queries |
| -------------------- | -----------------: | ----------------: |
| Rust                 |                    |                   |
| `std::str::rfind`    |         2.72 GiB/s |        5.94 GiB/s |
| `memmem::rfind`      |         2.70 GiB/s |        5.90 GiB/s |
| `memmem::FinderRev`  |         2.79 GiB/s |        5.81 GiB/s |
| `stringzilla::rfind` |    __10.34 GiB/s__ |   __10.66 GiB/s__ |
|                      |                    |                   |
| Python               |                    |                   |
| `str.rfind`          |         1.54 GiB/s |        3.84 GiB/s |
| `stringzilla.rfind`  |     __7.15 GiB/s__ |   __11.56 GiB/s__ |

## Byte-Set Search

StringWars takes a few representative examples of various character sets that appear in real parsing or string validation tasks:

- tabulation characters, like `\n\r\v\f`;
- HTML and XML markup characters, like `</>&'\"=[]`;
- numeric characters, like `0123456789`.

It's common in such cases, to pre-construct some library-specific filter-object or Finite State Machine (FSM) to search for a set of characters.
Once that object is constructed, all of its inclusions in each token (word or line) are counted.

| Library                         |    Short Words |     Long Lines |
| ------------------------------- | -------------: | -------------: |
| Rust                            |                |                |
| `bstr::iter`                    |     0.26 GiB/s |     0.25 GiB/s |
| `regex::find_iter`              |     0.23 GiB/s |     5.22 GiB/s |
| `aho_corasick::find_iter`       |     0.41 GiB/s |     0.50 GiB/s |
| `stringzilla::find_byteset`     | __1.61 GiB/s__ | __8.17 GiB/s__ |
|                                 |                |                |
| Python                          |                |                |
| `re.finditer`                   |     0.04 GiB/s |     0.19 GiB/s |
| `stringzilla.Str.find_first_of` | __0.11 GiB/s__ | __8.79 GiB/s__ |

---

See [README.md](README.md) for dataset information and replication instructions.
