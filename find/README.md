# Substring and Byte-Set Search Benchmarks

Benchmarks for substring search and character-set matching across Rust and Python implementations.

Substring search is one of the most common operations in text processing, and one of the slowest.
Most of the time, programmers don't think about replacing the `str::find` method, as it's already expected to be optimized.
In many languages it's offloaded to the C standard library [`memmem`](https://man7.org/linux/man-pages/man3/memmem.3.html) or [`strstr`](https://en.cppreference.com/w/c/string/byte/strstr) for `NULL`-terminated strings.
The C standard library is, however, also implemented by humans, and a better solution can be created.

Interestingly, the reverse order search, the `rfind` rows, is almost never implemented in SIMD, assuming fewer people ever need it.
Still, those are provided by StringZilla mostly for parsing tasks and feature parity.

The `find_byteset` and `find_first_of` rows search for a set of characters.
StringWars takes a few representative examples of various character sets that appear in real parsing or string validation tasks:

- tabulation characters, like `\n\r\v\f`;
- HTML and XML markup characters, like `</>&'\"=[]`;
- numeric characters, like `0123456789`.

It's common in such cases, to pre-construct some library-specific filter-object or Finite State Machine (FSM) to search for a set of characters.
Once that object is constructed, all of its inclusions in each token (word or line) are counted.

Every table reports single-core throughput in bytes per second, for queries over whitespace-delimited words (_short words_) and newline-delimited lines (_long lines_).
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `std::str::find`, `memmem::rfind`, `aho_corasick::find_iter`.

| StringZilla                              |     Speed | Speedup | Alternative          |     Speed |
| :--------------------------------------- | --------: | ------: | :------------------- | --------: |
| `stringzilla::find`, short words         |   10 GB/s |   1.11× | `memmem::Finder`     |    9 GB/s |
| `stringzilla::find`, long lines          |   17 GB/s |   1.00× | `memmem::find`       |   17 GB/s |
| `stringzilla::rfind`, short words        |   11 GB/s |   4.62× | `memmem::FinderRev`  | 2.38 GB/s |
| `stringzilla::rfind`, long lines         |   16 GB/s |   4.10× | `std::str::rfind`    |  3.9 GB/s |
| `stringzilla::find_byteset`, short words | 1.09 GB/s |   1.07× | `bstr::find_byteset` | 1.02 GB/s |
| `stringzilla::find_byteset`, long lines  |  8.3 GB/s |   1.47× | `regex::find_iter`   | 5.65 GB/s |

> Measured October 14, 2026.

### Python

| StringZilla                                  |     Speed | Speedup | Alternative   |     Speed |
| :------------------------------------------- | --------: | ------: | :------------ | --------: |
| `stringzilla.Str.find`, short words          |  2.0 GB/s |   1.40× | `str.find`    | 1.43 GB/s |
| `stringzilla.Str.find`, long lines           |   16 GB/s |   9.14× | `str.find`    | 1.75 GB/s |
| `stringzilla.Str.rfind`, short words         |  1.8 GB/s |   1.29× | `str.rfind`   |  1.4 GB/s |
| `stringzilla.Str.rfind`, long lines          |   16 GB/s |   5.54× | `str.rfind`   | 2.89 GB/s |
| `stringzilla.Str.find_first_of`, short words | 2.14 GB/s |   4.99× | `re.finditer` |  429 MB/s |
| `stringzilla.Str.find_first_of`, long lines  |  2.3 GB/s |   6.50× | `re.finditer` |  354 MB/s |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `std::str::find`, `memmem::find`, `std::str::rfind`, `aho_corasick::find_iter`.

| StringZilla                              |      Speed | Speedup | Alternative          |      Speed |
| :--------------------------------------- | ---------: | ------: | :------------------- | ---------: |
| `stringzilla::find`, short words         | 27.98 GB/s |   0.79× | `memmem::Finder`     | 35.25 GB/s |
| `stringzilla::find`, long lines          | 30.91 GB/s |   0.65× | `memmem::Finder`     | 47.23 GB/s |
| `stringzilla::rfind`, short words        | 26.93 GB/s |   4.72× | `memmem::FinderRev`  |  5.70 GB/s |
| `stringzilla::rfind`, long lines         | 30.59 GB/s |   7.35× | `memmem::rfind`      |  4.16 GB/s |
| `stringzilla::find_byteset`, short words |  1.43 GB/s |   1.04× | `bstr::find_byteset` |  1.37 GB/s |
| `stringzilla::find_byteset`, long lines  | 12.20 GB/s |   1.39× | `regex::find_iter`   |  8.76 GB/s |

> Measured July 29, 2026.

### Python

Also benchmarked: `pyahocorasick.iter`.

| StringZilla                                  |      Speed | Speedup | Alternative   |      Speed |
| :------------------------------------------- | ---------: | ------: | :------------ | ---------: |
| `stringzilla.Str.find`, short words          | 24.96 GB/s |   8.82× | `str.find`    |  2.83 GB/s |
| `stringzilla.Str.find`, long lines           | 35.46 GB/s |   2.11× | `str.find`    | 16.79 GB/s |
| `stringzilla.Str.rfind`, short words         | 21.01 GB/s |   7.96× | `str.rfind`   |  2.64 GB/s |
| `stringzilla.Str.rfind`, long lines          | 30.46 GB/s |   7.91× | `str.rfind`   |  3.85 GB/s |
| `stringzilla.Str.find_first_of`, short words |  3.73 GB/s |   5.20× | `re.finditer` |   717 MB/s |
| `stringzilla.Str.find_first_of`, long lines  |  4.07 GB/s |   6.91× | `re.finditer` |   589 MB/s |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
