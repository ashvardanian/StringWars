# UTF-8 Tokenization & Iteration Benchmarks

Benchmarks for UTF-8 segmentation and codepoint iteration — whitespace, newline, and TR29 word
splitting, UTF-8 character counting and decoding, and locating the Nth codepoint — across different
languages and hardware platforms.

## Tokenization

Different scripts stress UTF-8 processing in different ways:

- **Korean**: 3-byte Hangul syllables with single-byte whitespace between words - representative for tokenization workloads
- **Chinese**: 3-byte CJK characters with rare whitespace - tests raw byte throughput
- **Arabic**: 2-byte Arabic script with regular punctuation - good for newline splitting benchmarks
- **French**: Mixed 1-2 byte Latin with high diacritic density
- **English**: Mostly 1-byte ASCII baseline

### AMD Zen5 Turin

On AMD Zen5 Turin CPUs on different datasets, StringZilla provides the following throughput for splitting around whitespace and newline characters on 5 vastly different languages.

| Library                                   |    English |    Chinese |     Arabic |     French |     Korean |
| ----------------------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Split around 25 whitespace characters:    |            |            |            |            |            |
| `stringzilla::utf8_whitespace_splits`     |  0.82 GB/s |  2.40 GB/s |  2.40 GB/s |  0.92 GB/s |  1.88 GB/s |
| `stdlib::split(char::is_whitespace)`      |  0.77 GB/s |  1.87 GB/s |  1.04 GB/s |  0.72 GB/s |  0.98 GB/s |
| `icu::WhiteSpace`                         |  0.11 GB/s |  0.16 GB/s |  0.15 GB/s |  0.12 GB/s |  0.15 GB/s |
|                                           |            |            |            |            |            |
| Split around 8 newline combinations:      |            |            |            |            |            |
| `stringzilla::utf8_newline_splits`        | 15.45 GB/s | 16.65 GB/s | 18.34 GB/s | 14.52 GB/s | 16.71 GB/s |
| `stdlib::split(char::is_unicode_newline)` |  1.90 GB/s |  1.93 GB/s |  1.82 GB/s |  1.78 GB/s |  1.81 GB/s |

### Apple M2 Pro

| Library                                   |   English |   Chinese |    Arabic |    French |    Korean |
| ----------------------------------------- | --------: | --------: | --------: | --------: | --------: |
| Split around 25 whitespace characters:    |           |           |           |           |           |
| `stringzilla::utf8_whitespace_splits`     | 0.57 GB/s | 2.45 GB/s | 1.18 GB/s | 0.61 GB/s | 0.92 GB/s |
| `stdlib::split(char::is_whitespace)`      | 0.59 GB/s | 1.16 GB/s | 0.99 GB/s | 0.63 GB/s | 0.89 GB/s |
| `icu::WhiteSpace`                         | 0.10 GB/s | 0.16 GB/s | 0.14 GB/s | 0.11 GB/s | 0.14 GB/s |
|                                           |           |           |           |           |           |
| Split around 8 newline combinations:      |           |           |           |           |           |
| `stringzilla::utf8_newline_splits`        | 5.69 GB/s | 6.24 GB/s | 6.58 GB/s | 6.70 GB/s | 6.29 GB/s |
| `stdlib::split(char::is_unicode_newline)` | 1.12 GB/s | 1.11 GB/s | 1.11 GB/s | 1.11 GB/s | 1.13 GB/s |

To rerun the benchmarks for all languages:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bench bench_tokenization --features bench_tokenization
bin=$(find target/release/deps -name 'bench_tokenization-*' -executable -type f | head -1)

for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="tokenize" "$bin"
done
```

---

See [README.md](../README.md) for dataset information and replication instructions.
