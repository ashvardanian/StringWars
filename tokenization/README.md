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

### Intel Xeon4 Sapphire Rapids

| Library                                   |   English |   Chinese |    Arabic |    French |    Korean |
| ----------------------------------------- | --------: | --------: | --------: | --------: | --------: |
| Split around 25 whitespace characters:    |           |           |           |           |           |
| `stringzilla::utf8_whitespace_splits`     | 0.97 GB/s | 2.24 GB/s | 1.41 GB/s | 0.99 GB/s | 1.44 GB/s |
| `stdlib::split(char::is_whitespace)`      | 0.36 GB/s | 0.89 GB/s | 0.52 GB/s | 0.34 GB/s | 0.60 GB/s |
| `icu::WhiteSpace`                         | 0.09 GB/s | 0.27 GB/s | 0.18 GB/s | 0.09 GB/s | 0.22 GB/s |
|                                           |           |           |           |           |           |
| Split around 8 newline combinations:      |           |           |           |           |           |
| `stringzilla::utf8_newline_splits`        | 3.87 GB/s | 3.34 GB/s | 4.07 GB/s | 3.57 GB/s | 3.86 GB/s |
| `stdlib::split(char::is_unicode_newline)` | 0.96 GB/s | 1.20 GB/s | 0.51 GB/s | 0.73 GB/s | 0.68 GB/s |
|                                           |           |           |           |           |           |
| TR29 word segmentation:                   |           |           |           |           |           |
| `stringzilla::utf8_word_splits`           | 0.10 GB/s | 0.16 GB/s | 0.10 GB/s | 0.10 GB/s | 0.21 GB/s |
| `unicode-segmentation::unicode_words`     | 0.06 GB/s | 0.08 GB/s | 0.10 GB/s | 0.06 GB/s | 0.15 GB/s |
| `unicode-segmentation::split_word_bounds` | 0.06 GB/s | 0.12 GB/s | 0.11 GB/s | 0.06 GB/s | 0.19 GB/s |
| `icu::WordSegmenter`                      | 0.11 GB/s | 0.02 GB/s | 0.21 GB/s | 0.11 GB/s | 0.19 GB/s |

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

## Codepoint Operations

Counting codepoints (`count_utf8`) and locating the byte offset of the Nth codepoint (`find_nth_utf8`), on the full Leipzig corpora, single-threaded, decimal GB/s.
`find_nth_utf8` targets the last codepoint, so every implementation scans the whole buffer.

### Intel Xeon4 Sapphire Rapids

| Library                           |    English |    Chinese |     Arabic |     French |     Korean |
| --------------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Count UTF-8 codepoints:           |            |            |            |            |            |
| `stringzilla::count_utf8`         | 16.33 GB/s | 18.13 GB/s | 14.23 GB/s | 17.35 GB/s | 15.23 GB/s |
| `stringzilla::utf8_chars().len()` | 16.40 GB/s | 17.78 GB/s | 14.23 GB/s | 17.49 GB/s | 15.69 GB/s |
| `simdutf::count_utf8`             | 18.68 GB/s | 20.60 GB/s | 15.55 GB/s | 17.76 GB/s | 17.01 GB/s |
| `stdlib::chars().count()`         |  9.95 GB/s | 10.65 GB/s |  9.08 GB/s | 10.36 GB/s |  9.43 GB/s |
|                                   |            |            |            |            |            |
| Byte offset of the Nth codepoint: |            |            |            |            |            |
| `stringzilla::find_nth_utf8`      | 14.44 GB/s | 16.00 GB/s | 13.11 GB/s | 15.65 GB/s | 13.88 GB/s |
| `stdlib::char_indices().nth()`    |  1.60 GB/s |  1.06 GB/s |  0.67 GB/s |  1.09 GB/s |  0.72 GB/s |

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
