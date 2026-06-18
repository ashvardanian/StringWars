# UTF-8 Tokenization & Iteration Benchmarks

Benchmarks for UTF-8 segmentation and codepoint iteration — whitespace, newline, and TR29 word
splitting, UTF-8 character counting and decoding, and locating the Nth codepoint — across different
languages and hardware platforms.

## Tokenization

Different scripts stress UTF-8 processing in different ways:

- __Korean__: 3-byte Hangul syllables with single-byte whitespace between words - representative for tokenization workloads
- __Chinese__: 3-byte CJK characters with rare whitespace - tests raw byte throughput
- __Arabic__: 2-byte Arabic script with regular punctuation - good for newline splitting benchmarks
- __French__: Mixed 1-2 byte Latin with high diacritic density
- __English__: Mostly 1-byte ASCII baseline

### Intel Xeon4 Sapphire Rapids

| Library                                   |       English |       Chinese |        Arabic |        French |        Korean |
| ----------------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Split around 25 whitespace characters:    |               |               |               |               |               |
| `stringzilla::utf8_whitespace_splits`     | __0.44 GB/s__ | __1.10 GB/s__ | __0.66 GB/s__ | __0.43 GB/s__ | __0.70 GB/s__ |
| `std::split<is_whitespace>`               |     0.27 GB/s |     0.59 GB/s |     0.35 GB/s |     0.26 GB/s |     0.42 GB/s |
| `icu::WhiteSpace`                         |     0.05 GB/s |     0.15 GB/s |     0.10 GB/s |     0.06 GB/s |     0.20 GB/s |
|                                           |               |               |               |               |               |
| Split around 8 newline combinations:      |               |               |               |               |               |
| `stringzilla::utf8_newline_splits`        | __1.90 GB/s__ | __1.64 GB/s__ | __2.39 GB/s__ | __1.72 GB/s__ | __3.18 GB/s__ |
| `std::split<is_unicode_newline>`          |     0.44 GB/s |     0.75 GB/s |     0.40 GB/s |     0.39 GB/s |     0.70 GB/s |
|                                           |               |               |               |               |               |
| TR29 word segmentation:                   |               |               |               |               |               |
| `stringzilla::utf8_word_splits`           | __0.07 GB/s__ | __0.11 GB/s__ |     0.06 GB/s |     0.06 GB/s | __0.19 GB/s__ |
| `unicode-segmentation::unicode_words`     |     0.04 GB/s |     0.05 GB/s |     0.06 GB/s |     0.04 GB/s |     0.15 GB/s |
| `unicode-segmentation::split_word_bounds` |     0.04 GB/s |     0.08 GB/s |     0.07 GB/s |     0.04 GB/s |     0.17 GB/s |
| `icu::WordSegmenter`                      | __0.07 GB/s__ |     0.01 GB/s | __0.14 GB/s__ | __0.08 GB/s__ |     0.18 GB/s |

> Measured June 17, 2026 on an Intel Xeon4 Sapphire Rapids.

### AMD Zen5 Turin

On AMD Zen5 Turin CPUs on different datasets, StringZilla provides the following throughput for splitting around whitespace and newline characters on 5 vastly different languages.

| Library                                |        English |        Chinese |         Arabic |         French |         Korean |
| -------------------------------------- | -------------: | -------------: | -------------: | -------------: | -------------: |
| Split around 25 whitespace characters: |                |                |                |                |                |
| `stringzilla::utf8_whitespace_splits`  |  __0.82 GB/s__ |  __2.40 GB/s__ |  __2.40 GB/s__ |  __0.92 GB/s__ |  __1.88 GB/s__ |
| `std::split<is_whitespace>`            |      0.77 GB/s |      1.87 GB/s |      1.04 GB/s |      0.72 GB/s |      0.98 GB/s |
| `icu::WhiteSpace`                      |      0.11 GB/s |      0.16 GB/s |      0.15 GB/s |      0.12 GB/s |      0.15 GB/s |
|                                        |                |                |                |                |                |
| Split around 8 newline combinations:   |                |                |                |                |                |
| `stringzilla::utf8_newline_splits`     | __15.45 GB/s__ | __16.65 GB/s__ | __18.34 GB/s__ | __14.52 GB/s__ | __16.71 GB/s__ |
| `std::split<is_unicode_newline>`       |      1.90 GB/s |      1.93 GB/s |      1.82 GB/s |      1.78 GB/s |      1.81 GB/s |

### Apple M2 Pro

| Library                                |       English |       Chinese |        Arabic |        French |        Korean |
| -------------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Split around 25 whitespace characters: |               |               |               |               |               |
| `stringzilla::utf8_whitespace_splits`  |     0.57 GB/s | __2.45 GB/s__ | __1.18 GB/s__ |     0.61 GB/s | __0.92 GB/s__ |
| `std::split<is_whitespace>`            | __0.59 GB/s__ |     1.16 GB/s |     0.99 GB/s | __0.63 GB/s__ |     0.89 GB/s |
| `icu::WhiteSpace`                      |     0.10 GB/s |     0.16 GB/s |     0.14 GB/s |     0.11 GB/s |     0.14 GB/s |
|                                        |               |               |               |               |               |
| Split around 8 newline combinations:   |               |               |               |               |               |
| `stringzilla::utf8_newline_splits`     | __5.69 GB/s__ | __6.24 GB/s__ | __6.58 GB/s__ | __6.70 GB/s__ | __6.29 GB/s__ |
| `std::split<is_unicode_newline>`       |     1.12 GB/s |     1.11 GB/s |     1.11 GB/s |     1.11 GB/s | __1.13 GB/s__ |

## Codepoint Operations

Counting codepoints (`count_utf8`) and locating the byte offset of the Nth codepoint (`find_nth_utf8`), on the full Leipzig corpora, single-threaded, decimal GB/s.
`find_nth_utf8` targets the last codepoint, so every implementation scans the whole buffer.
Counting is memory-bandwidth-bound, so every implementation converges near 7–14 GB/s and `simdutf` edges ahead on the multi-byte scripts; the decisive win is `find_nth_utf8`, where StringZilla is an order of magnitude faster than the standard library.

### Intel Xeon4 Sapphire Rapids

| Library                           |       English |       Chinese |        Arabic |        French |         Korean |
| --------------------------------- | ------------: | ------------: | ------------: | ------------: | -------------: |
| Count UTF-8 codepoints:           |               |               |               |               |                |
| `stringzilla::count_utf8`         | __7.24 GB/s__ |     6.93 GB/s |     7.23 GB/s | __7.23 GB/s__ |     13.28 GB/s |
| `simdutf::count_utf8`             |     7.06 GB/s | __7.67 GB/s__ | __8.15 GB/s__ |     7.08 GB/s | __14.05 GB/s__ |
| `std::chars.count`                |     6.59 GB/s |     6.85 GB/s |     7.07 GB/s |     6.54 GB/s |      8.37 GB/s |
|                                   |               |               |               |               |                |
| Byte offset of the Nth codepoint: |               |               |               |               |                |
| `stringzilla::find_nth_utf8`      | __7.14 GB/s__ | __7.57 GB/s__ | __8.78 GB/s__ | __7.62 GB/s__ | __12.14 GB/s__ |
| `std::char_indices.nth`           |     0.83 GB/s |     0.97 GB/s |     0.57 GB/s |     0.67 GB/s |      0.72 GB/s |

> Measured June 17, 2026 on an Intel Xeon4 Sapphire Rapids.

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
