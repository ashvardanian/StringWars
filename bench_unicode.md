# UTF-8 Processing Benchmarks

Benchmarks for UTF-8 text processing, including whitespace and newline splitting across different languages and hardware platforms.

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

## Case Folding

### AMD Zen5 Turin

| Language     | Standard 🦀 | StringZilla 🦀 |      | Standard 🐍 | StringZilla 🐍 |      |
| :----------- | ---------: | ------------: | ---: | ---------: | ------------: | ---: |
| English 🇬🇧    |   482 MB/s |     7.53 GB/s |  16x |   257 MB/s |     3.14 GB/s |  12x |
| German 🇩🇪     |   432 MB/s |     2.59 GB/s |   6x |   260 MB/s |     1.81 GB/s |   7x |
| Russian 🇷🇺    |   217 MB/s |     2.20 GB/s |  10x |   470 MB/s |     1.56 GB/s |   3x |
| French 🇫🇷     |   346 MB/s |     1.84 GB/s |   5x |   274 MB/s |     1.37 GB/s |   5x |
| Greek 🇬🇷      |   220 MB/s |     1.00 GB/s |   5x |   431 MB/s |      779 MB/s |   2x |
| Armenian 🇦🇲   |   223 MB/s |      908 MB/s |   4x |   470 MB/s |      746 MB/s |   2x |
| Vietnamese 🇻🇳 |   265 MB/s |      352 MB/s |   1x |   340 MB/s |      291 MB/s |   1x |
| Arabic 🇸🇦     |   232 MB/s |     1004 MB/s |   4x |   467 MB/s |     1.80 GB/s |   4x |
| Bengali 🇧🇩    |   314 MB/s |     6.17 GB/s |  20x |   694 MB/s |     2.91 GB/s |   4x |
| Chinese 🇨🇳    |   325 MB/s |     1.21 GB/s |   4x |   697 MB/s |      886 MB/s |   1x |
| Czech 🇨🇿      |   322 MB/s |      827 MB/s |   3x |   292 MB/s |      688 MB/s |   2x |
| Dutch 🇳🇱      |   471 MB/s |     4.73 GB/s |  10x |   262 MB/s |     2.97 GB/s |  11x |
| Farsi 🇮🇷      |   235 MB/s |      858 MB/s |   4x |   475 MB/s |     1.42 GB/s |   3x |
| Georgian 🇬🇪   |   294 MB/s |      192 MB/s |   1x |   689 MB/s |      488 MB/s |   1x |
| Hebrew 🇮🇱     |   233 MB/s |     1.01 GB/s |   4x |   473 MB/s |     1.86 GB/s |   4x |
| Italian 🇮🇹    |   439 MB/s |     2.29 GB/s |   5x |   268 MB/s |     1.93 GB/s |   7x |
| Japanese 🇯🇵   |   330 MB/s |     3.51 GB/s |  11x |   726 MB/s |     2.00 GB/s |   3x |
| Korean 🇰🇷     |   314 MB/s |      861 MB/s |   3x |   623 MB/s |     2.80 GB/s |   4x |
| Lithuanian 🇱🇹 |   352 MB/s |      864 MB/s |   2x |   274 MB/s |      728 MB/s |   3x |
| Polish 🇵🇱     |   364 MB/s |      939 MB/s |   3x |   277 MB/s |      786 MB/s |   3x |
| Portuguese 🇧🇷 |   395 MB/s |     2.38 GB/s |   6x |   270 MB/s |     1.79 GB/s |   7x |
| Spanish 🇪🇸    |   414 MB/s |     2.38 GB/s |   6x |   272 MB/s |     1.80 GB/s |   7x |
| Tamil 🇮🇳      |   306 MB/s |     6.05 GB/s |  20x |   712 MB/s |     3.03 GB/s |   4x |
| Turkish 🇹🇷    |   326 MB/s |      852 MB/s |   3x |   284 MB/s |      706 MB/s |   2x |
| Ukrainian 🇺🇦  |   217 MB/s |     2.09 GB/s |  10x |   476 MB/s |     1.58 GB/s |   3x |

To rerun the benchmarks for all languages:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bench bench_unicode --features bench_unicode
bin=$(find target/release/deps -name 'bench_unicode-*' -executable -type f | head -1)

for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold" "$bin"
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold/" uv run bench_unicode.py
done
```

## Case-Insensitive Substring Search

| Language     | Standard 🦀 | StringZilla 🦀 |      | Standard 🐍 | StringZilla 🐍 |      |
| :----------- | ---------: | ------------: | ---: | ---------: | ------------: | ---: |
| Arabic 🇸🇦     |   200 MB/s |    38.55 GB/s | 193x |  3.01 GB/s |    14.78 GB/s |   5x |
| Armenian 🇦🇲   |   190 MB/s |      980 MB/s |   5x |  2.07 GB/s |      860 MB/s |   0x |
| Bengali 🇧🇩    |   300 MB/s |    28.20 GB/s |  94x |  4.51 GB/s |    21.19 GB/s |   5x |
| Chinese 🇨🇳    |   240 MB/s |    25.65 GB/s | 107x |  5.40 GB/s |    13.94 GB/s |   3x |
| Czech 🇨🇿      |    90 MB/s |     7.41 GB/s |  82x |  1.38 GB/s |     6.36 GB/s |   5x |
| Dutch 🇳🇱      |    90 MB/s |    12.61 GB/s | 140x |   860 MB/s |     7.99 GB/s |   9x |
| English 🇬🇧    |    80 MB/s |    12.79 GB/s | 160x |   770 MB/s |     5.61 GB/s |   7x |
| Farsi 🇮🇷      |   190 MB/s |    26.22 GB/s | 138x |  2.36 GB/s |    10.70 GB/s |   5x |
| French 🇫🇷     |    90 MB/s |    10.77 GB/s | 120x |  1.10 GB/s |     6.83 GB/s |   6x |
| Georgian 🇬🇪   |   190 MB/s |     1.03 GB/s |   5x |  3.20 GB/s |      620 MB/s |   0x |
| German 🇩🇪     |    80 MB/s |    10.67 GB/s | 133x |   900 MB/s |     6.08 GB/s |   7x |
| Greek 🇬🇷      |   130 MB/s |     2.57 GB/s |  20x |  1.38 GB/s |     2.48 GB/s |   2x |
| Hebrew 🇮🇱     |   190 MB/s |    34.54 GB/s | 182x |  2.92 GB/s |    15.72 GB/s |   5x |
| Italian 🇮🇹    |    80 MB/s |    12.99 GB/s | 162x |   970 MB/s |     8.87 GB/s |   9x |
| Japanese 🇯🇵   |   220 MB/s |    21.71 GB/s |  99x |  4.88 GB/s |    13.17 GB/s |   3x |
| Korean 🇰🇷     |   230 MB/s |    35.10 GB/s | 153x |  4.59 GB/s |    20.05 GB/s |   4x |
| Polish 🇵🇱     |    90 MB/s |    10.50 GB/s | 117x |  1.29 GB/s |     8.02 GB/s |   6x |
| Portuguese 🇧🇷 |    90 MB/s |    10.72 GB/s | 119x |  1.10 GB/s |     8.12 GB/s |   7x |
| Russian 🇷🇺    |   140 MB/s |     7.12 GB/s |  51x |  2.30 GB/s |     5.70 GB/s |   2x |
| Spanish 🇪🇸    |    90 MB/s |    11.62 GB/s | 129x |  1.02 GB/s |     6.33 GB/s |   6x |
| Tamil 🇮🇳      |   270 MB/s |    29.53 GB/s | 109x |  5.81 GB/s |    23.11 GB/s |   4x |
| Turkish 🇹🇷    |    90 MB/s |     8.18 GB/s |  91x |  1.49 GB/s |     5.25 GB/s |   4x |
| Ukrainian 🇺🇦  |   140 MB/s |     8.88 GB/s |  63x |  2.26 GB/s |     5.35 GB/s |   2x |
| Vietnamese 🇻🇳 |   110 MB/s |     4.25 GB/s |  39x |  1.07 GB/s |     1.12 GB/s |   1x |

To rerun the benchmarks for all languages:

```bash
for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=words STRINGWARS_FILTER="case-insensitive-find" STRINGWARS_UNIQUE=1 "$bin"
done
```

---

See [README.md](README.md) for dataset information and replication instructions.
