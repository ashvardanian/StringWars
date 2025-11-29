# UTF-8 Processing Benchmarks

Benchmarks for UTF-8 text processing, including whitespace and newline splitting across different languages and hardware platforms.

## Tokenization

Different scripts stress UTF-8 processing in different ways:

- __Korean__: 3-byte Hangul syllables with single-byte whitespace between words - representative for tokenization workloads
- __Chinese__: 3-byte CJK characters with rare whitespace - tests raw byte throughput
- __Arabic__: 2-byte Arabic script with regular punctuation - good for newline splitting benchmarks
- __French__: Mixed 1-2 byte Latin with high diacritic density
- __English__: Mostly 1-byte ASCII baseline

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

| Language      | `stdlib` | `stringzilla` | Speedup |     | Language      | `stdlib` | `stringzilla` | Speedup |
| :------------ | -------: | ------------: | ------: | --- | :------------ | -------: | ------------: | ------: |
| Arabic 🇸🇦     | 232 MB/s |     1004 MB/s |    4.3x |     | Japanese 🇯🇵   | 330 MB/s |     3.51 GB/s |   10.9x |
| Armenian 🇦🇲   | 223 MB/s |      908 MB/s |    4.1x |     | Korean 🇰🇷     | 314 MB/s |      861 MB/s |    2.7x |
| Bengali 🇧🇩    | 314 MB/s |     6.17 GB/s |   20.1x |     | Lithuanian 🇱🇹 | 352 MB/s |      864 MB/s |    2.5x |
| Chinese 🇨🇳    | 325 MB/s |     1.21 GB/s |    3.8x |     | Polish 🇵🇱     | 364 MB/s |      939 MB/s |    2.6x |
| Czech 🇨🇿      | 322 MB/s |      827 MB/s |    2.6x |     | Portuguese 🇧🇷 | 395 MB/s |     2.38 GB/s |    6.2x |
| Dutch 🇳🇱      | 471 MB/s |     4.73 GB/s |   10.3x |     | Russian 🇷🇺    | 217 MB/s |     2.20 GB/s |   10.4x |
| English 🇬🇧    | 482 MB/s |     7.53 GB/s |   16.0x |     | Spanish 🇪🇸    | 414 MB/s |     2.38 GB/s |    5.9x |
| Farsi 🇮🇷      | 235 MB/s |      858 MB/s |    3.7x |     | Tamil 🇮🇳      | 306 MB/s |     6.05 GB/s |   20.2x |
| French 🇫🇷     | 346 MB/s |     1.84 GB/s |    5.4x |     | Turkish 🇹🇷    | 326 MB/s |      852 MB/s |    2.7x |
| Georgian 🇬🇪   | 294 MB/s |      192 MB/s |    0.7x |     | Ukrainian 🇺🇦  | 217 MB/s |     2.09 GB/s |    9.9x |
| German 🇩🇪     | 432 MB/s |     2.59 GB/s |    6.1x |     | Vietnamese 🇻🇳 | 265 MB/s |      352 MB/s |    1.3x |
| Greek 🇬🇷      | 220 MB/s |     1.00 GB/s |    4.7x |     | Hebrew 🇮🇱     | 233 MB/s |     1.01 GB/s |    4.4x |
| Hindi 🇮🇳      | 293 MB/s |     6.32 GB/s |   22.1x |     | Italian 🇮🇹    | 439 MB/s |     2.29 GB/s |    5.3x |

To rerun the benchmarks for all languages:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bench bench_unicode --features bench_unicode
bin=$(find target/release/deps -name 'bench_unicode-*' -executable -type f | head -1)

for f in leipzig1M_*.txt; do
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold" "$bin"
done
```

---

See [README.md](README.md) for dataset information and replication instructions.
