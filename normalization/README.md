# Case Folding & Normalization Benchmarks

Benchmarks for Unicode case-insensitive operations and normalization — case folding, case-insensitive comparison and substring search, and NFC/NFD/NFKC/NFKD normalization — across different languages and hardware platforms.

## Case Folding

Measured on the full Leipzig corpora (`STRINGWARS_TOKENS=file`), single-threaded.
`Standard` is `std::to_lowercase` and `str.casefold()`, `StringZilla` is `utf8_uncased_fold`.

### Intel Xeon4 Sapphire Rapids

| Language     | Standard 🦀 | StringZilla 🦀 |      | Standard 🐍 | StringZilla 🐍 |      |
| :----------- | ---------: | ------------: | ---: | ---------: | ------------: | ---: |
| Arabic 🇸🇦     |   115 MB/s |     3.39 GB/s |  29x |   318 MB/s |     1.51 GB/s |   5x |
| Armenian 🇦🇲   |   112 MB/s |      551 MB/s |   5x |   320 MB/s |      543 MB/s |   2x |
| Bengali 🇧🇩    |   155 MB/s |     3.23 GB/s |  21x |   464 MB/s |     1.47 GB/s |   3x |
| Chinese 🇨🇳    |   163 MB/s |      622 MB/s |   4x |   464 MB/s |      452 MB/s |   1x |
| Czech 🇨🇿      |   137 MB/s |     1.87 GB/s |  14x |   196 MB/s |     1.11 GB/s |   6x |
| Dutch 🇳🇱      |   223 MB/s |     5.09 GB/s |  23x |   174 MB/s |     1.74 GB/s |  10x |
| English 🇬🇧    |   219 MB/s |     4.72 GB/s |  22x |   178 MB/s |     1.85 GB/s |  10x |
| Farsi 🇮🇷      |   116 MB/s |     1.62 GB/s |  14x |   317 MB/s |      933 MB/s |   3x |
| French 🇫🇷     |   194 MB/s |     2.45 GB/s |  13x |   182 MB/s |     1.23 GB/s |   7x |
| German 🇩🇪     |   207 MB/s |     2.99 GB/s |  14x |   179 MB/s |     1.34 GB/s |   7x |
| Greek 🇬🇷      |   112 MB/s |     2.05 GB/s |  18x |   293 MB/s |     1.22 GB/s |   4x |
| Hebrew 🇮🇱     |   115 MB/s |     3.52 GB/s |  31x |   309 MB/s |     1.49 GB/s |   5x |
| Hindi 🇮🇳      |   150 MB/s |     3.26 GB/s |  22x |   453 MB/s |     1.48 GB/s |   3x |
| Italian 🇮🇹    |   214 MB/s |     4.19 GB/s |  20x |   184 MB/s |     1.62 GB/s |   9x |
| Japanese 🇯🇵   |   163 MB/s |     1.70 GB/s |  10x |   491 MB/s |     1.05 GB/s |   2x |
| Korean 🇰🇷     |   155 MB/s |     2.27 GB/s |  15x |   429 MB/s |     1.44 GB/s |   3x |
| Polish 🇵🇱     |   178 MB/s |     1.72 GB/s |  10x |   186 MB/s |     1.01 GB/s |   5x |
| Portuguese 🇧🇷 |   199 MB/s |     3.22 GB/s |  16x |   182 MB/s |     1.32 GB/s |   7x |
| Russian 🇷🇺    |   110 MB/s |     1.87 GB/s |  17x |   313 MB/s |     1.06 GB/s |   3x |
| Spanish 🇪🇸    |   205 MB/s |     3.07 GB/s |  15x |   180 MB/s |     1.35 GB/s |   7x |
| Tamil 🇮🇳      |   157 MB/s |     3.32 GB/s |  21x |   467 MB/s |     1.35 GB/s |   3x |
| Turkish 🇹🇷    |   153 MB/s |     1.69 GB/s |  11x |   193 MB/s |      979 MB/s |   5x |
| Ukrainian 🇺🇦  |   110 MB/s |     1.77 GB/s |  16x |   316 MB/s |      996 MB/s |   3x |
| Vietnamese 🇻🇳 |   135 MB/s |     2.00 GB/s |  15x |   227 MB/s |     1.12 GB/s |   5x |

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
RUSTFLAGS="-C target-cpu=native" cargo build --release --bench bench_normalization --features bench_normalization
bin=$(find target/release/deps -name 'bench_normalization-*' -executable -type f | head -1)

for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold" "$bin"
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold/" uv run normalization/bench.py
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

See [README.md](../README.md) for dataset information and replication instructions.
