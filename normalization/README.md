# Case Folding & Normalization Benchmarks

Benchmarks for Unicode case-insensitive operations and normalization — case folding, case-insensitive comparison and substring search, and NFC/NFD/NFKC/NFKD normalization — across different languages and hardware platforms.

## Case Folding

Measured single-threaded on a 512 KB prefix of each Leipzig corpus, as one `file`-mode token.
The 512 KB size is set by `pyunormalize`, which sustains ~53 MChar/s to 800 K characters and then collapses by more than four orders of magnitude.
`Standard` is `std::to_lowercase` and `str.casefold()`, `StringZilla` is `utf8_uncased_fold`.

### Intel Xeon4 Sapphire Rapids

| Language      | Standard 🦀 | StringZilla 🦀 |     | Standard 🐍 | StringZilla 🐍 |     |
| ------------- | ----------: | -------------: | --: | ----------: | -------------: | --: |
| Arabic 🇸🇦     |     84 MB/s |      2.10 GB/s | 26x |    179 MB/s |       288 MB/s |  2x |
| Armenian 🇦🇲   |     69 MB/s |       356 MB/s |  5x |    199 MB/s |       172 MB/s |  1x |
| Bengali 🇧🇩    |     94 MB/s |      2.25 GB/s | 24x |    289 MB/s |       362 MB/s |  1x |
| Chinese 🇨🇳    |     92 MB/s |       395 MB/s |  4x |    228 MB/s |       181 MB/s |  1x |
| Czech 🇨🇿      |     92 MB/s |      1.31 GB/s | 15x |    119 MB/s |       217 MB/s |  2x |
| Dutch 🇳🇱      |    130 MB/s |      4.39 GB/s | 35x |    289 MB/s |       316 MB/s |  1x |
| English 🇬🇧    |    129 MB/s |      4.72 GB/s | 37x |    344 MB/s |       375 MB/s |  1x |
| Farsi 🇮🇷      |     80 MB/s |      1.07 GB/s | 14x |    209 MB/s |       258 MB/s |  1x |
| French 🇫🇷     |    119 MB/s |      1.59 GB/s | 14x |    116 MB/s |       227 MB/s |  2x |
| Georgian 🇬🇪   |           — |              — |   — |           — |              — |   — |
| German 🇩🇪     |    127 MB/s |      1.91 GB/s | 15x |    118 MB/s |       268 MB/s |  2x |
| Greek 🇬🇷      |     66 MB/s |      1.17 GB/s | 18x |    185 MB/s |       262 MB/s |  1x |
| Hebrew 🇮🇱     |     71 MB/s |      2.21 GB/s | 32x |    172 MB/s |       248 MB/s |  1x |
| Hindi 🇮🇳      |     93 MB/s |      2.28 GB/s | 25x |    278 MB/s |       350 MB/s |  1x |
| Italian 🇮🇹    |    134 MB/s |      2.86 GB/s | 22x |    145 MB/s |       327 MB/s |  2x |
| Japanese 🇯🇵   |     93 MB/s |      1.09 GB/s | 12x |    231 MB/s |       255 MB/s |  1x |
| Korean 🇰🇷     |    142 MB/s |      2.09 GB/s | 15x |    230 MB/s |       273 MB/s |  1x |
| Polish 🇵🇱     |    112 MB/s |      1.04 GB/s | 10x |    105 MB/s |       186 MB/s |  2x |
| Portuguese 🇧🇷 |    127 MB/s |      2.14 GB/s | 17x |    109 MB/s |       253 MB/s |  2x |
| Russian 🇷🇺    |     66 MB/s |      1.22 GB/s | 19x |    190 MB/s |       275 MB/s |  1x |
| Spanish 🇪🇸    |    124 MB/s |      2.02 GB/s | 17x |    104 MB/s |       267 MB/s |  3x |
| Tamil 🇮🇳      |    108 MB/s |      2.24 GB/s | 21x |    304 MB/s |       376 MB/s |  1x |
| Turkish 🇹🇷    |    101 MB/s |      1.04 GB/s | 11x |    118 MB/s |       217 MB/s |  2x |
| Ukrainian 🇺🇦  |     66 MB/s |      1.15 GB/s | 18x |    194 MB/s |       270 MB/s |  1x |
| Vietnamese 🇻🇳 |     82 MB/s |      1.16 GB/s | 15x |    148 MB/s |       243 MB/s |  2x |

> Measured June 17, 2026.

### AMD Zen5 Turin

| Language      | Standard 🦀 | StringZilla 🦀 |     | Standard 🐍 | StringZilla 🐍 |     |
| ------------- | ----------: | -------------: | --: | ----------: | -------------: | --: |
| English 🇬🇧    |    460 MB/s |      7.01 GB/s | 16x |    245 MB/s |      2.92 GB/s | 12x |
| German 🇩🇪     |    412 MB/s |      2.41 GB/s |  6x |    248 MB/s |      1.69 GB/s |  7x |
| Russian 🇷🇺    |    207 MB/s |      2.05 GB/s | 10x |    448 MB/s |      1.45 GB/s |  3x |
| French 🇫🇷     |    330 MB/s |      1.71 GB/s |  5x |    261 MB/s |      1.28 GB/s |  5x |
| Greek 🇬🇷      |    210 MB/s |     0.931 GB/s |  5x |    411 MB/s |       743 MB/s |  2x |
| Armenian 🇦🇲   |    213 MB/s |       866 MB/s |  4x |    448 MB/s |       711 MB/s |  2x |
| Vietnamese 🇻🇳 |    253 MB/s |       336 MB/s |  1x |    324 MB/s |       278 MB/s |  1x |
| Arabic 🇸🇦     |    221 MB/s |     957.5 MB/s |  4x |    445 MB/s |      1.68 GB/s |  4x |
| Bengali 🇧🇩    |    299 MB/s |      5.75 GB/s | 20x |    662 MB/s |      2.71 GB/s |  4x |
| Chinese 🇨🇳    |    310 MB/s |      1.13 GB/s |  4x |    665 MB/s |       845 MB/s |  1x |
| Czech 🇨🇿      |    307 MB/s |       789 MB/s |  3x |    278 MB/s |       656 MB/s |  2x |
| Dutch 🇳🇱      |    449 MB/s |      4.41 GB/s | 10x |    250 MB/s |      2.77 GB/s | 11x |
| Farsi 🇮🇷      |    224 MB/s |       818 MB/s |  4x |    453 MB/s |      1.32 GB/s |  3x |
| Georgian 🇬🇪   |    280 MB/s |       183 MB/s |  1x |    657 MB/s |       465 MB/s |  1x |
| Hebrew 🇮🇱     |    222 MB/s |     0.941 GB/s |  4x |    451 MB/s |      1.73 GB/s |  4x |
| Hindi 🇮🇳      |           — |              — |   — |           — |              — |   — |
| Italian 🇮🇹    |    419 MB/s |      2.13 GB/s |  5x |    256 MB/s |      1.80 GB/s |  7x |
| Japanese 🇯🇵   |    315 MB/s |      3.27 GB/s | 11x |    692 MB/s |      1.86 GB/s |  3x |
| Korean 🇰🇷     |    299 MB/s |       821 MB/s |  3x |    594 MB/s |      2.61 GB/s |  4x |
| Lithuanian 🇱🇹 |    336 MB/s |       824 MB/s |  2x |    261 MB/s |       694 MB/s |  3x |
| Polish 🇵🇱     |    347 MB/s |       896 MB/s |  3x |    264 MB/s |       750 MB/s |  3x |
| Portuguese 🇧🇷 |    377 MB/s |      2.22 GB/s |  6x |    257 MB/s |      1.67 GB/s |  7x |
| Spanish 🇪🇸    |    395 MB/s |      2.22 GB/s |  6x |    259 MB/s |      1.68 GB/s |  7x |
| Tamil 🇮🇳      |    292 MB/s |      5.63 GB/s | 20x |    679 MB/s |      2.82 GB/s |  4x |
| Turkish 🇹🇷    |    311 MB/s |       813 MB/s |  3x |    271 MB/s |       673 MB/s |  2x |
| Ukrainian 🇺🇦  |    207 MB/s |      1.95 GB/s | 10x |    454 MB/s |      1.47 GB/s |  3x |

### Apple M5 Pro

| Language      | Standard 🦀 | StringZilla 🦀 |       | Standard 🐍 |  PyICU 🐍 | StringZilla 🐍 |       |
| ------------- | ----------: | -------------: | ----: | ----------: | --------: | -------------: | ----: |
| Arabic 🇸🇦     |    304 MB/s |      2.20 GB/s |  7.4x |    873 MB/s |  434 MB/s |      2.17 GB/s |  2.5x |
| Armenian 🇦🇲   |    295 MB/s |       902 MB/s |  3.1x |    890 MB/s |  426 MB/s |       871 MB/s |  1.0x |
| Bengali 🇧🇩    |    421 MB/s |      6.29 GB/s | 15.3x |   1.39 GB/s | 1.32 GB/s |      6.21 GB/s |  4.5x |
| Chinese 🇨🇳    |    441 MB/s |       515 MB/s |  1.2x |   1.28 GB/s |  668 MB/s |       573 MB/s |  0.4x |
| Czech 🇨🇿      |    319 MB/s |      1.30 GB/s |  4.2x |    473 MB/s |  535 MB/s |      1.27 GB/s |  2.7x |
| Dutch 🇳🇱      |    676 MB/s |     14.64 GB/s | 22.2x |    474 MB/s |  478 MB/s |     14.21 GB/s | 30.7x |
| English 🇬🇧    |    680 MB/s |     14.88 GB/s | 22.4x |    535 MB/s |  426 MB/s |     14.85 GB/s | 28.4x |
| Farsi 🇮🇷      |    295 MB/s |      2.47 GB/s |  8.6x |    751 MB/s |  508 MB/s |      2.52 GB/s |  3.4x |
| French 🇫🇷     |    524 MB/s |      1.33 GB/s |  2.6x |    444 MB/s |  485 MB/s |      1.29 GB/s |  3.0x |
| Georgian 🇬🇪   |    424 MB/s |       774 MB/s |  1.8x |   1.01 GB/s |  392 MB/s |       714 MB/s |  0.7x |
| German 🇩🇪     |    625 MB/s |      2.19 GB/s |  3.6x |    544 MB/s |  362 MB/s |      2.10 GB/s |  4.0x |
| Greek 🇬🇷      |    298 MB/s |      1.62 GB/s |  5.6x |    649 MB/s |  196 MB/s |      1.46 GB/s |  2.3x |
| Hebrew 🇮🇱     |    308 MB/s |      2.65 GB/s |  8.8x |    686 MB/s |  391 MB/s |      2.65 GB/s |  4.0x |
| Hindi 🇮🇳      |    400 MB/s |      6.83 GB/s | 17.5x |   1.07 GB/s | 1.17 GB/s |      6.74 GB/s |  6.3x |
| Italian 🇮🇹    |    654 MB/s |      3.89 GB/s |  6.1x |    538 MB/s |  480 MB/s |      3.79 GB/s |  7.2x |
| Japanese 🇯🇵   |    423 MB/s |      1.68 GB/s |  4.1x |   1.32 GB/s |  749 MB/s |      1.79 GB/s |  1.4x |
| Korean 🇰🇷     |    391 MB/s |      1.73 GB/s |  4.5x |   1.01 GB/s | 1.05 GB/s |      1.63 GB/s |  1.6x |
| Polish 🇵🇱     |    492 MB/s |      1.29 GB/s |  2.7x |    452 MB/s |  472 MB/s |      1.25 GB/s |  2.8x |
| Portuguese 🇧🇷 |    582 MB/s |      1.69 GB/s |  3.0x |    441 MB/s |  464 MB/s |      1.62 GB/s |  3.8x |
| Russian 🇷🇺    |    299 MB/s |      1.04 GB/s |  3.6x |    913 MB/s |  254 MB/s |     0.950 GB/s |  1.1x |
| Spanish 🇪🇸    |    611 MB/s |      1.91 GB/s |  3.2x |    435 MB/s |  472 MB/s |      1.81 GB/s |  4.3x |
| Tamil 🇮🇳      |    432 MB/s |      5.84 GB/s | 13.8x |   1.02 GB/s |  710 MB/s |      5.81 GB/s |  5.7x |
| Turkish 🇹🇷    |    401 MB/s |      1.18 GB/s |  3.0x |    585 MB/s |  476 MB/s |      1.15 GB/s |  2.0x |
| Ukrainian 🇺🇦  |    293 MB/s |      1.05 GB/s |  3.7x |    723 MB/s |  289 MB/s |     0.969 GB/s |  1.4x |
| Vietnamese 🇻🇳 |    331 MB/s |      1.34 GB/s |  4.1x |    537 MB/s |  379 MB/s |      1.29 GB/s |  2.5x |

> Measured July 29, 2026.

To rerun the benchmarks for all languages:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bench bench_normalization --features bench_normalization
bin=$(find target/release/deps -name 'bench_normalization-*' -executable -type f | head -1)

for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold" "$bin"
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold/" uv run --group normalization normalization/bench.py
done
```

## Case-Insensitive Substring Search

### Intel Xeon4 Sapphire Rapids

| Language      | Standard 🦀 | StringZilla 🦀 |        | Standard 🐍 | StringZilla 🐍 |      |
| ------------- | ----------: | -------------: | -----: | ----------: | -------------: | ---: |
| Arabic 🇸🇦     |   98.2 MB/s |      6.74 GB/s |  70.3x |   2.80 GB/s |     13.76 GB/s | 4.9x |
| Armenian 🇦🇲   |    129 MB/s |       259 MB/s |   2.0x |   1.93 GB/s |       820 MB/s | 0.4x |
| Bengali 🇧🇩    |    182 MB/s |      6.49 GB/s |  36.5x |   4.20 GB/s |     19.73 GB/s | 4.7x |
| Chinese 🇨🇳    |   99.2 MB/s |      8.12 GB/s |  83.8x |   5.03 GB/s |     12.98 GB/s | 2.6x |
| Czech 🇨🇿      |     38 MB/s |      4.96 GB/s | 133.2x |   1.29 GB/s |      5.92 GB/s | 4.6x |
| Dutch 🇳🇱      |     39 MB/s |      4.03 GB/s | 105.6x |    820 MB/s |      7.44 GB/s | 9.3x |
| English 🇬🇧    |     41 MB/s |      4.57 GB/s | 114.2x |    734 MB/s |      5.22 GB/s | 7.3x |
| Farsi 🇮🇷      |    121 MB/s |      6.17 GB/s |  52.2x |   2.20 GB/s |     9.965 GB/s | 4.5x |
| French 🇫🇷     |     59 MB/s |      4.99 GB/s |  86.5x |   1.02 GB/s |      6.36 GB/s | 6.2x |
| Georgian 🇬🇪   |    181 MB/s |     0.959 GB/s |   5.4x |   2.98 GB/s |       591 MB/s | 0.2x |
| German 🇩🇪     |     45 MB/s |      4.16 GB/s |  95.1x |    858 MB/s |      5.66 GB/s | 6.8x |
| Greek 🇬🇷      |     53 MB/s |      1.55 GB/s |  29.6x |   1.29 GB/s |      2.31 GB/s | 1.8x |
| Hebrew 🇮🇱     |     73 MB/s |      6.39 GB/s |  89.1x |   2.72 GB/s |     14.64 GB/s | 5.4x |
| Hindi 🇮🇳      |           — |              — |      — |           — |              — |    — |
| Italian 🇮🇹    |     59 MB/s |      4.68 GB/s |  81.1x |    925 MB/s |      8.26 GB/s | 9.1x |
| Japanese 🇯🇵   |    101 MB/s |      8.76 GB/s |  88.8x |   4.54 GB/s |     12.27 GB/s | 2.7x |
| Korean 🇰🇷     |    147 MB/s |      9.26 GB/s |  64.5x |   4.27 GB/s |     18.67 GB/s | 4.4x |
| Polish 🇵🇱     |     40 MB/s |      4.13 GB/s | 105.5x |   1.20 GB/s |      7.47 GB/s | 6.2x |
| Portuguese 🇧🇷 |     39 MB/s |      4.59 GB/s | 120.2x |   1.02 GB/s |      7.56 GB/s | 7.4x |
| Russian 🇷🇺    |     57 MB/s |      3.30 GB/s |  59.0x |   2.14 GB/s |      5.31 GB/s | 2.5x |
| Spanish 🇪🇸    |     61 MB/s |      4.54 GB/s |  76.2x |  0.950 GB/s |      5.90 GB/s | 6.2x |
| Tamil 🇮🇳      |    111 MB/s |      6.50 GB/s |  60.2x |   5.41 GB/s |     21.52 GB/s | 4.0x |
| Turkish 🇹🇷    |     59 MB/s |      3.84 GB/s |  66.5x |   1.39 GB/s |      4.89 GB/s | 3.5x |
| Ukrainian 🇺🇦  |     93 MB/s |      2.77 GB/s |  30.6x |   2.10 GB/s |      4.98 GB/s | 2.4x |
| Vietnamese 🇻🇳 |     72 MB/s |      4.71 GB/s |  66.6x |  0.997 GB/s |      1.04 GB/s | 1.0x |

> Measured June 17, 2026.
> `Standard` is `pcre2::pre-jit` 🦀 / `regex.search<fullcase>` 🐍 — PCRE2/regex with full Unicode case folding, the fair baseline against StringZilla's own full case folding.

### Apple M5 Pro

| Language      | memchr+ICU 🦀 | pcre2 no-jit 🦀 | pcre2 jit-on-fly 🦀 | pcre2 pre-jit 🦀 | StringZilla 🦀 |         | casefold.find 🐍 |  PyICU 🐍 |   regex 🐍 | StringZilla 🐍 |      |
| ------------- | ------------: | --------------: | ------------------: | ---------------: | -------------: | ------: | ---------------: | --------: | ---------: | -------------: | ---: |
| Arabic 🇸🇦     |      282 MB/s |          9 MB/s |              9 MB/s |           9 MB/s |     10.44 GB/s | 1245.6x |         658 MB/s |  126 MB/s |  3.02 GB/s |     10.41 GB/s | 3.5x |
| Armenian 🇦🇲   |      290 MB/s |         50 MB/s |             60 MB/s |          59 MB/s |      1.07 GB/s |   18.5x |         668 MB/s |  170 MB/s |  2.57 GB/s |      1.01 GB/s | 0.4x |
| Bengali 🇧🇩    |      442 MB/s |         21 MB/s |             23 MB/s |          23 MB/s |     22.68 GB/s | 1014.6x |         913 MB/s |  224 MB/s |  4.85 GB/s |     27.87 GB/s | 5.7x |
| Chinese 🇨🇳    |      398 MB/s |        9.5 MB/s |            9.5 MB/s |         9.5 MB/s |     11.64 GB/s | 1250.0x |        1.11 GB/s |  134 MB/s |  5.16 GB/s |     13.88 GB/s | 2.7x |
| Czech 🇨🇿      |      159 MB/s |         23 MB/s |             24 MB/s |          24 MB/s |      4.07 GB/s |  174.8x |         389 MB/s | 97.3 MB/s |  1.56 GB/s |      2.70 GB/s | 1.7x |
| Dutch 🇳🇱      |      163 MB/s |          9 MB/s |              9 MB/s |           9 MB/s |      5.31 GB/s |  633.3x |         325 MB/s |  108 MB/s |  1.08 GB/s |      4.86 GB/s | 4.5x |
| English 🇬🇧    |      150 MB/s |         18 MB/s |             18 MB/s |          18 MB/s |      6.43 GB/s |  363.2x |         374 MB/s |   93 MB/s |  1.12 GB/s |      5.68 GB/s | 5.1x |
| Farsi 🇮🇷      |      318 MB/s |          8 MB/s |              8 MB/s |           8 MB/s |     20.96 GB/s | 2813.8x |         564 MB/s |  140 MB/s |  2.52 GB/s |     13.76 GB/s | 5.5x |
| French 🇫🇷     |      155 MB/s |         28 MB/s |             29 MB/s |          29 MB/s |      6.58 GB/s |  235.3x |         352 MB/s |  100 MB/s |  1.04 GB/s |      4.00 GB/s | 3.8x |
| Georgian 🇬🇪   |      287 MB/s |       99.2 MB/s |            131 MB/s |         131 MB/s |      3.32 GB/s |   26.0x |         857 MB/s |  257 MB/s |  3.08 GB/s |      2.82 GB/s | 0.9x |
| German 🇩🇪     |      155 MB/s |         14 MB/s |             15 MB/s |          15 MB/s |      7.00 GB/s |  470.0x |         398 MB/s | 98.2 MB/s | 0.950 GB/s |      5.32 GB/s | 5.6x |
| Greek 🇬🇷      |      195 MB/s |         40 MB/s |             46 MB/s |          45 MB/s |      1.61 GB/s |   36.8x |         528 MB/s |  144 MB/s |  2.34 GB/s |      1.89 GB/s | 0.8x |
| Hebrew 🇮🇱     |      267 MB/s |         23 MB/s |             25 MB/s |          26 MB/s |     15.77 GB/s |  627.0x |         597 MB/s |  163 MB/s |  2.83 GB/s |     10.10 GB/s | 3.6x |
| Hindi 🇮🇳      |      425 MB/s |          7 MB/s |              7 MB/s |           7 MB/s |     14.94 GB/s | 2291.4x |       0.997 GB/s |  258 MB/s |  4.58 GB/s |     12.59 GB/s | 2.7x |
| Italian 🇮🇹    |      154 MB/s |         41 MB/s |             42 MB/s |          43 MB/s |      5.23 GB/s |  124.9x |         419 MB/s |  104 MB/s |  1.04 GB/s |      6.04 GB/s | 5.8x |
| Japanese 🇯🇵   |      402 MB/s |         11 MB/s |             11 MB/s |          11 MB/s |     10.27 GB/s |  919.2x |        1.17 GB/s |  156 MB/s | 9.472 GB/s |      8.97 GB/s | 0.9x |
| Korean 🇰🇷     |      308 MB/s |         75 MB/s |             93 MB/s |          93 MB/s |     28.14 GB/s |  308.3x |         813 MB/s |   92 MB/s |  5.05 GB/s |     27.99 GB/s | 5.5x |
| Polish 🇵🇱     |      152 MB/s |         60 MB/s |             64 MB/s |          63 MB/s |      4.76 GB/s |   77.4x |         364 MB/s |  100 MB/s |  1.20 GB/s |      4.27 GB/s | 3.6x |
| Portuguese 🇧🇷 |      158 MB/s |         44 MB/s |             46 MB/s |          46 MB/s |      4.33 GB/s |   96.9x |         338 MB/s |  103 MB/s |  1.26 GB/s |      5.08 GB/s | 4.0x |
| Russian 🇷🇺    |      218 MB/s |         51 MB/s |             61 MB/s |          61 MB/s |      3.53 GB/s |   59.2x |         714 MB/s |  165 MB/s |  2.72 GB/s |      2.62 GB/s | 1.0x |
| Spanish 🇪🇸    |      154 MB/s |         19 MB/s |             20 MB/s |          20 MB/s |      4.94 GB/s |  252.4x |         340 MB/s |  100 MB/s |  1.07 GB/s |      4.57 GB/s | 4.3x |
| Tamil 🇮🇳      |      381 MB/s |         50 MB/s |             57 MB/s |          58 MB/s |     18.47 GB/s |  325.1x |       0.997 GB/s |  265 MB/s |  4.92 GB/s |     19.79 GB/s | 4.0x |
| Turkish 🇹🇷    |      148 MB/s |         13 MB/s |             13 MB/s |          13 MB/s |      2.55 GB/s |  195.7x |         374 MB/s |  106 MB/s |  1.54 GB/s |      2.78 GB/s | 1.8x |
| Ukrainian 🇺🇦  |      234 MB/s |         16 MB/s |             17 MB/s |          17 MB/s |      2.47 GB/s |  147.2x |         582 MB/s |  168 MB/s |  2.70 GB/s |      3.00 GB/s | 1.1x |
| Vietnamese 🇻🇳 |      176 MB/s |          4 MB/s |              4 MB/s |           4 MB/s |      1.85 GB/s |  497.5x |         422 MB/s |   90 MB/s |  1.43 GB/s |      1.81 GB/s | 1.3x |

> Measured July 29, 2026.

To rerun the benchmarks for all languages:

```bash
for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-insensitive-find" "$bin"
done
```

## Unicode Normalization

Normalizing each Leipzig corpus into NFC, single-threaded.
Rust compares `stringzilla::utf8_norm` against the `unicode-normalization` crate and ICU4X's `ComposingNormalizer`; Python compares `stringzilla.utf8_norm` against `unicodedata.normalize`, PyICU's `Normalizer2`, and [`pyunormalize`](https://github.com/mlodel/pyunormalize) — a pure-Python UAX #15 implementation that ships its own Unicode tables, included as the reference for what the algorithm costs with no native code behind it.
NFD/NFKC/NFKD were measured too and follow the same ordering; only NFC is tabulated here.

### Apple M5 Pro

| Language      | unicode-norm 🦀 |      ICU4X 🦀 | StringZilla 🦀 | unicodedata 🐍 |       PyICU 🐍 | pyunormalize 🐍 | StringZilla 🐍 |
| ------------- | --------------: | ------------: | -------------: | -------------: | -------------: | --------------: | -------------: |
| Arabic 🇸🇦     |        153 MB/s |      483 MB/s |  __1.09 GB/s__ |  __2.37 GB/s__ |     0.931 GB/s |               — |      1.13 GB/s |
| Armenian 🇦🇲   |        169 MB/s |      484 MB/s |  __1.04 GB/s__ |        80 MB/s |     0.959 GB/s |               — |  __1.07 GB/s__ |
| Bengali 🇧🇩    |        229 MB/s |      357 MB/s |   __404 MB/s__ |        93 MB/s |   __791 MB/s__ |               — |       387 MB/s |
| Chinese 🇨🇳    |        250 MB/s |      722 MB/s |  __1.08 GB/s__ |  __3.95 GB/s__ |      1.57 GB/s |         73 MB/s |      1.09 GB/s |
| Czech 🇨🇿      |        118 MB/s |     3.60 GB/s | __14.33 GB/s__ |      1.81 GB/s |     0.969 GB/s |         52 MB/s | __11.57 GB/s__ |
| Dutch 🇳🇱      |        132 MB/s |     3.73 GB/s | __35.89 GB/s__ |      1.67 GB/s |       899 MB/s |         50 MB/s | __21.23 GB/s__ |
| English 🇬🇧    |        128 MB/s |     3.56 GB/s | __20.48 GB/s__ |      1.74 GB/s |       891 MB/s |         45 MB/s | __14.92 GB/s__ |
| Farsi 🇮🇷      |        166 MB/s |      484 MB/s | __0.987 GB/s__ |        68 MB/s | __0.997 GB/s__ |               — |     0.997 GB/s |
| French 🇫🇷     |        128 MB/s |     3.45 GB/s | __12.66 GB/s__ |      1.69 GB/s |       924 MB/s |         51 MB/s | __9.993 GB/s__ |
| Georgian 🇬🇪   |        244 MB/s |      653 MB/s | __0.978 GB/s__ |        92 MB/s |  __1.46 GB/s__ |               — |     0.987 GB/s |
| German 🇩🇪     |        130 MB/s |     3.51 GB/s | __15.45 GB/s__ |      1.66 GB/s |       901 MB/s |         51 MB/s | __11.90 GB/s__ |
| Greek 🇬🇷      |        148 MB/s |      509 MB/s |  __1.19 GB/s__ |  __2.65 GB/s__ |     0.941 GB/s |         61 MB/s |      1.18 GB/s |
| Hebrew 🇮🇱     |        160 MB/s |      485 MB/s |  __1.02 GB/s__ |        84 MB/s |       925 MB/s |               — |  __1.03 GB/s__ |
| Hindi 🇮🇳      |        229 MB/s |      479 MB/s |   __731 MB/s__ |        93 MB/s |  __1.40 GB/s__ |               — |       726 MB/s |
| Italian 🇮🇹    |        131 MB/s |     3.71 GB/s | __27.25 GB/s__ |      1.64 GB/s |       909 MB/s |         51 MB/s | __18.81 GB/s__ |
| Japanese 🇯🇵   |        215 MB/s |      700 MB/s | __0.950 GB/s__ |  __4.07 GB/s__ |      1.61 GB/s |         81 MB/s |     0.969 GB/s |
| Korean 🇰🇷     |        128 MB/s |      570 MB/s |   __824 MB/s__ |  __3.96 GB/s__ |      1.31 GB/s |         71 MB/s |       758 MB/s |
| Polish 🇵🇱     |        123 MB/s |     3.43 GB/s | __11.00 GB/s__ |       117 MB/s |       934 MB/s |               — | __9.360 GB/s__ |
| Portuguese 🇧🇷 |        123 MB/s |     3.71 GB/s | __18.71 GB/s__ |      1.69 GB/s |       920 MB/s |         51 MB/s | __13.26 GB/s__ |
| Russian 🇷🇺    |        163 MB/s |      484 MB/s | __10.78 GB/s__ |        70 MB/s |     0.959 GB/s |               — |  __9.15 GB/s__ |
| Spanish 🇪🇸    |        128 MB/s |     3.72 GB/s | __20.19 GB/s__ |      1.67 GB/s |       926 MB/s |         51 MB/s | __13.90 GB/s__ |
| Tamil 🇮🇳      |        209 MB/s |      350 MB/s |   __526 MB/s__ |        82 MB/s |   __923 MB/s__ |               — |       508 MB/s |
| Turkish 🇹🇷    |        114 MB/s |     3.50 GB/s | __12.65 GB/s__ |      1.78 GB/s |     0.950 GB/s |         52 MB/s | __9.937 GB/s__ |
| Ukrainian 🇺🇦  |        160 MB/s |      515 MB/s |  __5.82 GB/s__ |        69 MB/s |     0.997 GB/s |               — |  __5.27 GB/s__ |
| Vietnamese 🇻🇳 |        106 MB/s | __1.40 GB/s__ |       821 MB/s |        72 MB/s |   __874 MB/s__ |               — |       779 MB/s |

> Measured July 29, 2026.

---

See [README.md](../README.md) for dataset information and replication instructions.
