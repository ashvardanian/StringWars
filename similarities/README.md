# Similarity Scoring Benchmarks

Benchmarks for string similarity and alignment algorithms across Rust and Python implementations, including CPU and GPU variants.

## Overview

Edit Distance calculation is a common component of Search Engines, Data Cleaning, and Natural Language Processing, as well as in Bioinformatics.
It's a computationally expensive operation, generally implemented using dynamic programming, with a quadratic time complexity upper bound.
For biological sequences, the Needleman-Wunsch and Smith-Waterman algorithms are more appropriate, as they allow overriding the default substitution costs.
Each of those has two flavors - with linear and affine gap penalties, also known as the "Gotoh" variation.

Performance is measured in MCUPS (Million Cell Updates Per Second).
Both harnesses (`bench.rs` and `bench.py`) score an all-pairs cross-product: a side × side matrix of disjoint query and candidate batches, where each axis is round(√(`STRINGWARS_BATCH_PER_CORE` × cores)).
A CPU core and a GPU streaming-multiprocessor (SM) each count as one core, so `<1xSPR>` runs one core, `<16xSPR>` all sixteen, and `<H100>` feeds the device a batch sized across the H100's 132 SMs.
MCUPS is the aggregate cell-update count over the whole matrix divided by the wall-clock time.

Datasets:

- __acgt_100 / acgt_1k__ — uniform synthetic DNA, lines of ~100 and ~1,000 bytes (~100 bytes is also the canonical protein-domain length).
- __XLSum words / lines__ — highly non-uniform multilingual text: ~5-byte word tokens and ~3.2 KB article lines.

The synthetic-DNA and word runs use `STRINGWARS_BATCH_PER_CORE=16384` to saturate the GPU (`<1xSPR>` side 128; `<16xSPR>` side 512; `<H100>` side 1,471).
The ~3.2 KB XLSum-lines run uses `STRINGWARS_BATCH_PER_CORE=256` (`<H100>` side 184) because each line-vs-line pair carries ~1,000× more cells than a word pair, so a smaller batch already saturates the device and keeps the single-core runs tractable.
StringZilla scores every column with the same unary 32-class match/mismatch costs, and the `bio` / `biopython` baselines use the same unary match/mismatch scoring, so the cross-language comparison stays apples-to-apples.

## Levenshtein Distance

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                                          |           ACGT 100B |            ACGT 1KB |       XLSum words |       XLSum lines |
| ------------------------------------------------ | ------------------: | ------------------: | ----------------: | ----------------: |
| Rust                                             |                     |                     |                   |                   |
| `bio::levenshtein<1xSPR>`                        |        590.86 MCUPS |         1,090 MCUPS |      327.67 MCUPS |      144.37 MCUPS |
| `rapidfuzz::levenshtein<Bytes,1xSPR>`            |         7,100 MCUPS |        18,010 MCUPS |       2,010 MCUPS |      15,580 MCUPS |
| `rapidfuzz::levenshtein<Chars,1xSPR>`            |         6,240 MCUPS |        17,730 MCUPS |      334.58 MCUPS |      15,260 MCUPS |
| `stringzillas::LevenshteinDistances<1xSPR>`      |        20,420 MCUPS |        16,120 MCUPS |       4,800 MCUPS |      10,210 MCUPS |
| `stringzillas::LevenshteinDistances<16xSPR>`     |       185,830 MCUPS |       161,390 MCUPS |      14,250 MCUPS |      58,630 MCUPS |
| `stringzillas::LevenshteinDistances<H100>`       | __6,420,010 MCUPS__ | __6,248,560 MCUPS__ | __246,820 MCUPS__ |     358,160 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<1xSPR>`  |        20,410 MCUPS |        15,990 MCUPS |      248.74 MCUPS |      10,620 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<16xSPR>` |       182,660 MCUPS |       161,500 MCUPS |       1,600 MCUPS |      55,290 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<H100>`   |       238,520 MCUPS |        80,690 MCUPS |      24,150 MCUPS | __522,420 MCUPS__ |
|                                                  |                     |                     |                   |                   |
| Python                                           |                     |                     |                   |                   |
| `rapidfuzz.Levenshtein.distance`                 |         4,830 MCUPS |        24,070 MCUPS |       54.18 MCUPS |      25,380 MCUPS |
| `Levenshtein.distance`                           |         4,570 MCUPS |        24,160 MCUPS |       45.28 MCUPS |      25,460 MCUPS |
| `jellyfish.levenshtein_distance`                 |        155.79 MCUPS |        214.51 MCUPS |       26.12 MCUPS |      278.63 MCUPS |
| `editdistance.eval`                              |         1,780 MCUPS |        454.75 MCUPS |       27.61 MCUPS |      453.58 MCUPS |
| `nltk.edit_distance`                             |          3.41 MCUPS |          2.90 MCUPS |        2.29 MCUPS |        2.51 MCUPS |
| `edlib.align`                                    |         2,830 MCUPS |        12,090 MCUPS |       13.38 MCUPS |               — ‖ |
| `polyleven.levenshtein`                          |         4,380 MCUPS |         6,460 MCUPS |      147.84 MCUPS |      16,240 MCUPS |
| `cudf.edit_distance<H100>` ‡                     |                   — |                   — |                 — |                 — |
| `stringzillas.LevenshteinDistances<1xSPR>`       |        17,470 MCUPS |        13,860 MCUPS |       2,340 MCUPS |       9,100 MCUPS |
| `stringzillas.LevenshteinDistances<16xSPR>`      |       130,630 MCUPS |       128,130 MCUPS |       2,520 MCUPS |      51,970 MCUPS |
| `stringzillas.LevenshteinDistances<H100>`        | __4,359,670 MCUPS__ | __6,079,630 MCUPS__ |  __43,210 MCUPS__ |     317,960 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<1xSPR>`   |        17,490 MCUPS |        13,450 MCUPS |      155.97 MCUPS |       6,760 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<16xSPR>`  |       133,340 MCUPS |       130,540 MCUPS |      656.42 MCUPS |      79,780 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<H100>`    |       234,510 MCUPS |        81,110 MCUPS |      12,270 MCUPS | __524,520 MCUPS__ |

> Measured on a 16-core Xeon Platinum 8468 and one idle H100 80GB HBM3.

### Apple M5 Pro

| Library                                         |        ACGT 100B |          ACGT 1KB |      XLSum words |       XLSum lines |
| ----------------------------------------------- | ---------------: | ----------------: | ---------------: | ----------------: |
| Rust                                            |                  |                   |                  |                   |
| `bio::levenshtein<1xM5>`                        |        900 MCUPS |       1,567 MCUPS |        436 MCUPS |         217 MCUPS |
| `rapidfuzz::levenshtein<Bytes,1xM5>`            |      9,119 MCUPS |      24,876 MCUPS |      2,037 MCUPS |      21,589 MCUPS |
| `rapidfuzz::levenshtein<Chars,1xM5>`            |      8,664 MCUPS |      24,464 MCUPS |        468 MCUPS |      21,075 MCUPS |
| `stringzillas::LevenshteinDistances<1xM5>`      |     22,422 MCUPS |      54,250 MCUPS |      2,482 MCUPS |      14,073 MCUPS |
| `stringzillas::LevenshteinDistances<18xM5>`     |     49,922 MCUPS | __715,914 MCUPS__ |      2,537 MCUPS |     150,332 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<1xM5>`  |     22,392 MCUPS |      54,255 MCUPS |        711 MCUPS |      28,444 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<18xM5>` | __53,979 MCUPS__ |     694,590 MCUPS | __11,028 MCUPS__ | __316,631 MCUPS__ |
|                                                 |                  |                   |                  |                   |
| Python                                          |                  |                   |                  |                   |
| `rapidfuzz.Levenshtein.distance`                |      9,704 MCUPS |      29,245 MCUPS |        215 MCUPS |      26,394 MCUPS |
| `Levenshtein.distance`                          |      9,189 MCUPS |      29,192 MCUPS |        140 MCUPS |      26,423 MCUPS |
| `jellyfish.levenshtein_distance`                |        214 MCUPS |         252 MCUPS |         59 MCUPS |         432 MCUPS |
| `editdistance.eval`                             |      4,652 MCUPS |         814 MCUPS |         51 MCUPS |         780 MCUPS |
| `nltk.edit_distance`                            |       5.94 MCUPS |        4.64 MCUPS |       3.78 MCUPS |                 — |
| `edlib.align`                                   |      4,864 MCUPS |      24,552 MCUPS |         19 MCUPS |      26,314 MCUPS |
| `polyleven.levenshtein`                         |     10,828 MCUPS |      11,316 MCUPS |        461 MCUPS |      26,301 MCUPS |
| `stringzillas.LevenshteinDistances<1xM5>`       |     22,084 MCUPS |      54,549 MCUPS |  __1,396 MCUPS__ |      14,478 MCUPS |
| `stringzillas.LevenshteinDistances<18xM5>`      | __46,893 MCUPS__ | __723,480 MCUPS__ |        445 MCUPS |     181,045 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<1xM5>`   |     22,074 MCUPS |      54,532 MCUPS |        479 MCUPS |      27,126 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<18xM5>`  |     46,047 MCUPS |     696,989 MCUPS |        166 MCUPS | __399,501 MCUPS__ |

> Measured July 29, 2026.

## Needleman-Wunsch for Global Alignment

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                                       |         ACGT 100B |          ACGT 1KB |      XLSum words |       XLSum lines |
| --------------------------------------------- | ----------------: | ----------------: | ---------------: | ----------------: |
| Rust                                          |                   |                   |                  |                   |
| `bio::pairwise::global<1xSPR>`                |       98.71 MCUPS |       91.81 MCUPS |      75.72 MCUPS |       28.77 MCUPS |
| `stringzillas::NeedlemanWunschScores<1xSPR>`  |      18,440 MCUPS |      13,020 MCUPS |      3,200 MCUPS |      612.28 MCUPS |
| `stringzillas::NeedlemanWunschScores<16xSPR>` |     135,550 MCUPS |     118,570 MCUPS |     14,460 MCUPS |       5,440 MCUPS |
| `stringzillas::NeedlemanWunschScores<H100>`   | __410,840 MCUPS__ | __701,600 MCUPS__ | __19,970 MCUPS__ | __227,040 MCUPS__ |
|                                               |                   |                   |                  |                   |
| Python                                        |                   |                   |                  |                   |
| `biopython.PairwiseAligner.global`            |      485.26 MCUPS |      462.33 MCUPS |      23.10 MCUPS |      970.86 MCUPS |
| `stringzillas.NeedlemanWunschScores<1xSPR>`   |      16,080 MCUPS |      10,720 MCUPS |      1,060 MCUPS |       1,260 MCUPS |
| `stringzillas.NeedlemanWunschScores<16xSPR>`  |     108,490 MCUPS |      94,360 MCUPS |      1,910 MCUPS |       2,830 MCUPS |
| `stringzillas.NeedlemanWunschScores<H100>`    | __401,040 MCUPS__ | __701,650 MCUPS__ | __13,040 MCUPS__ | __228,370 MCUPS__ |

> Measured on a 16-core Xeon Platinum 8468 and one idle H100 80GB HBM3.

### Apple M5 Pro

| Library                                      |        ACGT 100B |         ACGT 1KB |     XLSum words |     XLSum lines |
| -------------------------------------------- | ---------------: | ---------------: | --------------: | --------------: |
| Rust                                         |                  |                  |                 |                 |
| `bio::pairwise::global<1xM5>`                |        153 MCUPS |        157 MCUPS |       124 MCUPS |   __124 MCUPS__ |
| `stringzillas::NeedlemanWunschScores<1xM5>`  |        974 MCUPS |      1,071 MCUPS |       252 MCUPS |               — |
| `stringzillas::NeedlemanWunschScores<18xM5>` | __11,684 MCUPS__ | __15,581 MCUPS__ | __1,065 MCUPS__ |               — |
|                                              |                  |                  |                 |                 |
| Python                                       |                  |                  |                 |                 |
| `biopython.PairwiseAligner.global`           |        914 MCUPS |        842 MCUPS |        34 MCUPS | __1,894 MCUPS__ |
| `stringzillas.NeedlemanWunschScores<1xM5>`   |        823 MCUPS |      1,017 MCUPS |        24 MCUPS |               — |
| `stringzillas.NeedlemanWunschScores<18xM5>`  | __10,193 MCUPS__ | __14,331 MCUPS__ |   __210 MCUPS__ |               — |

> Measured July 29, 2026.

## Smith-Waterman for Local Alignment

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                                     |         ACGT 100B |          ACGT 1KB |      XLSum words |       XLSum lines |
| ------------------------------------------- | ----------------: | ----------------: | ---------------: | ----------------: |
| Rust                                        |                   |                   |                  |                   |
| `bio::pairwise::local<1xSPR>`               |       85.69 MCUPS |       83.76 MCUPS |      63.43 MCUPS |       52.21 MCUPS |
| `stringzillas::SmithWatermanScores<1xSPR>`  |      14,360 MCUPS |      11,430 MCUPS |      2,830 MCUPS |      581.24 MCUPS |
| `stringzillas::SmithWatermanScores<16xSPR>` |     104,680 MCUPS |     115,110 MCUPS |     13,180 MCUPS |       5,630 MCUPS |
| `stringzillas::SmithWatermanScores<H100>`   | __345,620 MCUPS__ | __605,930 MCUPS__ | __18,300 MCUPS__ | __213,850 MCUPS__ |
|                                             |                   |                   |                  |                   |
| Python                                      |                   |                   |                  |                   |
| `biopython.PairwiseAligner.local`           |      343.43 MCUPS |      445.88 MCUPS |      22.16 MCUPS |      803.53 MCUPS |
| `stringzillas.SmithWatermanScores<1xSPR>`   |      13,860 MCUPS |      10,240 MCUPS |     965.83 MCUPS |      911.30 MCUPS |
| `stringzillas.SmithWatermanScores<16xSPR>`  |      84,130 MCUPS |      90,890 MCUPS |      1,980 MCUPS |       2,910 MCUPS |
| `stringzillas.SmithWatermanScores<H100>`    | __337,690 MCUPS__ | __607,940 MCUPS__ | __11,760 MCUPS__ | __203,690 MCUPS__ |

> Measured on a 16-core Xeon Platinum 8468 and one idle H100 80GB HBM3.

### Apple M5 Pro

| Library                                    |        ACGT 100B |         ACGT 1KB |   XLSum words |     XLSum lines |
| ------------------------------------------ | ---------------: | ---------------: | ------------: | --------------: |
| Rust                                       |                  |                  |               |                 |
| `bio::pairwise::local<1xM5>`               |        144 MCUPS |        147 MCUPS |     128 MCUPS |   __127 MCUPS__ |
| `stringzillas::SmithWatermanScores<1xM5>`  |        960 MCUPS |      1,048 MCUPS |     249 MCUPS |               — |
| `stringzillas::SmithWatermanScores<18xM5>` | __11,465 MCUPS__ | __14,810 MCUPS__ | __931 MCUPS__ |               — |
|                                            |                  |                  |               |                 |
| Python                                     |                  |                  |               |                 |
| `biopython.PairwiseAligner.local`          |        776 MCUPS |        782 MCUPS |      34 MCUPS | __1,905 MCUPS__ |
| `stringzillas.SmithWatermanScores<1xM5>`   |        843 MCUPS |      1,055 MCUPS |      24 MCUPS |               — |
| `stringzillas.SmithWatermanScores<18xM5>`  | __10,436 MCUPS__ | __15,984 MCUPS__ | __209 MCUPS__ |               — |

> Measured July 29, 2026.

## Needleman-Wunsch-Gotoh for Global Alignment

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                                       |         ACGT 100B |          ACGT 1KB |      XLSum words |       XLSum lines |
| --------------------------------------------- | ----------------: | ----------------: | ---------------: | ----------------: |
| Rust                                          |                   |                   |                  |                   |
| `bio::pairwise::global<1xSPR>`                |       79.77 MCUPS |       75.63 MCUPS |      70.64 MCUPS |       62.73 MCUPS |
| `stringzillas::NeedlemanWunschScores<1xSPR>`  |       9,170 MCUPS |       3,910 MCUPS |      2,020 MCUPS |      239.53 MCUPS |
| `stringzillas::NeedlemanWunschScores<16xSPR>` |      67,870 MCUPS |      42,590 MCUPS |      8,630 MCUPS |       2,090 MCUPS |
| `stringzillas::NeedlemanWunschScores<H100>`   | __221,520 MCUPS__ | __393,910 MCUPS__ | __12,380 MCUPS__ | __113,490 MCUPS__ |
|                                               |                   |                   |                  |                   |
| Python                                        |                   |                   |                  |                   |
| `biopython.PairwiseAligner.global`            |      263.44 MCUPS |      266.76 MCUPS |      22.74 MCUPS |      563.38 MCUPS |
| `stringzillas.NeedlemanWunschScores<1xSPR>`   |       7,440 MCUPS |       3,920 MCUPS |     958.66 MCUPS |      420.45 MCUPS |
| `stringzillas.NeedlemanWunschScores<16xSPR>`  |      63,750 MCUPS |      48,570 MCUPS |      1,810 MCUPS |       1,310 MCUPS |
| `stringzillas.NeedlemanWunschScores<H100>`    | __216,910 MCUPS__ | __393,380 MCUPS__ |  __9,130 MCUPS__ | __121,330 MCUPS__ |

> Measured on a 16-core Xeon Platinum 8468 and one idle H100 80GB HBM3.

### Apple M5 Pro

| Library                                      |        ACGT 100B |         ACGT 1KB |   XLSum words |     XLSum lines |
| -------------------------------------------- | ---------------: | ---------------: | ------------: | --------------: |
| Rust                                         |                  |                  |               |                 |
| `bio::pairwise::global<1xM5>`                |        138 MCUPS |        142 MCUPS |     126 MCUPS |   __123 MCUPS__ |
| `stringzillas::NeedlemanWunschScores<1xM5>`  |        929 MCUPS |        986 MCUPS |     243 MCUPS |               — |
| `stringzillas::NeedlemanWunschScores<18xM5>` | __10,780 MCUPS__ | __14,184 MCUPS__ | __862 MCUPS__ |               — |
|                                              |                  |                  |               |                 |
| Python                                       |                  |                  |               |                 |
| `biopython.PairwiseAligner.global`           |        610 MCUPS |        700 MCUPS |      34 MCUPS | __1,631 MCUPS__ |
| `stringzillas.NeedlemanWunschScores<1xM5>`   |        902 MCUPS |      1,007 MCUPS |      43 MCUPS |               — |
| `stringzillas.NeedlemanWunschScores<18xM5>`  | __10,683 MCUPS__ | __15,219 MCUPS__ | __232 MCUPS__ |               — |

> Measured July 29, 2026.

## Smith-Waterman-Gotoh for Local Alignment

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                                     |         ACGT 100B |          ACGT 1KB |      XLSum words |       XLSum lines |
| ------------------------------------------- | ----------------: | ----------------: | ---------------: | ----------------: |
| Rust                                        |                   |                   |                  |                   |
| `bio::pairwise::local<1xSPR>`               |       69.85 MCUPS |       59.95 MCUPS |      56.62 MCUPS |       51.54 MCUPS |
| `stringzillas::SmithWatermanScores<1xSPR>`  |       8,350 MCUPS |       3,940 MCUPS |      1,930 MCUPS |      239.39 MCUPS |
| `stringzillas::SmithWatermanScores<16xSPR>` |      68,280 MCUPS |      41,630 MCUPS |      9,610 MCUPS |       2,010 MCUPS |
| `stringzillas::SmithWatermanScores<H100>`   | __219,970 MCUPS__ | __363,910 MCUPS__ | __12,100 MCUPS__ | __114,430 MCUPS__ |
|                                             |                   |                   |                  |                   |
| Python                                      |                   |                   |                  |                   |
| `biopython.PairwiseAligner.local`           |      162.48 MCUPS |      214.67 MCUPS |      22.80 MCUPS |      560.02 MCUPS |
| `stringzillas.SmithWatermanScores<1xSPR>`   |       7,920 MCUPS |       3,870 MCUPS |     907.42 MCUPS |      305.16 MCUPS |
| `stringzillas.SmithWatermanScores<16xSPR>`  |      62,420 MCUPS |      43,280 MCUPS |      1,870 MCUPS |       1,550 MCUPS |
| `stringzillas.SmithWatermanScores<H100>`    | __219,770 MCUPS__ | __364,020 MCUPS__ |  __9,590 MCUPS__ | __114,570 MCUPS__ |

> Measured on a 16-core Xeon Platinum 8468 and one idle H100 80GB HBM3.

### Apple M5 Pro

| Library                                    |        ACGT 100B |         ACGT 1KB |   XLSum words |   XLSum lines |
| ------------------------------------------ | ---------------: | ---------------: | ------------: | ------------: |
| Rust                                       |                  |                  |               |               |
| `bio::pairwise::local<1xM5>`               |        119 MCUPS |        124 MCUPS |     131 MCUPS | __117 MCUPS__ |
| `stringzillas::SmithWatermanScores<1xM5>`  |        886 MCUPS |        956 MCUPS |     232 MCUPS |             — |
| `stringzillas::SmithWatermanScores<18xM5>` | __10,565 MCUPS__ | __13,303 MCUPS__ | __887 MCUPS__ |             — |
|                                            |                  |                  |               |               |
| Python                                     |                  |                  |               |               |
| `biopython.PairwiseAligner.local`          |        387 MCUPS |        396 MCUPS |      33 MCUPS | __999 MCUPS__ |
| `stringzillas.SmithWatermanScores<1xM5>`   |        879 MCUPS |        975 MCUPS |      43 MCUPS |             — |
| `stringzillas.SmithWatermanScores<18xM5>`  | __10,421 MCUPS__ | __14,582 MCUPS__ | __260 MCUPS__ |             — |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
