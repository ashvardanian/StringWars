# Similarity Scoring Benchmarks

Benchmarks for string similarity and alignment algorithms across Rust and Python implementations, including CPU and GPU variants.

## Overview

Edit Distance calculation is a common component of Search Engines, Data Cleaning, and Natural Language Processing, as well as in Bioinformatics.
It's a computationally expensive operation, generally implemented using dynamic programming, with a quadratic time complexity upper bound.
For biological sequences, the Needleman-Wunsch and Smith-Waterman algorithms are more appropriate, as they allow overriding the default substitution costs.
Each of those has two flavors - with linear and affine gap penalties, also known as the "Gotoh" variation.

Performance is measured in MCUPS (Million Cell Updates Per Second).

## Levenshtein Distance

| Library                                              |  ~100 bytes lines |  ~1,000 bytes lines |
| ---------------------------------------------------- | ----------------: | ------------------: |
| Rust                                                 |                   |                     |
| `bio::levenshtein` on 1x SPR                         |         428 MCUPS |           823 MCUPS |
| `rapidfuzz::levenshtein<Bytes>` on 1x SPR            |       4,633 MCUPS |        14,316 MCUPS |
| `rapidfuzz::levenshtein<Chars>` on 1x SPR            |       3,877 MCUPS |        13,179 MCUPS |
| `stringzillas::LevenshteinDistances` on 1x SPR       |       3,315 MCUPS |        13,084 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8` on 1x SPR   |       3,283 MCUPS |        11,690 MCUPS |
| `stringzillas::LevenshteinDistances` on 16x SPR      |      29,430 MCUPS |       105,400 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8` on 16x SPR  |      38,954 MCUPS |       103,500 MCUPS |
| `stringzillas::LevenshteinDistances` on RTX6000      |  __32,030 MCUPS__ |   __901,990 MCUPS__ |
| `stringzillas::LevenshteinDistances` on H100         |  __31,913 MCUPS__ |   __925,890 MCUPS__ |
| `stringzillas::LevenshteinDistances` on B200         |  __32,960 MCUPS__ |   __998,620 MCUPS__ |
| `stringzillas::LevenshteinDistances` on 384x GNR     | __114,190 MCUPS__ | __3,084,270 MCUPS__ |
| `stringzillas::LevenshteinDistancesUtf8` on 384x GNR | __103,590 MCUPS__ | __2,938,320 MCUPS__ |
|                                                      |                   |                     |
| Python                                               |                   |                     |
| `nltk.edit_distance`                                 |           2 MCUPS |             2 MCUPS |
| `jellyfish.levenshtein_distance`                     |          81 MCUPS |           228 MCUPS |
| `rapidfuzz.Levenshtein.distance`                     |         108 MCUPS |         9,272 MCUPS |
| `editdistance.eval`                                  |          89 MCUPS |           660 MCUPS |
| `edlib.align`                                        |          82 MCUPS |         7,262 MCUPS |
| `polyleven.levenshtein`                              |          89 MCUPS |         3,887 MCUPS |
| `stringzillas.LevenshteinDistances` on 1x SPR        |          53 MCUPS |         3,407 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8` on 1x SPR    |          57 MCUPS |         3,693 MCUPS |
| `cudf.edit_distance` batch on H100                   |      24,754 MCUPS |         6,976 MCUPS |
| `stringzillas.LevenshteinDistances` batch on 1x SPR  |       2,343 MCUPS |        12,141 MCUPS |
| `stringzillas.LevenshteinDistances` batch on 16x SPR |       3,762 MCUPS |       119,261 MCUPS |
| `stringzillas.LevenshteinDistances` batch on H100    |  __18,081 MCUPS__ |   __320,109 MCUPS__ |

## Needleman-Wunsch (Global Alignment)

| Library                                               | ~100 bytes lines | ~1,000 bytes lines |
| ----------------------------------------------------- | ---------------: | -----------------: |
| Rust                                                  |                  |                    |
| `bio::pairwise::global` on 1x SPR                     |         51 MCUPS |           57 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 1x SPR       |        278 MCUPS |          612 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 16x SPR      |      4,057 MCUPS |        8,492 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 384x GNR     | __64,290 MCUPS__ |  __331,340 MCUPS__ |
| `stringzillas::NeedlemanWunschScores` on H100         |        131 MCUPS |   __12,113 MCUPS__ |
|                                                       |                  |                    |
| Python                                                |                  |                    |
| `biopython.PairwiseAligner.score` on 1x SPR           |         95 MCUPS |          557 MCUPS |
| `stringzillas.NeedlemanWunschScores` on 1x SPR        |         30 MCUPS |          481 MCUPS |
| `stringzillas.NeedlemanWunschScores` batch on 1x SPR  |        246 MCUPS |          570 MCUPS |
| `stringzillas.NeedlemanWunschScores` batch on 16x SPR |      3,103 MCUPS |        9,208 MCUPS |
| `stringzillas.NeedlemanWunschScores` batch on H100    |        127 MCUPS |       12,246 MCUPS |

## Smith-Waterman (Local Alignment)

| Library                                             | ~100 bytes lines | ~1,000 bytes lines |
| --------------------------------------------------- | ---------------: | -----------------: |
| Rust                                                |                  |                    |
| `bio::pairwise::local` on 1x SPR                    |         49 MCUPS |           50 MCUPS |
| `stringzillas::SmithWatermanScores` on 1x SPR       |        263 MCUPS |          552 MCUPS |
| `stringzillas::SmithWatermanScores` on 16x SPR      |      3,883 MCUPS |        8,011 MCUPS |
| `stringzillas::SmithWatermanScores` on 384x GNR     | __58,880 MCUPS__ |  __285,480 MCUPS__ |
| `stringzillas::SmithWatermanScores` on H100         |        143 MCUPS |   __12,921 MCUPS__ |
|                                                     |                  |                    |
| Python                                              |                  |                    |
| `biopython.PairwiseAligner.score` on 1x SPR         |         95 MCUPS |          557 MCUPS |
| `stringzillas.SmithWatermanScores` on 1x SPR        |         28 MCUPS |          440 MCUPS |
| `stringzillas.SmithWatermanScores` batch on 1x SPR  |        255 MCUPS |          582 MCUPS |
| `stringzillas.SmithWatermanScores` batch on 16x SPR |  __3,535 MCUPS__ |        8,235 MCUPS |
| `stringzillas.SmithWatermanScores` batch on H100    |        130 MCUPS |   __12,702 MCUPS__ |

## Needleman-Wunsch-Gotoh (Affine Gap Penalties)

| Library                                           | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------------------------- | ---------------: | -----------------: |
| Rust                                              |                  |                    |
| `stringzillas::NeedlemanWunschScores` on 1x SPR   |         83 MCUPS |          354 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 16x SPR  |      1,267 MCUPS |        4,694 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 384x GNR | __42,050 MCUPS__ |  __155,920 MCUPS__ |
| `stringzillas::NeedlemanWunschScores` on H100     |        128 MCUPS |   __13,799 MCUPS__ |

## Smith-Waterman-Gotoh (Local with Affine Gaps)

| Library                                         | ~100 bytes lines | ~1,000 bytes lines |
| ----------------------------------------------- | ---------------: | -----------------: |
| Rust                                            |                  |                    |
| `stringzillas::SmithWatermanScores` on 1x SPR   |         79 MCUPS |          284 MCUPS |
| `stringzillas::SmithWatermanScores` on 16x SPR  |      1,026 MCUPS |        3,776 MCUPS |
| `stringzillas::SmithWatermanScores` on 384x GNR | __38,430 MCUPS__ |  __129,140 MCUPS__ |
| `stringzillas::SmithWatermanScores` on H100     |        127 MCUPS |   __13,205 MCUPS__ |

---

See [README.md](README.md) for dataset information and replication instructions.
