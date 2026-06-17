# Similarity Scoring Benchmarks

Benchmarks for string similarity and alignment algorithms across Rust and Python implementations, including CPU and GPU variants.

## Overview

Edit Distance calculation is a common component of Search Engines, Data Cleaning, and Natural Language Processing, as well as in Bioinformatics.
It's a computationally expensive operation, generally implemented using dynamic programming, with a quadratic time complexity upper bound.
For biological sequences, the Needleman-Wunsch and Smith-Waterman algorithms are more appropriate, as they allow overriding the default substitution costs.
Each of those has two flavors - with linear and affine gap penalties, also known as the "Gotoh" variation.

Performance is measured in MCUPS (Million Cell Updates Per Second).

## Levenshtein Distance

| Library                                              |  ~100 bytes lines | ~1,000 bytes lines |
| ---------------------------------------------------- | ----------------: | -----------------: |
| Rust                                                 |                   |                    |
| `bio::levenshtein` on 1x SPR                         |         423 MCUPS |          710 MCUPS |
| `rapidfuzz::levenshtein<Bytes>` on 1x SPR            |       3,730 MCUPS |       12,760 MCUPS |
| `rapidfuzz::levenshtein<Chars>` on 1x SPR            |       3,200 MCUPS |       11,490 MCUPS |
| `stringzillas::LevenshteinDistances` on 1x SPR       |      11,360 MCUPS |       10,450 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8` on 1x SPR   |       3,190 MCUPS |       11,080 MCUPS |
| `stringzillas::LevenshteinDistances` on 16x SPR      |      71,440 MCUPS |       73,260 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8` on 16x SPR  |      30,470 MCUPS |      106,070 MCUPS |
| `stringzillas::LevenshteinDistances` on H100         | __147,100 MCUPS__ |  __969,170 MCUPS__ |
|                                                      |                   |                    |
| Python                                               |                   |                    |
| `nltk.edit_distance`                                 |           4 MCUPS |            3 MCUPS |
| `jellyfish.levenshtein_distance`                     |         187 MCUPS |          240 MCUPS |
| `rapidfuzz.Levenshtein.distance`                     |       6,932 MCUPS |       24,811 MCUPS |
| `editdistance.eval`                                  |       2,078 MCUPS |          472 MCUPS |
| `edlib.align`                                        |       2,869 MCUPS |       16,523 MCUPS |
| `polyleven.levenshtein`                              |       4,216 MCUPS |        6,868 MCUPS |
| `stringzillas.LevenshteinDistances` on 1x SPR        |       1,190 MCUPS |        6,737 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8` on 1x SPR    |       1,175 MCUPS |        6,857 MCUPS |
| `cudf.edit_distance` batch on H100                   |      24,754 MCUPS |        6,976 MCUPS |
| `stringzillas.LevenshteinDistances` batch on 1x SPR  |       7,814 MCUPS |       10,859 MCUPS |
| `stringzillas.LevenshteinDistances` batch on 16x SPR |      58,540 MCUPS |       72,011 MCUPS |
| `stringzillas.LevenshteinDistances` batch on H100    |   __7,398 MCUPS__ |  __149,410 MCUPS__ |

## Needleman-Wunsch (Global Alignment)

| Library                                               | ~100 bytes lines | ~1,000 bytes lines |
| ----------------------------------------------------- | ---------------: | -----------------: |
| Rust                                                  |                  |                    |
| `bio::pairwise::global` on 1x SPR                     |         53 MCUPS |           52 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 1x SPR       |      1,420 MCUPS |        3,980 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 16x SPR      |     15,300 MCUPS |       43,740 MCUPS |
| `stringzillas::NeedlemanWunschScores` on H100         | __44,270 MCUPS__ |  __766,880 MCUPS__ |
|                                                       |                  |                    |
| Python                                                |                  |                    |
| `biopython.PairwiseAligner.score` on 1x SPR           |        557 MCUPS |          480 MCUPS |
| `stringzillas.NeedlemanWunschScores` on 1x SPR        |        626 MCUPS |        1,581 MCUPS |
| `stringzillas.NeedlemanWunschScores` batch on 1x SPR  |        894 MCUPS |        1,662 MCUPS |
| `stringzillas.NeedlemanWunschScores` batch on 16x SPR |      5,392 MCUPS |       17,273 MCUPS |
| `stringzillas.NeedlemanWunschScores` batch on H100    |  __5,856 MCUPS__ |  __214,448 MCUPS__ |

## Smith-Waterman (Local Alignment)

| Library                                             | ~100 bytes lines | ~1,000 bytes lines |
| --------------------------------------------------- | ---------------: | -----------------: |
| Rust                                                |                  |                    |
| `bio::pairwise::local` on 1x SPR                    |         45 MCUPS |           48 MCUPS |
| `stringzillas::SmithWatermanScores` on 1x SPR       |      1,060 MCUPS |        3,160 MCUPS |
| `stringzillas::SmithWatermanScores` on 16x SPR      |     11,370 MCUPS |       36,690 MCUPS |
| `stringzillas::SmithWatermanScores` on H100         | __40,900 MCUPS__ |  __673,530 MCUPS__ |
|                                                     |                  |                    |
| Python                                              |                  |                    |
| `biopython.PairwiseAligner.score` on 1x SPR         |        557 MCUPS |          480 MCUPS |
| `stringzillas.SmithWatermanScores` on 1x SPR        |        437 MCUPS |          924 MCUPS |
| `stringzillas.SmithWatermanScores` batch on 1x SPR  |        594 MCUPS |          957 MCUPS |
| `stringzillas.SmithWatermanScores` batch on 16x SPR |      6,184 MCUPS |       11,975 MCUPS |
| `stringzillas.SmithWatermanScores` batch on H100    |  __5,852 MCUPS__ |  __230,374 MCUPS__ |

## Needleman-Wunsch-Gotoh (Affine Gap Penalties)

| Library                                          | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------------------------ | ---------------: | -----------------: |
| Rust                                             |                  |                    |
| `stringzillas::NeedlemanWunschScores` on 1x SPR  |      1,470 MCUPS |        3,380 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 16x SPR |     13,620 MCUPS |       33,340 MCUPS |
| `stringzillas::NeedlemanWunschScores` on H100    | __39,750 MCUPS__ |  __470,940 MCUPS__ |

## Smith-Waterman-Gotoh (Local with Affine Gaps)

| Library                                        | ~100 bytes lines | ~1,000 bytes lines |
| ---------------------------------------------- | ---------------: | -----------------: |
| Rust                                           |                  |                    |
| `stringzillas::SmithWatermanScores` on 1x SPR  |      1,070 MCUPS |        2,860 MCUPS |
| `stringzillas::SmithWatermanScores` on 16x SPR |     11,620 MCUPS |       30,440 MCUPS |
| `stringzillas::SmithWatermanScores` on H100    | __39,910 MCUPS__ |  __447,970 MCUPS__ |

---

See [README.md](README.md) for dataset information and replication instructions.
