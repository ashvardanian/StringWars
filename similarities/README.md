# Similarity Scoring Benchmarks

Benchmarks for string similarity and alignment algorithms across Rust and Python implementations, including CPU and GPU variants.

## Overview

Edit Distance calculation is a common component of Search Engines, Data Cleaning, and Natural Language Processing, as well as in Bioinformatics.
It's a computationally expensive operation, generally implemented using dynamic programming, with a quadratic time complexity upper bound.
For biological sequences, the Needleman-Wunsch and Smith-Waterman algorithms are more appropriate, as they allow overriding the default substitution costs.
Each of those has two flavors - with linear and affine gap penalties, also known as the "Gotoh" variation.

Performance is measured in MCUPS (Million Cell Updates Per Second), on `acgt_100.txt` and `acgt_1k.txt` lines of ~100 and ~1,000 bytes.
CPU rows run on one core (`<1xSPR>`) or all sixteen (`<16xSPR>`); GPU rows (`<H100>`) feed the device one batch of 33,792 pairs — 256 per streaming multiprocessor across the H100's 132 SMs — derived identically in the Rust and Python harnesses, so the residual Rust↔Python gap is binding overhead rather than a batch-size mismatch.
The Python `<batch,…>` rows score an entire batch per native call, whereas the `<1xSPR>` Python rows without `batch` measure single-pair latency.
StringZilla scores `acgt` sequences with the same unary match/mismatch costs in Rust and Python (BLOSUM62 is used only for the `biopython` baseline), so the throughput is comparable across languages.

## Levenshtein Distance

| Library                                           |  ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------------------------- | ----------------: | -----------------: |
| Rust                                              |                   |                    |
| `bio::levenshtein<1xSPR>`                         |         454 MCUPS |          718 MCUPS |
| `rapidfuzz::levenshtein<Bytes><1xSPR>`            |       4,270 MCUPS |       14,360 MCUPS |
| `rapidfuzz::levenshtein<Chars><1xSPR>`            |       3,510 MCUPS |       11,480 MCUPS |
| `stringzillas::LevenshteinDistances<1xSPR>`       |      12,520 MCUPS |       10,380 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<1xSPR>`   |       2,850 MCUPS |        9,550 MCUPS |
| `stringzillas::LevenshteinDistances<16xSPR>`      | __133,200 MCUPS__ |       74,090 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<16xSPR>`  |      34,630 MCUPS |      120,000 MCUPS |
| `stringzillas::LevenshteinDistances<H100>`        |     119,810 MCUPS |  __705,600 MCUPS__ |
|                                                   |                   |                    |
| Python                                            |                   |                    |
| `nltk.edit_distance`                              |           4 MCUPS |            3 MCUPS |
| `jellyfish.levenshtein_distance`                  |         195 MCUPS |          237 MCUPS |
| `rapidfuzz.Levenshtein.distance`                  |       7,270 MCUPS |       27,760 MCUPS |
| `editdistance.eval`                               |       2,210 MCUPS |          504 MCUPS |
| `edlib.align`                                     |       3,290 MCUPS |       13,660 MCUPS |
| `polyleven.levenshtein`                           |       4,490 MCUPS |        7,290 MCUPS |
| `stringzillas.LevenshteinDistances<1xSPR>`        |       2,040 MCUPS |       11,100 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<1xSPR>`    |       1,530 MCUPS |       11,120 MCUPS |
| `cudf.edit_distance<batch,H100>` ‡                |      24,754 MCUPS |        6,976 MCUPS |
| `stringzillas.LevenshteinDistances<batch,1xSPR>`  |      13,760 MCUPS |       11,780 MCUPS |
| `stringzillas.LevenshteinDistances<batch,16xSPR>` | __112,280 MCUPS__ |      120,130 MCUPS |
| `stringzillas.LevenshteinDistances<batch,H100>`   |      13,510 MCUPS |  __245,980 MCUPS__ |

## Needleman-Wunsch for Global Alignment

| Library                                            | ~100 bytes lines | ~1,000 bytes lines |
| -------------------------------------------------- | ---------------: | -----------------: |
| Rust                                               |                  |                    |
| `bio::pairwise::global<1xSPR>`                     |         56 MCUPS |           48 MCUPS |
| `stringzillas::NeedlemanWunschScores<1xSPR>`       |      1,600 MCUPS |        3,890 MCUPS |
| `stringzillas::NeedlemanWunschScores<16xSPR>`      |     15,600 MCUPS |       39,340 MCUPS |
| `stringzillas::NeedlemanWunschScores<H100>`        | __37,260 MCUPS__ |  __557,340 MCUPS__ |
|                                                    |                  |                    |
| Python                                             |                  |                    |
| `biopython.PairwiseAligner.score<1xSPR>`           |        576 MCUPS |          506 MCUPS |
| `stringzillas.NeedlemanWunschScores<1xSPR>`        |        752 MCUPS |        1,480 MCUPS |
| `stringzillas.NeedlemanWunschScores<batch,1xSPR>`  |      1,170 MCUPS |        1,430 MCUPS |
| `stringzillas.NeedlemanWunschScores<batch,16xSPR>` |     10,450 MCUPS |   __17,160 MCUPS__ |
| `stringzillas.NeedlemanWunschScores<batch,H100>`   | __25,200 MCUPS__ |                — † |

## Smith-Waterman for Local Alignment

| Library                                          | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------------------------ | ---------------: | -----------------: |
| Rust                                             |                  |                    |
| `bio::pairwise::local<1xSPR>`                    |         50 MCUPS |           43 MCUPS |
| `stringzillas::SmithWatermanScores<1xSPR>`       |      1,170 MCUPS |        2,670 MCUPS |
| `stringzillas::SmithWatermanScores<16xSPR>`      |     13,620 MCUPS |       31,300 MCUPS |
| `stringzillas::SmithWatermanScores<H100>`        | __36,550 MCUPS__ |  __490,680 MCUPS__ |
|                                                  |                  |                    |
| Python                                           |                  |                    |
| `biopython.PairwiseAligner.score<1xSPR>`         |        576 MCUPS |          506 MCUPS |
| `stringzillas.SmithWatermanScores<1xSPR>`        |        583 MCUPS |          887 MCUPS |
| `stringzillas.SmithWatermanScores<batch,1xSPR>`  |        749 MCUPS |          906 MCUPS |
| `stringzillas.SmithWatermanScores<batch,16xSPR>` |      8,360 MCUPS |   __11,610 MCUPS__ |
| `stringzillas.SmithWatermanScores<batch,H100>`   | __25,020 MCUPS__ |                — † |

## Needleman-Wunsch-Gotoh for Global Alignment with Affine Gap Penalties

| Library                                            | ~100 bytes lines | ~1,000 bytes lines |
| -------------------------------------------------- | ---------------: | -----------------: |
| Rust                                               |                  |                    |
| `stringzillas::NeedlemanWunschScores<1xSPR>`       |      1,630 MCUPS |        3,380 MCUPS |
| `stringzillas::NeedlemanWunschScores<16xSPR>`      |     15,920 MCUPS |       37,080 MCUPS |
| `stringzillas::NeedlemanWunschScores<H100>`        | __33,550 MCUPS__ |  __354,540 MCUPS__ |
|                                                    |                  |                    |
| Python                                             |                  |                    |
| `biopython.PairwiseAligner.score<1xSPR>`           |        305 MCUPS |          297 MCUPS |
| `stringzillas.NeedlemanWunschScores<1xSPR>`        |        636 MCUPS |        1,470 MCUPS |
| `stringzillas.NeedlemanWunschScores<batch,1xSPR>`  |        843 MCUPS |        1,500 MCUPS |
| `stringzillas.NeedlemanWunschScores<batch,16xSPR>` |      9,750 MCUPS |   __14,040 MCUPS__ |
| `stringzillas.NeedlemanWunschScores<batch,H100>`   | __23,830 MCUPS__ |                — † |

## Smith-Waterman-Gotoh for Local Alignment with Affine Gaps

| Library                                          | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------------------------ | ---------------: | -----------------: |
| Rust                                             |                  |                    |
| `stringzillas::SmithWatermanScores<1xSPR>`       |      1,220 MCUPS |        3,260 MCUPS |
| `stringzillas::SmithWatermanScores<16xSPR>`      |     11,720 MCUPS |       35,280 MCUPS |
| `stringzillas::SmithWatermanScores<H100>`        | __30,750 MCUPS__ |  __340,060 MCUPS__ |
|                                                  |                  |                    |
| Python                                           |                  |                    |
| `biopython.PairwiseAligner.score<1xSPR>`         |        305 MCUPS |          297 MCUPS |
| `stringzillas.SmithWatermanScores<1xSPR>`        |        494 MCUPS |          904 MCUPS |
| `stringzillas.SmithWatermanScores<batch,1xSPR>`  |        579 MCUPS |          903 MCUPS |
| `stringzillas.SmithWatermanScores<batch,16xSPR>` |      7,810 MCUPS |   __10,180 MCUPS__ |
| `stringzillas.SmithWatermanScores<batch,H100>`   | __23,810 MCUPS__ |                — † |

> Measured June 17, 2026 on an Intel Xeon4 Sapphire Rapids with an NVIDIA H100, over `acgt_100.txt` and `acgt_1k.txt`.
> † The Python `stringzillas` GPU score path raises `CUDA_ERROR_ILLEGAL_ADDRESS` on ~1,000-byte inputs, so that cell is omitted; the Rust GPU path is unaffected and reaches 354,540–557,340 MCUPS there.
> ‡ `cudf` (RAPIDS) was not installed in the re-measured environment; its row is carried over from the prior run.

---

See [README.md](README.md) for dataset information and replication instructions.
