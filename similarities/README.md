# Similarity Scoring Benchmarks

Benchmarks for string similarity and alignment algorithms across Rust and Python implementations, including CPU and GPU variants.

## Overview

Edit Distance calculation is a common component of Search Engines, Data Cleaning, and Natural Language Processing, as well as in Bioinformatics.
It's a computationally expensive operation, generally implemented using dynamic programming, with a quadratic time complexity upper bound.
For biological sequences, the Needleman-Wunsch and Smith-Waterman algorithms are more appropriate, as they allow overriding the default substitution costs.
Each of those has two flavors - with linear and affine gap penalties, also known as the "Gotoh" variation.

Performance is measured in MCUPS (Million Cell Updates Per Second).
The Xeon5 tables keep the prefix the harness prints, where 1 GCUPS is 10³ MCUPS and 1 TCUPS is 10⁶ MCUPS.
Both harnesses (`bench.rs` and `bench.py`) score an all-pairs cross-product: a side × side matrix of disjoint query and candidate batches, where each axis is round(√(`STRINGWARS_BATCH_PER_CORE` × cores)).
A CPU core and a GPU streaming-multiprocessor (SM) each count as one core, so CPU rows run one core and the GPU rows feed the 148× Nvidia SM100 device a batch sized across all its SMs.
MCUPS is the aggregate cell-update count over the whole matrix divided by the wall-clock time.

Datasets:

- __acgt_100 / acgt_1k__ — uniform synthetic DNA, lines of ~100 and ~1,000 bytes (~100 bytes is also the canonical protein-domain length).
- __XLSum words / lines__ — highly non-uniform multilingual text: ~5-byte word tokens and ~3.2 KB article lines.

The synthetic-DNA and word runs use `STRINGWARS_BATCH_PER_CORE=16384` to saturate the GPU (single-core side 128, four-core side 256, GPU side 1,557).
The ~3.2 KB XLSum-lines run uses `STRINGWARS_BATCH_PER_CORE=256` (GPU side 195) because each line-vs-line pair carries ~1,000× more cells than a word pair, so a smaller batch already saturates the device and keeps the single-core runs tractable.
StringZilla scores every column with the same unary 32-class match/mismatch costs, and the `bio` / `biopython` baselines use the same unary match/mismatch scoring, so the cross-language comparison stays apples-to-apples.

The `LevenshteinDistances` rows score Levenshtein distance, and the `NeedlemanWunschScores` and `SmithWatermanScores` rows score global and local alignment, with _linear_ gaps for the plain Needleman-Wunsch and Smith-Waterman algorithms and _affine_ gaps for their Gotoh variations.
Every CPU table reports single-core MCUPS, GCUPS or TCUPS, with the harness prefix kept on Xeon5 and MCUPS on Apple M5 Pro.
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.
Byte-level StringZilla rows race the byte-level rivals and the UTF-8 rows race the character-level ones.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `bio::levenshtein`.

| StringZilla                                              |      Speed | Speedup | Alternative                     |      Speed |
| :------------------------------------------------------- | ---------: | ------: | :------------------------------ | ---------: |
| `stringzilla::LevenshteinDistances`, ACGT 100B           |   83 GCUPS |   9.33× | `rapidfuzz::levenshtein<Bytes>` |  8.9 GCUPS |
| `stringzilla::LevenshteinDistances`, ACGT 1KB            |  150 GCUPS |   6.52× | `rapidfuzz::levenshtein<Bytes>` |   23 GCUPS |
| `stringzilla::LevenshteinDistances`, XLSum words         |  6.6 GCUPS |   2.54× | `rapidfuzz::levenshtein<Bytes>` |  2.6 GCUPS |
| `stringzilla::LevenshteinDistances`, XLSum lines         | 46.4 GCUPS |   2.44× | `rapidfuzz::levenshtein<Bytes>` |   19 GCUPS |
| `stringzilla::LevenshteinDistancesUtf8`, ACGT 100B       |   47 GCUPS |   6.03× | `rapidfuzz::levenshtein<Chars>` |  7.8 GCUPS |
| `stringzilla::LevenshteinDistancesUtf8`, ACGT 1KB        |  140 GCUPS |   6.09× | `rapidfuzz::levenshtein<Chars>` |   23 GCUPS |
| `stringzilla::LevenshteinDistancesUtf8`, XLSum words     |  490 MCUPS |   0.94× | `strsim::levenshtein<Chars>`    |  520 MCUPS |
| `stringzilla::LevenshteinDistancesUtf8`, XLSum lines     | 53.0 GCUPS |   2.79× | `rapidfuzz::levenshtein<Chars>` |   19 GCUPS |
| `stringzilla::NeedlemanWunschScores`, ACGT 100B linear   |          ✗ |       – | `bio::pairwise::global`         |  110 MCUPS |
| `stringzilla::NeedlemanWunschScores`, ACGT 1KB linear    |          ✗ |       – | `bio::pairwise::global`         |  129 MCUPS |
| `stringzilla::NeedlemanWunschScores`, XLSum words linear |          ✗ |       – | `bio::pairwise::global`         |  100 MCUPS |
| `stringzilla::NeedlemanWunschScores`, XLSum lines linear |          ✗ |       – | –                               |          – |
| `stringzilla::SmithWatermanScores`, ACGT 100B linear     |          ✗ |       – | `bio::pairwise::local`          | 95.2 MCUPS |
| `stringzilla::SmithWatermanScores`, ACGT 1KB linear      |          ✗ |       – | `bio::pairwise::local`          |  120 MCUPS |
| `stringzilla::SmithWatermanScores`, XLSum words linear   |          ✗ |       – | `bio::pairwise::local`          |   85 MCUPS |
| `stringzilla::SmithWatermanScores`, XLSum lines linear   |          ✗ |       – | –                               |          – |
| `stringzilla::NeedlemanWunschScores`, ACGT 100B affine   |          ✗ |       – | `bio::pairwise::global`         |   94 MCUPS |
| `stringzilla::NeedlemanWunschScores`, ACGT 1KB affine    |          ✗ |       – | `bio::pairwise::global`         |  111 MCUPS |
| `stringzilla::NeedlemanWunschScores`, XLSum words affine |          ✗ |       – | `bio::pairwise::global`         |   90 MCUPS |
| `stringzilla::NeedlemanWunschScores`, XLSum lines affine |          ✗ |       – | –                               |          – |
| `stringzilla::SmithWatermanScores`, ACGT 100B affine     |          ✗ |       – | `bio::pairwise::local`          |   70 MCUPS |
| `stringzilla::SmithWatermanScores`, ACGT 1KB affine      |          ✗ |       – | `bio::pairwise::local`          | 95.8 MCUPS |
| `stringzilla::SmithWatermanScores`, XLSum words affine   |          ✗ |       – | `bio::pairwise::local`          |   74 MCUPS |
| `stringzilla::SmithWatermanScores`, XLSum lines affine   |          ✗ |       – | –                               |          – |

> Measured October 14, 2026.

### Python

Also benchmarked: `jellyfish.levenshtein_distance`, `editdistance.eval`, `nltk.edit_distance`, `edlib.align`.

| StringZilla                                               |      Speed | Speedup | Alternative                        |      Speed |
| :-------------------------------------------------------- | ---------: | ------: | :--------------------------------- | ---------: |
| `stringzilla.LevenshteinDistances`, ACGT 100B             |   84 GCUPS |   9.03× | `rapidfuzz.Levenshtein.distance`   |  9.3 GCUPS |
| `stringzilla.LevenshteinDistances`, ACGT 1KB              |  150 GCUPS |   4.84× | `rapidfuzz.Levenshtein.distance`   |   31 GCUPS |
| `stringzilla.LevenshteinDistances`, XLSum words           |  7.1 GCUPS |   5.92× | `polyleven.levenshtein`            |  1.2 GCUPS |
| `stringzilla.LevenshteinDistances`, XLSum lines           | 46.0 GCUPS |   1.65× | `Levenshtein.distance`             | 27.9 GCUPS |
| `stringzilla.LevenshteinDistancesUTF8`, ACGT 100B         |   58 GCUPS |   6.24× | `rapidfuzz.Levenshtein.distance`   |  9.3 GCUPS |
| `stringzilla.LevenshteinDistancesUTF8`, ACGT 1KB          |  140 GCUPS |   4.52× | `rapidfuzz.Levenshtein.distance`   |   31 GCUPS |
| `stringzilla.LevenshteinDistancesUTF8`, XLSum words       |  2.0 GCUPS |   1.67× | `polyleven.levenshtein`            |  1.2 GCUPS |
| `stringzilla.LevenshteinDistancesUTF8`, XLSum lines       | 53.3 GCUPS |   1.91× | `Levenshtein.distance`             | 27.9 GCUPS |
| `stringzilla.NeedlemanWunschScores`, ACGT 100B linear     |          ✗ |       – | `biopython.PairwiseAligner.global` |  590 MCUPS |
| `stringzilla.NeedlemanWunschScores`, ACGT 1KB linear      |          ✗ |       – | `biopython.PairwiseAligner.global` |  573 MCUPS |
| `stringzilla.NeedlemanWunschScores`, XLSum words linear   |          ✗ |       – | `biopython.PairwiseAligner.global` |   81 MCUPS |
| `stringzilla.NeedlemanWunschScores`, XLSum lines linear   |          ✗ |       – | `biopython.PairwiseAligner.global` | 1.21 GCUPS |
| `stringzilla.SmithWatermanScores`, ACGT 100B linear       |          ✗ |       – | `biopython.PairwiseAligner.local`  |  400 MCUPS |
| `stringzilla.SmithWatermanScores`, ACGT 1KB linear        |          ✗ |       – | `biopython.PairwiseAligner.local`  |  532 MCUPS |
| `stringzilla.SmithWatermanScores`, XLSum words linear     |          ✗ |       – | `biopython.PairwiseAligner.local`  |   81 MCUPS |
| `stringzilla.SmithWatermanScores`, XLSum lines linear     |          ✗ |       – | `biopython.PairwiseAligner.local`  |  930 MCUPS |
| `stringzilla.NeedlemanWunschScores`, ACGT 100B affine     |          ✗ |       – | `biopython.PairwiseAligner.global` |  325 MCUPS |
| `stringzilla.NeedlemanWunschScores`, ACGT 1KB affine      |          ✗ |       – | `biopython.PairwiseAligner.global` |  330 MCUPS |
| `stringzilla.NeedlemanWunschScores`, XLSum words affine   |          ✗ |       – | `biopython.PairwiseAligner.global` |   78 MCUPS |
| `stringzilla.NeedlemanWunschScores`, XLSum lines affine   |          ✗ |       – | `biopython.PairwiseAligner.global` |  700 MCUPS |
| `stringzilla.SmithWatermanScores`, ACGT 100B affine       |          ✗ |       – | `biopython.PairwiseAligner.local`  |  140 MCUPS |
| `stringzilla.SmithWatermanScores`, ACGT 1KB affine        |          ✗ |       – | `biopython.PairwiseAligner.local`  |  278 MCUPS |
| `stringzilla.SmithWatermanScores`, XLSum words affine     |          ✗ |       – | `biopython.PairwiseAligner.local`  |   76 MCUPS |
| `stringzilla.SmithWatermanScores`, XLSum lines affine     |          ✗ |       – | `biopython.PairwiseAligner.local`  |  609 MCUPS |

> Measured October 14, 2026.

## 148× Nvidia SM100

### Rust

| StringZilla                                              |      Speed | Speedup | Alternative | Speed |
| :------------------------------------------------------- | ---------: | ------: | :---------- | ----: |
| `stringzilla::LevenshteinDistances`, ACGT 100B           |   20 TCUPS |       – | –           |     – |
| `stringzilla::LevenshteinDistances`, XLSum words         |  600 GCUPS |       – | –           |     – |
| `stringzilla::LevenshteinDistancesUtf8`, ACGT 100B       |   17 TCUPS |       – | –           |     – |
| `stringzilla::LevenshteinDistancesUtf8`, XLSum words     |  300 GCUPS |       – | –           |     – |
| `stringzilla::NeedlemanWunschScores`, ACGT 100B linear   |          ✗ |       – | –           |     – |
| `stringzilla::NeedlemanWunschScores`, ACGT 1KB linear    |          ✗ |       – | –           |     – |
| `stringzilla::NeedlemanWunschScores`, XLSum words linear |          ✗ |       – | –           |     – |
| `stringzilla::NeedlemanWunschScores`, XLSum lines linear |          ✗ |       – | –           |     – |
| `stringzilla::SmithWatermanScores`, ACGT 100B linear     |          ✗ |       – | –           |     – |
| `stringzilla::SmithWatermanScores`, ACGT 1KB linear      |          ✗ |       – | –           |     – |
| `stringzilla::SmithWatermanScores`, XLSum words linear   |          ✗ |       – | –           |     – |
| `stringzilla::SmithWatermanScores`, XLSum lines linear   |          ✗ |       – | –           |     – |
| `stringzilla::NeedlemanWunschScores`, ACGT 100B affine   |          ✗ |       – | –           |     – |
| `stringzilla::NeedlemanWunschScores`, ACGT 1KB affine    |          ✗ |       – | –           |     – |
| `stringzilla::NeedlemanWunschScores`, XLSum words affine |          ✗ |       – | –           |     – |
| `stringzilla::NeedlemanWunschScores`, XLSum lines affine |          ✗ |       – | –           |     – |
| `stringzilla::SmithWatermanScores`, ACGT 100B affine     |          ✗ |       – | –           |     – |
| `stringzilla::SmithWatermanScores`, ACGT 1KB affine      |          ✗ |       – | –           |     – |
| `stringzilla::SmithWatermanScores`, XLSum words affine   |          ✗ |       – | –           |     – |
| `stringzilla::SmithWatermanScores`, XLSum lines affine   |          ✗ |       – | –           |     – |

> Measured October 14, 2026.

### Python

| StringZilla                                               |      Speed | Speedup | Alternative          |      Speed |
| :-------------------------------------------------------- | ---------: | ------: | :------------------- | ---------: |
| `stringzilla.LevenshteinDistances`, ACGT 100B             |   20 TCUPS |  4,000× | `cudf.edit_distance` |    5 GCUPS |
| `stringzilla.LevenshteinDistances`, XLSum words           |  710 GCUPS | 35,500× | `cudf.edit_distance` |   20 MCUPS |
| `stringzilla.LevenshteinDistancesUTF8`, ACGT 100B         |   20 TCUPS |  4,000× | `cudf.edit_distance` |    5 GCUPS |
| `stringzilla.LevenshteinDistancesUTF8`, XLSum words       |  360 GCUPS | 18,000× | `cudf.edit_distance` |   20 MCUPS |
| `stringzilla.NeedlemanWunschScores`, ACGT 100B linear     |          ✗ |       – | –                    |          – |
| `stringzilla.NeedlemanWunschScores`, ACGT 1KB linear      |          ✗ |       – | –                    |          – |
| `stringzilla.NeedlemanWunschScores`, XLSum words linear   |          ✗ |       – | –                    |          – |
| `stringzilla.NeedlemanWunschScores`, XLSum lines linear   |          ✗ |       – | –                    |          – |
| `stringzilla.SmithWatermanScores`, ACGT 100B linear       |          ✗ |       – | –                    |          – |
| `stringzilla.SmithWatermanScores`, ACGT 1KB linear        |          ✗ |       – | –                    |          – |
| `stringzilla.SmithWatermanScores`, XLSum words linear     |          ✗ |       – | –                    |          – |
| `stringzilla.SmithWatermanScores`, XLSum lines linear     |          ✗ |       – | –                    |          – |
| `stringzilla.NeedlemanWunschScores`, ACGT 100B affine     |          ✗ |       – | –                    |          – |
| `stringzilla.NeedlemanWunschScores`, ACGT 1KB affine      |          ✗ |       – | –                    |          – |
| `stringzilla.NeedlemanWunschScores`, XLSum words affine   |          ✗ |       – | –                    |          – |
| `stringzilla.NeedlemanWunschScores`, XLSum lines affine   |          ✗ |       – | –                    |          – |
| `stringzilla.SmithWatermanScores`, ACGT 100B affine       |          ✗ |       – | –                    |          – |
| `stringzilla.SmithWatermanScores`, ACGT 1KB affine        |          ✗ |       – | –                    |          – |
| `stringzilla.SmithWatermanScores`, XLSum words affine     |          ✗ |       – | –                    |          – |
| `stringzilla.SmithWatermanScores`, XLSum lines affine     |          ✗ |       – | –                    |          – |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `bio::levenshtein`.

| StringZilla                                              |        Speed | Speedup | Alternative                     |        Speed |
| :------------------------------------------------------- | -----------: | ------: | :------------------------------ | -----------: |
| `stringzilla::LevenshteinDistances`, ACGT 100B           | 22,420 MCUPS |   2.46× | `rapidfuzz::levenshtein<Bytes>` |  9,119 MCUPS |
| `stringzilla::LevenshteinDistances`, ACGT 1KB            | 54,250 MCUPS |   2.18× | `rapidfuzz::levenshtein<Bytes>` | 24,880 MCUPS |
| `stringzilla::LevenshteinDistances`, XLSum words         |  2,482 MCUPS |   1.22× | `rapidfuzz::levenshtein<Bytes>` |  2,037 MCUPS |
| `stringzilla::LevenshteinDistances`, XLSum lines         | 14,070 MCUPS |   0.65× | `rapidfuzz::levenshtein<Bytes>` | 21,590 MCUPS |
| `stringzilla::LevenshteinDistancesUtf8`, ACGT 100B       | 22,390 MCUPS |   2.58× | `rapidfuzz::levenshtein<Chars>` |  8,664 MCUPS |
| `stringzilla::LevenshteinDistancesUtf8`, ACGT 1KB        | 54,260 MCUPS |   2.22× | `rapidfuzz::levenshtein<Chars>` | 24,460 MCUPS |
| `stringzilla::LevenshteinDistancesUtf8`, XLSum words     |    711 MCUPS |   1.52× | `rapidfuzz::levenshtein<Chars>` |    468 MCUPS |
| `stringzilla::LevenshteinDistancesUtf8`, XLSum lines     | 28,440 MCUPS |   1.35× | `rapidfuzz::levenshtein<Chars>` | 21,080 MCUPS |
| `stringzilla::NeedlemanWunschScores`, ACGT 100B linear   |    974 MCUPS |   6.37× | `bio::pairwise::global`         |    153 MCUPS |
| `stringzilla::NeedlemanWunschScores`, ACGT 1KB linear    |  1,071 MCUPS |   6.82× | `bio::pairwise::global`         |    157 MCUPS |
| `stringzilla::NeedlemanWunschScores`, XLSum words linear |    252 MCUPS |   2.03× | `bio::pairwise::global`         |    124 MCUPS |
| `stringzilla::SmithWatermanScores`, ACGT 100B linear     |    960 MCUPS |   6.67× | `bio::pairwise::local`          |    144 MCUPS |
| `stringzilla::SmithWatermanScores`, ACGT 1KB linear      |  1,048 MCUPS |   7.13× | `bio::pairwise::local`          |    147 MCUPS |
| `stringzilla::SmithWatermanScores`, XLSum words linear   |    249 MCUPS |   1.95× | `bio::pairwise::local`          |    128 MCUPS |
| `stringzilla::NeedlemanWunschScores`, ACGT 100B affine   |    929 MCUPS |   6.73× | `bio::pairwise::global`         |    138 MCUPS |
| `stringzilla::NeedlemanWunschScores`, ACGT 1KB affine    |    986 MCUPS |   6.94× | `bio::pairwise::global`         |    142 MCUPS |
| `stringzilla::NeedlemanWunschScores`, XLSum words affine |    243 MCUPS |   1.93× | `bio::pairwise::global`         |    126 MCUPS |
| `stringzilla::SmithWatermanScores`, ACGT 100B affine     |    886 MCUPS |   7.45× | `bio::pairwise::local`          |    119 MCUPS |
| `stringzilla::SmithWatermanScores`, ACGT 1KB affine      |    956 MCUPS |   7.71× | `bio::pairwise::local`          |    124 MCUPS |
| `stringzilla::SmithWatermanScores`, XLSum words affine   |    232 MCUPS |   1.77× | `bio::pairwise::local`          |    131 MCUPS |

> Measured July 29, 2026.

### Python

Also benchmarked: `jellyfish.levenshtein_distance`, `editdistance.eval`, `nltk.edit_distance`, `edlib.align`.

| StringZilla                                               |        Speed | Speedup | Alternative                        |        Speed |
| :-------------------------------------------------------- | -----------: | ------: | :--------------------------------- | -----------: |
| `stringzilla.LevenshteinDistances`, ACGT 100B             | 22,080 MCUPS |   2.04× | `polyleven.levenshtein`            | 10,830 MCUPS |
| `stringzilla.LevenshteinDistances`, ACGT 1KB              | 54,550 MCUPS |   1.87× | `rapidfuzz.Levenshtein.distance`   | 29,250 MCUPS |
| `stringzilla.LevenshteinDistances`, XLSum words           |  1,396 MCUPS |   3.03× | `polyleven.levenshtein`            |    461 MCUPS |
| `stringzilla.LevenshteinDistances`, XLSum lines           | 14,480 MCUPS |   0.55× | `Levenshtein.distance`             | 26,420 MCUPS |
| `stringzilla.LevenshteinDistancesUTF8`, ACGT 100B         | 22,070 MCUPS |   2.04× | `polyleven.levenshtein`            | 10,830 MCUPS |
| `stringzilla.LevenshteinDistancesUTF8`, ACGT 1KB          | 54,530 MCUPS |   1.86× | `rapidfuzz.Levenshtein.distance`   | 29,250 MCUPS |
| `stringzilla.LevenshteinDistancesUTF8`, XLSum words       |    479 MCUPS |   1.04× | `polyleven.levenshtein`            |    461 MCUPS |
| `stringzilla.LevenshteinDistancesUTF8`, XLSum lines       | 27,130 MCUPS |   1.03× | `Levenshtein.distance`             | 26,420 MCUPS |
| `stringzilla.NeedlemanWunschScores`, ACGT 100B linear     |    823 MCUPS |   0.90× | `biopython.PairwiseAligner.global` |    914 MCUPS |
| `stringzilla.NeedlemanWunschScores`, ACGT 1KB linear      |  1,017 MCUPS |   1.21× | `biopython.PairwiseAligner.global` |    842 MCUPS |
| `stringzilla.NeedlemanWunschScores`, XLSum words linear   |     24 MCUPS |   0.71× | `biopython.PairwiseAligner.global` |     34 MCUPS |
| `stringzilla.SmithWatermanScores`, ACGT 100B linear       |    843 MCUPS |   1.09× | `biopython.PairwiseAligner.local`  |    776 MCUPS |
| `stringzilla.SmithWatermanScores`, ACGT 1KB linear        |  1,055 MCUPS |   1.35× | `biopython.PairwiseAligner.local`  |    782 MCUPS |
| `stringzilla.SmithWatermanScores`, XLSum words linear     |     24 MCUPS |   0.71× | `biopython.PairwiseAligner.local`  |     34 MCUPS |
| `stringzilla.NeedlemanWunschScores`, ACGT 100B affine     |    902 MCUPS |   1.48× | `biopython.PairwiseAligner.global` |    610 MCUPS |
| `stringzilla.NeedlemanWunschScores`, ACGT 1KB affine      |  1,007 MCUPS |   1.44× | `biopython.PairwiseAligner.global` |    700 MCUPS |
| `stringzilla.NeedlemanWunschScores`, XLSum words affine   |     43 MCUPS |   1.26× | `biopython.PairwiseAligner.global` |     34 MCUPS |
| `stringzilla.SmithWatermanScores`, ACGT 100B affine       |    879 MCUPS |   2.27× | `biopython.PairwiseAligner.local`  |    387 MCUPS |
| `stringzilla.SmithWatermanScores`, ACGT 1KB affine        |    975 MCUPS |   2.46× | `biopython.PairwiseAligner.local`  |    396 MCUPS |
| `stringzilla.SmithWatermanScores`, XLSum words affine     |     43 MCUPS |   1.30× | `biopython.PairwiseAligner.local`  |     33 MCUPS |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
