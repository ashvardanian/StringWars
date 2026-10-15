# Window Overlap Benchmarks

Benchmarks for the exact n-gram window overlap between short texts, across CPU and GPU implementations.

## Overview

In large-scale retrieval workloads a common technique is to compare variable-length messy strings by the byte windows they share.
`OverlapEngine` scores, for every query, candidate and n-gram width, the share of the candidate's byte windows found among the query's distinct ones, over the window count of the longer side.
It scores the first `STRINGWARS_BATCH_PER_CORE` documents as queries against every document as a candidate.

- `serial::window_overlap` and `sklearn.CountVectorizer` compute the same shares exactly, the latter as the sparse product `Q × Cᵀ` of binary query windows and counted candidate windows, both checked against the engine before timing.
- `cudf.Series.str.jaccard_index<aligned>` scores the same pairs laid out as aligned rows, since it has no cross product, and it scores __Jaccard over distinct character windows__, a different normalization, so only its zero pattern is checked against the engine and it is not an alternative.

Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.
Every table reports single-core throughput.

## Intel Xeon5 Emerald Rapids

### Rust

| StringZilla                                           |    Speed | Speedup | Alternative              |    Speed |
| :---------------------------------------------------- | -------: | ------: | :----------------------- | -------: |
| `stringzilla::OverlapEngine::scores`, ~100-byte lines | 940 KB/s |   6.39× | `serial::window_overlap` | 147 KB/s |

> Measured October 14, 2026.

### Python

| StringZilla                                         |    Speed | Speedup | Alternative               |    Speed |
| :-------------------------------------------------- | -------: | ------: | :------------------------ | -------: |
| `stringzilla.OverlapEngine.scores`, ~100-byte lines | 908 KB/s |   0.61× | `sklearn.CountVectorizer` | 1.5 MB/s |

> Measured October 14, 2026.

## 148× Nvidia SM100

### Rust

| StringZilla                                           | Speed | Speedup | Alternative | Speed |
| :---------------------------------------------------- | ----: | ------: | :---------- | ----: |
| `stringzilla::OverlapEngine::scores`, ~100-byte lines |     ✗ |       – | –           |     – |

> Measured October 14, 2026.

### Python

Also benchmarked: `cudf.Series.str.jaccard_index<aligned>`.

| StringZilla                                         | Speed | Speedup | Alternative | Speed |
| :-------------------------------------------------- | ----: | ------: | :---------- | ----: |
| `stringzilla.OverlapEngine.scores`, ~100-byte lines |     ✗ |       – | –           |     – |

> Measured October 14, 2026.

## Datasets

`acgt_100` stays the default: it is what the tables above were measured on, and the serial reference bounds the working set to 1 MB.
Its lines are random, so they hold no near-duplicates, and overlap scores over it mostly measure noise.
For real near-duplicates, point `STRINGWARS_DATASET` at the corpus below; like the others, it does not ship with the repository.

- `data/core-deduplication/train.jsonl`: the [CORE deduplication dataset](https://huggingface.co/datasets/pinecone/core-2020-05-10-deduplication) of scholarly articles with labelled near-duplicate versions, one JSON record per line, from `https://huggingface.co/datasets/pinecone/core-2020-05-10-deduplication/resolve/main/train.jsonl` at 203,599,780 bytes. The labelled pairs are spread across the file, so a 1 MB working set holds few of them.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
