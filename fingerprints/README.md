# Fingerprinting and Sketching Benchmarks

Benchmarks for byte-level fingerprinting and sketching algorithms across CPU and GPU implementations.

## Overview

In large-scale Retrieval workloads a common technique is to convert variable-length messy strings into some fixed-length representations.
Those are often called "fingerprints" or "sketches", like "Min-Hashing" or "Count-Min-Sketching".
There are a million variations of those algorithms, all resulting in different speed-vs-accuracy tradeoffs.

Two of the approximations worth considering are:

- The number of collisions of produced individual hashes within fingerprints
- The bit-distribution entropy of the produced fingerprints

Adjusting all implementations to the same tokenization scheme, one may experience the following numbers:

## Performance and Quality Metrics

| Library                                  |  ~100 bytes lines | ~1,000 bytes lines |
| ---------------------------------------- | ----------------: | -----------------: |
| serial `<ByteGrams>` on 1x SPR           |         0.44 MB/s |          0.47 MB/s |
|                                          | 92.81% collisions |  94.58% collisions |
|                                          |    0.8528 entropy |     0.7979 entropy |
|                                          |                   |                    |
| `pc::MinHash<ByteGrams>` on 1x SPR       |         2.41 MB/s |          3.16 MB/s |
|                                          | 91.80% collisions |  93.17% collisions |
|                                          |    0.9343 entropy |     0.8779 entropy |
|                                          |                   |                    |
| `stringzillas::Fingerprints` on 1x SPR   |         0.56 MB/s |          0.51 MB/s |
| `stringzillas::Fingerprints` on 16x SPR  |         6.62 MB/s |          8.03 MB/s |
| `stringzillas::Fingerprints` on 384x GNR |   **231.13 MB/s** |    **302.30 MB/s** |
| `stringzillas::Fingerprints` on RTX6000  |      **138 MB/s** |        162.99 MB/s |
| `stringzillas::Fingerprints` on H100     |       102.07 MB/s |    **392.37 MB/s** |
|                                          | 86.80% collisions |  93.21% collisions |
|                                          |    0.9992 entropy |     0.9967 entropy |

## Quality Analysis

The trickiest part, however, is analyzing the retrieval quality of those fingerprints and comparing them to other approaches.
So, how many bits per fingerprint are needed to achieve a specific recall rate for a given dataset?
Or, how does the average Levenshtein distance among the top-k nearest neighbors change with the fingerprint size?
It must clearly decrease, but how fast, and how does that compare to ground truth?

For detailed quality analysis, please check out the [HashEvals](https://github.com/ashvardanian/HashEvals) repository.

---

See [README.md](README.md) for dataset information and replication instructions.
