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

| Library                                 |  ~100 bytes lines | ~1,000 bytes lines |
| --------------------------------------- | ----------------: | -----------------: |
| serial `<ByteGrams>` on 1x SPR          |         0.43 MB/s |          0.39 MB/s |
|                                         | 91.25% collisions |  94.70% collisions |
|                                         |    0.8529 entropy |     0.7980 entropy |
|                                         |                   |                    |
| `pc::MinHash<ByteGrams>` on 1x SPR      |         3.12 MB/s |          2.62 MB/s |
|                                         | 88.82% collisions |  93.05% collisions |
|                                         |    0.9346 entropy |     0.8775 entropy |
|                                         |                   |                    |
| `stringzillas::Fingerprints` on 1x SPR  |         0.57 MB/s |          0.52 MB/s |
| `stringzillas::Fingerprints` on 16x SPR |         7.32 MB/s |          7.66 MB/s |
| `stringzillas::Fingerprints` on H100    |   **632.41 MB/s** |   **2630.00 MB/s** |
|                                         | 86.08% collisions |  90.12% collisions |
|                                         |    0.9978 entropy |     0.9968 entropy |

## Quality Analysis

The trickiest part, however, is analyzing the retrieval quality of those fingerprints and comparing them to other approaches.
So, how many bits per fingerprint are needed to achieve a specific recall rate for a given dataset?
Or, how does the average Levenshtein distance among the top-k nearest neighbors change with the fingerprint size?
It must clearly decrease, but how fast, and how does that compare to ground truth?

For detailed quality analysis, please check out the [HashEvals](https://github.com/ashvardanian/HashEvals) repository.

---

See [README.md](README.md) for dataset information and replication instructions.
