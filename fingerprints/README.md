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

Fingerprint throughput is measured at __512 dimensions__.

### Intel Xeon4 Sapphire Rapids

| Library                              |  ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------------ | ----------------: | -----------------: |
| Rust                                 |                   |                    |
| `serial::MinHash<ByteGrams,1xSPR>`   |         0.22 MB/s |          0.19 MB/s |
|                                      | 54.72% collisions |  30.03% collisions |
|                                      |    0.8530 entropy |     0.7916 entropy |
|                                      |                   |                    |
| `pc::MinHash<ByteGrams,1xSPR>`       |         1.51 MB/s |          1.95 MB/s |
|                                      | 63.68% collisions |  46.80% collisions |
|                                      |    0.9343 entropy |     0.8704 entropy |
|                                      |                   |                    |
| `stringzillas::Fingerprints<1xSPR>`  |         0.30 MB/s |          0.25 MB/s |
| `stringzillas::Fingerprints<16xSPR>` |         3.83 MB/s |          3.88 MB/s |
| `stringzillas::Fingerprints<H100>`   |    __93.98 MB/s__ |    __673.90 MB/s__ |
|                                      | 64.64% collisions |  48.30% collisions |
|                                      |    0.9980 entropy |     0.9977 entropy |
|                                      |                   |                    |
| Python                               |                   |                    |
| `datasketch.MinHash<1xSPR>`          |                 — |                  — |
| `stringzillas.Fingerprints<1xSPR>`   |                 — |                  — |
| `stringzillas.Fingerprints<16xSPR>`  |                 — |                  — |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                             |  ~100 bytes lines | ~1,000 bytes lines |
| ----------------------------------- | ----------------: | -----------------: |
| Rust                                |                   |                    |
| `serial::MinHash<ByteGrams,1xM5>`   |         0.53 MB/s |          0.47 MB/s |
|                                     | 55.33% collisions |  40.68% collisions |
|                                     |    0.8530 entropy |     0.7978 entropy |
|                                     |                   |                    |
| `pc::MinHash<ByteGrams,1xM5>`       |         2.53 MB/s |          2.80 MB/s |
|                                     | 56.80% collisions |  48.59% collisions |
|                                     |    0.9334 entropy |     0.8775 entropy |
|                                     |                   |                    |
| `stringzillas::Fingerprints<1xM5>`  |         0.93 MB/s |          0.83 MB/s |
| `stringzillas::Fingerprints<18xM5>` |     __9.23 MB/s__ |     __12.06 MB/s__ |
|                                     | 54.57% collisions |  45.33% collisions |
|                                     |    0.9980 entropy |     0.9972 entropy |
|                                     |                   |                    |
| Python                              |                   |                    |
| `datasketch.MinHash<1xM5>`          |         0.65 MB/s |          0.72 MB/s |
| `stringzillas.Fingerprints<1xM5>`   |         0.92 MB/s |          0.84 MB/s |
| `stringzillas.Fingerprints<18xM5>`  |     __9.25 MB/s__ |     __12.26 MB/s__ |

> Measured July 29, 2026.

## Quality Analysis

The trickiest part, however, is analyzing the retrieval quality of those fingerprints and comparing them to other approaches.
So, how many bits per fingerprint are needed to achieve a specific recall rate for a given dataset?
Or, how does the average Levenshtein distance among the top-k nearest neighbors change with the fingerprint size?
It must clearly decrease, but how fast, and how does that compare to ground truth?

For detailed quality analysis, please check out the [HashEvals](https://github.com/ashvardanian/HashEvals) repository.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
