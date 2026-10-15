# Multi-Way Hashing & Probabilistic Membership Benchmarks

Benchmarks for the hashing that backs probabilistic membership structures — Bloom filters and XOR/binary-fuse filters — across Rust and Python.

## Overview

A membership filter needs several independent hashes of the same short key: a _d_-ary cuckoo table probes _d_ buckets, a Bloom filter sets _k_ bits, a Count-Min sketch updates _k_ counters.
Most hash functions can only produce those by being called _k_ times, re-reading the key on each call.
StringZilla's `hash_multiseed` instead normalizes the key into AES blocks once and replays cheap per-seed rounds, emitting all _k_ hashes in a single pass.
This suite measures that primitive directly, then checks whether feeding StringZilla hashes into real filters actually helps.

Only decision-relevant comparisons are kept.
A hash that is slower, weaker, _and_ lacks a multi-seed path adds nothing, so there are no strawman columns for every library — the two baselines that change the conclusion are StringZilla's own naive per-seed calls (isolating what the multi-seed path amortizes) and `xxh3_128` using its full 128-bit output per call (the strongest single-pass alternative).
The cheap `g_i = h1 + i·h2` double-hashing shortcut that production Bloom filters use is deliberately left out: it fabricates extra _linearly dependent_ bits from one hash, so crediting them as digest bits would flatter it.

The `hash` and `hash_multiseed` rows produce a digest of independent hash bits per word over `xlsum.csv`, where the input is the digest size in bits.
Throughput is reported in produced digest bits/s so the 64-bit and 128-bit hashes line up on one scale: `stringzilla::hash` is StringZilla's hash called once per 64-bit seed; `xxh3::xxh3_128` is called once per seed and keeps its full 128-bit output, so it re-prepares the input only every 128 bits.
Every value is independent and each variant writes into one preallocated buffer (a NumPy array in Python), so neither double-hashing nor per-call allocation skews the comparison.

The `build` and `query` rows build each filter from the unique words, then query a held-out 20% to measure the false-positive rate, in keys per second.
Each filter is compared StringZilla-fed against its practical default with the structure held fixed, and each pair matches in bits per key and in false-positive rate.

Every table reports single-core throughput.
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.

The multi-seed path prepares the input once and replays cheap per-seed rounds, so its throughput climbs almost linearly with the digest width while the naive variants plateau — StringZilla's own `hash` flattens near 12 G bits/s on Xeon5 because it re-prepares the key every 64 bits.
`xxh3_128` keeps its full 128-bit output, so it re-prepares only every 128 bits and stays ahead of `stringzilla::hash` throughout.
Whether it also beats `hash_multiseed` is machine-dependent: on Xeon5 it leads only at 128 bits and trails from 256 bits up, and on M5 it leads at every width.
In Python the picture inverts for the baselines: per-call interpreter dispatch dominates, so `stringzilla.hash` (one native call per 64 bits) stays ahead of `xxhash.xxh3_128` (whose 128-bit output costs an extra big-integer split), while `hash_multiseed` — a single native call that fills the whole buffer — runs an order of magnitude ahead of both.

Feeding StringZilla helps exactly where the filter accepts a precomputed hash.
In Rust, `fastbloom`'s `insert_hash` / `contains_hash` take a single `sz::hash` and expand it internally, roughly doubling build and query throughput at identical bits-per-key and FPR, and `xorf` — built from a deduplicated `u64` array — queries faster with StringZilla keys.
In Python, `pyprobables` opens the same door: `add_alt` / `check_alt` take precomputed hashes, so one native `hash_multiseed` call replaces its default per-key FNV-1a loop at matching bits-per-key and FPR.
The lesson mirrors Layer 1 — StringZilla's hashing wins whenever a structure lets it hash each key once and hand over the result, in either language; it loses only when the API dribbles the key through a per-element hash callback (as `rbloom`'s `hash_func` does, with no precomputed-hash entry point).

## Intel Xeon5 Emerald Rapids

### Rust

| StringZilla                                  |         Speed | Speedup | Alternative                 |         Speed |
| :------------------------------------------- | ------------: | ------: | :-------------------------- | ------------: |
| `stringzilla::hash`, 128 bits                | 10.4 G bits/s |   0.69× | `xxh3::xxh3_128`            | 15.0 G bits/s |
| `stringzilla::hash`, 256 bits                | 11.2 G bits/s |   0.46× | `xxh3::xxh3_128`            | 24.6 G bits/s |
| `stringzilla::hash`, 512 bits                | 11.6 G bits/s |   0.34× | `xxh3::xxh3_128`            | 34.5 G bits/s |
| `stringzilla::hash`, 1024 bits               | 12.0 G bits/s |   0.29× | `xxh3::xxh3_128`            | 41.7 G bits/s |
| `stringzilla::hash_multiseed`, 128 bits      | 13.7 G bits/s |   0.91× | `xxh3::xxh3_128`            | 15.0 G bits/s |
| `stringzilla::hash_multiseed`, 256 bits      | 26.1 G bits/s |   1.06× | `xxh3::xxh3_128`            | 24.6 G bits/s |
| `stringzilla::hash_multiseed`, 512 bits      | 43.1 G bits/s |   1.25× | `xxh3::xxh3_128`            | 34.5 G bits/s |
| `stringzilla::hash_multiseed`, 1024 bits     | 67.9 G bits/s |   1.63× | `xxh3::xxh3_128`            | 41.7 G bits/s |
| `fastbloom<stringzilla>`, build              |   25 M keys/s |   2.50× | `fastbloom<siphash>`        |   10 M keys/s |
| `fastbloom<stringzilla>`, query              |   20 M keys/s |   2.00× | `fastbloom<siphash>`        |   10 M keys/s |
| `xorf::BinaryFuse8<stringzilla>`, build      |   14 M keys/s |   0.82× | `xorf::BinaryFuse8<xxh3>`   |   17 M keys/s |
| `xorf::BinaryFuse8<stringzilla>`, query      |   33 M keys/s |   1.06× | `xorf::BinaryFuse8<xxh3>`   |   31 M keys/s |

> Measured October 14, 2026.

### Python

| StringZilla                                 |          Speed | Speedup | Alternative        |          Speed |
| :------------------------------------------ | -------------: | ------: | :----------------- | -------------: |
| `stringzilla.hash`, 128 bits                |   500 M bits/s |   1.12× | `xxhash.xxh3_128`  |   445 M bits/s |
| `stringzilla.hash`, 256 bits                |   604 M bits/s |   1.18× | `xxhash.xxh3_128`  |   512 M bits/s |
| `stringzilla.hash`, 512 bits                |   649 M bits/s |   1.18× | `xxhash.xxh3_128`  |   550 M bits/s |
| `stringzilla.hash`, 1024 bits               |   691 M bits/s |   1.19× | `xxhash.xxh3_128`  |   583 M bits/s |
| `stringzilla.hash_multiseed`, 128 bits      |   1.9 G bits/s |   4.27× | `xxhash.xxh3_128`  |   445 M bits/s |
| `stringzilla.hash_multiseed`, 256 bits      |   3.8 G bits/s |   7.42× | `xxhash.xxh3_128`  |   512 M bits/s |
| `stringzilla.hash_multiseed`, 512 bits      |   7.4 G bits/s |   13.5× | `xxhash.xxh3_128`  |   550 M bits/s |
| `stringzilla.hash_multiseed`, 1024 bits     |    14 G bits/s |   24.0× | `xxhash.xxh3_128`  |   583 M bits/s |
| `pyprobables<stringzilla>`, build           | 0.446 M keys/s |   3.95× | `pyprobables<fnv>` | 0.113 M keys/s |
| `pyprobables<stringzilla>`, query           | 0.475 M keys/s |   4.13× | `pyprobables<fnv>` | 0.115 M keys/s |

> Measured October 14, 2026.

## Apple M5 Pro

### Rust

| StringZilla                              |          Speed | Speedup | Alternative               |          Speed |
| :--------------------------------------- | -------------: | ------: | :------------------------ | -------------: |
| `stringzilla::hash`, 128 bits            | 13.62 G bits/s |   0.50× | `xxh3::xxh3_128`          | 27.01 G bits/s |
| `stringzilla::hash`, 256 bits            | 18.11 G bits/s |   0.41× | `xxh3::xxh3_128`          | 44.01 G bits/s |
| `stringzilla::hash`, 512 bits            | 23.10 G bits/s |   0.35× | `xxh3::xxh3_128`          | 65.47 G bits/s |
| `stringzilla::hash`, 1024 bits           | 26.90 G bits/s |   0.34× | `xxh3::xxh3_128`          | 79.79 G bits/s |
| `stringzilla::hash_multiseed`, 128 bits  | 14.90 G bits/s |   0.55× | `xxh3::xxh3_128`          | 27.01 G bits/s |
| `stringzilla::hash_multiseed`, 256 bits  | 25.73 G bits/s |   0.58× | `xxh3::xxh3_128`          | 44.01 G bits/s |
| `stringzilla::hash_multiseed`, 512 bits  | 39.67 G bits/s |   0.61× | `xxh3::xxh3_128`          | 65.47 G bits/s |
| `stringzilla::hash_multiseed`, 1024 bits | 55.96 G bits/s |   0.70× | `xxh3::xxh3_128`          | 79.79 G bits/s |
| `fastbloom<stringzilla>`, build          | 35.58 M keys/s |   1.85× | `fastbloom<siphash>`      | 19.24 M keys/s |
| `fastbloom<stringzilla>`, query          | 35.09 M keys/s |   1.85× | `fastbloom<siphash>`      | 18.98 M keys/s |
| `xorf::BinaryFuse8<stringzilla>`, build  | 25.97 M keys/s |   0.91× | `xorf::BinaryFuse8<xxh3>` | 28.42 M keys/s |
| `xorf::BinaryFuse8<stringzilla>`, query  | 64.96 M keys/s |   1.06× | `xorf::BinaryFuse8<xxh3>` | 61.29 M keys/s |

> Measured July 29, 2026.

### Python

| StringZilla                             |          Speed | Speedup | Alternative        |          Speed |
| :-------------------------------------- | -------------: | ------: | :----------------- | -------------: |
| `stringzilla.hash`, 128 bits            | 841.7 M bits/s |   0.98× | `xxhash.xxh3_128`  | 855.4 M bits/s |
| `stringzilla.hash`, 256 bits            | 963.0 M bits/s |   1.03× | `xxhash.xxh3_128`  | 932.5 M bits/s |
| `stringzilla.hash`, 512 bits            |  1.07 G bits/s |   1.05× | `xxhash.xxh3_128`  |  1.02 G bits/s |
| `stringzilla.hash_multiseed`, 128 bits  |  3.34 G bits/s |   3.90× | `xxhash.xxh3_128`  | 855.4 M bits/s |
| `stringzilla.hash_multiseed`, 256 bits  |  6.35 G bits/s |   6.81× | `xxhash.xxh3_128`  | 932.5 M bits/s |
| `stringzilla.hash_multiseed`, 512 bits  | 11.88 G bits/s |   11.6× | `xxhash.xxh3_128`  |  1.02 G bits/s |
| `stringzilla.hash_multiseed`, 1024 bits | 21.30 G bits/s |       – | `xxhash.xxh3_128`  |              ⋯ |
| `pyprobables<stringzilla>`, build       |  0.89 M keys/s |   6.36× | `pyprobables<fnv>` |  0.14 M keys/s |
| `pyprobables<stringzilla>`, query       |  0.95 M keys/s |   6.33× | `pyprobables<fnv>` |  0.15 M keys/s |

> Measured July 29, 2026.

---

See [README.md](../README.md) for dataset information and replication instructions.
