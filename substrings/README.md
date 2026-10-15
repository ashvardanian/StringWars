# Multi-Pattern Search Benchmarks

Benchmarks for matching a __whole dictionary of needles__ against a __whole collection of documents__ in one pass — the workhorse of log scanning, protocol dispatch, content filtering, and signature matching.

## Overview

Every engine here is an Aho-Corasick automaton compiled once from the needle set and reused across every later call, so the dictionary is paid for outside the measured loop.
What separates them is the automaton's memory layout and what the walk is allowed to report.

- `stringzilla::Substrings` splits a goto-completed automaton into a dense 256-wide row per frequently-visited state and a double array for the rest, so a step is one load with no failure-following at runtime. It runs across a slice of CPU cores or a CUDA GPU, advancing many documents concurrently rather than making any single document faster.
- `aho-corasick` is the reference Rust implementation, benchmarked as a dense `DFA` and as a `ContiguousNFA` — the two ends of the same speed-for-memory trade the tier split makes internally.
- `daachorse` is a double-array Aho-Corasick, the direct analog of our cold tier on its own.
- `regex` over an alternation of escaped literals is the baseline most people actually write, and its literal prefilters make it a real competitor rather than a strawman.
- `pyahocorasick` and `ahocorasick_rs` are the Python side's C and Rust automata.
- `bm25`, `bm25s`, `rank_bm25`, `sklearn.TfidfVectorizer` and `cuml.TfidfVectorizer` are index-then-query scorers, present in the scoring group only. `rank_bm25` is the most installed of them, `TfidfVectorizer` is the TF-IDF reference rather than a BM25, and cuML is its GPU counterpart. No comparable GPU BM25 library exists.

Throughput is reported in __MB/s of document bytes consumed__, the rate at which the automaton advances over the corpus; the tables below come from the shared harness, which counts megabytes in powers of two.
CPU rows are single-core.
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.
Rows are named by vocabulary slice and operation, and a row is left out where StringZilla's kernel was never measured.

## Methodology

Each table fixes one input shape: `xlsum.csv` split into lines as documents, searched with one slice of that corpus's own frequency-ordered vocabulary as the dictionary.
The vocabulary is whitespace-cut words of 3 to 32 bytes, with every term occurring once and the most frequent one percent dropped; slices are taken by term count, so the frequent and the rare slice of the same percentage hold the same number of needles and differ only in byte totals, since Zipf makes frequent terms short and their automata correspondingly shallower.
This is the recipe `bench/substrings.cpp` uses inside StringZilla, so these numbers sit beside the cross-backend tables in `include/stringzilla/substrings/README.md` rather than floating free.

Each table below fixes one vocabulary slice, so a row is an engine and a column is the question asked of it.

Five operations are measured.
__Count__ tallies every overlapping match.
__Cover__ resolves a leftmost-longest cover, which is a different walk and should not be differenced against counting.
__Find__ materializes every overlapping match rather than tallying it, which is where output bandwidth starts to show.
__Replace__ rewrites every document, substituting each needle with itself so the output stays the size of the input and the cell measures the rewrite rather than an expansion factor.
__BM25__ reduces each document to one float, weighting every needle uniformly, so each slice heading doubles as a query size.

Two framing rules, without which the table lies.

__The head-to-head rows are single-core.__
Every rival is single-threaded and CPU-only, so the multi-core and GPU runs are what the engine adds, not what it beats them by, and the tables keep only the single-core CPU rows and the GPU rows.

__Scoring is scan versus index.__
`bm25`, `bm25s`, `rank_bm25` and `sklearn.TfidfVectorizer` tokenize the corpus and build an inverted index once, then answer many queries against it; `stringzilla::Substrings` scans raw bytes against a fixed dictionary and builds nothing per query.
Build and query are therefore separate rows, and a query rate is only meaningful once the build above it has been paid for.
The regime this engine is for is a corpus seen once — a firehose being filtered, or a candidate set handed over by a first-stage retriever — where the index does not exist yet and building it is the whole cost.

Uncased matching is not benchmarked here.
`aho-corasick`'s `ascii_case_insensitive` folds ASCII only, while `stringzilla::Substrings` applies full Unicode case folding to both sides of the comparison, so the two solve different problems.
The cross-backend uncased numbers live in StringZilla's own tables.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `aho_corasick<ContiguousNFA>`, `regex::find_iter`.

| StringZilla                                           |     Speed | Speedup | Alternative           |     Speed |
| :---------------------------------------------------- | --------: | ------: | :-------------------- | --------: |
| `stringzilla::Substrings`, most frequent 1% count     |  500 MB/s |   2.38× | `aho_corasick<DFA>`   |  210 MB/s |
| `stringzilla::Substrings`, most frequent 1% cover     |  282 MB/s |   1.86× | `daachorse`           |  152 MB/s |
| `stringzilla::Substrings`, most frequent 1% find      |  380 MB/s |   3.04× | `aho_corasick`        |  125 MB/s |
| `stringzilla::Substrings`, most frequent 1% replace   |  246 MB/s |   2.24× | `aho_corasick`        |  110 MB/s |
| `stringzilla::Substrings`, most frequent 1% BM25      |  361 MB/s |       – | –                     |         – |
| `stringzilla::Substrings`, most frequent 10% count    |  139 MB/s |   1.05× | `daachorse`           |  132 MB/s |
| `stringzilla::Substrings`, most frequent 10% cover    |   80 MB/s |   0.53× | `daachorse`           |  150 MB/s |
| `stringzilla::Substrings`, most frequent 10% find     |  106 MB/s |   1.23× | `aho_corasick`        | 86.0 MB/s |
| `stringzilla::Substrings`, most frequent 10% replace  | 67.4 MB/s |   0.89× | `aho_corasick`        | 75.7 MB/s |
| `stringzilla::Substrings`, most frequent 10% BM25     |   94 MB/s |       – | –                     |         – |
| `stringzilla::Substrings`, least frequent 1% count    | 4.39 GB/s |   3.14× | `daachorse`           |  1.4 GB/s |
| `stringzilla::Substrings`, least frequent 1% cover    | 3.30 GB/s |   2.36× | `daachorse`           |  1.4 GB/s |
| `stringzilla::Substrings`, least frequent 1% find     | 3.85 GB/s |   7.55× | `aho_corasick`        |  510 MB/s |
| `stringzilla::Substrings`, least frequent 1% replace  |  2.7 GB/s |   5.09× | `aho_corasick`        |  530 MB/s |
| `stringzilla::Substrings`, least frequent 1% BM25     | 3.41 GB/s |       – | –                     |         – |
| `stringzilla::Substrings`, least frequent 10% count   |  340 MB/s |   1.05× | `daachorse`           |  324 MB/s |
| `stringzilla::Substrings`, least frequent 10% cover   |  253 MB/s |   0.78× | `daachorse`           |  326 MB/s |
| `stringzilla::Substrings`, least frequent 10% find    |  301 MB/s |   1.48× | `aho_corasick`        |  203 MB/s |
| `stringzilla::Substrings`, least frequent 10% replace |  226 MB/s |   1.26× | `aho_corasick`        |  180 MB/s |
| `stringzilla::Substrings`, least frequent 10% BM25    |  288 MB/s |       – | –                     |         – |
| `stringzilla::Substrings`, entire vocabulary count    | 95.3 MB/s |   1.25× | `daachorse`           | 76.0 MB/s |
| `stringzilla::Substrings`, entire vocabulary cover    |   39 MB/s |   0.40× | `daachorse`           |   97 MB/s |
| `stringzilla::Substrings`, entire vocabulary find     | 60.6 MB/s |   1.33× | `aho_corasick`        | 45.5 MB/s |
| `stringzilla::Substrings`, entire vocabulary replace  |   33 MB/s |   0.76× | `aho_corasick`        | 43.2 MB/s |
| `stringzilla::Substrings`, entire vocabulary BM25     | 48.1 MB/s |       – | –                     |         – |

> Measured October 14, 2026.

### Python

Also benchmarked: `pyahocorasick.Automaton.iter`, `bm25s.BM25.index`, `sklearn.TfidfVectorizer.fit`.

| StringZilla                                           |     Speed | Speedup | Alternative      |     Speed |
| :---------------------------------------------------- | --------: | ------: | :--------------- | --------: |
| `stringzilla.Substrings`, most frequent 1% count      |  530 MB/s |   5.64× | `ahocorasick_rs` |   94 MB/s |
| `stringzilla.Substrings`, most frequent 1% cover      |  276 MB/s |   2.79× | `ahocorasick_rs` | 98.9 MB/s |
| `stringzilla.Substrings`, most frequent 1% find       |  380 MB/s |   4.04× | `ahocorasick_rs` |   94 MB/s |
| `stringzilla.Substrings`, most frequent 1% replace    |  243 MB/s |       – | –                |         – |
| `stringzilla.Substrings`, most frequent 1% BM25       |  359 MB/s |  0.033× | `bm25s.BM25`     |   11 GB/s |
| `stringzilla.Substrings`, most frequent 10% count     |  152 MB/s |   2.76× | `ahocorasick_rs` |   55 MB/s |
| `stringzilla.Substrings`, most frequent 10% cover     | 86.2 MB/s |   1.39× | `ahocorasick_rs` |   62 MB/s |
| `stringzilla.Substrings`, most frequent 10% find      |  113 MB/s |   2.09× | `ahocorasick_rs` |   54 MB/s |
| `stringzilla.Substrings`, most frequent 10% replace   | 75.3 MB/s |       – | –                |         – |
| `stringzilla.Substrings`, most frequent 10% BM25      | 97.3 MB/s |  0.085× | `bm25s.BM25`     | 1.14 GB/s |
| `stringzilla.Substrings`, least frequent 1% count     | 4.70 GB/s |   18.0× | `ahocorasick_rs` |  261 MB/s |
| `stringzilla.Substrings`, least frequent 1% cover     |  3.3 GB/s |   11.6× | `ahocorasick_rs` |  285 MB/s |
| `stringzilla.Substrings`, least frequent 1% find      |  4.1 GB/s |   16.0× | `ahocorasick_rs` |  256 MB/s |
| `stringzilla.Substrings`, least frequent 1% replace   |  2.8 GB/s |       – | –                |         – |
| `stringzilla.Substrings`, least frequent 1% BM25      | 3.40 GB/s |   0.30× | `bm25s.BM25`     | 11.4 GB/s |
| `stringzilla.Substrings`, least frequent 10% count    |  376 MB/s |   2.89× | `ahocorasick_rs` |  130 MB/s |
| `stringzilla.Substrings`, least frequent 10% cover    |  266 MB/s |   1.94× | `ahocorasick_rs` |  137 MB/s |
| `stringzilla.Substrings`, least frequent 10% find     |  320 MB/s |   2.41× | `ahocorasick_rs` |  133 MB/s |
| `stringzilla.Substrings`, least frequent 10% replace  |  240 MB/s |       – | –                |         – |
| `stringzilla.Substrings`, least frequent 10% BM25     |  297 MB/s |   0.23× | `bm25s.BM25`     | 1.27 GB/s |
| `stringzilla.Substrings`, entire vocabulary count     | 93.6 MB/s |   3.34× | `ahocorasick_rs` |   28 MB/s |
| `stringzilla.Substrings`, entire vocabulary cover     | 39.2 MB/s |   0.87× | `ahocorasick_rs` |   45 MB/s |
| `stringzilla.Substrings`, entire vocabulary find      |   61 MB/s |   2.10× | `ahocorasick_rs` |   29 MB/s |
| `stringzilla.Substrings`, entire vocabulary replace   |   34 MB/s |       – | –                |         – |
| `stringzilla.Substrings`, entire vocabulary BM25      | 48.7 MB/s |   0.44× | `bm25s.BM25`     |  110 MB/s |

> Measured October 14, 2026.

## 148× Nvidia SM100

### Rust

| StringZilla                                        |     Speed | Speedup | Alternative | Speed |
| :------------------------------------------------- | --------: | ------: | :---------- | ----: |
| `stringzilla::Substrings`, most frequent 1% count  |   40 GB/s |       – | –           |     – |
| `stringzilla::Substrings`, most frequent 1% BM25   |   20 GB/s |       – | –           |     – |
| `stringzilla::Substrings`, most frequent 10% count |   30 GB/s |       – | –           |     – |
| `stringzilla::Substrings`, most frequent 10% BM25  |  7.6 GB/s |       – | –           |     – |
| `stringzilla::Substrings`, least frequent 1% count |   50 GB/s |       – | –           |     – |
| `stringzilla::Substrings`, least frequent 1% BM25  |   60 GB/s |       – | –           |     – |
| `stringzilla::Substrings`, least frequent 10% count |  39 GB/s |       – | –           |     – |
| `stringzilla::Substrings`, least frequent 10% BM25 |   24 GB/s |       – | –           |     – |
| `stringzilla::Substrings`, entire vocabulary count |   28 GB/s |       – | –           |     – |
| `stringzilla::Substrings`, entire vocabulary BM25  | 4.17 GB/s |       – | –           |     – |

> Measured October 14, 2026.

### Python

| StringZilla                                        |      Speed | Speedup | Alternative | Speed |
| :------------------------------------------------- | ---------: | ------: | :---------- | ----: |
| `stringzilla.Substrings`, most frequent 1% count   |    56 GB/s |       – | –           |     – |
| `stringzilla.Substrings`, most frequent 1% BM25    |  26.2 GB/s |       – | –           |     – |
| `stringzilla.Substrings`, most frequent 10% count  |    39 GB/s |       – | –           |     – |
| `stringzilla.Substrings`, most frequent 10% BM25   |  7.87 GB/s |       – | –           |     – |
| `stringzilla.Substrings`, least frequent 1% count  |    68 GB/s |       – | –           |     – |
| `stringzilla.Substrings`, least frequent 1% BM25   |  86.3 GB/s |       – | –           |     – |
| `stringzilla.Substrings`, least frequent 10% count |    48 GB/s |       – | –           |     – |
| `stringzilla.Substrings`, least frequent 10% BM25  |  25.8 GB/s |       – | –           |     – |
| `stringzilla.Substrings`, entire vocabulary count  |  32.2 GB/s |       – | –           |     – |
| `stringzilla.Substrings`, entire vocabulary BM25   | 4.233 GB/s |       – | –           |     – |

> Measured October 14, 2026.

## Reproducing

The suite reads the first 64 MB of `data/xlsum/xlsum.csv` as lines unless `STRINGWARS_DATASET`, `STRINGWARS_BYTES` or `STRINGWARS_TOKENS` say otherwise:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --features bench_substrings --bench bench_substrings
```

With a CUDA GPU, add the feature; `STRINGWARS_DEVICES=cuda:0,cuda:1` puts one row on each named device:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --features "cuda bench_substrings" --bench bench_substrings
```

The Python side mirrors the same sweep:

```sh
uv run --group substrings substrings/bench.py
```

`STRINGWARS_FILTER` narrows a run to one operation, one vocabulary slice, or one library, since every benchmark name is `<operation>-<slice>/<library>::<call>`:

```sh
STRINGWARS_FILTER='cover-most_frequent_1%' cargo bench --features bench_substrings --bench bench_substrings
STRINGWARS_FILTER='stringzilla' uv run --group substrings substrings/bench.py
```

A note on the Python numbers: `pyahocorasick` and `ahocorasick_rs` return one Python object per match, so a dense dictionary charges them interpreter overhead the byte-level engines never pay, and their gap widens with the match count rather than with the corpus size.
`stringzilla.Substrings` returns NumPy arrays, so its per-match cost stays in C.

`re.finditer` carries no cell in the tables above, and the code measures it on the narrowest dictionary alone.
Python's `re` walks an alternation branch by branch with no literal prefilter, so at 3,462 needles a single pass over 64 MiB ran past twelve minutes without finishing, so those rows are omitted.
Rust's `regex` carries such a prefilter and sweeps all five slices, which is the comparison worth drawing.
Reproduce the Python row with a smaller `STRINGWARS_BYTES`, where it costs seconds rather than hours.
