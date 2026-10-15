# Regular Expression Benchmarks

Benchmarks for matching a __fixed vocabulary of regular expressions__ against a __whole collection of documents__ in one pass: the shape of log parsing, PII redaction, and entity extraction.

## Overview

Every engine compiles the vocabulary once, outside the measured loop, and walks every document leftmost-first across the union of its patterns.

- `stringzilla::RegexEngine` compiles RE2 syntax plus bounded lookarounds into one automaton over all patterns, and runs on one core, one engine per core, or a CUDA GPU.
- `regex` is the reference Rust engine, benchmarked over one alternation of the vocabulary and as a `RegexSet`.
- `fancy-regex` and `pcre2` backtrack, so they are the Rust rivals that also express lookarounds.
- `aho-corasick` answers the literal vocabulary alone, since it matches no metacharacters.
- `re` and the `regex` package are the Python standard and its drop-in replacement, both run with their ASCII flags.
- `google-re2` and `hyperscan` are the automata most Python deployments reach for when `re` is too slow.
- `cudf.Series.str.count`, `str.contains` and `str.extract` are the GPU dataframe counterparts, where RAPIDS is installed.

Throughput is reported in __bytes/s of document bytes consumed__.
CPU rows are single-core.
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a pattern the engine rejected at compile time, and `⋯` an alternative not benchmarked on that chip.
A `–` alternative is an operation that engine is never asked for or cannot express, not a slow result.

## Methodology

Every table fixes one vocabulary over `xlsum.csv` split into lines as documents, spelled identically in `bench.rs` and `bench.py`:

- __literal__: `government`, `minister`, `president` and `police` as four separate patterns.
- __plain__: emails, `http` and `https` URLs, ISO dates, decimal numbers, and the four keywords between `\b` boundaries.
- __lookaround__: amounts after a `$`, numbers before a `%`, a capitalized word before another one, and all-capital acronyms.

Four operations are measured.
__Count__ tallies every leftmost-first match of the vocabulary in each document.
__Find__ materializes every match with the index of the pattern that produced it.
__Replace__ rewrites every match with a short marker naming its pattern, so the output is another corpus.
__Extract__ reads the year, month and day groups of each document's first ISO date.

Three framing rules, without which the tables lie.

__The head-to-head rows are single-core.__
Every rival is single-threaded and CPU-only, so the multi-core and GPU runs are what the engine adds, not what it beats them by, and the tables keep only the single-core CPU rows and the GPU rows.

__A rival only runs a vocabulary it can spell as written.__
The `regex` crate, `RegexSet`, `google-re2`, `hyperscan` and `aho-corasick` refuse lookarounds, so they carry `–` there rather than a rewritten pattern.
The rivals see the vocabulary as one alternation with a capturing group per pattern, which is how they report which pattern matched.

__Not every count is the same count.__
`RegexSet` reports which patterns match a document, not how often, and `str.contains` reports whether any does.
`hyperscan` reports every match end it finds rather than a leftmost-first cover.
`\b` is an ASCII word boundary in StringZilla, `regex` with Unicode off, `pcre2`, `google-re2` and the ASCII-flagged Python engines, and a Unicode one in `fancy-regex`, which matters only on documents with non-ASCII letters beside a keyword.

## Datasets

`xlsum.csv` stays the default, since the __literal__ and __lookaround__ vocabularies are news words and figures.
News text holds few emails, URLs or log lines, so the __plain__ vocabulary is better measured over the corpora below.
None ships with the repository; the suite expects them under `data/` and reads one through `STRINGWARS_DATASET`.

- `data/rebar/lh3lh3-reb-howto.txt`: the HOWTO concatenation of Heng Li's regex benchmark, dense in emails, URLs and dates, curated by [rebar](https://github.com/BurntSushi/rebar), from `https://raw.githubusercontent.com/BurntSushi/rebar/master/benchmarks/haystacks/imported/lh3lh3-reb-howto.txt` at 39,422,105 bytes.
- `data/rebar/mariomka.txt`: the haystack of the [mariomka regex benchmark](https://github.com/mariomka/regex-benchmark) for emails, URIs and IPv4 addresses, from `https://raw.githubusercontent.com/BurntSushi/rebar/master/benchmarks/haystacks/imported/mariomka.txt` at 6,839,410 bytes.
- `data/rebar/opensubtitles-en-sampled.txt`: sampled English subtitles, rebar's general-text haystack, from `https://raw.githubusercontent.com/BurntSushi/rebar/master/benchmarks/haystacks/opensubtitles/en-sampled.txt` at 899,232 bytes.
- `data/loghub-openstack/openstack.log`: the OpenStack logs of [LogHub](https://github.com/logpai/loghub), timestamped request lines with IPs and paths, its `.log` files concatenated out of `https://zenodo.org/records/8196385/files/OpenStack.tar.gz` at 5,394,691 bytes compressed.

For a multi-pattern vocabulary of production size, the [Emerging Threats Open](https://rules.emergingthreats.net) Suricata rules are at `https://rules.emergingthreats.net/open/suricata-7.0.3/emerging.rules.tar.gz`, 5,630,528 bytes compressed.
Their `content` and `pcre` fields need extracting into patterns, and no small public payload capture was found to scan them over, so the logs above stand in for the traffic.

The rows are named by vocabulary and operation.
`regex::bytes::RegexSet`, `hyperscan` and `cudf.Series.str.contains` count something other than a leftmost-first cover, so they are never alternatives to a count row.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `regex::bytes::RegexSet`, `fancy_regex::Regex::captures`, `pcre2::bytes::Regex::captures`.

| StringZilla                                      |      Speed | Speedup | Alternative                     |      Speed |
| :----------------------------------------------- | ---------: | ------: | :------------------------------ | ---------: |
| `stringzilla::RegexEngine`, literal count        | 1.010 GB/s |  0.099× | `aho_corasick::AhoCorasick`     |  10.2 GB/s |
| `stringzilla::RegexEngine`, literal find         |  1.00 GB/s |       – | –                               |          – |
| `stringzilla::RegexEngine`, literal replace      |   989 MB/s |   0.13× | `regex::bytes::Regex`           |  7.41 GB/s |
| `stringzilla::RegexEngine`, plain count          |   540 MB/s |   0.90× | `regex::bytes::Regex`           | 600.7 MB/s |
| `stringzilla::RegexEngine`, plain find           |   540 MB/s |       – | –                               |          – |
| `stringzilla::RegexEngine`, plain replace        |   531 MB/s |   1.09× | `regex::bytes::Regex`           |   485 MB/s |
| `stringzilla::RegexEngine`, lookaround count     |  13.0 MB/s |  0.013× | `pcre2::bytes::Regex`           | 1.021 GB/s |
| `stringzilla::RegexEngine`, lookaround find      |  12.8 MB/s |       – | –                               |          – |
| `stringzilla::RegexEngine`, lookaround replace   |  12.9 MB/s |   0.90× | `fancy_regex::Regex`            |  14.4 MB/s |
| `stringzilla::RegexEngine`, date groups extract  |  7.87 GB/s |   0.59× | `regex::bytes::Regex::captures` |  13.4 GB/s |

> Measured October 14, 2026.

### Python

Also benchmarked: `hyperscan`, `re.search`, `re2.search`.

| StringZilla                                      |     Speed | Speedup | Alternative    |      Speed |
| :----------------------------------------------- | --------: | ------: | :------------- | ---------: |
| `stringzilla.RegexEngine`, literal count         | 1.03 GB/s |   2.61× | `re2`          |   394 MB/s |
| `stringzilla.RegexEngine`, literal find          |  1.0 GB/s |       – | –              |          – |
| `stringzilla.RegexEngine`, literal replace       |  901 MB/s |   4.90× | `regex`        |   184 MB/s |
| `stringzilla.RegexEngine`, plain count           |  540 MB/s |   5.24× | `re2`          |   103 MB/s |
| `stringzilla.RegexEngine`, plain find            |  530 MB/s |       – | –              |          – |
| `stringzilla.RegexEngine`, plain replace         |  518 MB/s |   23.5× | `re`           |    22 MB/s |
| `stringzilla.RegexEngine`, lookaround count      | 13.1 MB/s |   0.54× | `re`           | 24.15 MB/s |
| `stringzilla.RegexEngine`, lookaround find       | 13.0 MB/s |       – | –              |          – |
| `stringzilla.RegexEngine`, lookaround replace    | 12.9 MB/s |   0.54× | `re`           |  23.8 MB/s |
| `stringzilla.RegexEngine`, date groups extract   |  8.1 GB/s |   3.19× | `regex.search` |  2.54 GB/s |

> Measured October 14, 2026.

## 148× Nvidia SM100

### Rust

| StringZilla                                      |     Speed | Speedup | Alternative | Speed |
| :----------------------------------------------- | --------: | ------: | :---------- | ----: |
| `stringzilla::RegexEngine`, literal count        |   10 GB/s |       – | –           |     – |
| `stringzilla::RegexEngine`, literal find         |   11 GB/s |       – | –           |     – |
| `stringzilla::RegexEngine`, plain count          |  9.2 GB/s |       – | –           |     – |
| `stringzilla::RegexEngine`, plain find           |  9.3 GB/s |       – | –           |     – |
| `stringzilla::RegexEngine`, lookaround count     |  353 MB/s |       – | –           |     – |
| `stringzilla::RegexEngine`, lookaround find      |  350 MB/s |       – | –           |     – |
| `stringzilla::RegexEngine`, date groups extract  |  6.3 GB/s |       – | –           |     – |

> Measured October 14, 2026.

### Python

Also benchmarked: `cudf.Series.str.contains`.

| StringZilla                                      |     Speed | Speedup | Alternative                |      Speed |
| :----------------------------------------------- | --------: | ------: | :------------------------- | ---------: |
| `stringzilla.RegexEngine`, literal count         |  9.9 GB/s |   19.0× | `cudf.Series.str.count`    | 520.2 MB/s |
| `stringzilla.RegexEngine`, literal find          |   10 GB/s |       – | –                          |          – |
| `stringzilla.RegexEngine`, plain count           |  8.7 GB/s |    103× | `cudf.Series.str.count`    |  84.3 MB/s |
| `stringzilla.RegexEngine`, plain find            |  9.0 GB/s |       – | –                          |          – |
| `stringzilla.RegexEngine`, lookaround count      |  355 MB/s |       – | `cudf.Series.str.count`    |          ✗ |
| `stringzilla.RegexEngine`, lookaround find       |  354 MB/s |       – | –                          |          – |
| `stringzilla.RegexEngine`, date groups extract   |  6.3 GB/s |   9.71× | `cudf.Series.str.extract`  | 648.9 MB/s |

> Measured October 14, 2026.

## Reproducing

The suite reads the first 64 MB of `data/xlsum/xlsum.csv` as lines unless `STRINGWARS_DATASET`, `STRINGWARS_BYTES` or `STRINGWARS_TOKENS` say otherwise:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --features bench_regex --bench bench_regex
```

With a CUDA GPU, add the feature; `STRINGWARS_DEVICES=cuda:0,cuda:1` puts one row on each named device:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --features "cuda bench_regex" --bench bench_regex
```

The Python side mirrors the same sweep:

```sh
uv run --group regex regex/bench.py
```

`STRINGWARS_FILTER` narrows a run to one operation, one vocabulary, or one library, since every benchmark name is `<operation>-<vocabulary>/<library>::<call>`:

```sh
STRINGWARS_FILTER='count-lookaround' cargo bench --features bench_regex --bench bench_regex
STRINGWARS_FILTER='RegexEngine' uv run --group regex regex/bench.py
```

---

See the [top-level README](../README.md) for dataset information and replication instructions.
