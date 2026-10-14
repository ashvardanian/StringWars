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

__The head-to-head row is `<1cpu>`.__
Every rival is single-threaded and CPU-only, so `<Ncpu>` and `<cuda:0>` are what the engine adds, not what it beats them by.

__A rival only runs a vocabulary it can spell as written.__
The `regex` crate, `RegexSet`, `google-re2`, `hyperscan` and `aho-corasick` refuse lookarounds, so they carry `–` there rather than a rewritten pattern.
The rivals see the vocabulary as one alternation with a capturing group per pattern, which is how they report which pattern matched.

__Not every count is the same count.__
`RegexSet` reports which patterns match a document, not how often, and `str.contains` reports whether any does.
`hyperscan` reports every match end it finds rather than a leftmost-first cover.
`\b` is an ASCII word boundary in StringZilla, `regex` with Unicode off, `pcre2`, `google-re2` and the ASCII-flagged Python engines, and a Unicode one in `fancy-regex`, which matters only on documents with non-ASCII letters beside a keyword.

## Performance

No run has filled these tables yet; `<Ncpu>` stands for `STRINGWARS_THREADS` engines, one per pool thread.
A `–` cell is an operation that engine is never asked for or cannot express, not a slow result.

### Literal Keywords

| Library                            | Count | Find | Replace |
| ---------------------------------- | ----: | ---: | ------: |
| Rust                               |       |      |         |
| `stringzilla::RegexEngine<1cpu>`   |       |      |         |
| `stringzilla::RegexEngine<Ncpu>`   |       |      |         |
| `stringzilla::RegexEngine<cuda:0>` |       |      |       – |
| `regex::bytes::Regex`              |       |    – |         |
| `regex::bytes::RegexSet`           |       |    – |       – |
| `fancy_regex::Regex`               |       |    – |         |
| `pcre2::bytes::Regex`              |       |    – |       – |
| `aho_corasick::AhoCorasick`        |       |    – |       – |
|                                    |       |      |         |
| Python                             |       |      |         |
| `stringzilla.RegexEngine<1cpu>`    |       |      |         |
| `stringzilla.RegexEngine<Ncpu>`    |       |      |         |
| `stringzilla.RegexEngine<cuda:0>`  |       |      |       – |
| `re`                               |       |    – |         |
| `regex`                            |       |    – |         |
| `re2`                              |       |    – |       – |
| `hyperscan`                        |       |    – |       – |
| `cudf.Series.str.count<cuda:0>`    |       |    – |       – |
| `cudf.Series.str.contains<cuda:0>` |       |    – |       – |

### Everyday Patterns

| Library                            | Count | Find | Replace |
| ---------------------------------- | ----: | ---: | ------: |
| Rust                               |       |      |         |
| `stringzilla::RegexEngine<1cpu>`   |       |      |         |
| `stringzilla::RegexEngine<Ncpu>`   |       |      |         |
| `stringzilla::RegexEngine<cuda:0>` |       |      |       – |
| `regex::bytes::Regex`              |       |    – |         |
| `regex::bytes::RegexSet`           |       |    – |       – |
| `fancy_regex::Regex`               |       |    – |         |
| `pcre2::bytes::Regex`              |       |    – |       – |
|                                    |       |      |         |
| Python                             |       |      |         |
| `stringzilla.RegexEngine<1cpu>`    |       |      |         |
| `stringzilla.RegexEngine<Ncpu>`    |       |      |         |
| `stringzilla.RegexEngine<cuda:0>`  |       |      |       – |
| `re`                               |       |    – |         |
| `regex`                            |       |    – |         |
| `re2`                              |       |    – |       – |
| `hyperscan`                        |       |    – |       – |
| `cudf.Series.str.count<cuda:0>`    |       |    – |       – |
| `cudf.Series.str.contains<cuda:0>` |       |    – |       – |

### Lookarounds

| Library                            | Count | Find | Replace |
| ---------------------------------- | ----: | ---: | ------: |
| Rust                               |       |      |         |
| `stringzilla::RegexEngine<1cpu>`   |       |      |         |
| `stringzilla::RegexEngine<Ncpu>`   |       |      |         |
| `stringzilla::RegexEngine<cuda:0>` |       |      |       – |
| `fancy_regex::Regex`               |       |    – |         |
| `pcre2::bytes::Regex`              |       |    – |       – |
|                                    |       |      |         |
| Python                             |       |      |         |
| `stringzilla.RegexEngine<1cpu>`    |       |      |         |
| `stringzilla.RegexEngine<Ncpu>`    |       |      |         |
| `stringzilla.RegexEngine<cuda:0>`  |       |      |       – |
| `re`                               |       |    – |         |
| `regex`                            |       |    – |         |
| `cudf.Series.str.count<cuda:0>`    |       |    – |       – |
| `cudf.Series.str.contains<cuda:0>` |       |    – |       – |

### Date Groups

| Library                            | Extract |
| ---------------------------------- | ------: |
| Rust                               |         |
| `stringzilla::RegexEngine<1cpu>`   |         |
| `stringzilla::RegexEngine<Ncpu>`   |         |
| `stringzilla::RegexEngine<cuda:0>` |         |
| `regex::bytes::Regex::captures`    |         |
| `fancy_regex::Regex::captures`     |         |
| `pcre2::bytes::Regex::captures`    |         |
|                                    |         |
| Python                             |         |
| `stringzilla.RegexEngine<1cpu>`    |         |
| `stringzilla.RegexEngine<Ncpu>`    |         |
| `stringzilla.RegexEngine<cuda:0>`  |         |
| `re.search`                        |         |
| `regex.search`                     |         |
| `re2.search`                       |         |
| `cudf.Series.str.extract<cuda:0>`  |         |

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
