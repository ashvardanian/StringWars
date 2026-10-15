# Subdocument Query Benchmarks

Benchmarks for selecting nodes out of a __whole collection of JSON or XML documents__ with a __fixed vocabulary of JSONPath or XPath queries__: the shape of log enrichment, event routing, and columnar ingestion of semi-structured feeds.

## Overview

Every engine compiles the vocabulary once, outside the measured loop, and answers it for every document.
What separates them is whether they build a tree first.

- `stringzilla::SubdocumentsEngine` walks each document once against every compiled path, without building a tree, and runs on one core, one engine per core, or a CUDA GPU.
- `serde_json_path` and `jsonpath-rust` are the RFC 9535 JSONPath crates, both over a `serde_json` DOM.
- `serde_json::Value::pointer`, `sonic-rs` and `simd-json` read JSON pointers, so they answer only the vocabulary a pointer can spell; `sonic-rs` skips to the path without building a tree, the other two build one.
- `sxd-xpath` evaluates XPath 1.0 over an `sxd-document` DOM.
- `jsonpath-ng` over `orjson` documents, `orjson` with a manual walk, and `pysimdjson` pointers are the Python JSON rivals, and `lxml` the XPath one.
- `cudf.Series.str.get_json_object` is the GPU dataframe counterpart, where RAPIDS is installed.

Throughput is reported in __bytes/s of document bytes consumed__.
CPU rows are single-core.
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a query the rival's parser rejected, and `⋯` an alternative not benchmarked on that chip.
A `–` alternative is an operation that engine is never asked for, not a slow result.

## Methodology

Each line of the dataset is one document, and a line that is not a whole document is left out by the same rule in `bench.rs` and `bench.py`: a JSON line must open with `{` and close with `}`, an XML line must open with `<row` and close with `/>`.
That drops the header and footer of a Stack Exchange dump and the line a byte budget cuts.
The dataset's extension picks the format, so one suite covers both.

Three vocabularies, spelled identically in both languages:

- __json-pointer__: `$.type`, `$.actor.login` and `$.repo.name`, one value per GH Archive event, which every JSON rival can answer.
- __json-query__: every commit hash under `$.payload.commits[*]`, every `url` at any depth through `$..url`, and the messages of commits filtered on `distinct == true`.
- __xml-attributes__: the `Id`, `Score`, `Tags` and `OwnerUserId` attributes of each Stack Exchange `<row>`.

Two operations are measured.
__Count__ tallies the nodes each path selects in each document.
__Find__ materializes each selected node's span with the index of the path that selected it.

Two framing rules, without which the tables lie.

__The head-to-head rows are single-core.__
Every rival is single-threaded and CPU-only, so the multi-core and GPU runs are what the engine adds, not what it beats them by, and the tables keep only the single-core CPU rows and the GPU rows.

__The DOM rivals pay their parse in every pass.__
Selecting nodes is what the parse is for, so a DOM built outside the loop would time a lookup in a structure the engine never builds.
A rival that cannot spell a vocabulary carries `–` rather than a rewritten query: pointers spell no wildcards, descendants or filters, and `jsonpath-ng` and `get_json_object` read a subset of JSONPath that the suite probes before timing.

## Datasets

Neither corpus ships with the repository; the suite expects them under `data/` as follows.

- `data/gharchive/2024-01-01-15.json`: one hour of public GitHub events from [GH Archive](https://www.gharchive.org), one JSON object per line, from `https://data.gharchive.org/2024-01-01-15.json.gz` at 83,156,149 bytes compressed. The suite reads its first 64 MB by default.
- `data/stackexchange-bioinformatics/Posts.xml`: the posts of [bioinformatics.stackexchange.com](https://bioinformatics.stackexchange.com) from the [Stack Exchange data dump](https://archive.org/details/stackexchange), one `<row>` element per line, from `https://archive.org/download/stackexchange/bioinformatics.stackexchange.com.7z` at 17,525,174 bytes compressed.

Three more fit the suite, none of them read by default.

- `data/jsonbench-bluesky/file_0001.json`: the first of the 1,000 Bluesky event files of [ClickHouse JSONBench](https://github.com/ClickHouse/JSONBench), a million events, one JSON object per line, from `https://clickhouse-public-datasets.s3.amazonaws.com/bluesky/file_0001.json.gz` at 135,176,827 bytes compressed, the later files numbered up to `file_1000.json.gz`. Its events share none of GH Archive's top-level keys, so it needs a vocabulary of its own paths, like `$.commit.collection`.
- `data/xmark/standard.xml`: the 100 MB reference document of the [XMark](https://projects.cwi.nl/xmark/) auction benchmark, from `https://projects.cwi.nl/xmark/Assets/standard.gz` at 37,924,307 bytes compressed, for deep XPath. It is one document rather than a `<row>` per line, so it runs under `STRINGWARS_TOKENS=file` and with XPath queries of its own.
- `data/jsonpath-compliance/cts.json`: the [RFC 9535 JSONPath Compliance Test Suite](https://github.com/jsonpath-standard/jsonpath-compliance-test-suite), from `https://raw.githubusercontent.com/jsonpath-standard/jsonpath-compliance-test-suite/main/cts.json` at 236,904 bytes. It holds queries, documents and expected results, so it cross-checks the JSONPath rivals and the engine against the standard rather than timing them.

The rows are named by vocabulary and operation.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `jsonpath_rust::JsonPath::query`, `serde_json::Value::pointer`, `simd_json::to_borrowed_value`.

| StringZilla                                             |    Speed | Speedup | Alternative                        |     Speed |
| :------------------------------------------------------ | -------: | ------: | :--------------------------------- | --------: |
| `stringzilla::SubdocumentsEngine`, json-pointer count   | 3.8 GB/s |   0.79× | `sonic_rs::get`                    | 4.84 GB/s |
| `stringzilla::SubdocumentsEngine`, json-pointer find    | 3.6 GB/s |       – | –                                  |         – |
| `stringzilla::SubdocumentsEngine`, json-query count     | 1.1 GB/s |   5.00× | `serde_json_path::JsonPath::query` |  220 MB/s |
| `stringzilla::SubdocumentsEngine`, json-query find      | 1.1 GB/s |       – | –                                  |         – |
| `stringzilla::SubdocumentsEngine`, xml-attributes count | 4.6 GB/s |   65.7× | `sxd_xpath::XPath::evaluate`       |   70 MB/s |
| `stringzilla::SubdocumentsEngine`, xml-attributes find  |   4 GB/s |       – | –                                  |         – |

> Measured October 14, 2026.

### Python

Also benchmarked: `orjson.loads`.

| StringZilla                                            |    Speed | Speedup | Alternative                 |     Speed |
| :----------------------------------------------------- | -------: | ------: | :-------------------------- | --------: |
| `stringzilla.SubdocumentsEngine`, json-pointer count   |   4 GB/s |   2.00× | `simdjson.Parser.parse`     |    2 GB/s |
| `stringzilla.SubdocumentsEngine`, json-pointer find    |   4 GB/s |       – | –                           |         – |
| `stringzilla.SubdocumentsEngine`, json-query count     | 1.1 GB/s |       – | `jsonpath_ng.JSONPath.find` |         ✗ |
| `stringzilla.SubdocumentsEngine`, json-query find      |   1 GB/s |       – | –                           |         – |
| `stringzilla.SubdocumentsEngine`, xml-attributes count |   3 GB/s |   42.9× | `lxml.etree.XPath`          |   70 MB/s |
| `stringzilla.SubdocumentsEngine`, xml-attributes find  |   4 GB/s |       – | –                           |         – |

> Measured October 14, 2026.

## 148× Nvidia SM100

### Rust

| StringZilla                                             |     Speed | Speedup | Alternative | Speed |
| :------------------------------------------------------ | --------: | ------: | :---------- | ----: |
| `stringzilla::SubdocumentsEngine`, json-pointer count   | 4.07 GB/s |       – | –           |     – |
| `stringzilla::SubdocumentsEngine`, json-pointer find    | 2.06 GB/s |       – | –           |     – |
| `stringzilla::SubdocumentsEngine`, json-query count     | 2.34 GB/s |       – | –           |     – |
| `stringzilla::SubdocumentsEngine`, json-query find      | 1.18 GB/s |       – | –           |     – |
| `stringzilla::SubdocumentsEngine`, xml-attributes count |  5.5 GB/s |       – | –           |     – |
| `stringzilla::SubdocumentsEngine`, xml-attributes find  |  2.8 GB/s |       – | –           |     – |

> Measured October 14, 2026.

### Python

| StringZilla                                            |      Speed | Speedup | Alternative                       |   Speed |
| :----------------------------------------------------- | ---------: | ------: | :-------------------------------- | ------: |
| `stringzilla.SubdocumentsEngine`, json-pointer count   |  4.11 GB/s |  0.057× | `cudf.Series.str.get_json_object` | 72 GB/s |
| `stringzilla.SubdocumentsEngine`, json-pointer find    |  2.07 GB/s |       – | –                                 |       – |
| `stringzilla.SubdocumentsEngine`, json-query count     |  2.33 GB/s |       – | `cudf.Series.str.get_json_object` |       ✗ |
| `stringzilla.SubdocumentsEngine`, json-query find      | 1.177 GB/s |       – | –                                 |       – |
| `stringzilla.SubdocumentsEngine`, xml-attributes count |   5.4 GB/s |       – | –                                 |       – |
| `stringzilla.SubdocumentsEngine`, xml-attributes find  |   2.8 GB/s |       – | –                                 |       – |

> Measured October 14, 2026.

## Reproducing

The suite reads the first 64 MB of `data/gharchive/2024-01-01-15.json` as lines unless `STRINGWARS_DATASET`, `STRINGWARS_BYTES` or `STRINGWARS_TOKENS` say otherwise:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --features bench_subdocuments --bench bench_subdocuments
```

Pointing `STRINGWARS_DATASET` at an `.xml` file runs the XPath vocabulary instead:

```sh
STRINGWARS_DATASET=data/stackexchange-bioinformatics/Posts.xml cargo bench --features bench_subdocuments --bench bench_subdocuments
```

With a CUDA GPU, add the feature; `STRINGWARS_DEVICES=cuda:0,cuda:1` puts one row on each named device:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --features "cuda bench_subdocuments" --bench bench_subdocuments
```

The Python side mirrors the same sweep:

```sh
uv run --group subdocuments subdocuments/bench.py
```

`STRINGWARS_FILTER` narrows a run to one operation, one vocabulary, or one library, since every benchmark name is `<operation>-<vocabulary>/<library>::<call>`:

```sh
STRINGWARS_FILTER='count-json-query' cargo bench --features bench_subdocuments --bench bench_subdocuments
STRINGWARS_FILTER='SubdocumentsEngine' uv run --group subdocuments subdocuments/bench.py
```

---

See the [top-level README](../README.md) for dataset information and replication instructions.
