#![doc = r#"# StringWars: Subdocuments

Batched path-query benchmarks: a vocabulary of JSONPath or XPath queries, compiled once, selecting
nodes out of every document in one walk, without building a tree.

- StringZilla `SubdocumentsEngine` on one core, one engine per core, and each selected GPU
- `serde_json_path` and `jsonpath-rust`, RFC 9535 JSONPath over a `serde_json` DOM
- `serde_json` pointers, `sonic-rs` and `simd-json`, for the paths a JSON pointer can spell
- `sxd-xpath` over an `sxd-document` DOM, for XPath

One document per line. A `.xml` dataset runs the XPath vocabulary over Stack Exchange `<row>`
elements, anything else the JSONPath vocabularies over JSON events. A line that is not a whole
document, like the header of an XML dump or the line a byte budget cuts, is left out by the same
rule in `subdocuments/bench.py`. The DOM rivals pay their parse in every pass, since selecting
nodes is what the parse is for. A rival that cannot spell a vocabulary notes its row as unavailable.

- `STRINGWARS_THREADS` sets how many engines the multi-core rows run, one per pool thread.
- `STRINGWARS_DEVICES` names the GPUs that carry rows, in a `cuda` build.

```sh
STRINGWARS_DATASET=data/gharchive/2024-01-01-15.json cargo bench --features bench_subdocuments --bench bench_subdocuments
```
"#]
use std::alloc::{Allocator, Global};
use std::fmt;
use std::hint::black_box;
use std::ops::Range;
use std::process::ExitCode;

use jsonpath_rust::JsonPath as _;
use simd_json::prelude::*;
use stringzilla::sz::{
    Capabilities, Match, Status, Stream, Strings, StringsView, SubdocumentsEngine,
    SubdocumentsFormat, UnifiedAllocator,
};

use stringwars::{
    expect_ok, finish, gpu_ordinals, gpu_stream, install_panic_hook, log_timing_overhead, measure,
    measure_per_thread, note_unavailable, print_machine, resolve_dataset, Bytes, MeasureSpec,
    ResultExt, Settings, Threads, Unit, WorkUnits,
};

/// One fixed vocabulary of paths, identical in `subdocuments/bench.py`, with the JSON pointers that
/// spell the same selection where pointers can.
struct Vocabulary {
    name: &'static str,
    paths: &'static [&'static str],
    pointers: Option<&'static [&'static str]>,
}

const JSON_VOCABULARIES: [Vocabulary; 2] = [
    Vocabulary {
        name: "json-pointer",
        paths: &["$.type", "$.actor.login", "$.repo.name"],
        pointers: Some(&["/type", "/actor/login", "/repo/name"]),
    },
    Vocabulary {
        name: "json-query",
        paths: &[
            "$.payload.commits[*].sha",
            "$..url",
            "$.payload.commits[?@.distinct == true].message",
        ],
        pointers: None,
    },
];

const XML_VOCABULARY: Vocabulary = Vocabulary {
    name: "xml-attributes",
    paths: &["/row/@Id", "/row/@Score", "/row/@Tags", "/row/@OwnerUserId"],
    pointers: None,
};

/// Whether a line is one whole document: a JSON object, or a self-closing XML `<row>` element.
fn is_document(line: &[u8], format: SubdocumentsFormat) -> bool {
    match format {
        SubdocumentsFormat::Json => line.starts_with(b"{") && line.ends_with(b"}"),
        SubdocumentsFormat::Xml => line.starts_with(b"<row") && line.ends_with(b"/>"),
    }
}

/// Where one row runs, in the order the rows run.
#[derive(Clone, Copy)]
enum Device {
    OneCore,
    AllCores(Threads),
    Gpu(i32),
}

impl fmt::Display for Device {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OneCore => formatter.write_str("1cpu"),
            Self::AllCores(threads) => write!(formatter, "{threads}cpu"),
            Self::Gpu(ordinal) => write!(formatter, "cuda:{ordinal}"),
        }
    }
}

/// The single core, every core, then each GPU the run selects.
fn selected_devices(settings: &Settings) -> Vec<Device> {
    let mut devices = vec![Device::OneCore, Device::AllCores(settings.threads)];
    for ordinal in gpu_ordinals(settings) {
        if gpu_stream(settings, ordinal).is_some() {
            devices.push(Device::Gpu(ordinal));
        }
    }
    devices
}

/// Every engine walks the same documents, so one pass over all of them is every row's work.
struct Corpus {
    documents: Vec<&'static [u8]>,
    texts: Vec<&'static str>,
    bytes: Bytes,
    format: SubdocumentsFormat,
}

impl Corpus {
    fn spec(&self, concurrency: Threads) -> MeasureSpec {
        MeasureSpec::new(
            Unit::Bytes,
            WorkUnits::new(self.documents.len() as u64, self.bytes),
        )
        .with_concurrency(concurrency)
    }

    /// The documents one of `threads` engines walks, contiguous so its outputs are too.
    fn share(&self, thread: usize, threads: usize) -> Range<usize> {
        let chunk = self.documents.len().div_ceil(threads);
        let first = (thread * chunk).min(self.documents.len());
        first..(first + chunk).min(self.documents.len())
    }
}

/// The two verbs one compiled vocabulary answers, each a group of rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verb {
    Count,
    Find,
}

/// What one round writes, all from one allocator, so a GPU round finds every buffer where its
/// device reaches.
struct Buffers<A: Allocator> {
    counts: Vec<usize, A>,
    matches: Vec<Match, A>,
    offsets: Vec<usize, A>,
}

/// A vector of `length` copies of `value` in `allocator`.
fn filled<T: Clone, A: Allocator>(length: usize, value: T, allocator: A) -> Vec<T, A> {
    let mut vector = Vec::new_in(allocator);
    vector.resize(length, value);
    vector
}

impl<A: Allocator + Clone> Buffers<A> {
    /// Buffers for `verb` over `documents`, the matches sized by an empty round whose last offset
    /// is what the filled one will need.
    fn sized(
        verb: Verb,
        engine: &mut SubdocumentsEngine,
        stream: &Stream,
        documents: StringsView<'_>,
        allocator: A,
    ) -> Self {
        let mut buffers = Buffers {
            counts: filled(documents.len(), 0, allocator.clone()),
            matches: Vec::new_in(allocator.clone()),
            offsets: filled(documents.len() + 1, 0, allocator),
        };
        if verb == Verb::Find {
            expect_ok(buffers.round(verb, engine, stream, documents));
            let total = buffers.offsets[documents.len()];
            buffers.matches.resize(total, Match::default());
        }
        buffers
    }

    /// One round of `verb`, joined before it returns.
    fn round(
        &mut self,
        verb: Verb,
        engine: &mut SubdocumentsEngine,
        stream: &Stream,
        documents: StringsView<'_>,
    ) -> Result<(), Status> {
        let mut malformed_index = 0usize;
        stream.scope(|scope| match verb {
            Verb::Count => engine.counts(scope, documents, &mut self.counts, 1, &mut malformed_index),
            Verb::Find => engine.find(
                scope,
                documents,
                &mut self.matches,
                &mut self.offsets,
                &mut malformed_index,
            ),
        })
    }

    /// Whether a device round wrote what the host engine did.
    fn agrees<B: Allocator>(&self, host: &Buffers<B>, verb: Verb) -> bool {
        match verb {
            Verb::Count => self.counts[..] == host.counts[..],
            Verb::Find => self.offsets[..] == host.offsets[..] && self.matches[..] == host.matches[..],
        }
    }
}

/// Times `verb` over `paths` on every device: one host engine, one engine per pool thread over a
/// contiguous share of the documents, and one engine per GPU whose first round is checked against
/// the host.
fn sweep(
    settings: &Settings,
    corpus: &Corpus,
    group: &str,
    paths: &[&str],
    devices: &[Device],
    verb: Verb,
) {
    let call = match verb {
        Verb::Count => "counts",
        Verb::Find => "find",
    };
    let cpu = Stream::default(Capabilities::cpu_enabled());
    for &device in devices {
        let name = format!("{group}/stringzilla::SubdocumentsEngine::{call}<{device}>");
        match device {
            Device::OneCore => {
                if !settings.selects(&name) {
                    continue;
                }
                let documents = StringsView::from_slices(&corpus.documents);
                let mut engine = match SubdocumentsEngine::new(paths, corpus.format, &cpu) {
                    Ok(engine) => engine,
                    Err(error) => {
                        note_unavailable(&name, &format!("{error:?}"));
                        continue;
                    }
                };
                let mut buffers = Buffers::sized(verb, &mut engine, &cpu, documents, Global);
                measure(settings, &name, corpus.spec(Threads::ONE), || {
                    expect_ok(buffers.round(verb, &mut engine, &cpu, documents));
                    black_box(&buffers.counts);
                });
            }
            Device::AllCores(threads) => {
                let count = threads.0.get();
                measure_per_thread(
                    settings,
                    &name,
                    corpus.spec(threads),
                    |thread| {
                        let share = corpus.share(thread, count);
                        let documents = StringsView::from_slices(&corpus.documents[share.clone()]);
                        let stream = Stream::default(Capabilities::cpu_enabled());
                        let mut engine =
                            expect_ok(SubdocumentsEngine::new(paths, corpus.format, &stream));
                        let buffers = Buffers::sized(verb, &mut engine, &stream, documents, Global);
                        (stream, engine, buffers, share)
                    },
                    |(stream, engine, buffers, share), _| {
                        let documents = StringsView::from_slices(&corpus.documents[share.clone()]);
                        expect_ok(buffers.round(verb, engine, stream, documents));
                    },
                );
            }
            Device::Gpu(ordinal) => {
                if !settings.selects(&name) {
                    continue;
                }
                let Some(stream) = gpu_stream(settings, ordinal) else {
                    continue;
                };
                let mut engine = match SubdocumentsEngine::new(paths, corpus.format, &stream) {
                    Ok(engine) => engine,
                    Err(error) => {
                        note_unavailable(&name, &format!("{error:?}"));
                        continue;
                    }
                };
                let unified = UnifiedAllocator::new(&stream);
                let documents = expect_ok(Strings::copy_in(
                    StringsView::from_slices(&corpus.documents),
                    unified,
                    &cpu,
                ));
                let documents = documents.view();
                let mut buffers = Buffers::sized(verb, &mut engine, &stream, documents, unified);
                expect_ok(buffers.round(verb, &mut engine, &stream, documents));

                let host_documents = StringsView::from_slices(&corpus.documents);
                let mut host = expect_ok(SubdocumentsEngine::new(paths, corpus.format, &cpu));
                let mut expected = Buffers::sized(verb, &mut host, &cpu, host_documents, Global);
                expect_ok(expected.round(verb, &mut host, &cpu, host_documents));
                assert!(
                    buffers.agrees(&expected, verb),
                    "{name} disagrees with the host engine on its first round"
                );

                measure(settings, &name, corpus.spec(Threads::ONE), || {
                    expect_ok(buffers.round(verb, &mut engine, &stream, documents));
                });
            }
        }
    }
}

/// Builds a single-threaded rival and times `walk` over it, or notes the row when it cannot spell
/// the vocabulary.
fn measure_rival<Engine, Error: fmt::Display>(
    settings: &Settings,
    corpus: &Corpus,
    name: &str,
    build: impl FnOnce() -> Result<Engine, Error>,
    mut walk: impl FnMut(&Engine),
) {
    if !settings.selects(name) {
        return;
    }
    match build() {
        Ok(engine) => {
            measure(settings, name, corpus.spec(Threads::ONE), || walk(&engine));
        }
        Err(error) => note_unavailable(name, &error.to_string()),
    }
}

/// The JSON rivals: two JSONPath crates over a `serde_json` DOM for every vocabulary, and the
/// pointer readers only where the vocabulary is pointers.
fn bench_json_rivals(settings: &Settings, corpus: &Corpus, group: &str, vocabulary: &Vocabulary) {
    let paths = vocabulary.paths;
    measure_rival(
        settings,
        corpus,
        &format!("{group}/serde_json_path::JsonPath::query"),
        || {
            paths
                .iter()
                .map(|path| serde_json_path::JsonPath::parse(path))
                .collect::<Result<Vec<_>, _>>()
        },
        |queries| {
            for document in &corpus.documents {
                let value: serde_json::Value = expect_ok(serde_json::from_slice(document));
                let selected: usize = queries.iter().map(|query| query.query(&value).len()).sum();
                black_box(selected);
            }
        },
    );

    measure_rival(
        settings,
        corpus,
        &format!("{group}/jsonpath_rust::JsonPath::query"),
        || {
            // Paths parse as they run, so one parse of each up front surfaces a refusal here.
            let probe = serde_json::Value::Null;
            paths
                .iter()
                .try_for_each(|path| probe.query(path).map(drop))
                .map(|()| paths)
        },
        |paths| {
            for document in &corpus.documents {
                let value: serde_json::Value = expect_ok(serde_json::from_slice(document));
                let selected: usize = paths
                    .iter()
                    .map(|path| value.query(path).map_or(0, |nodes| nodes.len()))
                    .sum();
                black_box(selected);
            }
        },
    );

    let pointer_rows = [
        "serde_json::Value::pointer",
        "sonic_rs::get",
        "simd_json::to_borrowed_value",
    ];
    let Some(pointers) = vocabulary.pointers else {
        for call in pointer_rows {
            let name = format!("{group}/{call}");
            if settings.selects(&name) {
                note_unavailable(&name, "a JSON pointer cannot spell wildcards, descendants or filters");
            }
        }
        return;
    };
    let keys: Vec<Vec<&str>> = pointers
        .iter()
        .map(|pointer| pointer.split('/').skip(1).collect())
        .collect();

    let name = format!("{group}/serde_json::Value::pointer");
    if settings.selects(&name) {
        measure(settings, &name, corpus.spec(Threads::ONE), || {
            for document in &corpus.documents {
                let value: serde_json::Value = expect_ok(serde_json::from_slice(document));
                let selected = pointers.iter().filter(|pointer| value.pointer(pointer).is_some());
                black_box(selected.count());
            }
        });
    }

    let name = format!("{group}/sonic_rs::get");
    if settings.selects(&name) {
        measure(settings, &name, corpus.spec(Threads::ONE), || {
            for document in &corpus.documents {
                let selected = keys
                    .iter()
                    .filter(|path| sonic_rs::get(*document, path.iter()).is_ok());
                black_box(selected.count());
            }
        });
    }

    // `simd-json` parses in place, so each document is copied into a reused buffer first.
    let name = format!("{group}/simd_json::to_borrowed_value");
    if settings.selects(&name) {
        let mut buffer = Vec::new();
        measure(settings, &name, corpus.spec(Threads::ONE), || {
            for document in &corpus.documents {
                buffer.clear();
                buffer.extend_from_slice(document);
                let value = expect_ok(simd_json::to_borrowed_value(&mut buffer));
                let selected = keys.iter().filter(|path| {
                    path.iter()
                        .try_fold(&value, |node, key| node.get(*key))
                        .is_some()
                });
                black_box(selected.count());
            }
        });
    }
}

/// The XML rival: compiled XPath queries over an `sxd-document` DOM.
fn bench_xml_rivals(settings: &Settings, corpus: &Corpus, group: &str, vocabulary: &Vocabulary) {
    let paths = vocabulary.paths;
    measure_rival(
        settings,
        corpus,
        &format!("{group}/sxd_xpath::XPath::evaluate"),
        || {
            let factory = sxd_xpath::Factory::new();
            paths
                .iter()
                .map(|path| {
                    factory
                        .build(path)
                        .map_err(|error| error.to_string())?
                        .ok_or_else(|| format!("{path} compiles to nothing"))
                })
                .collect::<Result<Vec<_>, String>>()
        },
        |queries| {
            let context = sxd_xpath::Context::new();
            for text in &corpus.texts {
                let package = expect_ok(sxd_document::parser::parse(text));
                let document = package.as_document();
                let selected: usize = queries
                    .iter()
                    .map(|query| match query.evaluate(&context, document.root()) {
                        Ok(sxd_xpath::Value::Nodeset(nodes)) => nodes.size(),
                        _ => 0,
                    })
                    .sum();
                black_box(selected);
            }
        },
    );
}

fn main() -> ExitCode {
    install_panic_hook();
    print_machine();
    let settings = Settings::read("subdocuments");
    settings.print();

    // Leaked so the documents borrow for the whole run.
    let tape: &'static _ = Box::leak(Box::new(resolve_dataset(&settings).unwrap_nice()));
    log_timing_overhead(&settings);

    let format = match settings.dataset_path.extension() {
        Some(extension) if extension == "xml" => SubdocumentsFormat::Xml,
        _ => SubdocumentsFormat::Json,
    };
    let documents: Vec<&'static [u8]> = tape
        .iter()
        .map(|line| line.trim_ascii())
        .filter(|line| is_document(line, format))
        .collect();
    if documents.is_empty() {
        eprintln!("The working set holds no whole document; is the dataset path right?");
        return ExitCode::FAILURE;
    }
    // The DOM rivals read `&str`, and the working set is cut at character boundaries.
    let texts: Vec<&'static str> = documents
        .iter()
        .map(|document| std::str::from_utf8(document).expect("the working set is UTF-8"))
        .collect();
    let corpus = Corpus {
        bytes: Bytes(documents.iter().map(|document| document.len() as u64).sum()),
        documents,
        texts,
        format,
    };
    let devices = selected_devices(&settings);

    println!("Benchmark configuration:");
    println!(
        "- Corpus: {} documents, {} bytes",
        corpus.documents.len(),
        corpus.bytes.0
    );
    for device in &devices {
        println!("- Device: {device}");
    }

    let vocabularies: &[Vocabulary] = match format {
        SubdocumentsFormat::Json => &JSON_VOCABULARIES,
        SubdocumentsFormat::Xml => std::slice::from_ref(&XML_VOCABULARY),
    };
    for vocabulary in vocabularies {
        println!("\n# {} - {} paths", vocabulary.name, vocabulary.paths.len());
        let count_group = format!("count-{}", vocabulary.name);
        sweep(&settings, &corpus, &count_group, vocabulary.paths, &devices, Verb::Count);
        match format {
            SubdocumentsFormat::Json => bench_json_rivals(&settings, &corpus, &count_group, vocabulary),
            SubdocumentsFormat::Xml => bench_xml_rivals(&settings, &corpus, &count_group, vocabulary),
        }
        let find_group = format!("find-{}", vocabulary.name);
        sweep(&settings, &corpus, &find_group, vocabulary.paths, &devices, Verb::Find);
    }

    finish()
}
