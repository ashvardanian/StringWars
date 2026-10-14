#![doc = r#"# StringWars: Regex

Multi-pattern regular-expression benchmarks: a fixed vocabulary of patterns, compiled once, matched
leftmost-first across its union against every document in one pass.

- StringZilla `RegexEngine` on one core, one engine per core, and each selected GPU
- `regex`, as one alternation of the vocabulary and as a `RegexSet`
- `fancy-regex` and `pcre2`, which also express lookarounds
- `aho-corasick`, for the literal vocabulary only

Three vocabularies, spelled identically in `regex/bench.py`: literal keywords, everyday patterns
(emails, URLs, ISO dates, numbers, word-bounded keywords), and lookarounds. A rival that cannot
express a vocabulary notes its row as unavailable rather than running a rewritten pattern. `\b` is
an ASCII word boundary in StringZilla, `regex` with Unicode off, and PCRE2, and a Unicode one in
`fancy-regex`. `RegexSet` answers which patterns match a document, not how often.

- `STRINGWARS_THREADS` sets how many engines the multi-core rows run, one per pool thread.
- `STRINGWARS_DEVICES` names the GPUs that carry rows, in a `cuda` build.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_regex --bench bench_regex
```
"#]
use std::alloc::{Allocator, Global};
use std::fmt;
use std::hint::black_box;
use std::ops::Range;
use std::process::ExitCode;

use aho_corasick::{AhoCorasick, MatchKind};
use regex::bytes::{Captures, RegexBuilder, RegexSetBuilder};
use stringzilla::sz::{
    Capabilities, CaseSensitivity, Match, RegexEngine, Status, Stream, Strings, StringsView,
    UnifiedAllocator,
};

use stringwars::{
    expect_ok, finish, gpu_ordinals, gpu_stream, install_panic_hook, log_timing_overhead, measure,
    measure_per_thread, note_unavailable, print_machine, resolve_dataset, Bytes, MeasureSpec,
    ResultExt, Settings, Threads, Unit, WorkUnits,
};

/// One fixed vocabulary of patterns, none holding a capturing group, so wrapping each in one
/// numbers the alternatives of a rival's rewrite.
struct Vocabulary {
    name: &'static str,
    patterns: &'static [&'static str],
}

const VOCABULARIES: [Vocabulary; 3] = [
    Vocabulary {
        name: "literal",
        patterns: &["government", "minister", "president", "police"],
    },
    Vocabulary {
        name: "plain",
        patterns: &[
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            r"https?://[A-Za-z0-9.-]+(?:/[A-Za-z0-9._~%/-]*)?",
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
            r"[0-9]+(?:[.,][0-9]+)*",
            r"\b(?:government|minister|president|police)\b",
        ],
    },
    Vocabulary {
        name: "lookaround",
        patterns: &[
            r"(?<=\$)[0-9]+(?:\.[0-9]{2})?",
            r"[0-9]+(?=%)",
            r"[A-Z][a-z]+(?= [A-Z][a-z]+)",
            r"(?<![A-Za-z])[A-Z]{2,}(?![a-z])",
        ],
    },
];

/// The one pattern `extract` reads the groups of: an ISO date's year, month and day.
const DATE_GROUPS: &str = r"([0-9]{4})-([0-9]{2})-([0-9]{2})";

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

/// The verbs one compiled vocabulary answers, each a group of rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verb {
    Count,
    Find,
    Extract,
    Replace,
}

impl Verb {
    fn call(self) -> &'static str {
        match self {
            Self::Count => "counts",
            Self::Find => "find",
            Self::Extract => "extract",
            Self::Replace => "replace",
        }
    }
}

/// What one round writes, all from one allocator, so a GPU round finds every buffer where its
/// device reaches.
struct Buffers<A: Allocator> {
    counts: Vec<usize, A>,
    matches: Vec<Match, A>,
    offsets: Vec<usize, A>,
    target: Vec<u8, A>,
}

/// A vector of `length` copies of `value` in `allocator`.
fn filled<T: Clone, A: Allocator>(length: usize, value: T, allocator: A) -> Vec<T, A> {
    let mut vector = Vec::new_in(allocator);
    vector.resize(length, value);
    vector
}

impl<A: Allocator + Clone> Buffers<A> {
    /// Buffers for `verb` over `haystacks`, the variable-length ones sized by an empty round,
    /// whose last offset is what the filled one will need.
    fn sized(
        verb: Verb,
        engine: &mut RegexEngine,
        stream: &Stream,
        haystacks: StringsView<'_>,
        replacements: StringsView<'_>,
        allocator: A,
    ) -> Self {
        let mut buffers = Buffers {
            counts: filled(haystacks.len(), 0, allocator.clone()),
            matches: Vec::new_in(allocator.clone()),
            offsets: filled(haystacks.len() + 1, 0, allocator.clone()),
            target: Vec::new_in(allocator),
        };
        if verb != Verb::Count {
            expect_ok(buffers.round(verb, engine, stream, haystacks, replacements));
            let total = buffers.offsets[haystacks.len()];
            match verb {
                Verb::Replace => buffers.target.resize(total, 0),
                _ => buffers.matches.resize(total, Match::default()),
            }
        }
        buffers
    }

    /// One round of `verb`, joined before it returns.
    fn round(
        &mut self,
        verb: Verb,
        engine: &mut RegexEngine,
        stream: &Stream,
        haystacks: StringsView<'_>,
        replacements: StringsView<'_>,
    ) -> Result<(), Status> {
        stream.scope(|scope| match verb {
            Verb::Count => engine.counts(scope, haystacks, &mut self.counts, 1),
            Verb::Find => engine.find(scope, haystacks, &mut self.matches, &mut self.offsets),
            Verb::Extract => {
                engine.extract(scope, haystacks, 0, &mut self.matches, &mut self.offsets)
            }
            Verb::Replace => {
                engine.replace(scope, haystacks, replacements, &mut self.target, &mut self.offsets)
            }
        })
    }

    /// Whether a device round wrote what the host engine did.
    fn agrees<B: Allocator>(&self, host: &Buffers<B>, verb: Verb) -> bool {
        match verb {
            Verb::Count => self.counts[..] == host.counts[..],
            _ => self.offsets[..] == host.offsets[..] && self.matches[..] == host.matches[..],
        }
    }
}

/// Times `verb` over `patterns` on every device: one host engine, one engine per pool thread over
/// a contiguous share of the documents, and one engine per GPU whose first round is checked
/// against the host. A rewrite runs on the CPU alone, so its GPU rows are noted instead.
fn sweep(
    settings: &Settings,
    corpus: &Corpus,
    group: &str,
    patterns: &[&str],
    replacements: &[String],
    devices: &[Device],
    verb: Verb,
) {
    let cpu = Stream::default(Capabilities::cpu_enabled());
    let replacements_view = StringsView::from_slices(replacements);
    for &device in devices {
        let name = format!("{group}/stringzilla::RegexEngine::{}<{device}>", verb.call());
        match device {
            Device::OneCore => {
                if !settings.selects(&name) {
                    continue;
                }
                let haystacks = StringsView::from_slices(&corpus.documents);
                let mut engine = match RegexEngine::new(patterns, CaseSensitivity::Cased, &cpu) {
                    Ok(engine) => engine,
                    Err(error) => {
                        note_unavailable(&name, &format!("{error:?}"));
                        continue;
                    }
                };
                let mut buffers =
                    Buffers::sized(verb, &mut engine, &cpu, haystacks, replacements_view, Global);
                measure(settings, &name, corpus.spec(Threads::ONE), || {
                    expect_ok(buffers.round(verb, &mut engine, &cpu, haystacks, replacements_view));
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
                        let haystacks = StringsView::from_slices(&corpus.documents[share.clone()]);
                        let stream = Stream::default(Capabilities::cpu_enabled());
                        let mut engine =
                            expect_ok(RegexEngine::new(patterns, CaseSensitivity::Cased, &stream));
                        let buffers = Buffers::sized(
                            verb,
                            &mut engine,
                            &stream,
                            haystacks,
                            StringsView::from_slices(replacements),
                            Global,
                        );
                        (stream, engine, buffers, share)
                    },
                    |(stream, engine, buffers, share), _| {
                        let haystacks = StringsView::from_slices(&corpus.documents[share.clone()]);
                        let replacements = StringsView::from_slices(replacements);
                        expect_ok(buffers.round(verb, engine, stream, haystacks, replacements));
                    },
                );
            }
            Device::Gpu(ordinal) => {
                if !settings.selects(&name) {
                    continue;
                }
                if verb == Verb::Replace {
                    note_unavailable(&name, "RegexEngine::replace runs on the CPU only");
                    continue;
                }
                let Some(stream) = gpu_stream(settings, ordinal) else {
                    continue;
                };
                let mut engine = match RegexEngine::new(patterns, CaseSensitivity::Cased, &stream) {
                    Ok(engine) => engine,
                    Err(error) => {
                        note_unavailable(&name, &format!("{error:?}"));
                        continue;
                    }
                };
                let unified = UnifiedAllocator::new(&stream);
                let haystacks = expect_ok(Strings::copy_in(
                    StringsView::from_slices(&corpus.documents),
                    unified,
                    &cpu,
                ));
                let haystacks = haystacks.view();
                let mut buffers =
                    Buffers::sized(verb, &mut engine, &stream, haystacks, replacements_view, unified);
                expect_ok(buffers.round(verb, &mut engine, &stream, haystacks, replacements_view));

                let host_haystacks = StringsView::from_slices(&corpus.documents);
                let mut host = expect_ok(RegexEngine::new(patterns, CaseSensitivity::Cased, &cpu));
                let mut expected =
                    Buffers::sized(verb, &mut host, &cpu, host_haystacks, replacements_view, Global);
                expect_ok(expected.round(verb, &mut host, &cpu, host_haystacks, replacements_view));
                assert!(
                    buffers.agrees(&expected, verb),
                    "{name} disagrees with the host engine on its first round"
                );

                measure(settings, &name, corpus.spec(Threads::ONE), || {
                    expect_ok(buffers.round(verb, &mut engine, &stream, haystacks, replacements_view));
                });
            }
        }
    }
}

/// Builds a single-threaded rival and times `walk` over it, or notes the row when it cannot express
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

/// The vocabulary as one leftmost-first alternation, every pattern its own capturing group.
fn alternation(patterns: &[&str]) -> String {
    let groups: Vec<String> = patterns.iter().map(|pattern| format!("({pattern})")).collect();
    groups.join("|")
}

/// The replacement of whichever pattern matched, read from the numbered alternative it lives in.
fn replacement_of<'a>(replacements: &'a [String], matched: impl Fn(usize) -> bool) -> &'a str {
    (0..replacements.len())
        .find(|&index| matched(index + 1))
        .map_or("", |index| replacements[index].as_str())
}

/// Counts every leftmost-first match of the vocabulary, the walk every engine offers.
fn bench_counting(
    settings: &Settings,
    corpus: &Corpus,
    vocabulary: &Vocabulary,
    devices: &[Device],
) {
    let group = format!("count-{}", vocabulary.name);
    let patterns = vocabulary.patterns;
    sweep(settings, corpus, &group, patterns, &[], devices, Verb::Count);

    measure_rival(
        settings,
        corpus,
        &format!("{group}/regex::bytes::Regex::find_iter"),
        || RegexBuilder::new(&alternation(patterns)).unicode(false).build(),
        |expression| {
            let total: usize = corpus
                .documents
                .iter()
                .map(|document| expression.find_iter(document).count())
                .sum();
            black_box(total);
        },
    );

    measure_rival(
        settings,
        corpus,
        &format!("{group}/regex::bytes::RegexSet::matches"),
        || RegexSetBuilder::new(patterns).unicode(false).build(),
        |set| {
            let total: usize = corpus
                .documents
                .iter()
                .map(|document| set.matches(document).iter().count())
                .sum();
            black_box(total);
        },
    );

    measure_rival(
        settings,
        corpus,
        &format!("{group}/fancy_regex::Regex::find_iter"),
        || fancy_regex::Regex::new(&alternation(patterns)),
        |expression| {
            let total: usize = corpus
                .texts
                .iter()
                .map(|text| expression.find_iter(text).count())
                .sum();
            black_box(total);
        },
    );

    measure_rival(
        settings,
        corpus,
        &format!("{group}/pcre2::bytes::Regex::find_iter"),
        || {
            pcre2::bytes::RegexBuilder::new()
                .jit_if_available(true)
                .build(&alternation(patterns))
        },
        |expression| {
            let total: usize = corpus
                .documents
                .iter()
                .map(|document| expression.find_iter(document).count())
                .sum();
            black_box(total);
        },
    );

    // An automaton over literals, so only the literal vocabulary has this row.
    if vocabulary.name == "literal" {
        measure_rival(
            settings,
            corpus,
            &format!("{group}/aho_corasick::AhoCorasick::find_iter"),
            || {
                AhoCorasick::builder()
                    .match_kind(MatchKind::LeftmostFirst)
                    .build(patterns)
            },
            |automaton| {
                let total: usize = corpus
                    .documents
                    .iter()
                    .map(|document| automaton.find_iter(document).count())
                    .sum();
                black_box(total);
            },
        );
    }
}

/// Reads the year, month and day of each document's first ISO date.
fn bench_extracting(settings: &Settings, corpus: &Corpus, devices: &[Device]) {
    let group = "extract-date";
    sweep(settings, corpus, group, &[DATE_GROUPS], &[], devices, Verb::Extract);

    measure_rival(
        settings,
        corpus,
        &format!("{group}/regex::bytes::Regex::captures"),
        || RegexBuilder::new(DATE_GROUPS).unicode(false).build(),
        |expression| {
            for document in &corpus.documents {
                black_box(expression.captures(document));
            }
        },
    );

    measure_rival(
        settings,
        corpus,
        &format!("{group}/fancy_regex::Regex::captures"),
        || fancy_regex::Regex::new(DATE_GROUPS),
        |expression| {
            for text in &corpus.texts {
                black_box(expression.captures(text).ok());
            }
        },
    );

    measure_rival(
        settings,
        corpus,
        &format!("{group}/pcre2::bytes::Regex::captures"),
        || {
            pcre2::bytes::RegexBuilder::new()
                .jit_if_available(true)
                .build(DATE_GROUPS)
        },
        |expression| {
            for document in &corpus.documents {
                black_box(expression.captures(document).ok());
            }
        },
    );
}

/// Rewrites every match with its pattern's replacement, the one verb whose output is another corpus.
fn bench_rewriting(
    settings: &Settings,
    corpus: &Corpus,
    vocabulary: &Vocabulary,
    devices: &[Device],
) {
    let group = format!("replace-{}", vocabulary.name);
    let patterns = vocabulary.patterns;
    let replacements: Vec<String> = (0..patterns.len()).map(|index| format!("<{index}>")).collect();
    sweep(settings, corpus, &group, patterns, &replacements, devices, Verb::Replace);

    measure_rival(
        settings,
        corpus,
        &format!("{group}/regex::bytes::Regex::replace_all"),
        || RegexBuilder::new(&alternation(patterns)).unicode(false).build(),
        |expression| {
            let written: usize = corpus
                .documents
                .iter()
                .map(|document| {
                    let rewrite = |captures: &Captures| {
                        replacement_of(&replacements, |group| captures.get(group).is_some())
                            .as_bytes()
                            .to_vec()
                    };
                    expression.replace_all(document, rewrite).len()
                })
                .sum();
            black_box(written);
        },
    );

    measure_rival(
        settings,
        corpus,
        &format!("{group}/fancy_regex::Regex::replace_all"),
        || fancy_regex::Regex::new(&alternation(patterns)),
        |expression| {
            let written: usize = corpus
                .texts
                .iter()
                .map(|text| {
                    let rewrite = |captures: &fancy_regex::Captures<'_, str>| {
                        replacement_of(&replacements, |group| captures.get(group).is_some())
                            .to_string()
                    };
                    expression.replace_all(text, rewrite).len()
                })
                .sum();
            black_box(written);
        },
    );
}

fn main() -> ExitCode {
    install_panic_hook();
    print_machine();
    let settings = Settings::read("regex");
    settings.print();

    // Leaked so the documents borrow for the whole run.
    let tape: &'static _ = Box::leak(Box::new(resolve_dataset(&settings).unwrap_nice()));
    log_timing_overhead(&settings);

    let documents: Vec<&'static [u8]> = tape.iter().collect();
    // `fancy-regex` rewrites only `&str`, and the working set is cut at character boundaries.
    let texts: Vec<&'static str> = documents
        .iter()
        .map(|document| std::str::from_utf8(document).expect("the working set is UTF-8"))
        .collect();
    let corpus = Corpus {
        bytes: Bytes(documents.iter().map(|document| document.len() as u64).sum()),
        documents,
        texts,
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

    for vocabulary in &VOCABULARIES {
        println!(
            "\n# {} - {} patterns",
            vocabulary.name,
            vocabulary.patterns.len()
        );
        bench_counting(&settings, &corpus, vocabulary, &devices);
        let find_group = format!("find-{}", vocabulary.name);
        sweep(&settings, &corpus, &find_group, vocabulary.patterns, &[], &devices, Verb::Find);
        bench_rewriting(&settings, &corpus, vocabulary, &devices);
    }

    println!("\n# extract - 1 pattern");
    bench_extracting(&settings, &corpus, &devices);

    finish()
}
