#![doc = r#"# StringWars: Substrings

Multi-pattern search benchmarks: one dictionary of needles matched against every document in one pass.
Every engine compiles its needle set once and reuses it, so the dictionary is paid for outside the
measured loop.

- StringZilla `SubstringsEngine` on one core, one engine per core, and each selected GPU
- `aho-corasick`, the reference Rust implementation, as a dense DFA and as a contiguous NFA
- `daachorse`, a double-array Aho-Corasick
- `regex` over an alternation of literals, the baseline most people actually write
- `bm25`, an index-then-query scorer, for the scoring group only

Every rival is single-threaded and CPU-only, so the head-to-head row is `<1cpu>`. Scoring is scan
versus index: `bm25` tokenizes the corpus into an inverted index once and answers many queries,
while `SubstringsEngine` scans raw bytes against a fixed dictionary, so build and query are separate
rows. Uncased matching is left out: `aho-corasick` folds ASCII only, while `SubstringsEngine` folds
the whole Unicode table.

The dictionary comes from the working set itself: words of 3 to 32 bytes under the normative `words`
rule, every term occurring once and the most frequent one percent dropped, then sliced by frequency.

- `STRINGWARS_THREADS` sets how many engines the multi-core rows run, one per pool thread.
- `STRINGWARS_DEVICES` names the GPUs that carry rows, in a `cuda` build.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_substrings --bench bench_substrings
```
"#]
use std::alloc::{Allocator, Global};
use std::collections::HashMap;
use std::fmt;
use std::hint::black_box;
use std::ops::Range;
use std::process::ExitCode;

use aho_corasick::{AhoCorasick, AhoCorasickKind, MatchKind};
use daachorse::{
    DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind as DoubleArrayMatchKind,
};
use regex::bytes::RegexBuilder;
use stringzilla::sz::{
    Bm25Params, Capabilities, CaseSensitivity, Match, Status, Stream, Strings, StringsView,
    SubstringsEngine, SubstringsOverlapPolicy, UnifiedAllocator, SUBSTRINGS_HOT_STATES_AUTO,
    SUBSTRINGS_MATCHES_BUDGET_AUTO,
};

use stringwars::{
    expect_ok, finish, gpu_ordinals, gpu_stream, install_panic_hook, log_timing_overhead, measure,
    measure_per_thread, note_unavailable, print_machine, resolve_dataset, token_ranges, Bytes,
    MeasureSpec, ResultExt, Settings, Threads, Tokenization, Unit, WorkUnits,
};

/// Shortest word admitted into the dictionary: anything under three bytes matches at nearly every
/// position, so the benchmark would measure match materialization rather than the automaton walk.
const VOCABULARY_MIN_WORD_BYTES: usize = 3;

/// Longest word admitted. Longer whitespace-cut tokens are unsegmented CJK runs and URLs rather than
/// words, and the longest needle sets the warm-up every GPU chunk re-walks.
const VOCABULARY_MAX_WORD_BYTES: usize = 32;

/// Which end of the frequency-ordered vocabulary a dictionary is drawn from, and how much of it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VocabularySlice {
    MostFrequentOnePercent,
    MostFrequentTenPercent,
    LeastFrequentOnePercent,
    LeastFrequentTenPercent,
    Entire,
}

impl VocabularySlice {
    fn name(self) -> &'static str {
        match self {
            Self::MostFrequentOnePercent => "most_frequent_1%",
            Self::MostFrequentTenPercent => "most_frequent_10%",
            Self::LeastFrequentOnePercent => "least_frequent_1%",
            Self::LeastFrequentTenPercent => "least_frequent_10%",
            Self::Entire => "entire",
        }
    }
}

const VOCABULARY_SLICES: [VocabularySlice; 5] = [
    VocabularySlice::MostFrequentOnePercent,
    VocabularySlice::MostFrequentTenPercent,
    VocabularySlice::LeastFrequentOnePercent,
    VocabularySlice::LeastFrequentTenPercent,
    VocabularySlice::Entire,
];

/// Counts word frequencies over every document and drops both noisy ends, returning the survivors
/// ordered by descending frequency with ties broken by content so the ranking is reproducible.
fn build_vocabulary(documents: &[&'static [u8]]) -> Vec<&'static [u8]> {
    let mut frequencies: HashMap<&'static [u8], usize> = HashMap::new();
    for &document in documents {
        for range in token_ranges(document, Tokenization::Words) {
            let word = &document[range];
            if (VOCABULARY_MIN_WORD_BYTES..=VOCABULARY_MAX_WORD_BYTES).contains(&word.len()) {
                *frequencies.entry(word).or_insert(0) += 1;
            }
        }
    }

    // The hapax filter runs before the sort, so the ranking touches only the few percent that
    // survive. The top cutoff keeps its base: a fraction of all distinct terms, hapax included.
    let distinct_count = frequencies.len();
    let mut ranked: Vec<(&'static [u8], usize)> = frequencies
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .collect();
    ranked.sort_unstable_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));

    let dropped_from_top = (distinct_count / 100).min(ranked.len());
    ranked.drain(..dropped_from_top);
    ranked.into_iter().map(|(term, _)| term).collect()
}

/// One contiguous slice of a non-empty frequency-ordered vocabulary, never empty itself.
fn needle_slice_of(vocabulary: &[&'static [u8]], slice: VocabularySlice) -> Vec<&'static [u8]> {
    let available = vocabulary.len();
    let count = match slice {
        VocabularySlice::MostFrequentOnePercent | VocabularySlice::LeastFrequentOnePercent => {
            available / 100
        }
        VocabularySlice::MostFrequentTenPercent | VocabularySlice::LeastFrequentTenPercent => {
            available / 10
        }
        VocabularySlice::Entire => available,
    }
    .clamp(1, available);
    let first = match slice {
        VocabularySlice::LeastFrequentOnePercent | VocabularySlice::LeastFrequentTenPercent => {
            available - count
        }
        _ => 0,
    };
    vocabulary[first..first + count].to_vec()
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

/// The verbs one compiled dictionary answers, each a group of rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verb {
    Count,
    Cover,
    Find,
    Replace,
    Score,
}

impl Verb {
    fn name(self, slice: &str, device: Device) -> String {
        let (group, call) = match self {
            Self::Count => ("count", "counts"),
            Self::Cover => ("cover", "counts"),
            Self::Find => ("find", "find"),
            Self::Replace => ("replace", "replace"),
            Self::Score => ("bm25", "bm25_scores"),
        };
        format!("{group}-{slice}/stringzilla::SubstringsEngine::{call}<{device}>")
    }

    /// The overlap policy is fixed at construction: covers and rewrites need a leftmost one.
    fn policy(self) -> SubstringsOverlapPolicy {
        match self {
            Self::Cover | Self::Replace => SubstringsOverlapPolicy::LeftmostLongest,
            _ => SubstringsOverlapPolicy::Overlapping,
        }
    }
}

/// Compiles `needles` for `verb` on `stream`. A device tier sizes one round's arena for
/// `haystacks_budget` documents, while a host engine walks straight into the caller's outputs.
fn compile(
    needles: &[&[u8]],
    verb: Verb,
    haystacks_budget: usize,
    stream: &Stream,
) -> Result<SubstringsEngine, Status> {
    SubstringsEngine::new(
        needles,
        CaseSensitivity::Cased,
        verb.policy(),
        SUBSTRINGS_HOT_STATES_AUTO,
        SUBSTRINGS_MATCHES_BUDGET_AUTO,
        haystacks_budget,
        stream,
    )
}

/// What one round reads beyond the documents and writes, all from one allocator, so a GPU round
/// finds every buffer where its device reaches.
struct Buffers<A: Allocator> {
    counts: Vec<usize, A>,
    matches: Vec<Match, A>,
    offsets: Vec<usize, A>,
    target: Vec<u8, A>,
    weights: Vec<f32, A>,
    scores: Vec<f32, A>,
}

impl<A: Allocator + Clone> Buffers<A> {
    /// Buffers for `verb` over `haystacks`, the variable-length ones sized by an empty round,
    /// which reports what the filled one will need without failing.
    fn sized(
        verb: Verb,
        engine: &mut SubstringsEngine,
        stream: &Stream,
        haystacks: StringsView<'_>,
        replacements: StringsView<'_>,
        allocator: A,
    ) -> Self {
        let mut buffers = Buffers {
            counts: filled(haystacks.len(), 0, allocator.clone()),
            matches: Vec::new_in(allocator.clone()),
            offsets: filled(haystacks.len() + 1, 0, allocator.clone()),
            target: Vec::new_in(allocator.clone()),
            weights: filled(engine.needles_count(), 1.0, allocator.clone()),
            scores: filled(haystacks.len(), 0.0, allocator),
        };
        if matches!(verb, Verb::Find | Verb::Replace) {
            expect_ok(buffers.round(verb, engine, stream, haystacks, replacements));
            let report = engine.report();
            buffers.matches.resize(report.matches_emitted, Match::default());
            buffers.target.resize(report.target_length, 0);
        }
        buffers
    }

    /// One round of `verb`, joined before it returns.
    fn round(
        &mut self,
        verb: Verb,
        engine: &mut SubstringsEngine,
        stream: &Stream,
        haystacks: StringsView<'_>,
        replacements: StringsView<'_>,
    ) -> Result<(), Status> {
        let parameters = Bm25Params::normalized(mean_length(haystacks));
        stream.scope(|scope| match verb {
            Verb::Count | Verb::Cover => engine.counts(scope, haystacks, &mut self.counts, 1),
            Verb::Find => engine.find(scope, haystacks, &mut self.matches, &mut self.offsets),
            Verb::Replace => {
                engine.replace(scope, haystacks, replacements, &mut self.target, &mut self.offsets)
            }
            Verb::Score => {
                let (weights, scores) = (&self.weights, &mut self.scores);
                engine.bm25_scores(scope, haystacks, None, &parameters, weights, scores, 1)
            }
        })
    }

    /// Whether a device round wrote what the host engine did, scores to a float tolerance.
    fn agrees<B: Allocator>(&self, host: &Buffers<B>, verb: Verb) -> bool {
        match verb {
            Verb::Count | Verb::Cover => self.counts[..] == host.counts[..],
            Verb::Find => self.offsets[..] == host.offsets[..],
            Verb::Replace => self.offsets[..] == host.offsets[..] && self.target[..] == host.target[..],
            Verb::Score => self.scores.iter().zip(host.scores.iter()).all(|(device, host)| {
                (device - host).abs() <= 1e-3 * host.abs().max(1.0)
            }),
        }
    }
}

/// A vector of `length` copies of `value` in `allocator`.
fn filled<T: Clone, A: Allocator>(length: usize, value: T, allocator: A) -> Vec<T, A> {
    let mut vector = Vec::new_in(allocator);
    vector.resize(length, value);
    vector
}

/// The mean document length BM25 normalizes against, in bytes.
fn mean_length(haystacks: StringsView<'_>) -> f32 {
    let bytes: usize = haystacks.iter().map(|haystack| haystack.len()).sum();
    bytes as f32 / haystacks.len().max(1) as f32
}

/// Times `verb` on every device: one host engine, one engine per pool thread over a contiguous
/// share of the documents, and one engine per GPU whose first round is checked against the host.
fn sweep(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    devices: &[Device],
    verb: Verb,
) {
    let cpu = Stream::default(Capabilities::cpu_enabled());
    let replacements = StringsView::from_slices(needles);
    for &device in devices {
        let name = verb.name(slice, device);
        match device {
            Device::OneCore => {
                if !settings.selects(&name) {
                    continue;
                }
                let haystacks = StringsView::from_slices(&corpus.documents);
                let mut engine = match compile(needles, verb, 0, &cpu) {
                    Ok(engine) => engine,
                    Err(error) => {
                        note_unavailable(&name, &error.to_string());
                        continue;
                    }
                };
                let mut buffers =
                    Buffers::sized(verb, &mut engine, &cpu, haystacks, replacements, Global);
                measure(settings, &name, corpus.spec(Threads::ONE), || {
                    expect_ok(buffers.round(verb, &mut engine, &cpu, haystacks, replacements));
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
                        let mut engine = expect_ok(compile(needles, verb, 0, &stream));
                        let buffers = Buffers::sized(
                            verb,
                            &mut engine,
                            &stream,
                            haystacks,
                            StringsView::from_slices(needles),
                            Global,
                        );
                        (stream, engine, buffers, share)
                    },
                    |(stream, engine, buffers, share), _| {
                        let haystacks = StringsView::from_slices(&corpus.documents[share.clone()]);
                        let replacements = StringsView::from_slices(needles);
                        expect_ok(buffers.round(verb, engine, stream, haystacks, replacements));
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
                let unified = UnifiedAllocator::new(&stream);
                let haystacks = expect_ok(Strings::copy_in(
                    StringsView::from_slices(&corpus.documents),
                    unified,
                    &cpu,
                ));
                let device_replacements = expect_ok(Strings::copy_in(replacements, unified, &cpu));
                let mut engine = match compile(needles, verb, corpus.documents.len(), &stream) {
                    Ok(engine) => engine,
                    Err(error) => {
                        note_unavailable(&name, &error.to_string());
                        continue;
                    }
                };
                let (haystacks, device_replacements) = (haystacks.view(), device_replacements.view());
                let mut buffers = Buffers::sized(
                    verb,
                    &mut engine,
                    &stream,
                    haystacks,
                    device_replacements,
                    unified,
                );
                expect_ok(buffers.round(verb, &mut engine, &stream, haystacks, device_replacements));

                let host_haystacks = StringsView::from_slices(&corpus.documents);
                let mut host = expect_ok(compile(needles, verb, 0, &cpu));
                let mut expected =
                    Buffers::sized(verb, &mut host, &cpu, host_haystacks, replacements, Global);
                expect_ok(expected.round(verb, &mut host, &cpu, host_haystacks, replacements));
                assert!(
                    buffers.agrees(&expected, verb),
                    "{name} disagrees with the host engine on its first round"
                );

                measure(settings, &name, corpus.spec(Threads::ONE), || {
                    expect_ok(buffers.round(verb, &mut engine, &stream, haystacks, device_replacements));
                });
            }
        }
    }
}

/// Builds a single-threaded rival and times `walk` over it, or notes the row when it does not build.
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

/// Counts every overlapping match of every needle, the cheapest walk each engine offers.
fn bench_counting(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    devices: &[Device],
) {
    sweep(settings, corpus, needles, slice, devices, Verb::Count);

    for (kind_name, kind) in [
        ("DFA", AhoCorasickKind::DFA),
        ("ContiguousNFA", AhoCorasickKind::ContiguousNFA),
    ] {
        measure_rival(
            settings,
            corpus,
            &format!("count-{slice}/aho_corasick::find_overlapping_iter<{kind_name}>"),
            || {
                AhoCorasick::builder()
                    .match_kind(MatchKind::Standard)
                    .kind(Some(kind))
                    .build(needles)
            },
            |automaton| {
                let total: usize = corpus
                    .documents
                    .iter()
                    .map(|document| automaton.find_overlapping_iter(document).count())
                    .sum();
                black_box(total);
            },
        );
    }

    measure_rival(
        settings,
        corpus,
        &format!("count-{slice}/daachorse::find_overlapping_iter"),
        || DoubleArrayAhoCorasick::<u32>::new(needles),
        |automaton| {
            let total: usize = corpus
                .documents
                .iter()
                .map(|document| automaton.find_overlapping_iter(document).count())
                .sum();
            black_box(total);
        },
    );
}

/// Counts a leftmost cover, the walk a rewrite needs and the one every engine spells differently.
fn bench_cover(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    devices: &[Device],
) {
    sweep(settings, corpus, needles, slice, devices, Verb::Cover);

    measure_rival(
        settings,
        corpus,
        &format!("cover-{slice}/aho_corasick::find_iter<LeftmostLongest>"),
        || {
            AhoCorasick::builder()
                .match_kind(MatchKind::LeftmostLongest)
                .build(needles)
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

    measure_rival(
        settings,
        corpus,
        &format!("cover-{slice}/daachorse::leftmost_find_iter"),
        || {
            DoubleArrayAhoCorasickBuilder::new()
                .match_kind(DoubleArrayMatchKind::LeftmostLongest)
                .build(needles)
        },
        |automaton: &DoubleArrayAhoCorasick<u32>| {
            let total: usize = corpus
                .documents
                .iter()
                .map(|document| automaton.leftmost_find_iter(document).count())
                .sum();
            black_box(total);
        },
    );

    // An alternation of escaped literals resolves leftmost-first, so it sits beside the covers.
    measure_rival(
        settings,
        corpus,
        &format!("cover-{slice}/regex::find_iter"),
        || {
            let alternation: Vec<String> = needles
                .iter()
                .map(|needle| regex::escape(&String::from_utf8_lossy(needle)))
                .collect();
            RegexBuilder::new(&alternation.join("|"))
                .size_limit(1 << 30)
                .build()
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
}

/// Materializes every match rather than tallying it, which is where the output bandwidth shows.
fn bench_locating(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    devices: &[Device],
) {
    sweep(settings, corpus, needles, slice, devices, Verb::Find);

    let mut oracle_matches: Vec<(usize, usize, usize)> = Vec::new();
    measure_rival(
        settings,
        corpus,
        &format!("find-{slice}/aho_corasick::find_overlapping_iter"),
        || {
            AhoCorasick::builder()
                .match_kind(MatchKind::Standard)
                .build(needles)
        },
        |automaton| {
            oracle_matches.clear();
            for (document_index, document) in corpus.documents.iter().enumerate() {
                for one_match in automaton.find_overlapping_iter(document) {
                    oracle_matches.push((document_index, one_match.start(), one_match.len()));
                }
            }
            black_box(&oracle_matches);
        },
    );
}

/// Substitutes every match under a cover, the one verb whose output is another corpus. Replacing
/// every needle with itself keeps the output the size of the input, so the row measures the
/// rewrite rather than an expansion factor.
fn bench_rewriting(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    devices: &[Device],
) {
    sweep(settings, corpus, needles, slice, devices, Verb::Replace);

    measure_rival(
        settings,
        corpus,
        &format!("replace-{slice}/aho_corasick::replace_all_bytes"),
        || {
            AhoCorasick::builder()
                .match_kind(MatchKind::LeftmostLongest)
                .build(needles)
        },
        |automaton| {
            let written: usize = corpus
                .documents
                .iter()
                .map(|document| automaton.replace_all_bytes(document, needles).len())
                .sum();
            black_box(written);
        },
    );
}

/// The index-then-query rival. The build tokenizes the corpus and never sees the dictionary, so it
/// is the same work for every slice and only the first slice measures it; the query is the slice
/// itself, so it widens with the dictionary the way the scan does.
fn bench_scoring_index(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: VocabularySlice,
) {
    use bm25::{EmbedderBuilder, Scorer};

    let build_name = format!("bm25-{}/bm25::Scorer::build", slice.name());
    let query_name = format!("bm25-{}/bm25::Scorer::matches", slice.name());
    let measures_build = slice == VOCABULARY_SLICES[0] && settings.selects(&build_name);
    if !measures_build && !settings.selects(&query_name) {
        return;
    }

    let documents: Vec<String> = corpus
        .documents
        .iter()
        .map(|document| String::from_utf8_lossy(document).into_owned())
        .collect();
    let mean_document_length = corpus.bytes.0 as f32 / corpus.documents.len().max(1) as f32;
    let embedder = EmbedderBuilder::<u32>::with_avgdl(mean_document_length).build();
    let build = || {
        let mut scorer = Scorer::<usize>::new();
        for (index, document) in documents.iter().enumerate() {
            scorer.upsert(&index, embedder.embed(document));
        }
        scorer
    };

    if measures_build {
        measure(settings, &build_name, corpus.spec(Threads::ONE), || {
            black_box(build());
        });
    }

    let scorer = build();
    let query: Vec<String> = needles
        .iter()
        .map(|needle| String::from_utf8_lossy(needle).into_owned())
        .collect();
    let query_embedding = embedder.embed(&query.join(" "));
    measure(settings, &query_name, corpus.spec(Threads::ONE), || {
        black_box(scorer.matches(&query_embedding));
    });
}

fn main() -> ExitCode {
    install_panic_hook();
    print_machine();
    let settings = Settings::read("substrings");
    settings.print();

    // Leaked so the vocabulary and the documents borrow for the whole run.
    let tape: &'static _ = Box::leak(Box::new(resolve_dataset(&settings).unwrap_nice()));
    log_timing_overhead(&settings);

    let documents: Vec<&'static [u8]> = tape.iter().collect();
    let vocabulary = build_vocabulary(&documents);
    if vocabulary.is_empty() {
        eprintln!("The working set yielded no vocabulary; is the dataset path right?");
        return ExitCode::FAILURE;
    }
    let corpus = Corpus {
        bytes: Bytes(documents.iter().map(|document| document.len() as u64).sum()),
        documents,
    };
    let devices = selected_devices(&settings);

    println!("Benchmark configuration:");
    println!(
        "- Vocabulary: {} terms after dropping hapax and the top one percent",
        vocabulary.len()
    );
    println!(
        "- Corpus: {} documents, {} bytes",
        corpus.documents.len(),
        corpus.bytes.0
    );
    for device in &devices {
        println!("- Device: {device}");
    }

    for slice in VOCABULARY_SLICES {
        let needles = needle_slice_of(&vocabulary, slice);
        let needle_bytes: usize = needles.iter().map(|needle| needle.len()).sum();
        println!(
            "\n# {} - {} needles, {} bytes",
            slice.name(),
            needles.len(),
            needle_bytes
        );

        bench_counting(&settings, &corpus, &needles, slice.name(), &devices);
        bench_cover(&settings, &corpus, &needles, slice.name(), &devices);
        bench_locating(&settings, &corpus, &needles, slice.name(), &devices);
        bench_rewriting(&settings, &corpus, &needles, slice.name(), &devices);
        sweep(&settings, &corpus, &needles, slice.name(), &devices, Verb::Score);
        bench_scoring_index(&settings, &corpus, &needles, slice);
    }

    finish()
}
