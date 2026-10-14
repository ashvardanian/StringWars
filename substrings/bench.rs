#![doc = r#"# StringWars: Substrings

Multi-pattern search benchmarks: one dictionary of needles matched against every document in one pass.
Every engine compiles its needle set once and reuses it, so the dictionary is paid for outside the
measured loop.

- StringZilla `szs::Substrings` on one core, every core, and each selected GPU
- `aho-corasick`, the reference Rust implementation, as a dense DFA and as a contiguous NFA
- `daachorse`, a double-array Aho-Corasick
- `regex` over an alternation of literals, the baseline most people actually write
- `bm25`, an index-then-query scorer, for the scoring group only

Every rival is single-threaded and CPU-only, so the head-to-head row is `<1cpu>`, and the rivals run
right after it, before any thread pool exists. Scoring is scan versus index: `bm25` tokenizes the
corpus into an inverted index once and answers many queries, while `szs::Substrings` scans raw bytes
against a fixed dictionary, so build and query are separate rows. Uncased matching is left out:
`aho-corasick` folds ASCII only, while `szs::Substrings` folds the whole Unicode table.

The dictionary comes from the working set itself: words of 3 to 32 bytes under the normative `words`
rule, every term occurring once and the most frequent one percent dropped, then sliced by frequency.

- `STRINGWARS_THREADS` overrides the multi-core scope width.
- `STRINGWARS_DEVICES` names the GPUs that carry rows, in a `cuda` build.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_substrings --bench bench_substrings
```
"#]
use std::collections::HashMap;
use std::fmt;
use std::hint::black_box;
use std::process::ExitCode;

use aho_corasick::{AhoCorasick, AhoCorasickKind, MatchKind};
use daachorse::{
    DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind as DoubleArrayMatchKind,
};
use regex::bytes::RegexBuilder;
use stringtape::BytesTape;
use stringzilla::szs::{
    AnyBytesTape, Bm25Params, CaseSensitivity, DeviceScope, OverlapPolicy, Substrings,
    SubstringsMatch, UnifiedAlloc,
};

use stringwars::{
    expect_ok, finish, gpu_ordinals, gpu_require, install_panic_hook, log_timing_overhead, measure,
    note_unavailable, print_machine, resolve_dataset, token_ranges, Bytes, MeasureSpec, ResultExt,
    Settings, Threads, Tokenization, Unit, WorkUnits,
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

/// Where one row runs. Single core comes first, so the single-threaded rivals measured right after
/// it never share the machine with a thread pool.
#[derive(Clone, Copy)]
enum Device {
    OneCore,
    AllCores(Threads),
    Gpu(i32),
}

impl Device {
    fn concurrency(self) -> Threads {
        match self {
            Self::AllCores(threads) => threads,
            _ => Threads::ONE,
        }
    }

    /// Opens this row's scope. It lives only as long as the row: an idle ForkUnion pool spin-waits,
    /// so a multi-core scope alive during a single-threaded row would consume the very cores that
    /// row is measured on.
    fn open(self) -> DeviceScope {
        match self {
            Self::OneCore => DeviceScope::cpu_cores(1),
            Self::AllCores(threads) => DeviceScope::cpu_cores(threads.0.get()),
            Self::Gpu(ordinal) => DeviceScope::gpu_device(ordinal as usize),
        }
        .unwrap_or_else(|error| panic!("Failed to open the <{}> device scope: {error}", self))
    }
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
        if !gpu_require(settings, ordinal) {
            continue;
        }
        match DeviceScope::gpu_device(ordinal as usize) {
            Ok(_) => devices.push(Device::Gpu(ordinal)),
            Err(error) => assert!(
                settings.devices.is_none(),
                "Requested CUDA device cuda:{ordinal} is unavailable: {error}"
            ),
        }
    }
    devices
}

/// Calls `body` once per device with a scope opened for that call alone.
fn for_each_device(
    settings: &Settings,
    devices: &[Device],
    mut body: impl FnMut(Device, &DeviceScope),
) {
    for &device in devices {
        if let Device::Gpu(ordinal) = device {
            assert!(
                gpu_require(settings, ordinal),
                "CUDA device cuda:{ordinal} is unavailable"
            );
        }
        body(device, &device.open());
    }
}

/// Every engine walks the same documents, so one pass over all of them is every row's work.
struct Corpus {
    tape: BytesTape<u64, UnifiedAlloc>,
    documents: Vec<&'static [u8]>,
    bytes: Bytes,
}

impl Corpus {
    fn haystacks(&self) -> AnyBytesTape<'_> {
        AnyBytesTape::View64(self.tape.view())
    }

    fn spec(&self, concurrency: Threads) -> MeasureSpec {
        MeasureSpec::new(
            Unit::Bytes,
            WorkUnits::new(self.documents.len() as u64, self.bytes),
        )
        .with_concurrency(concurrency)
    }
}

/// Compiles `needles` on `scope` and times `walk` over it, or notes the row when the engine declines.
fn measure_engine(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    name: &str,
    device: Device,
    scope: &DeviceScope,
    mut walk: impl FnMut(&Substrings),
) {
    if !settings.selects(name) {
        return;
    }
    match Substrings::new(scope, needles, CaseSensitivity::Cased) {
        Ok(engine) => {
            measure(settings, name, corpus.spec(device.concurrency()), || {
                walk(&engine)
            });
        }
        Err(error) => note_unavailable(name, &error.to_string()),
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
    let haystacks = corpus.haystacks();
    let mut counts = vec![0usize; corpus.documents.len()];
    for_each_device(settings, devices, |device, scope| {
        let name = format!("count-{slice}/stringzillas::Substrings<{device}>");
        measure_engine(settings, corpus, needles, &name, device, scope, |engine| {
            black_box(expect_ok(engine.count_into(
                scope,
                &haystacks,
                OverlapPolicy::Overlapping,
                &mut counts,
            )));
        });
        if !matches!(device, Device::OneCore) {
            return;
        }

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
    });
}

/// Counts a leftmost cover, the walk a rewrite needs and the one every engine spells differently.
fn bench_cover(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    devices: &[Device],
) {
    let haystacks = corpus.haystacks();
    let mut counts = vec![0usize; corpus.documents.len()];
    for_each_device(settings, devices, |device, scope| {
        let name = format!("cover-{slice}/stringzillas::Substrings<{device}>");
        measure_engine(settings, corpus, needles, &name, device, scope, |engine| {
            black_box(expect_ok(engine.count_into(
                scope,
                &haystacks,
                OverlapPolicy::LeftmostLongest,
                &mut counts,
            )));
        });
        if !matches!(device, Device::OneCore) {
            return;
        }

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
    });
}

/// Materializes every match rather than tallying it, which is where the output bandwidth shows.
fn bench_locating(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    devices: &[Device],
) {
    let haystacks = corpus.haystacks();

    // The output is sized once, outside every measured loop, as a pipeline would size it.
    let matches_total = {
        let scope = Device::OneCore.open();
        let mut counts = vec![0usize; corpus.documents.len()];
        Substrings::new(&scope, needles, CaseSensitivity::Cased)
            .and_then(|probe| {
                probe.count_into(&scope, &haystacks, OverlapPolicy::Overlapping, &mut counts)
            })
            .expect_display("Failed to size the located matches")
    };
    let mut matches = vec![SubstringsMatch::default(); matches_total];
    let mut oracle_matches: Vec<(usize, usize, usize)> = Vec::with_capacity(matches_total);

    for_each_device(settings, devices, |device, scope| {
        let name = format!("find-{slice}/stringzillas::Substrings<{device}>");
        measure_engine(settings, corpus, needles, &name, device, scope, |engine| {
            black_box(expect_ok(engine.find_into(
                scope,
                &haystacks,
                OverlapPolicy::Overlapping,
                &mut matches,
            )));
        });
        if !matches!(device, Device::OneCore) {
            return;
        }

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
    });
}

/// Substitutes every match under a cover, the one verb whose output is another corpus.
fn bench_rewriting(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    devices: &[Device],
) {
    let haystacks = corpus.haystacks();
    // Replacing every needle with itself keeps the output the size of the input, so the row measures
    // the rewrite rather than an expansion factor.
    let replacements: Vec<&[u8]> = needles.to_vec();

    let bound = {
        let scope = Device::OneCore.open();
        Substrings::new(&scope, needles, CaseSensitivity::Cased)
            .and_then(|probe| probe.replace_bound(&replacements, corpus.bytes.0 as usize))
            .expect_display("Failed to bound the rewritten corpus")
    };
    let mut output_data = vec![0u8; bound];
    let mut output_offsets = vec![0u64; corpus.documents.len() + 1];

    for_each_device(settings, devices, |device, scope| {
        let name = format!("replace-{slice}/stringzillas::Substrings<{device}>");
        measure_engine(settings, corpus, needles, &name, device, scope, |engine| {
            black_box(expect_ok(engine.replace_into(
                scope,
                &haystacks,
                OverlapPolicy::LeftmostLongest,
                &replacements,
                &mut output_data,
                &mut output_offsets,
            )));
        });
        if !matches!(device, Device::OneCore) {
            return;
        }

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
                    .map(|document| automaton.replace_all_bytes(document, &replacements).len())
                    .sum();
                black_box(written);
            },
        );
    });
}

/// Scores every document against the dictionary, which is the query.
fn bench_scoring(
    settings: &Settings,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    devices: &[Device],
) {
    let haystacks = corpus.haystacks();
    let weights = vec![1.0f32; needles.len()];
    let mut scores = vec![0.0f32; corpus.documents.len()];
    let mean_document_length = corpus.bytes.0 as f32 / corpus.documents.len().max(1) as f32;
    let parameters = Bm25Params::normalized(mean_document_length);
    for_each_device(settings, devices, |device, scope| {
        let name = format!("bm25-{slice}/stringzillas::Substrings<{device}>");
        measure_engine(settings, corpus, needles, &name, device, scope, |engine| {
            expect_ok(engine.score_bm25_into(
                scope,
                &haystacks,
                &weights,
                None,
                parameters,
                &mut scores,
            ));
            black_box(&scores);
        });
    });
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

    // One unified-memory tape serves every scope, so the GPU rows need no copy of their own.
    let mut unified_tape: BytesTape<u64, UnifiedAlloc> = BytesTape::new_in(UnifiedAlloc);
    unified_tape
        .extend(documents.iter().copied())
        .expect("Failed to build the unified-memory corpus tape");
    let corpus = Corpus {
        tape: unified_tape,
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
        bench_scoring(&settings, &corpus, &needles, slice.name(), &devices);
        bench_scoring_index(&settings, &corpus, &needles, slice);
    }

    finish()
}
