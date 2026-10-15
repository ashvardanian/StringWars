#![doc = r#"# StringWars: Fingerprints

Exact n-gram window overlap benchmarks: how many of one text's byte windows the other spells.

- `STRINGWARS_BATCH_PER_CORE` sets how many documents the overlap rows take as queries.
- `STRINGWARS_THREADS` sets how many overlap engines the multi-core row runs, one per pool thread.
- `STRINGWARS_DEVICES` names the GPUs that carry rows, in a `cuda` build.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_fingerprints --bench bench_fingerprints
```
"#]
#![allow(clippy::needless_range_loop)]

use std::collections::HashSet;
use std::process::ExitCode;

use stringzilla::sz::{
    Capabilities, OverlapEngine, Stream, Strings, StringsView, UnifiedAllocator,
};

use stringwars::{
    batch_size, expect_ok, finish, gpu_ordinals, gpu_stream, install_panic_hook,
    log_timing_overhead, measure, measure_per_thread, note_unavailable, print_machine,
    resolve_dataset, Bytes, MeasureSpec, ResultExt, Settings, Unit, WorkUnits,
};

const NGRAM_WIDTHS: [usize; 4] = [5, 9, 17, 33];

/// The candidates one of `threads` engines scores, contiguous so its block of scores is too.
fn share(count: usize, thread: usize, threads: usize) -> std::ops::Range<usize> {
    let chunk = count.div_ceil(threads);
    let first = (thread * chunk).min(count);
    first..(first + chunk).min(count)
}

/// Scores the first `batch_per_core` documents as queries against every document as a candidate,
/// at every n-gram width. One pass streams every candidate past every query, so the byte rate is
/// candidate bytes per query batch.
fn bench_overlap(settings: &Settings, documents: &[&[u8]], total_bytes: Bytes) {
    let queries = &documents[..batch_size(settings, 1).min(documents.len())];
    let widths = NGRAM_WIDTHS.len();
    let work = WorkUnits::new((queries.len() * documents.len()) as u64, total_bytes);
    let call = "overlap/stringzilla::OverlapEngine::scores";
    println!("\n# overlap - {} queries, {} candidates", queries.len(), documents.len());

    // One engine, every candidate, on this core.
    let cpu = Stream::default(Capabilities::cpu_enabled());
    let name = format!("{call}<1cpu>");
    let candidate_stride = widths;
    let query_stride = documents.len() * widths;
    if settings.selects(&name) {
        let mut engine = expect_ok(OverlapEngine::new(queries, &NGRAM_WIDTHS, 0, &cpu));
        let mut scores = vec![0f32; queries.len() * query_stride];
        measure(settings, &name, MeasureSpec::new(Unit::Bytes, work), || {
            let candidates = StringsView::from_slices(documents);
            expect_ok(cpu.scope(|scope| {
                engine.scores(scope, candidates, &mut scores, query_stride, candidate_stride)
            }));
        });
    }

    // One engine per pool thread, each holding every query against its share of the candidates.
    let threads = settings.threads;
    let count = threads.0.get();
    measure_per_thread(
        settings,
        &format!("{call}<{threads}cpu>"),
        MeasureSpec::new(Unit::Bytes, work).with_concurrency(threads),
        |thread| {
            let candidates = share(documents.len(), thread, count);
            let stream = Stream::default(Capabilities::cpu_enabled());
            let engine = expect_ok(OverlapEngine::new(queries, &NGRAM_WIDTHS, 0, &stream));
            let scores = vec![0f32; queries.len() * candidates.len() * widths];
            (stream, engine, scores, candidates)
        },
        |(stream, engine, scores, candidates), _| {
            let query_stride = candidates.len() * widths;
            let view = StringsView::from_slices(&documents[candidates.clone()]);
            expect_ok(stream.scope(|scope| {
                engine.scores(scope, view, scores, query_stride.max(widths), widths)
            }));
        },
    );

    // One engine per GPU, its first round checked against the host engine.
    for ordinal in gpu_ordinals(settings) {
        let name = format!("{call}<cuda:{ordinal}>");
        if !settings.selects(&name) {
            continue;
        }
        let Some(stream) = gpu_stream(settings, ordinal) else {
            continue;
        };
        let mut engine =
            match OverlapEngine::new(queries, &NGRAM_WIDTHS, documents.len(), &stream) {
                Ok(engine) => engine,
                Err(error) => {
                    note_unavailable(&name, &error.to_string());
                    continue;
                }
            };
        let unified = UnifiedAllocator::new(&stream);
        let candidates = expect_ok(Strings::copy_in(
            StringsView::from_slices(documents),
            unified,
            &cpu,
        ));
        let mut scores = Vec::new_in(unified);
        scores.resize(queries.len() * query_stride, 0f32);
        expect_ok(stream.scope(|scope| {
            engine.scores(scope, candidates.view(), &mut scores, query_stride, candidate_stride)
        }));
        let mut host = expect_ok(OverlapEngine::new(queries, &NGRAM_WIDTHS, 0, &cpu));
        let mut expected = vec![0f32; queries.len() * query_stride];
        let host_candidates = StringsView::from_slices(documents);
        expect_ok(cpu.scope(|scope| {
            host.scores(scope, host_candidates, &mut expected, query_stride, candidate_stride)
        }));
        assert!(
            scores.iter().zip(expected.iter()).all(|(device, host)| (device - host).abs() <= 1e-4),
            "{name} disagrees with the host engine on its first round"
        );
        measure(settings, &name, MeasureSpec::new(Unit::Bytes, work), || {
            let view = candidates.view();
            expect_ok(stream.scope(|scope| {
                engine.scores(scope, view, &mut scores, query_stride, candidate_stride)
            }));
        });
    }

    // The same shares computed exactly with one set of distinct windows per query and width, the
    // way an index-free deduplication pass would.
    let name = "overlap/serial::window_overlap<1cpu>";
    if settings.selects(name) {
        let query_windows: Vec<Vec<HashSet<&[u8]>>> = queries
            .iter()
            .map(|query| {
                NGRAM_WIDTHS
                    .iter()
                    .map(|&width| query.windows(width).collect())
                    .collect()
            })
            .collect();
        let mut scores = vec![0f32; queries.len() * query_stride];
        measure(settings, name, MeasureSpec::new(Unit::Bytes, work), || {
            for (query_index, (query, windows_by_width)) in
                queries.iter().zip(&query_windows).enumerate()
            {
                for (candidate_index, candidate) in documents.iter().enumerate() {
                    for (width_index, &width) in NGRAM_WIDTHS.iter().enumerate() {
                        let distinct = &windows_by_width[width_index];
                        let windows = candidate.len().saturating_sub(width - 1);
                        let hits = candidate
                            .windows(width)
                            .filter(|window| distinct.contains(window))
                            .count();
                        // The engine divides by every window of the longer side, repeats included.
                        let total = query.len().saturating_sub(width - 1).max(windows).max(1);
                        scores[query_index * query_stride
                            + candidate_index * candidate_stride
                            + width_index] = hits as f32 / total as f32;
                    }
                }
            }
            std::hint::black_box(&scores);
        });
    }
}

fn main() -> ExitCode {
    install_panic_hook();
    print_machine();
    let settings = Settings::read("fingerprints");
    settings.print();
    println!("Window Overlap Benchmarks");
    println!("- stringzilla::OverlapEngine: exact n-gram window overlap");

    let tape = resolve_dataset(&settings).unwrap_nice();
    log_timing_overhead(&settings);
    let documents: Vec<&[u8]> = tape.iter().collect();
    let total_bytes = Bytes(documents.iter().map(|token| token.len() as u64).sum());

    bench_overlap(&settings, &documents, total_bytes);

    finish()
}
