#![doc = r#"# StringWars: Similarities

String-similarity benchmarks in CUPS: Levenshtein, Needleman-Wunsch, Smith-Waterman.

The engines evaluate a square `side × side` cross-product: the first `side` tokens
against the next `side` disjoint ones. The pairs per batch on each device are
`STRINGWARS_BATCH_PER_CORE * cores` and `side = round(sqrt(pairs_per_batch))`, so the
matrix holds about that many pairs.

- `STRINGWARS_THREADS` sets how many engines the multi-core rows run, one per pool thread,
  each holding its share of the queries.
- `STRINGWARS_DEVICES` names the GPUs that carry rows, in a `cuda` build.

StringZilla v6 scores unit-cost Levenshtein only, so the Needleman-Wunsch and Smith-Waterman
groups keep their `bio` baselines and note the StringZilla rows as unavailable.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_similarities --bench bench_similarities
```
"#]
#![allow(clippy::too_many_arguments)]
use core::convert::TryInto;
use std::process::ExitCode;

use stringtape::{BytesTape, BytesTapeView, CharsTapeView};

use bio::alignment::{distance as bio_distance, pairwise::Aligner};
use rapidfuzz::distance::levenshtein;
use stringzilla::sz::{
    Capabilities, LevenshteinEngine, LevenshteinSymbol, Stream, Strings, StringsView,
    UnifiedAllocator,
};

use stringwars::{
    batch_size, expect_ok, finish, gpu_batch_size, gpu_ordinals, gpu_stream, install_panic_hook,
    log_timing_overhead, measure, measure_per_thread, note_unavailable, print_machine,
    resolve_dataset, Bytes, MeasureSpec, ResultExt, Settings, Threads, Unit, WorkUnits,
};

/// Computes the square cross-product side for `pairs_per_batch` on one device and `token_count`
/// available tokens. A `side × side` cross-product holds ~`pairs_per_batch` pairs, clamped so
/// queries `[0, side)` and candidates `[side, 2×side)` are disjoint, i.e. `2 * side <= token_count`.
fn crossproduct_side(pairs_per_batch: usize, token_count: usize) -> usize {
    let target = ((pairs_per_batch as f64).sqrt().round() as usize).max(1);
    let max_side = token_count / 2;
    target.min(max_side).max(1)
}

/// Work of the cross-product of a `[0, side)` query slice and `[side, 2×side)` candidate slice of
/// a bytes view: `elements = sum_query_bytes * sum_candidate_bytes` cells over
/// `bytes = sum_query_bytes + sum_candidate_bytes`.
fn crossproduct_work_bytes(full_view: &BytesTapeView<u64>, side: usize) -> WorkUnits {
    let (sum_query, sum_candidate) = (0..side).fold((0u64, 0u64), |(query, candidate), index| {
        (
            query + full_view[index].len() as u64,
            candidate + full_view[side + index].len() as u64,
        )
    });
    WorkUnits::new(sum_query * sum_candidate, Bytes(sum_query + sum_candidate))
}

/// Work spanned by the diagonal `(query_i, candidate_i)` pairs of the `[0, side)` and
/// `[side, 2×side)` slices, which is what the one-pair-per-call baselines score:
/// `elements = sum(len_query * len_candidate)`, `bytes = sum(len_query + len_candidate)`.
fn diagonal_work_bytes(full_view: &BytesTapeView<u64>, side: usize) -> WorkUnits {
    (0..side).fold(WorkUnits::default(), |accumulated, index| {
        let query = &full_view[index];
        let candidate = &full_view[side + index];
        WorkUnits::new(
            accumulated.elements + (query.len() * candidate.len()) as u64,
            Bytes(accumulated.bytes.0 + (query.len() + candidate.len()) as u64),
        )
    })
}

/// Work of the cross-product of a `[0, side)` query slice and `[side, 2×side)` candidate slice of
/// a chars view: `elements = sum_query_chars * sum_candidate_chars` cells, over the UTF-8 bytes
/// of both slices for the secondary byte rate.
fn crossproduct_work_chars(full_view: &CharsTapeView<u64>, side: usize) -> WorkUnits {
    let mut sum_query_chars = 0u64;
    let mut sum_candidate_chars = 0u64;
    let mut sum_query_bytes = 0u64;
    let mut sum_candidate_bytes = 0u64;
    for index in 0..side {
        let query = &full_view[index];
        let candidate = &full_view[side + index];
        sum_query_chars += query.chars().count() as u64;
        sum_candidate_chars += candidate.chars().count() as u64;
        sum_query_bytes += query.len() as u64;
        sum_candidate_bytes += candidate.len() as u64;
    }
    WorkUnits::new(
        sum_query_chars * sum_candidate_chars,
        Bytes(sum_query_bytes + sum_candidate_bytes),
    )
}

/// The `[0, side)` queries and the `[side, 2×side)` candidates of one cross-product.
fn crossproduct_halves<'a>(
    full_view: &'a BytesTapeView<u64>,
    side: usize,
) -> (Vec<&'a [u8]>, Vec<&'a [u8]>) {
    (
        (0..side).map(|index| &full_view[index]).collect(),
        (side..2 * side).map(|index| &full_view[index]).collect(),
    )
}

/// The rows one of `threads` engines holds, contiguous so its block of the matrix is too.
fn share(rows: usize, thread: usize, threads: usize) -> std::ops::Range<usize> {
    let chunk = rows.div_ceil(threads);
    let first = (thread * chunk).min(rows);
    first..(first + chunk).min(rows)
}

fn bench_similarities(settings: &Settings) {
    let tape_bytes = resolve_dataset(settings).unwrap_nice();
    log_timing_overhead(settings);
    let tape = tape_bytes
        .as_chars()
        .expect("Dataset must be valid UTF-8 for similarities");

    if tape.len() < 2 {
        panic!("Dataset must contain at least two items for comparisons.");
    }

    // Core-aware batch sizing: each variant scales `STRINGWARS_BATCH_PER_CORE` by its own core count.
    // A CPU core is one core; a GPU streaming multiprocessor (SM) is one core.
    let threads = settings.threads;
    let batch_single_cpu = batch_size(settings, 1);
    let batch_multi_cpu = batch_size(settings, threads.0.get());

    let mut units_tape = BytesTape::<u64>::new();
    units_tape
        .extend(tape.iter().map(|string| string.as_bytes()))
        .expect("Failed to extend BytesTape");

    // Full zero-copy bytes view over every token (the cross-product slices disjoint halves of it).
    let tape_bytes_view = units_tape.view();
    // Full zero-copy chars view over every token (UTF-8 variant of the same tape).
    let chars_view: CharsTapeView<u64> = units_tape
        .view()
        .try_into()
        .expect("Failed to convert to CharsTapeView");

    let token_count = units_tape.len();
    let side_single_cpu = crossproduct_side(batch_single_cpu, token_count);
    let side_multi_cpu = crossproduct_side(batch_multi_cpu, token_count);

    println!("Benchmark configuration:");
    println!(
        "- Single-core batch: {} ({}×{} cross-product)",
        batch_single_cpu, side_single_cpu, side_single_cpu
    );
    println!(
        "- {}-core batch: {} ({}×{} cross-product)",
        threads, batch_multi_cpu, side_multi_cpu, side_multi_cpu
    );
    println!("- Tokens available: {}", token_count);

    // Each selected GPU and the side of its cross-product. A stream opens per row, so this only
    // checks that the device answers.
    let mut gpu_sides: Vec<(i32, usize)> = Vec::new();
    for ordinal in gpu_ordinals(settings) {
        if gpu_stream(settings, ordinal).is_none() {
            continue;
        }
        let side = crossproduct_side(gpu_batch_size(settings, ordinal), token_count);
        println!("- GPU device: cuda:{ordinal} with a {side}×{side} cross-product");
        gpu_sides.push((ordinal, side));
    }
    let side_gpu = gpu_sides.iter().map(|&(_, side)| side).max().unwrap_or(0);
    println!();

    println!("# uniform");
    perform_uniform_benchmarks(
        settings,
        &tape_bytes_view,
        &chars_view,
        &gpu_sides,
        threads,
        side_single_cpu,
        side_multi_cpu,
    );

    // NW/SW with linear (open=-2, extend=-2) then affine (open=-5, extend=-1) gap costs,
    // both under unary match=2 / mismatch=-1 scoring.
    for (group_name, open_cost, extend_cost) in [("linear", -2, -2), ("affine", -5, -1)] {
        println!("# {}", group_name);
        perform_score_benchmarks(
            settings,
            group_name,
            &tape_bytes_view,
            side_single_cpu,
            side_multi_cpu,
            side_gpu,
            open_cost,
            extend_cost,
        );
    }
}

/// Uniform cost benchmarks: Classic Levenshtein distance (match=0, mismatch=1, open=1, extend=1)
fn perform_uniform_benchmarks(
    settings: &Settings,
    tape_bytes_view: &BytesTapeView<u64>,
    chars_view: &CharsTapeView<u64>,
    gpu_sides: &[(i32, usize)],
    threads: Threads,
    side_single_cpu: usize,
    side_multi_cpu: usize,
) {
    // RapidFuzz baselines (no batching; scan one-by-one). One pair per call across the
    // query/candidate diagonal of the single-core cross-product. One pass scores every
    // baseline pair, so the pair mixture is identical in every sample rather than
    // depending on how far the deadline happened to reach.
    let baseline_side = side_single_cpu;
    let baseline_chars_work =
        (0..baseline_side).fold(WorkUnits::default(), |accumulated, index| {
            let query = &chars_view[index];
            let candidate = &chars_view[baseline_side + index];
            WorkUnits::new(
                accumulated.elements + (query.chars().count() * candidate.chars().count()) as u64,
                Bytes(accumulated.bytes.0 + (query.len() + candidate.len()) as u64),
            )
        });
    let baseline_pass_work = diagonal_work_bytes(tape_bytes_view, baseline_side);
    measure(
        settings,
        "uniform/rapidfuzz::levenshtein<Bytes,1cpu>",
        MeasureSpec::new(Unit::Cups, baseline_pass_work),
        || {
            for index in 0..baseline_side {
                let a_bytes = &tape_bytes_view[index];
                let b_bytes = &tape_bytes_view[baseline_side + index];
                std::hint::black_box(levenshtein::distance(
                    a_bytes.iter().copied(),
                    b_bytes.iter().copied(),
                ));
            }
        },
    );

    measure(
        settings,
        "uniform/rapidfuzz::levenshtein<Chars,1cpu>",
        MeasureSpec::new(Unit::Cups, baseline_chars_work),
        || {
            for index in 0..baseline_side {
                let a_str = &chars_view[index];
                let b_str = &chars_view[baseline_side + index];
                std::hint::black_box(levenshtein::distance(a_str.chars(), b_str.chars()));
            }
        },
    );

    measure(
        settings,
        "uniform/bio::levenshtein<1cpu>",
        MeasureSpec::new(Unit::Cups, baseline_pass_work),
        || {
            for index in 0..baseline_side {
                let a_bytes = &tape_bytes_view[index];
                let b_bytes = &tape_bytes_view[baseline_side + index];
                std::hint::black_box(bio_distance::levenshtein(a_bytes, b_bytes));
            }
        },
    );

    // The most depended-upon edit-distance crate, checked against the engine on the same pairs.
    let name = "uniform/strsim::levenshtein<Chars,1cpu>";
    if settings.selects(name) {
        let cpu = Stream::default(Capabilities::cpu_enabled());
        let (queries, candidates) = crossproduct_halves(tape_bytes_view, baseline_side);
        let mut engine = expect_ok(LevenshteinEngine::new(
            &queries,
            LevenshteinSymbol::Runes,
            &cpu,
        ));
        let mut distances = vec![0usize; baseline_side * baseline_side];
        expect_ok(cpu.scope(|scope| {
            let candidates = StringsView::from_slices(&candidates);
            engine.distances(scope, candidates, &mut distances, baseline_side)
        }));
        let pair = |index: usize| (&chars_view[index], &chars_view[baseline_side + index]);
        assert!(
            (0..baseline_side).all(|index| {
                let (query, candidate) = pair(index);
                strsim::levenshtein(query, candidate) == distances[index * baseline_side + index]
            }),
            "{name} disagrees with StringZilla"
        );
        measure(
            settings,
            name,
            MeasureSpec::new(Unit::Cups, baseline_chars_work),
            || {
                for index in 0..baseline_side {
                    let (query, candidate) = pair(index);
                    std::hint::black_box(strsim::levenshtein(query, candidate));
                }
            },
        );
    }

    for (symbol, label) in [
        (LevenshteinSymbol::Bytes, "Bytes"),
        (LevenshteinSymbol::Runes, "Runes"),
    ] {
        let work = |side: usize| match symbol {
            LevenshteinSymbol::Runes => crossproduct_work_chars(chars_view, side),
            _ => crossproduct_work_bytes(tape_bytes_view, side),
        };
        let call = "uniform/stringzilla::LevenshteinEngine::distances";

        // One engine, every query, on this core.
        let cpu = Stream::default(Capabilities::cpu_enabled());
        let name = format!("{call}<{label},1cpu>");
        if settings.selects(&name) {
            let side = side_single_cpu;
            let (queries, candidates) = crossproduct_halves(tape_bytes_view, side);
            let mut engine = expect_ok(LevenshteinEngine::new(&queries, symbol, &cpu));
            let mut distances = vec![0usize; side * side];
            measure(settings, &name, MeasureSpec::new(Unit::Cups, work(side)), || {
                let candidates = StringsView::from_slices(&candidates);
                expect_ok(cpu.scope(|scope| engine.distances(scope, candidates, &mut distances, side)));
            });
        }

        // One engine per pool thread, each holding its share of the queries against every candidate.
        {
            let side = side_multi_cpu;
            let (queries, candidates) = crossproduct_halves(tape_bytes_view, side);
            let count = threads.0.get();
            measure_per_thread(
                settings,
                &format!("{call}<{label},{threads}cpu>"),
                MeasureSpec::new(Unit::Cups, work(side)).with_concurrency(threads),
                |thread| {
                    let rows = share(side, thread, count);
                    let stream = Stream::default(Capabilities::cpu_enabled());
                    let engine = (!rows.is_empty())
                        .then(|| expect_ok(LevenshteinEngine::new(&queries[rows.clone()], symbol, &stream)));
                    (stream, engine, vec![0usize; rows.len() * side])
                },
                |(stream, engine, distances), _| {
                    if let Some(engine) = engine {
                        let candidates = StringsView::from_slices(&candidates);
                        expect_ok(stream.scope(|scope| engine.distances(scope, candidates, distances, side)));
                    }
                },
            );
        }

        // One engine per GPU, its first round checked against the host engine.
        for &(ordinal, side) in gpu_sides {
            let name = format!("{call}<{label},cuda:{ordinal}>");
            if !settings.selects(&name) {
                continue;
            }
            let Some(stream) = gpu_stream(settings, ordinal) else {
                continue;
            };
            let (queries, candidates) = crossproduct_halves(tape_bytes_view, side);
            let mut engine = match LevenshteinEngine::new(&queries, symbol, &stream) {
                Ok(engine) => engine,
                Err(error) => {
                    note_unavailable(&name, &error.to_string());
                    continue;
                }
            };
            let unified = UnifiedAllocator::new(&stream);
            let device_candidates = expect_ok(Strings::copy_in(
                StringsView::from_slices(&candidates),
                unified,
                &cpu,
            ));
            let mut distances = Vec::new_in(unified);
            distances.resize(side * side, 0usize);
            let first_round = stream.scope(|scope| {
                engine.distances(scope, device_candidates.view(), &mut distances, side)
            });
            if let Err(error) = first_round {
                note_unavailable(&name, &error.to_string());
                continue;
            }
            let mut host = expect_ok(LevenshteinEngine::new(&queries, symbol, &cpu));
            let mut expected = vec![0usize; side * side];
            let host_candidates = StringsView::from_slices(&candidates);
            expect_ok(cpu.scope(|scope| host.distances(scope, host_candidates, &mut expected, side)));
            assert!(
                distances[..] == expected[..],
                "{name} disagrees with the host engine on its first round"
            );
            measure(settings, &name, MeasureSpec::new(Unit::Cups, work(side)), || {
                let candidates = device_candidates.view();
                expect_ok(stream.scope(|scope| engine.distances(scope, candidates, &mut distances, side)));
            });
        }
    }
}

/// Largest token byte length across the union of the query/candidate slices of every variant,
/// used to pre-size the `bio` aligner capacity.
fn max_token_len(
    tape_bytes_view: &BytesTapeView<u64>,
    side_single_cpu: usize,
    side_multi_cpu: usize,
    side_gpu: usize,
) -> usize {
    let widest_side = side_single_cpu.max(side_multi_cpu).max(side_gpu);
    (0..2 * widest_side)
        .map(|index| tape_bytes_view[index].len())
        .max()
        .unwrap_or(0)
        .max(1)
}

/// NW/SW score benchmarks for one gap-cost group. `linear` and `affine` differ only in the
/// `(open_cost, extend_cost)` pair and the label, so they share this body.
fn perform_score_benchmarks(
    settings: &Settings,
    group_name: &str,
    tape_bytes_view: &BytesTapeView<u64>,
    side_single_cpu: usize,
    side_multi_cpu: usize,
    side_gpu: usize,
    open_cost: i8,
    extend_cost: i8,
) {
    let max_len = max_token_len(tape_bytes_view, side_single_cpu, side_multi_cpu, side_gpu);
    let baseline_side = side_single_cpu;
    // One pass scores every baseline pair, so the pair mixture is identical in
    // every sample instead of depending on how far the deadline reached.
    let baseline_work = diagonal_work_bytes(tape_bytes_view, baseline_side);
    let unary = |a: u8, b: u8| if a == b { 2 } else { -1 };
    {
        let mut aligner =
            Aligner::with_capacity(max_len, max_len, open_cost as i32, extend_cost as i32, unary);
        measure(
            settings,
            &format!("{group_name}/bio::pairwise::global<1cpu>"),
            MeasureSpec::new(Unit::Cups, baseline_work),
            || {
                for index in 0..baseline_side {
                    let a_bytes = &tape_bytes_view[index];
                    let b_bytes = &tape_bytes_view[baseline_side + index];
                    std::hint::black_box(aligner.global(a_bytes, b_bytes).score);
                }
            },
        );
    }

    {
        let mut aligner =
            Aligner::with_capacity(max_len, max_len, open_cost as i32, extend_cost as i32, unary);
        measure(
            settings,
            &format!("{group_name}/bio::pairwise::local<1cpu>"),
            MeasureSpec::new(Unit::Cups, baseline_work),
            || {
                for index in 0..baseline_side {
                    let a_bytes = &tape_bytes_view[index];
                    let b_bytes = &tape_bytes_view[baseline_side + index];
                    std::hint::black_box(aligner.local(a_bytes, b_bytes).score);
                }
            },
        );
    }

    for engine in ["NeedlemanWunschScores", "SmithWatermanScores"] {
        let name = format!("{group_name}/stringzillas::{engine}");
        if settings.selects(&name) {
            note_unavailable(&name, "StringZilla v6 scores unit-cost Levenshtein only");
        }
    }
}

fn main() -> ExitCode {
    install_panic_hook();
    print_machine();
    let settings = Settings::read("similarities");
    settings.print();
    bench_similarities(&settings);

    finish()
}
