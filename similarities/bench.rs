#![doc = r#"# StringWars: Similarities

String-similarity benchmarks in CUPS: Levenshtein, Needleman-Wunsch, Smith-Waterman.

The engines evaluate a square `side x side` cross-product: the first `side` tokens
against the next `side` disjoint ones. The pairs per batch on each device are
`STRINGWARS_BATCH_PER_CORE * cores` and `side = round(sqrt(pairs_per_batch))`, so the
matrix holds about that many pairs.

- `STRINGWARS_THREADS` overrides the multi-core scope width.

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
use stringzilla::szs::{
    AnyBytesTape, AnyCharsTape, DeviceScope, LevenshteinDistances, LevenshteinDistancesUtf8,
    NeedlemanWunschScores, SmithWatermanScores, UnifiedAlloc, UnifiedMat,
};

use stringwars::{
    batch_size, finish, gpu_multiprocessor_count, install_panic_hook, log_timing_overhead, measure,
    note_unavailable, print_machine, resolve_dataset, Bytes, MeasureSpec, ResultExt, Settings,
    Threads, Unit, WorkUnits, FALLBACK_GPU_MULTIPROCESSORS,
};

/// Builds a substitution table for classic unary scoring: `match_cost` on the diagonal,
/// `mismatch_cost` everywhere else. Bytes are folded into 32 classes via `i % 32`, which keeps
/// the table compact; throughput (MCUPS) is invariant to the actual costs.
fn unary_class_costs(match_cost: i8, mismatch_cost: i8) -> ([u8; 256], [[i8; 32]; 32]) {
    let byte_to_class = core::array::from_fn(|byte_value| (byte_value % 32) as u8);
    let class_costs = core::array::from_fn(|row| {
        core::array::from_fn(|column| {
            if row == column {
                match_cost
            } else {
                mismatch_cost
            }
        })
    });
    (byte_to_class, class_costs)
}

/// Computes the square cross-product side for `pairs_per_batch` on one device and `token_count`
/// available tokens. A `side x side` cross-product holds ~`pairs_per_batch` pairs, clamped so
/// queries `[0, side)` and candidates `[side, 2*side)` are disjoint, i.e. `2 * side <= token_count`.
fn crossproduct_side(pairs_per_batch: usize, token_count: usize) -> usize {
    let target = ((pairs_per_batch as f64).sqrt().round() as usize).max(1);
    let max_side = token_count / 2;
    target.min(max_side).max(1)
}

/// Runs one `measure_throughput` block for a bytes-tape cross-product engine that writes `usize`
/// results (e.g. `LevenshteinDistances`). The disjoint query slice `[0, side)` and candidate
/// slice `[side, 2*side)` are rebuilt from `full_view` each iteration (views are not `Clone`),
/// wrapped as `AnyBytesTape::View64`, and written into the pre-allocated `matrix` via `compute`.
fn measure_crossproduct_bytes_usize(
    settings: &Settings,
    name: &str,
    full_view: &BytesTapeView<u64>,
    side: usize,
    spec: MeasureSpec,
    matrix: &mut UnifiedMat<usize>,
    mut compute: impl FnMut(AnyBytesTape<'_>, Option<AnyBytesTape<'_>>, &mut UnifiedMat<usize>),
) {
    measure(settings, name, spec, || {
        let query_view = full_view
            .subview(0, side)
            .expect("Failed to create query subview");
        let candidate_view = full_view
            .subview(side, 2 * side)
            .expect("Failed to create candidate subview");
        compute(
            AnyBytesTape::View64(query_view),
            Some(AnyBytesTape::View64(candidate_view)),
            matrix,
        );
        std::hint::black_box(&matrix);
    });
}

/// Runs one `measure_throughput` block for a chars-tape cross-product engine that writes `usize`
/// results (e.g. `LevenshteinDistancesUtf8`). The disjoint query slice `[0, side)` and candidate
/// slice `[side, 2*side)` are rebuilt from `full_view` each iteration (views are not `Clone`),
/// wrapped as `AnyCharsTape::View64`, and written into the pre-allocated `matrix` via `compute`.
fn measure_crossproduct_chars_usize(
    settings: &Settings,
    name: &str,
    full_view: &CharsTapeView<u64>,
    side: usize,
    spec: MeasureSpec,
    matrix: &mut UnifiedMat<usize>,
    mut compute: impl FnMut(AnyCharsTape<'_>, Option<AnyCharsTape<'_>>, &mut UnifiedMat<usize>),
) {
    measure(settings, name, spec, || {
        let query_view = full_view
            .subview(0, side)
            .expect("Failed to create query subview");
        let candidate_view = full_view
            .subview(side, 2 * side)
            .expect("Failed to create candidate subview");
        compute(
            AnyCharsTape::View64(query_view),
            Some(AnyCharsTape::View64(candidate_view)),
            matrix,
        );
        std::hint::black_box(&matrix);
    });
}

/// Runs one `measure_throughput` block for a bytes-tape cross-product engine that writes `isize`
/// results (e.g. `NeedlemanWunschScores`, `SmithWatermanScores`). The disjoint query slice
/// `[0, side)` and candidate slice `[side, 2*side)` are rebuilt from `full_view` each iteration
/// (views are not `Clone`), wrapped as `AnyBytesTape::View64`, and written into the pre-allocated
/// `matrix` via `compute`.
fn measure_crossproduct_bytes_isize(
    settings: &Settings,
    name: &str,
    full_view: &BytesTapeView<u64>,
    side: usize,
    spec: MeasureSpec,
    matrix: &mut UnifiedMat<isize>,
    mut compute: impl FnMut(AnyBytesTape<'_>, Option<AnyBytesTape<'_>>, &mut UnifiedMat<isize>),
) {
    measure(settings, name, spec, || {
        let query_view = full_view
            .subview(0, side)
            .expect("Failed to create query subview");
        let candidate_view = full_view
            .subview(side, 2 * side)
            .expect("Failed to create candidate subview");
        compute(
            AnyBytesTape::View64(query_view),
            Some(AnyBytesTape::View64(candidate_view)),
            matrix,
        );
        std::hint::black_box(&matrix);
    });
}

/// Work of the cross-product of a `[0, side)` query slice and `[side, 2*side)` candidate slice of
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
/// `[side, 2*side)` slices, which is what the one-pair-per-call baselines score:
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

/// Work of the cross-product of a `[0, side)` query slice and `[side, 2*side)` candidate slice of
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

/// Collects the `[0, side)` query slice of a chars view into an owned `Vec<&str>`.
fn chars_query_vec<'a>(full_view: &'a CharsTapeView<u64>, side: usize) -> Vec<&'a str> {
    (0..side).map(|index| &full_view[index]).collect()
}

/// Collects the `[side, 2*side)` candidate slice of a chars view into an owned `Vec<&str>`.
fn chars_candidate_vec<'a>(full_view: &'a CharsTapeView<u64>, side: usize) -> Vec<&'a str> {
    (0..side).map(|index| &full_view[side + index]).collect()
}

fn bench_similarities(settings: &Settings) {
    // Load dataset using unified loader
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
    let batch_gpu = batch_size(
        settings,
        gpu_multiprocessor_count(0).unwrap_or(FALLBACK_GPU_MULTIPROCESSORS),
    );

    let mut units_tape: BytesTape<u64, UnifiedAlloc> = BytesTape::new_in(UnifiedAlloc);
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
    let side_gpu = crossproduct_side(batch_gpu, token_count);

    // Log benchmark-specific configuration
    println!("Benchmark configuration:");
    println!(
        "- Single-core batch: {} ({}x{} cross-product)",
        batch_single_cpu, side_single_cpu, side_single_cpu
    );
    println!(
        "- {}-core batch: {} ({}x{} cross-product)",
        threads, batch_multi_cpu, side_multi_cpu, side_multi_cpu
    );
    println!(
        "- GPU batch: {} ({}x{} cross-product)",
        batch_gpu, side_gpu, side_gpu
    );
    println!("- Tokens available: {}", token_count);
    println!();

    // One thread pool per device width for the whole run: each `DeviceScope` spins up its own,
    // and the three benchmark groups used to build a fresh set apiece.
    let cpu_single = DeviceScope::cpu_cores(1).expect("Failed to create single-core device scope");
    let cpu_parallel =
        DeviceScope::cpu_cores(threads.0.get()).expect("Failed to create multi-core device scope");
    let maybe_gpu = DeviceScope::gpu_device(0).ok();

    // Uniform cost benchmarks (classic Levenshtein: match=0, mismatch=1, open=1, extend=1)
    println!("# uniform");
    perform_uniform_benchmarks(
        settings,
        &tape_bytes_view,
        &chars_view,
        &cpu_single,
        &cpu_parallel,
        maybe_gpu.as_ref(),
        threads,
        side_single_cpu,
        side_multi_cpu,
        side_gpu,
    );

    // NW/SW with linear (open=-2, extend=-2) then affine (open=-5, extend=-1) gap costs,
    // both under unary match=2 / mismatch=-1 scoring.
    for (group_name, open_cost, extend_cost) in [("linear", -2, -2), ("affine", -5, -1)] {
        println!("# {}", group_name);
        perform_score_benchmarks(
            settings,
            group_name,
            &tape_bytes_view,
            &cpu_single,
            &cpu_parallel,
            maybe_gpu.as_ref(),
            threads,
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
    cpu_single: &DeviceScope,
    cpu_parallel: &DeviceScope,
    maybe_gpu: Option<&DeviceScope>,
    threads: Threads,
    side_single_cpu: usize,
    side_multi_cpu: usize,
    side_gpu: usize,
) {
    let lev_single = LevenshteinDistances::new(cpu_single, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistances single");
    let lev_parallel = LevenshteinDistances::new(cpu_parallel, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistances parallel");
    let lev_utf8_single = LevenshteinDistancesUtf8::new(cpu_single, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistancesUtf8 single");
    let lev_utf8_parallel = LevenshteinDistancesUtf8::new(cpu_parallel, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistancesUtf8 parallel");
    let maybe_lev_gpu = maybe_gpu.and_then(|gpu| LevenshteinDistances::new(gpu, 0, 1, 1, 1).ok());
    // GPU UTF-8 Levenshtein: the engine may decline inputs beyond its supported length, returning an error we skip
    // on rather than aborting the suite.
    let maybe_lev_utf8_gpu =
        maybe_gpu.and_then(|gpu| LevenshteinDistancesUtf8::new(gpu, 0, 1, 1, 1).ok());

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
    {
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
    }

    {
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
    }

    {
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
    }

    // StringZilla byte-level Levenshtein distance (uniform costs: 0,1,1,1)
    {
        let work = crossproduct_work_bytes(tape_bytes_view, side_single_cpu);
        let mut matrix = UnifiedMat::<usize>::try_allocate(side_single_cpu, side_single_cpu)
            .expect("Failed to allocate LevenshteinDistances matrix (single)");
        measure_crossproduct_bytes_usize(
            settings,
            "uniform/stringzillas::LevenshteinDistances<1cpu>",
            tape_bytes_view,
            side_single_cpu,
            MeasureSpec::new(Unit::Cups, work),
            &mut matrix,
            |queries, candidates, matrix| {
                lev_single
                    .compute_into(cpu_single, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute LevenshteinDistances on CPU (single-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    {
        let work = crossproduct_work_bytes(tape_bytes_view, side_multi_cpu);
        let mut matrix = UnifiedMat::<usize>::try_allocate(side_multi_cpu, side_multi_cpu)
            .expect("Failed to allocate LevenshteinDistances matrix (parallel)");
        measure_crossproduct_bytes_usize(
            settings,
            &format!("uniform/stringzillas::LevenshteinDistances<{threads}cpu>"),
            tape_bytes_view,
            side_multi_cpu,
            MeasureSpec::new(Unit::Cups, work).with_concurrency(threads),
            &mut matrix,
            |queries, candidates, matrix| {
                lev_parallel
                    .compute_into(cpu_parallel, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute LevenshteinDistances on CPU (multi-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    // StringZilla UTF-8 Levenshtein Distance (uniform costs: 0,1,1,1)
    {
        let work = crossproduct_work_chars(chars_view, side_single_cpu);
        let mut matrix = UnifiedMat::<usize>::try_allocate(side_single_cpu, side_single_cpu)
            .expect("Failed to allocate LevenshteinDistancesUtf8 matrix (single)");
        measure_crossproduct_chars_usize(
            settings,
            "uniform/stringzillas::LevenshteinDistancesUtf8<1cpu>",
            chars_view,
            side_single_cpu,
            MeasureSpec::new(Unit::Cups, work),
            &mut matrix,
            |queries, candidates, matrix| {
                lev_utf8_single
                    .compute_into(cpu_single, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute LevenshteinDistancesUtf8 on CPU (single-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    {
        let work = crossproduct_work_chars(chars_view, side_multi_cpu);
        let mut matrix = UnifiedMat::<usize>::try_allocate(side_multi_cpu, side_multi_cpu)
            .expect("Failed to allocate LevenshteinDistancesUtf8 matrix (parallel)");
        measure_crossproduct_chars_usize(
            settings,
            &format!("uniform/stringzillas::LevenshteinDistancesUtf8<{threads}cpu>"),
            chars_view,
            side_multi_cpu,
            MeasureSpec::new(Unit::Cups, work).with_concurrency(threads),
            &mut matrix,
            |queries, candidates, matrix| {
                lev_utf8_parallel
                    .compute_into(cpu_parallel, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute LevenshteinDistancesUtf8 on CPU (multi-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    if let (Some(gpu), Some(engine)) = (maybe_gpu, maybe_lev_gpu.as_ref()) {
        let work = crossproduct_work_bytes(tape_bytes_view, side_gpu);
        let mut matrix = UnifiedMat::<usize>::try_allocate(side_gpu, side_gpu)
            .expect("Failed to allocate LevenshteinDistances matrix (GPU)");
        measure_crossproduct_bytes_usize(
            settings,
            "uniform/stringzillas::LevenshteinDistances<1gpu>",
            tape_bytes_view,
            side_gpu,
            MeasureSpec::new(Unit::Cups, work),
            &mut matrix,
            |queries, candidates, matrix| {
                engine
                    .compute_into(gpu, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
            },
        );
    }

    // GPU UTF-8 Levenshtein: the engine may decline inputs beyond its supported length, returning an error we skip on.
    if let (Some(gpu), Some(engine)) = (maybe_gpu, maybe_lev_utf8_gpu.as_ref()) {
        let work = crossproduct_work_chars(chars_view, side_gpu);
        let queries = chars_query_vec(chars_view, side_gpu);
        let candidates = chars_candidate_vec(chars_view, side_gpu);
        match engine.compute(gpu, &queries, &candidates) {
            Ok(mut matrix) => measure_crossproduct_chars_usize(
                settings,
                "uniform/stringzillas::LevenshteinDistancesUtf8<1gpu>",
                chars_view,
                side_gpu,
                MeasureSpec::new(Unit::Cups, work),
                &mut matrix,
                |queries, candidates, matrix| {
                    engine
                        .compute_into(gpu, queries, candidates, matrix)
                        .unwrap_or_else(|error| {
                            panic!("Failed to compute UTF-8 Levenshtein on GPU: {}", error)
                        });
                },
            ),
            Err(error) => note_unavailable(
                "uniform/stringzillas::LevenshteinDistancesUtf8<1gpu>",
                &error.to_string(),
            ),
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
    cpu_single: &DeviceScope,
    cpu_parallel: &DeviceScope,
    maybe_gpu: Option<&DeviceScope>,
    threads: Threads,
    side_single_cpu: usize,
    side_multi_cpu: usize,
    side_gpu: usize,
    open_cost: i8,
    extend_cost: i8,
) {
    // Unary scoring (match=2, mismatch=-1) folded into the 32-class table.
    let (byte_to_class, class_costs) = unary_class_costs(2, -1);

    let nw_single = NeedlemanWunschScores::new(
        cpu_single,
        &byte_to_class,
        &class_costs,
        open_cost,
        extend_cost,
    )
    .expect("Failed to create NW single");
    let nw_parallel = NeedlemanWunschScores::new(
        cpu_parallel,
        &byte_to_class,
        &class_costs,
        open_cost,
        extend_cost,
    )
    .expect("Failed to create NW parallel");
    let sw_single = SmithWatermanScores::new(
        cpu_single,
        &byte_to_class,
        &class_costs,
        open_cost,
        extend_cost,
    )
    .expect("Failed to create SW single");
    let sw_parallel = SmithWatermanScores::new(
        cpu_parallel,
        &byte_to_class,
        &class_costs,
        open_cost,
        extend_cost,
    )
    .expect("Failed to create SW parallel");
    let maybe_nw_gpu = maybe_gpu.and_then(|gpu| {
        NeedlemanWunschScores::new(gpu, &byte_to_class, &class_costs, open_cost, extend_cost).ok()
    });
    let maybe_sw_gpu = maybe_gpu.and_then(|gpu| {
        SmithWatermanScores::new(gpu, &byte_to_class, &class_costs, open_cost, extend_cost).ok()
    });

    let max_len = max_token_len(tape_bytes_view, side_single_cpu, side_multi_cpu, side_gpu);
    let baseline_side = side_single_cpu;
    // One pass scores every baseline pair, so the pair mixture is identical in
    // every sample instead of depending on how far the deadline reached.
    let baseline_work = diagonal_work_bytes(tape_bytes_view, baseline_side);
    {
        let mut aligner = Aligner::with_capacity(
            max_len,
            max_len,
            open_cost as i32,
            extend_cost as i32,
            |a: u8, b: u8| {
                if a == b {
                    2
                } else {
                    -1
                }
            },
        );
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
        let mut aligner = Aligner::with_capacity(
            max_len,
            max_len,
            open_cost as i32,
            extend_cost as i32,
            |a: u8, b: u8| {
                if a == b {
                    2
                } else {
                    -1
                }
            },
        );
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

    // Needleman-Wunsch (Global alignment)
    {
        let work = crossproduct_work_bytes(tape_bytes_view, side_single_cpu);
        let mut matrix = UnifiedMat::<isize>::try_allocate(side_single_cpu, side_single_cpu)
            .expect("Failed to allocate NeedlemanWunschScores matrix (single)");
        measure_crossproduct_bytes_isize(
            settings,
            &format!("{group_name}/stringzillas::NeedlemanWunschScores<1cpu>"),
            tape_bytes_view,
            side_single_cpu,
            MeasureSpec::new(Unit::Cups, work),
            &mut matrix,
            |queries, candidates, matrix| {
                nw_single
                    .compute_into(cpu_single, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute NeedlemanWunschScores on CPU (single-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    {
        let work = crossproduct_work_bytes(tape_bytes_view, side_multi_cpu);
        let mut matrix = UnifiedMat::<isize>::try_allocate(side_multi_cpu, side_multi_cpu)
            .expect("Failed to allocate NeedlemanWunschScores matrix (parallel)");
        measure_crossproduct_bytes_isize(
            settings,
            &format!("{group_name}/stringzillas::NeedlemanWunschScores<{threads}cpu>"),
            tape_bytes_view,
            side_multi_cpu,
            MeasureSpec::new(Unit::Cups, work).with_concurrency(threads),
            &mut matrix,
            |queries, candidates, matrix| {
                nw_parallel
                    .compute_into(cpu_parallel, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute NeedlemanWunschScores on CPU (multi-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    if let (Some(gpu), Some(engine)) = (maybe_gpu, maybe_nw_gpu.as_ref()) {
        let work = crossproduct_work_bytes(tape_bytes_view, side_gpu);
        let mut matrix = UnifiedMat::<isize>::try_allocate(side_gpu, side_gpu)
            .expect("Failed to allocate NeedlemanWunschScores matrix (GPU)");
        measure_crossproduct_bytes_isize(
            settings,
            &format!("{group_name}/stringzillas::NeedlemanWunschScores<1gpu>"),
            tape_bytes_view,
            side_gpu,
            MeasureSpec::new(Unit::Cups, work),
            &mut matrix,
            |queries, candidates, matrix| {
                engine
                    .compute_into(gpu, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
            },
        );
    }

    // Smith-Waterman (Local alignment)
    {
        let work = crossproduct_work_bytes(tape_bytes_view, side_single_cpu);
        let mut matrix = UnifiedMat::<isize>::try_allocate(side_single_cpu, side_single_cpu)
            .expect("Failed to allocate SmithWatermanScores matrix (single)");
        measure_crossproduct_bytes_isize(
            settings,
            &format!("{group_name}/stringzillas::SmithWatermanScores<1cpu>"),
            tape_bytes_view,
            side_single_cpu,
            MeasureSpec::new(Unit::Cups, work),
            &mut matrix,
            |queries, candidates, matrix| {
                sw_single
                    .compute_into(cpu_single, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute SmithWatermanScores on CPU (single-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    {
        let work = crossproduct_work_bytes(tape_bytes_view, side_multi_cpu);
        let mut matrix = UnifiedMat::<isize>::try_allocate(side_multi_cpu, side_multi_cpu)
            .expect("Failed to allocate SmithWatermanScores matrix (parallel)");
        measure_crossproduct_bytes_isize(
            settings,
            &format!("{group_name}/stringzillas::SmithWatermanScores<{threads}cpu>"),
            tape_bytes_view,
            side_multi_cpu,
            MeasureSpec::new(Unit::Cups, work).with_concurrency(threads),
            &mut matrix,
            |queries, candidates, matrix| {
                sw_parallel
                    .compute_into(cpu_parallel, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute SmithWatermanScores on CPU (multi-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    if let (Some(gpu), Some(engine)) = (maybe_gpu, maybe_sw_gpu.as_ref()) {
        let work = crossproduct_work_bytes(tape_bytes_view, side_gpu);
        let mut matrix = UnifiedMat::<isize>::try_allocate(side_gpu, side_gpu)
            .expect("Failed to allocate SmithWatermanScores matrix (GPU)");
        measure_crossproduct_bytes_isize(
            settings,
            &format!("{group_name}/stringzillas::SmithWatermanScores<1gpu>"),
            tape_bytes_view,
            side_gpu,
            MeasureSpec::new(Unit::Cups, work),
            &mut matrix,
            |queries, candidates, matrix| {
                engine
                    .compute_into(gpu, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
            },
        );
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
