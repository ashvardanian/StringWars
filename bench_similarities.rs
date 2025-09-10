#![doc = r#"# StringWa.rs: String Similarity Benchmarks

This file benchmarks different libraries implementing string alignment and edit
distance calculation, comparing single-threaded, multi-threaded, and GPU-accelerated
implementations across three gap cost models:

- **Uniform**: Classic Levenshtein distance with uniform substitution costs (match=0, mismatch=1)
- **Linear**: Needleman-Wunsch and Smith-Waterman with linear gap penalties (open_cost == extend_cost)
- **Affine**: Advanced alignment with different gap opening vs extension costs (open_cost != extend_cost)

The input file is tokenized into lines or words and each consecutive pair of tokens
is evaluated for similarity. As most algorithms have quadratic complexity and use
Dynamic Programming techniques, their throughput is evaluated in the number of CUPS,
or Cell Updates Per Second.

## Usage Examples

The benchmarks use environment variables to control the input dataset and mode:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line.
  - `words`: Process the dataset word by word.
- `STRINGWARS_BATCH`: Number of pairs to process in each batch (default: 2048).

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH=2048 \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_similarities bench_similarities --jobs 8
```
"#]
use core::convert::TryInto;
use std::env;
use std::fs;

use criterion::{Criterion, Throughput};
use fork_union::count_logical_cores;

use rapidfuzz::distance::levenshtein;
use stringtape::{BytesTape, BytesTapeView, CharsTapeView};
use stringzilla::szs::{
    error_costs_256x256_unary, AnyBytesTape, AnyCharsTape, DeviceScope, LevenshteinDistances,
    LevenshteinDistancesUtf8, NeedlemanWunschScores, SmithWatermanScores, UnifiedAlloc, UnifiedVec,
};

// Pull some metadata logging functionality
use stringzilla::sz::dynamic_dispatch as sz_dynamic_dispatch;
use stringzilla::szs::{capabilities as szs_capabilities, version as szs_version};

fn log_stringzilla_metadata() {
    let v = szs_version();
    println!("StringZilla v{}.{}.{}", v.major, v.minor, v.patch);
    println!("- uses dynamic dispatch: {}", sz_dynamic_dispatch());
    println!("- capabilities: {}", szs_capabilities().as_str());
}

fn configure_bench() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(10))
}

/// Creates batch subviews from bytes tape views for processing
fn bytes_tape_slice<'a>(
    tape_a_view: &'a BytesTapeView<u64>,
    tape_b_view: &'a BytesTapeView<u64>,
    start_idx: &mut usize,
    batch_size: usize,
    pairs_count: usize,
) -> (BytesTapeView<'a, u64>, BytesTapeView<'a, u64>, usize) {
    let current_start = *start_idx;
    let end_idx = std::cmp::min(current_start + batch_size, pairs_count);
    let actual_batch_size = end_idx - current_start;
    let batch_a_view = tape_a_view.subview(current_start, end_idx).unwrap();
    let batch_b_view = tape_b_view.subview(current_start, end_idx).unwrap();
    *start_idx = (current_start + actual_batch_size) % pairs_count;
    (batch_a_view, batch_b_view, actual_batch_size)
}

/// Creates batch subviews from chars tape views for processing
fn chars_tape_slice<'a>(
    chars_a_view: &'a CharsTapeView<u64>,
    chars_b_view: &'a CharsTapeView<u64>,
    start_idx: &mut usize,
    batch_size: usize,
    pairs_count: usize,
) -> (CharsTapeView<'a, u64>, CharsTapeView<'a, u64>, usize) {
    let current_start = *start_idx;
    let end_idx = std::cmp::min(current_start + batch_size, pairs_count);
    let actual_batch_size = end_idx - current_start;
    let batch_a_view = chars_a_view.subview(current_start, end_idx).unwrap();
    let batch_b_view = chars_b_view.subview(current_start, end_idx).unwrap();
    *start_idx = (current_start + actual_batch_size) % pairs_count;
    (batch_a_view, batch_b_view, actual_batch_size)
}

fn bench_similarities(c: &mut Criterion) {
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_TOKENS").unwrap_or_else(|_| "lines".to_string());
    let content = fs::read_to_string(&dataset_path).expect("Could not read dataset");

    let batch_size = env::var("STRINGWARS_BATCH")
        .unwrap_or_else(|_| "2048".to_string())
        .parse::<usize>()
        .expect("STRINGWARS_BATCH must be a number");

    let max_pairs = env::var("STRINGWARS_MAX_PAIRS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10000);

    let units: Vec<&str> = match mode.as_str() {
        "words" => content.split_whitespace().collect(),
        "lines" => content.lines().collect(),
        other => panic!(
            "Unknown STRINGWARS_TOKENS: {}. Use 'lines' or 'words'.",
            other
        ),
    };

    if units.len() < 2 {
        panic!("Dataset must contain at least two items for comparisons.");
    }

    // Limit units if max_pairs is specified (since pairs = units.len() - 1)
    let mut truncated_units = units.clone();
    if max_pairs < units.len() - 1 {
        truncated_units.truncate(max_pairs + 1);
    }

    if truncated_units.len() < 2 {
        panic!("Need at least 2 units to form consecutive pairs.");
    }

    // Create BytesTape and populate it with all units in batch
    let mut units_tape: BytesTape<u64, UnifiedAlloc> = BytesTape::new_in(UnifiedAlloc);
    units_tape
        .extend(truncated_units.iter().map(|s| s.as_bytes()))
        .expect("Failed to extend BytesTape");

    // Create zero-copy views for consecutive pairs
    // tape_a_view: elements 0..n-1 (all except last)
    // tape_b_view: elements 1..n (all except first)
    let tape_a_view = units_tape
        .subview(0, units_tape.len() - 1)
        .expect("Failed to create tape_a_view");
    let tape_b_view = units_tape
        .subview(1, units_tape.len())
        .expect("Failed to create tape_b_view");

    // Create CharsTapeView from the full tape view (zero-copy view casting)
    let full_chars_view: CharsTapeView<u64> = units_tape
        .view()
        .try_into()
        .expect("Failed to convert to CharsTapeView");
    let chars_a_view = full_chars_view
        .subview(0, full_chars_view.len() - 1)
        .expect("Failed to create chars_a_view");
    let chars_b_view = full_chars_view
        .subview(1, full_chars_view.len())
        .expect("Failed to create chars_b_view");

    let pairs_count = tape_a_view.len();

    // Calculate average matrix sizes for throughput reporting
    let mut total_cells_bytes = 0u64;
    let mut total_cells_utf8 = 0u64;

    for i in 0..pairs_count {
        let a_bytes = &tape_a_view[i];
        let b_bytes = &tape_b_view[i];
        total_cells_bytes += (a_bytes.len() * b_bytes.len()) as u64;

        let a_str = &chars_a_view[i];
        let b_str = &chars_b_view[i];
        total_cells_utf8 += (a_str.chars().count() as u64) * (b_str.chars().count() as u64);
    }

    let avg_cells_bytes = total_cells_bytes / pairs_count as u64;
    let avg_cells_utf8 = total_cells_utf8 / pairs_count as u64;

    // Uniform cost benchmarks (classic Levenshtein: match=0, mismatch=1, open=1, extend=1)
    let mut g = c.benchmark_group("uniform");
    perform_uniform_benchmarks(
        &mut g,
        &tape_a_view,
        &tape_b_view,
        &chars_a_view,
        &chars_b_view,
        batch_size,
        avg_cells_bytes,
        avg_cells_utf8,
    );
    g.finish();

    // Linear gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-2, extend=-2)
    let mut g = c.benchmark_group("linear");
    perform_linear_benchmarks(
        &mut g,
        &tape_a_view,
        &tape_b_view,
        batch_size,
        avg_cells_bytes,
        pairs_count,
    );
    g.finish();

    // Affine gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-5, extend=-1)
    let mut g = c.benchmark_group("affine");
    perform_affine_benchmarks(
        &mut g,
        &tape_a_view,
        &tape_b_view,
        batch_size,
        avg_cells_bytes,
        pairs_count,
    );
    g.finish();
}

/// Uniform cost benchmarks: Classic Levenshtein distance (match=0, mismatch=1, open=1, extend=1)
fn perform_uniform_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tape_a_view: &BytesTapeView<u64>,
    tape_b_view: &BytesTapeView<u64>,
    chars_a_view: &CharsTapeView<u64>,
    chars_b_view: &CharsTapeView<u64>,
    batch_size: usize,
    avg_cells_bytes: u64,
    avg_cells_utf8: u64,
) {
    // No tapes needed - use simple string arrays

    // Create device scopes
    let num_cores = count_logical_cores();
    let cpu_single = DeviceScope::cpu_cores(1).expect("Failed to create single-core device scope");
    let cpu_parallel =
        DeviceScope::cpu_cores(num_cores).expect("Failed to create multi-core device scope");
    let maybe_gpu = DeviceScope::gpu_device(0);

    // Create engines once
    let lev_single = LevenshteinDistances::new(&cpu_single, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistances single");
    let lev_parallel = LevenshteinDistances::new(&cpu_parallel, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistances parallel");
    let lev_utf8_single = LevenshteinDistancesUtf8::new(&cpu_single, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistancesUtf8 single");
    let lev_utf8_parallel = LevenshteinDistancesUtf8::new(&cpu_parallel, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistancesUtf8 parallel");
    let maybe_lev_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| LevenshteinDistances::new(gpu, 0, 1, 1, 1).ok());

    let per_pair_bytes = avg_cells_bytes;
    let per_pair_utf8 = avg_cells_utf8;
    let per_batch_bytes = (batch_size as u64) * avg_cells_bytes;
    let per_batch_utf8 = (batch_size as u64) * avg_cells_utf8;

    // RapidFuzz baselines (no batching; scan one-by-one)
    let pairs_count = tape_a_view.len();
    g.throughput(Throughput::Elements(per_pair_bytes));
    g.bench_function("rapidfuzz::levenshtein<Bytes>(1xCPU)", |b| {
        let mut pair_index = 0;
        b.iter(|| {
            let a_bytes = &tape_a_view[pair_index % pairs_count];
            let b_bytes = &tape_b_view[pair_index % pairs_count];
            pair_index = (pair_index + 1) % pairs_count;
            levenshtein::distance(a_bytes.iter().copied(), b_bytes.iter().copied())
        })
    });

    g.throughput(Throughput::Elements(per_pair_utf8));
    g.bench_function("rapidfuzz::levenshtein<Chars>(1xCPU)", |b| {
        let mut pair_index = 0;
        b.iter(|| {
            let a_str = &chars_a_view[pair_index % pairs_count];
            let b_str = &chars_b_view[pair_index % pairs_count];
            pair_index = (pair_index + 1) % pairs_count;
            levenshtein::distance(a_str.chars(), b_str.chars())
        })
    });

    // StringZilla Binary Levenshtein Distance (uniform costs: 0,1,1,1)
    g.throughput(Throughput::Elements(per_batch_bytes));
    g.bench_function("szs::LevenshteinDistances(1xCPU)", |b| {
        let mut results = UnifiedVec::<usize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        let mut start_idx = 0;
        b.iter(|| {
            let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                &tape_a_view,
                &tape_b_view,
                &mut start_idx,
                batch_size,
                pairs_count,
            );
            lev_single
                .compute_into(
                    &cpu_single,
                    AnyBytesTape::View64(batch_a_view),
                    AnyBytesTape::View64(batch_b_view),
                    &mut results[..actual_batch_size],
                )
                .unwrap();
            std::hint::black_box(&results);
        })
    });

    g.throughput(Throughput::Elements(per_batch_bytes));
    g.bench_function(
        &format!("szs::LevenshteinDistances({}xCPU)", num_cores),
        |b| {
            let mut results = UnifiedVec::<usize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                    &tape_a_view,
                    &tape_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                lev_parallel
                    .compute_into(
                        &cpu_parallel,
                        AnyBytesTape::View64(batch_a_view),
                        AnyBytesTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_lev_gpu.as_ref()) {
        g.throughput(Throughput::Elements(per_batch_bytes));
        let mut results = UnifiedVec::<usize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        g.bench_function("szs::LevenshteinDistances(1xGPU)", |b| {
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                    &tape_a_view,
                    &tape_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_a_view),
                        AnyBytesTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        });
    }

    // StringZilla UTF-8 Levenshtein Distance (uniform costs: 0,1,1,1)
    g.throughput(Throughput::Elements(per_batch_utf8));
    g.bench_function("szs::LevenshteinDistancesUtf8(1xCPU)", |b| {
        let mut results = UnifiedVec::<usize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        let mut start_idx = 0;
        b.iter(|| {
            let (batch_a_view, batch_b_view, actual_batch_size) = chars_tape_slice(
                &chars_a_view,
                &chars_b_view,
                &mut start_idx,
                batch_size,
                pairs_count,
            );
            lev_utf8_single
                .compute_into(
                    &cpu_single,
                    AnyCharsTape::View64(batch_a_view),
                    AnyCharsTape::View64(batch_b_view),
                    &mut results[..actual_batch_size],
                )
                .unwrap();
            std::hint::black_box(&results);
        })
    });

    g.throughput(Throughput::Elements(per_batch_utf8));
    g.bench_function(
        &format!("szs::LevenshteinDistancesUtf8({}xCPU)", num_cores),
        |b| {
            let mut results = UnifiedVec::<usize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = chars_tape_slice(
                    &chars_a_view,
                    &chars_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                lev_utf8_parallel
                    .compute_into(
                        &cpu_parallel,
                        AnyCharsTape::View64(batch_a_view),
                        AnyCharsTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        },
    );
}

/// Linear gap cost benchmarks: NW/SW with linear penalties (match=2, mismatch=-1, open=-2, extend=-2)
fn perform_linear_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tape_a_view: &BytesTapeView<u64>,
    tape_b_view: &BytesTapeView<u64>,
    batch_size: usize,
    avg_cells_bytes: u64,
    pairs_count: usize,
) {
    let num_cores = count_logical_cores();
    let cpu_single = DeviceScope::cpu_cores(1).expect("Failed to create single-core device scope");
    let cpu_parallel =
        DeviceScope::cpu_cores(num_cores).expect("Failed to create multi-core device scope");
    let maybe_gpu = DeviceScope::gpu_device(0);

    // Create scoring matrix for linear gap costs (match=2, mismatch=-1)
    let matrix = error_costs_256x256_unary();

    // Create engines once (linear gap costs: open=-2, extend=-2)
    let nw_single = NeedlemanWunschScores::new(&cpu_single, &matrix, -2, -2)
        .expect("Failed to create NW single");
    let nw_parallel = NeedlemanWunschScores::new(&cpu_parallel, &matrix, -2, -2)
        .expect("Failed to create NW parallel");
    let sw_single =
        SmithWatermanScores::new(&cpu_single, &matrix, -2, -2).expect("Failed to create SW single");
    let sw_parallel = SmithWatermanScores::new(&cpu_parallel, &matrix, -2, -2)
        .expect("Failed to create SW parallel");
    let maybe_nw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| NeedlemanWunschScores::new(gpu, &matrix, -2, -2).ok());
    let maybe_sw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| SmithWatermanScores::new(gpu, &matrix, -2, -2).ok());

    let per_batch = (batch_size as u64) * avg_cells_bytes;

    // Needleman-Wunsch (Global alignment)
    g.throughput(Throughput::Elements(per_batch));
    g.bench_function("szs::NeedlemanWunschScores(1xCPU)", |b| {
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        let mut start_idx = 0;
        b.iter(|| {
            let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                &tape_a_view,
                &tape_b_view,
                &mut start_idx,
                batch_size,
                pairs_count,
            );
            nw_single
                .compute_into(
                    &cpu_single,
                    AnyBytesTape::View64(batch_a_view),
                    AnyBytesTape::View64(batch_b_view),
                    &mut results[..actual_batch_size],
                )
                .unwrap();
            std::hint::black_box(&results);
        })
    });

    g.throughput(Throughput::Elements(per_batch));
    g.bench_function(
        &format!("szs::NeedlemanWunschScores({}xCPU)", num_cores),
        |b| {
            let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                    &tape_a_view,
                    &tape_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                nw_parallel
                    .compute_into(
                        &cpu_parallel,
                        AnyBytesTape::View64(batch_a_view),
                        AnyBytesTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_nw_gpu.as_ref()) {
        g.throughput(Throughput::Elements(per_batch));
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        g.bench_function("szs::NeedlemanWunschScores(1xGPU)", |b| {
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                    &tape_a_view,
                    &tape_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_a_view),
                        AnyBytesTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        });
    }

    // Smith-Waterman (Local alignment)
    g.throughput(Throughput::Elements(per_batch));
    g.bench_function("szs::SmithWatermanScores(1xCPU)", |b| {
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        let mut start_idx = 0;
        b.iter(|| {
            let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                &tape_a_view,
                &tape_b_view,
                &mut start_idx,
                batch_size,
                pairs_count,
            );
            sw_single
                .compute_into(
                    &cpu_single,
                    AnyBytesTape::View64(batch_a_view),
                    AnyBytesTape::View64(batch_b_view),
                    &mut results[..actual_batch_size],
                )
                .unwrap();
            std::hint::black_box(&results);
        })
    });

    g.throughput(Throughput::Elements(per_batch));
    g.bench_function(
        &format!("szs::SmithWatermanScores({}xCPU)", num_cores),
        |b| {
            let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                    &tape_a_view,
                    &tape_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                sw_parallel
                    .compute_into(
                        &cpu_parallel,
                        AnyBytesTape::View64(batch_a_view),
                        AnyBytesTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_sw_gpu.as_ref()) {
        g.throughput(Throughput::Elements(per_batch));
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        g.bench_function("szs::SmithWatermanScores(1xGPU)", |b| {
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                    &tape_a_view,
                    &tape_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_a_view),
                        AnyBytesTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        });
    }
}

/// Affine gap cost benchmarks: NW/SW with affine penalties (match=2, mismatch=-1, open=-5, extend=-1)
fn perform_affine_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tape_a_view: &BytesTapeView<u64>,
    tape_b_view: &BytesTapeView<u64>,
    batch_size: usize,
    avg_cells_bytes: u64,
    pairs_count: usize,
) {
    let num_cores = count_logical_cores();
    let cpu_single = DeviceScope::cpu_cores(1).expect("Failed to create single-core device scope");
    let cpu_parallel =
        DeviceScope::cpu_cores(num_cores).expect("Failed to create multi-core device scope");
    let maybe_gpu = DeviceScope::gpu_device(0);

    // Create scoring matrix for affine gap costs (match=2, mismatch=-1)
    let matrix = error_costs_256x256_unary();

    // Create engines once (affine gap costs: open=-5, extend=-1)
    let nw_single = NeedlemanWunschScores::new(&cpu_single, &matrix, -5, -1)
        .expect("Failed to create NW single");
    let nw_parallel = NeedlemanWunschScores::new(&cpu_parallel, &matrix, -5, -1)
        .expect("Failed to create NW parallel");
    let sw_single =
        SmithWatermanScores::new(&cpu_single, &matrix, -5, -1).expect("Failed to create SW single");
    let sw_parallel = SmithWatermanScores::new(&cpu_parallel, &matrix, -5, -1)
        .expect("Failed to create SW parallel");
    let maybe_nw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| NeedlemanWunschScores::new(gpu, &matrix, -5, -1).ok());
    let maybe_sw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| SmithWatermanScores::new(gpu, &matrix, -5, -1).ok());

    let per_batch = (batch_size as u64) * avg_cells_bytes;

    // Needleman-Wunsch (Global alignment)
    g.throughput(Throughput::Elements(per_batch));
    g.bench_function("szs::NeedlemanWunschScores(1xCPU)", |b| {
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        let mut start_idx = 0;
        b.iter(|| {
            let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                &tape_a_view,
                &tape_b_view,
                &mut start_idx,
                batch_size,
                pairs_count,
            );
            nw_single
                .compute_into(
                    &cpu_single,
                    AnyBytesTape::View64(batch_a_view),
                    AnyBytesTape::View64(batch_b_view),
                    &mut results[..actual_batch_size],
                )
                .unwrap();
            std::hint::black_box(&results);
        })
    });

    g.throughput(Throughput::Elements(per_batch));
    g.bench_function(
        &format!("szs::NeedlemanWunschScores({}xCPU)", num_cores),
        |b| {
            let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                    &tape_a_view,
                    &tape_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                nw_parallel
                    .compute_into(
                        &cpu_parallel,
                        AnyBytesTape::View64(batch_a_view),
                        AnyBytesTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_nw_gpu.as_ref()) {
        g.throughput(Throughput::Elements(per_batch));
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        g.bench_function("szs::NeedlemanWunschScores(1xGPU)", |b| {
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                    &tape_a_view,
                    &tape_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_a_view),
                        AnyBytesTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        });
    }

    // Smith-Waterman (Local alignment)
    g.throughput(Throughput::Elements(per_batch));
    g.bench_function("szs::SmithWatermanScores(1xCPU)", |b| {
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        let mut start_idx = 0;
        b.iter(|| {
            let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                &tape_a_view,
                &tape_b_view,
                &mut start_idx,
                batch_size,
                pairs_count,
            );
            sw_single
                .compute_into(
                    &cpu_single,
                    AnyBytesTape::View64(batch_a_view),
                    AnyBytesTape::View64(batch_b_view),
                    &mut results[..actual_batch_size],
                )
                .unwrap();
            std::hint::black_box(&results);
        })
    });

    g.throughput(Throughput::Elements(per_batch));
    g.bench_function(
        &format!("szs::SmithWatermanScores({}xCPU)", num_cores),
        |b| {
            let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                    &tape_a_view,
                    &tape_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                sw_parallel
                    .compute_into(
                        &cpu_parallel,
                        AnyBytesTape::View64(batch_a_view),
                        AnyBytesTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_sw_gpu.as_ref()) {
        g.throughput(Throughput::Elements(per_batch));
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        g.bench_function("szs::SmithWatermanScores(1xGPU)", |b| {
            let mut start_idx = 0;
            b.iter(|| {
                let (batch_a_view, batch_b_view, actual_batch_size) = bytes_tape_slice(
                    &tape_a_view,
                    &tape_b_view,
                    &mut start_idx,
                    batch_size,
                    pairs_count,
                );
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_a_view),
                        AnyBytesTape::View64(batch_b_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap();
                std::hint::black_box(&results);
            })
        });
    }
}

fn main() {
    log_stringzilla_metadata();
    let mut criterion = configure_bench();
    bench_similarities(&mut criterion);
    criterion.final_summary();
}
