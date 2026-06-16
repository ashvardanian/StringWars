#![doc = r#"# StringWars: String Similarity Benchmarks

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
    cargo criterion --features bench_similarities bench_similarities --jobs 1
```

To run on a GPU-capable machine, enable the CUDA feature and consider larger batches:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH=32768 \
    STRINGWARS_TOKENS=lines \
    STRINGWARS_FILTER=1xGPU \
    cargo criterion --features "cuda bench_similarities" bench_similarities --jobs 1
```
"#]
use core::convert::TryInto;
use std::env;

use criterion::{Criterion, Throughput};
use fork_union::count_logical_cores;
use stringtape::{BytesTape, BytesTapeView, CharsTapeView};

use bio::alignment::{distance as bio_distance, pairwise::Aligner};
use rapidfuzz::distance::levenshtein;
use stringzilla::szs::{
    AnyBytesTape, AnyCharsTape, DeviceScope, LevenshteinDistances, LevenshteinDistancesUtf8,
    NeedlemanWunschScores, SmithWatermanScores, UnifiedAlloc, UnifiedVec,
};

#[path = "../utils.rs"]
mod utils;
use utils::{
    configure_bench, install_panic_hook, load_dataset, log_stringzilla_metadata, should_run,
    CupsWallTime, ResultExt,
};

/// Builds a substitution table for classic unary scoring: `match_cost` on the diagonal,
/// `mismatch_cost` everywhere else. Bytes are folded into 32 classes via `i % 32`, which keeps
/// the table compact; throughput (MCUPS) is invariant to the actual costs.
fn unary_class_costs(match_cost: i8, mismatch_cost: i8) -> ([u8; 256], [[i8; 32]; 32]) {
    let mut byte_to_class = [0u8; 256];
    for byte_value in 0..256 {
        byte_to_class[byte_value] = (byte_value % 32) as u8;
    }
    let mut class_costs = [[mismatch_cost; 32]; 32];
    for class_index in 0..32 {
        class_costs[class_index][class_index] = match_cost;
    }
    (byte_to_class, class_costs)
}

/// Returns the next `batch_size` pairs from the two byte tapes, cycling back to the start once
/// exhausted, plus the actual batch length.
fn bytes_tape_slice<'a>(
    first: &'a BytesTapeView<u64>,
    second: &'a BytesTapeView<u64>,
    start: &mut usize,
    batch_size: usize,
    pairs_count: usize,
) -> (BytesTapeView<'a, u64>, BytesTapeView<'a, u64>, usize) {
    assert!(
        pairs_count > 0 && batch_size > 0,
        "pairs_count and batch_size must be positive"
    );
    let begin = *start % pairs_count;
    let count = batch_size.min(pairs_count - begin);
    let end = begin + count;
    let first_batch = first.subview(begin, end).unwrap_or_else(|error| {
        panic!(
            "first subview({begin}, {end}) failed: {error} (len={})",
            first.len()
        )
    });
    let second_batch = second.subview(begin, end).unwrap_or_else(|error| {
        panic!(
            "second subview({begin}, {end}) failed: {error} (len={})",
            second.len()
        )
    });
    *start = end % pairs_count;
    (first_batch, second_batch, count)
}

/// Returns the next `batch_size` pairs from the two character tapes, cycling back to the start
/// once exhausted, plus the actual batch length.
fn chars_tape_slice<'a>(
    first: &'a CharsTapeView<u64>,
    second: &'a CharsTapeView<u64>,
    start: &mut usize,
    batch_size: usize,
    pairs_count: usize,
) -> (CharsTapeView<'a, u64>, CharsTapeView<'a, u64>, usize) {
    assert!(
        pairs_count > 0 && batch_size > 0,
        "pairs_count and batch_size must be positive"
    );
    let begin = *start % pairs_count;
    let count = batch_size.min(pairs_count - begin);
    let end = begin + count;
    let first_batch = first.subview(begin, end).unwrap_or_else(|error| {
        panic!(
            "first subview({begin}, {end}) failed: {error} (len={})",
            first.len()
        )
    });
    let second_batch = second.subview(begin, end).unwrap_or_else(|error| {
        panic!(
            "second subview({begin}, {end}) failed: {error} (len={})",
            second.len()
        )
    });
    *start = end % pairs_count;
    (first_batch, second_batch, count)
}

fn bench_similarities(criterion: &mut Criterion<CupsWallTime>) {
    // Load dataset using unified loader
    let tape_bytes = load_dataset().unwrap_nice();
    let tape = tape_bytes
        .as_chars()
        .expect("Dataset must be valid UTF-8 for similarities");

    let batch_size = env::var("STRINGWARS_BATCH")
        .unwrap_or_else(|_| "2048".to_string())
        .parse::<usize>()
        .expect("STRINGWARS_BATCH must be a number");

    if tape.len() < 2 {
        panic!("Dataset must contain at least two items for comparisons.");
    }

    // Log benchmark-specific configuration
    println!("Benchmark configuration:");
    println!("- Batch size: {}", batch_size);

    // Create BytesTape and populate it with all tokens (already limited by STRINGWARS_MAX_TOKENS in load_dataset)
    let mut units_tape: BytesTape<u64, UnifiedAlloc> = BytesTape::new_in(UnifiedAlloc);
    units_tape
        .extend(tape.iter().map(|string| string.as_bytes()))
        .expect("Failed to extend BytesTape");

    // Create zero-copy views for consecutive pairs
    // tape_first_view: elements 0..n-1 (all except last)
    // tape_second_view: elements 1..n (all except first)
    let tape_first_view = units_tape
        .subview(0, units_tape.len() - 1)
        .expect("Failed to create tape_first_view");
    let tape_second_view = units_tape
        .subview(1, units_tape.len())
        .expect("Failed to create tape_second_view");

    // Create CharsTapeView from the full tape view (zero-copy view casting)
    let full_chars_view: CharsTapeView<u64> = units_tape
        .view()
        .try_into()
        .expect("Failed to convert to CharsTapeView");
    let chars_first_view = full_chars_view
        .subview(0, full_chars_view.len() - 1)
        .expect("Failed to create chars_first_view");
    let chars_second_view = full_chars_view
        .subview(1, full_chars_view.len())
        .expect("Failed to create chars_second_view");

    let pairs_count = tape_first_view.len();
    println!("- Consecutive pairs to process: {}", pairs_count);

    // Validate batch size against pairs count
    let effective_batch_size = std::cmp::min(batch_size, pairs_count);
    println!();

    // Calculate average matrix sizes for throughput reporting
    let mut total_cells_bytes = 0u64;
    let mut total_cells_utf8 = 0u64;

    for pair_index in 0..pairs_count {
        let a_bytes = &tape_first_view[pair_index];
        let b_bytes = &tape_second_view[pair_index];
        total_cells_bytes += (a_bytes.len() * b_bytes.len()) as u64;

        let a_str = &chars_first_view[pair_index];
        let b_str = &chars_second_view[pair_index];
        total_cells_utf8 += (a_str.chars().count() as u64) * (b_str.chars().count() as u64);
    }

    let avg_cells_bytes = total_cells_bytes / pairs_count as u64;
    let avg_cells_utf8 = total_cells_utf8 / pairs_count as u64;

    // Uniform cost benchmarks (classic Levenshtein: match=0, mismatch=1, open=1, extend=1)
    let mut group = criterion.benchmark_group("uniform");
    perform_uniform_benchmarks(
        &mut group,
        &tape_first_view,
        &tape_second_view,
        &chars_first_view,
        &chars_second_view,
        effective_batch_size,
        avg_cells_bytes,
        avg_cells_utf8,
        pairs_count,
    );
    group.finish();

    // Linear gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-2, extend=-2)
    let mut group = criterion.benchmark_group("linear");
    perform_linear_benchmarks(
        &mut group,
        &tape_first_view,
        &tape_second_view,
        effective_batch_size,
        avg_cells_bytes,
        pairs_count,
    );
    group.finish();

    // Affine gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-5, extend=-1)
    let mut group = criterion.benchmark_group("affine");
    perform_affine_benchmarks(
        &mut group,
        &tape_first_view,
        &tape_second_view,
        effective_batch_size,
        avg_cells_bytes,
        pairs_count,
    );
    group.finish();
}

/// Uniform cost benchmarks: Classic Levenshtein distance (match=0, mismatch=1, open=1, extend=1)
fn perform_uniform_benchmarks(
    group: &mut criterion::BenchmarkGroup<'_, CupsWallTime>,
    tape_first_view: &BytesTapeView<u64>,
    tape_second_view: &BytesTapeView<u64>,
    chars_first_view: &CharsTapeView<u64>,
    chars_second_view: &CharsTapeView<u64>,
    batch_size: usize,
    avg_cells_bytes: u64,
    avg_cells_utf8: u64,
    pairs_count: usize,
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
    if should_run("uniform/rapidfuzz::levenshtein<Bytes>(1xCPU)") {
        group.throughput(Throughput::Elements(per_pair_bytes));
        group.bench_function("rapidfuzz::levenshtein<Bytes>(1xCPU)", |bencher| {
            let mut pair_index = 0;
            bencher.iter(|| {
                let a_bytes = &tape_first_view[pair_index % pairs_count];
                let b_bytes = &tape_second_view[pair_index % pairs_count];
                pair_index = (pair_index + 1) % pairs_count;
                levenshtein::distance(a_bytes.iter().copied(), b_bytes.iter().copied())
            })
        });
    }

    if should_run("uniform/rapidfuzz::levenshtein<Chars>(1xCPU)") {
        group.throughput(Throughput::Elements(per_pair_utf8));
        group.bench_function("rapidfuzz::levenshtein<Chars>(1xCPU)", |bencher| {
            let mut pair_index = 0;
            bencher.iter(|| {
                let a_str = &chars_first_view[pair_index % pairs_count];
                let b_str = &chars_second_view[pair_index % pairs_count];
                pair_index = (pair_index + 1) % pairs_count;
                levenshtein::distance(a_str.chars(), b_str.chars())
            })
        });
    }

    if should_run("uniform/bio::levenshtein(1xCPU)") {
        group.throughput(Throughput::Elements(per_pair_bytes));
        group.bench_function("bio::levenshtein(1xCPU)", |bencher| {
            let mut pair_index = 0;
            bencher.iter(|| {
                let a_bytes = &tape_first_view[pair_index % pairs_count];
                let b_bytes = &tape_second_view[pair_index % pairs_count];
                pair_index = (pair_index + 1) % pairs_count;
                std::hint::black_box(bio_distance::levenshtein(a_bytes, b_bytes))
            })
        });
    }

    if should_run("uniform/stringzillas::LevenshteinDistances(1xCPU)") {
        group.throughput(Throughput::Elements(per_batch_bytes));
        group.bench_function("stringzillas::LevenshteinDistances(1xCPU)", |bencher| {
            let mut results = UnifiedVec::<usize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                    &tape_first_view,
                    &tape_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                lev_single
                    .compute_into(
                        &cpu_single,
                        AnyBytesTape::View64(batch_first_view),
                        AnyBytesTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute LevenshteinDistances on CPU (single-threaded): {}",
                            error
                        );
                    });
                std::hint::black_box(&results);
            })
        });
    }

    if should_run(&format!(
        "uniform/stringzillas::LevenshteinDistances({}xCPU)",
        num_cores
    )) {
        group.throughput(Throughput::Elements(per_batch_bytes));
        group.bench_function(
            &format!("stringzillas::LevenshteinDistances({}xCPU)", num_cores),
            |bencher| {
                let mut results = UnifiedVec::<usize>::with_capacity_in(batch_size, UnifiedAlloc);
                results.resize(batch_size, 0);
                let mut start_index = 0;
                bencher.iter(|| {
                    let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                        &tape_first_view,
                        &tape_second_view,
                        &mut start_index,
                        batch_size,
                        pairs_count,
                    );
                    lev_parallel
                        .compute_into(
                            &cpu_parallel,
                            AnyBytesTape::View64(batch_first_view),
                            AnyBytesTape::View64(batch_second_view),
                            &mut results[..actual_batch_size],
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                            "Failed to compute LevenshteinDistances on CPU (multi-threaded): {}",
                            error
                        );
                        });
                    std::hint::black_box(&results);
                })
            },
        );
    }

    // StringZilla UTF-8 Levenshtein Distance (uniform costs: 0,1,1,1)
    if should_run("uniform/stringzillas::LevenshteinDistancesUtf8(1xCPU)") {
        group.throughput(Throughput::Elements(per_batch_utf8));
        group.bench_function("stringzillas::LevenshteinDistancesUtf8(1xCPU)", |bencher| {
            let mut results = UnifiedVec::<usize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = chars_tape_slice(
                    &chars_first_view,
                    &chars_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                lev_utf8_single
                    .compute_into(
                        &cpu_single,
                        AnyCharsTape::View64(batch_first_view),
                        AnyCharsTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                        "Failed to compute LevenshteinDistancesUtf8 on CPU (single-threaded): {}",
                        error
                    );
                    });
                std::hint::black_box(&results);
            })
        });
    }

    if should_run(&format!(
        "uniform/stringzillas::LevenshteinDistancesUtf8({}xCPU)",
        num_cores
    )) {
        group.throughput(Throughput::Elements(per_batch_utf8));
        group.bench_function(
            &format!("stringzillas::LevenshteinDistancesUtf8({}xCPU)", num_cores),
        |bencher| {
            let mut results = UnifiedVec::<usize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = chars_tape_slice(
                    &chars_first_view,
                    &chars_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                lev_utf8_parallel
                    .compute_into(
                        &cpu_parallel,
                        AnyCharsTape::View64(batch_first_view),
                        AnyCharsTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute LevenshteinDistancesUtf8 on CPU (multi-threaded): {}", error);
                    });
                std::hint::black_box(&results);
            })
        },
    );
    }

    if maybe_gpu.is_ok()
        && maybe_lev_gpu.is_some()
        && should_run("uniform/stringzillas::LevenshteinDistances(1xGPU)")
    {
        let gpu = maybe_gpu.as_ref().unwrap();
        let engine = maybe_lev_gpu.as_ref().unwrap();
        group.throughput(Throughput::Elements(per_batch_bytes));
        let mut results = UnifiedVec::<usize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        group.bench_function("stringzillas::LevenshteinDistances(1xGPU)", |bencher| {
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                    &tape_first_view,
                    &tape_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_first_view),
                        AnyBytesTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
                std::hint::black_box(&results);
            })
        });
    }
}

/// Linear gap cost benchmarks: NW/SW with linear penalties (match=2, mismatch=-1, open=-2, extend=-2)
fn perform_linear_benchmarks(
    group: &mut criterion::BenchmarkGroup<'_, CupsWallTime>,
    tape_first_view: &BytesTapeView<u64>,
    tape_second_view: &BytesTapeView<u64>,
    batch_size: usize,
    avg_cells_bytes: u64,
    pairs_count: usize,
) {
    let num_cores = count_logical_cores();
    let cpu_single = DeviceScope::cpu_cores(1).expect("Failed to create single-core device scope");
    let cpu_parallel =
        DeviceScope::cpu_cores(num_cores).expect("Failed to create multi-core device scope");
    let maybe_gpu = DeviceScope::gpu_device(0);

    // Unary scoring (match=2, mismatch=-1) folded into the 32-class table.
    let (byte_to_class, class_costs) = unary_class_costs(2, -1);

    // Create engines once (linear gap costs: open=-2, extend=-2)
    let nw_single = NeedlemanWunschScores::new(&cpu_single, &byte_to_class, &class_costs, -2, -2)
        .expect("Failed to create NW single");
    let nw_parallel =
        NeedlemanWunschScores::new(&cpu_parallel, &byte_to_class, &class_costs, -2, -2)
            .expect("Failed to create NW parallel");
    let sw_single = SmithWatermanScores::new(&cpu_single, &byte_to_class, &class_costs, -2, -2)
        .expect("Failed to create SW single");
    let sw_parallel = SmithWatermanScores::new(&cpu_parallel, &byte_to_class, &class_costs, -2, -2)
        .expect("Failed to create SW parallel");
    let maybe_nw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| NeedlemanWunschScores::new(gpu, &byte_to_class, &class_costs, -2, -2).ok());
    let maybe_sw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| SmithWatermanScores::new(gpu, &byte_to_class, &class_costs, -2, -2).ok());

    let mut max_len = 0usize;
    for pair_index in 0..pairs_count {
        let a_len = tape_first_view[pair_index].len();
        let b_len = tape_second_view[pair_index].len();
        if a_len > max_len {
            max_len = a_len;
        }
        if b_len > max_len {
            max_len = b_len;
        }
    }
    let max_len = std::cmp::max(1, max_len);

    let per_batch = (batch_size as u64) * avg_cells_bytes;
    let per_pair = avg_cells_bytes;

    if should_run("linear/bio::pairwise::global(1xCPU)") {
        group.throughput(Throughput::Elements(per_pair));
        group.bench_function("bio::pairwise::global(1xCPU)", |bencher| {
            let mut aligner =
                Aligner::with_capacity(
                    max_len,
                    max_len,
                    -2,
                    -2,
                    |a: u8, b: u8| if a == b { 2 } else { -1 },
                );
            let mut pair_index = 0;
            bencher.iter(|| {
                let a_bytes = &tape_first_view[pair_index % pairs_count];
                let b_bytes = &tape_second_view[pair_index % pairs_count];
                pair_index = (pair_index + 1) % pairs_count;
                let score = aligner.global(a_bytes, b_bytes).score;
                std::hint::black_box(score);
            })
        });
    }

    if should_run("linear/bio::pairwise::local(1xCPU)") {
        group.throughput(Throughput::Elements(per_pair));
        group.bench_function("bio::pairwise::local(1xCPU)", |bencher| {
            let mut aligner =
                Aligner::with_capacity(
                    max_len,
                    max_len,
                    -2,
                    -2,
                    |a: u8, b: u8| if a == b { 2 } else { -1 },
                );
            let mut pair_index = 0;
            bencher.iter(|| {
                let a_bytes = &tape_first_view[pair_index % pairs_count];
                let b_bytes = &tape_second_view[pair_index % pairs_count];
                pair_index = (pair_index + 1) % pairs_count;
                let score = aligner.local(a_bytes, b_bytes).score;
                std::hint::black_box(score);
            })
        });
    }

    // Needleman-Wunsch (Global alignment)
    if should_run("stringzillas::NeedlemanWunschScores(1xCPU)") {
        group.throughput(Throughput::Elements(per_batch));
        group.bench_function("stringzillas::NeedlemanWunschScores(1xCPU)", |bencher| {
            let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                    &tape_first_view,
                    &tape_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                nw_single
                    .compute_into(
                        &cpu_single,
                        AnyBytesTape::View64(batch_first_view),
                        AnyBytesTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute NeedlemanWunschScores (linear gap) on CPU (single-threaded): {}", error);
                    });
                std::hint::black_box(&results);
            })
        });
    }

    if should_run(&format!(
        "stringzillas::NeedlemanWunschScores({}xCPU)",
        num_cores
    )) {
        group.throughput(Throughput::Elements(per_batch));
        group.bench_function(
            &format!("stringzillas::NeedlemanWunschScores({}xCPU)", num_cores),
            |bencher| {
                let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
                results.resize(batch_size, 0);
                let mut start_index = 0;
                bencher.iter(|| {
                    let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                        &tape_first_view,
                        &tape_second_view,
                        &mut start_index,
                        batch_size,
                        pairs_count,
                    );
                    nw_parallel
                        .compute_into(
                            &cpu_parallel,
                            AnyBytesTape::View64(batch_first_view),
                            AnyBytesTape::View64(batch_second_view),
                            &mut results[..actual_batch_size],
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "Failed to compute NeedlemanWunschScores on CPU (multi-threaded): {}",
                                error
                            );
                        });
                    std::hint::black_box(&results);
                })
            },
        );
    }

    if maybe_gpu.is_ok()
        && maybe_nw_gpu.is_some()
        && should_run("stringzillas::NeedlemanWunschScores(1xGPU)")
    {
        let gpu = maybe_gpu.as_ref().unwrap();
        let engine = maybe_nw_gpu.as_ref().unwrap();
        group.throughput(Throughput::Elements(per_batch));
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        group.bench_function("stringzillas::NeedlemanWunschScores(1xGPU)", |bencher| {
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                    &tape_first_view,
                    &tape_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_first_view),
                        AnyBytesTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
                std::hint::black_box(&results);
            })
        });
    }

    // Smith-Waterman (Local alignment)
    if should_run("stringzillas::SmithWatermanScores(1xCPU)") {
        group.throughput(Throughput::Elements(per_batch));
        group.bench_function("stringzillas::SmithWatermanScores(1xCPU)", |bencher| {
            let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                    &tape_first_view,
                    &tape_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                sw_single
                    .compute_into(
                        &cpu_single,
                        AnyBytesTape::View64(batch_first_view),
                        AnyBytesTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute SmithWatermanScores on CPU (single-threaded): {}",
                            error
                        );
                    });
                std::hint::black_box(&results);
            })
        });
    }

    if should_run(&format!(
        "stringzillas::SmithWatermanScores({}xCPU)",
        num_cores
    )) {
        group.throughput(Throughput::Elements(per_batch));
        group.bench_function(
            &format!("stringzillas::SmithWatermanScores({}xCPU)", num_cores),
            |bencher| {
                let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
                results.resize(batch_size, 0);
                let mut start_index = 0;
                bencher.iter(|| {
                    let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                        &tape_first_view,
                        &tape_second_view,
                        &mut start_index,
                        batch_size,
                        pairs_count,
                    );
                    sw_parallel
                        .compute_into(
                            &cpu_parallel,
                            AnyBytesTape::View64(batch_first_view),
                            AnyBytesTape::View64(batch_second_view),
                            &mut results[..actual_batch_size],
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "Failed to compute SmithWatermanScores on CPU (multi-threaded): {}",
                                error
                            );
                        });
                    std::hint::black_box(&results);
                })
            },
        );
    }

    if maybe_gpu.is_ok()
        && maybe_sw_gpu.is_some()
        && should_run("stringzillas::SmithWatermanScores(1xGPU)")
    {
        let gpu = maybe_gpu.as_ref().unwrap();
        let engine = maybe_sw_gpu.as_ref().unwrap();
        group.throughput(Throughput::Elements(per_batch));
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        group.bench_function("stringzillas::SmithWatermanScores(1xGPU)", |bencher| {
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                    &tape_first_view,
                    &tape_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_first_view),
                        AnyBytesTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
                std::hint::black_box(&results);
            })
        });
    }
}

/// Affine gap cost benchmarks: NW/SW with affine penalties (match=2, mismatch=-1, open=-5, extend=-1)
fn perform_affine_benchmarks(
    group: &mut criterion::BenchmarkGroup<'_, CupsWallTime>,
    tape_first_view: &BytesTapeView<u64>,
    tape_second_view: &BytesTapeView<u64>,
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
    // Unary scoring (match=2, mismatch=-1) folded into the 32-class table.
    let (byte_to_class, class_costs) = unary_class_costs(2, -1);

    // Create engines once (affine gap costs: open=-5, extend=-1)
    let nw_single = NeedlemanWunschScores::new(&cpu_single, &byte_to_class, &class_costs, -5, -1)
        .expect("Failed to create NW single");
    let nw_parallel =
        NeedlemanWunschScores::new(&cpu_parallel, &byte_to_class, &class_costs, -5, -1)
            .expect("Failed to create NW parallel");
    let sw_single = SmithWatermanScores::new(&cpu_single, &byte_to_class, &class_costs, -5, -1)
        .expect("Failed to create SW single");
    let sw_parallel = SmithWatermanScores::new(&cpu_parallel, &byte_to_class, &class_costs, -5, -1)
        .expect("Failed to create SW parallel");
    let maybe_nw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| NeedlemanWunschScores::new(gpu, &byte_to_class, &class_costs, -5, -1).ok());
    let maybe_sw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| SmithWatermanScores::new(gpu, &byte_to_class, &class_costs, -5, -1).ok());

    let mut max_len = 0usize;
    for pair_index in 0..pairs_count {
        let a_len = tape_first_view[pair_index].len();
        let b_len = tape_second_view[pair_index].len();
        if a_len > max_len {
            max_len = a_len;
        }
        if b_len > max_len {
            max_len = b_len;
        }
    }
    let max_len = std::cmp::max(1, max_len);

    let per_batch = (batch_size as u64) * avg_cells_bytes;
    let per_pair = avg_cells_bytes;

    if should_run("affine/bio::pairwise::global(1xCPU)") {
        group.throughput(Throughput::Elements(per_pair));
        group.bench_function("bio::pairwise::global(1xCPU)", |bencher| {
            let mut aligner =
                Aligner::with_capacity(
                    max_len,
                    max_len,
                    -5,
                    -1,
                    |a: u8, b: u8| if a == b { 2 } else { -1 },
                );
            let mut pair_index = 0;
            bencher.iter(|| {
                let a_bytes = &tape_first_view[pair_index % pairs_count];
                let b_bytes = &tape_second_view[pair_index % pairs_count];
                pair_index = (pair_index + 1) % pairs_count;
                let score = aligner.global(a_bytes, b_bytes).score;
                std::hint::black_box(score);
            })
        });
    }

    if should_run("affine/bio::pairwise::local(1xCPU)") {
        group.throughput(Throughput::Elements(per_pair));
        group.bench_function("bio::pairwise::local(1xCPU)", |bencher| {
            let mut aligner =
                Aligner::with_capacity(
                    max_len,
                    max_len,
                    -5,
                    -1,
                    |a: u8, b: u8| if a == b { 2 } else { -1 },
                );
            let mut pair_index = 0;
            bencher.iter(|| {
                let a_bytes = &tape_first_view[pair_index % pairs_count];
                let b_bytes = &tape_second_view[pair_index % pairs_count];
                pair_index = (pair_index + 1) % pairs_count;
                let score = aligner.local(a_bytes, b_bytes).score;
                std::hint::black_box(score);
            })
        });
    }

    // Needleman-Wunsch (Global alignment)
    if should_run("stringzillas::NeedlemanWunschScores(1xCPU)") {
        group.throughput(Throughput::Elements(per_batch));
        group.bench_function("stringzillas::NeedlemanWunschScores(1xCPU)", |bencher| {
            let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                    &tape_first_view,
                    &tape_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                nw_single
                    .compute_into(
                        &cpu_single,
                        AnyBytesTape::View64(batch_first_view),
                        AnyBytesTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute NeedlemanWunschScores (linear gap) on CPU (single-threaded): {}", error);
                    });
                std::hint::black_box(&results);
            })
        });
    }

    if should_run(&format!(
        "stringzillas::NeedlemanWunschScores({}xCPU)",
        num_cores
    )) {
        group.throughput(Throughput::Elements(per_batch));
        group.bench_function(
            &format!("stringzillas::NeedlemanWunschScores({}xCPU)", num_cores),
            |bencher| {
                let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
                results.resize(batch_size, 0);
                let mut start_index = 0;
                bencher.iter(|| {
                    let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                        &tape_first_view,
                        &tape_second_view,
                        &mut start_index,
                        batch_size,
                        pairs_count,
                    );
                    nw_parallel
                        .compute_into(
                            &cpu_parallel,
                            AnyBytesTape::View64(batch_first_view),
                            AnyBytesTape::View64(batch_second_view),
                            &mut results[..actual_batch_size],
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "Failed to compute NeedlemanWunschScores on CPU (multi-threaded): {}",
                                error
                            );
                        });
                    std::hint::black_box(&results);
                })
            },
        );
    }

    if maybe_gpu.is_ok()
        && maybe_nw_gpu.is_some()
        && should_run("stringzillas::NeedlemanWunschScores(1xGPU)")
    {
        let gpu = maybe_gpu.as_ref().unwrap();
        let engine = maybe_nw_gpu.as_ref().unwrap();
        group.throughput(Throughput::Elements(per_batch));
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        group.bench_function("stringzillas::NeedlemanWunschScores(1xGPU)", |bencher| {
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                    &tape_first_view,
                    &tape_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_first_view),
                        AnyBytesTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
                std::hint::black_box(&results);
            })
        });
    }

    // Smith-Waterman (Local alignment)
    if should_run("stringzillas::SmithWatermanScores(1xCPU)") {
        group.throughput(Throughput::Elements(per_batch));
        group.bench_function("stringzillas::SmithWatermanScores(1xCPU)", |bencher| {
            let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
            results.resize(batch_size, 0);
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                    &tape_first_view,
                    &tape_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                sw_single
                    .compute_into(
                        &cpu_single,
                        AnyBytesTape::View64(batch_first_view),
                        AnyBytesTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute SmithWatermanScores on CPU (single-threaded): {}",
                            error
                        );
                    });
                std::hint::black_box(&results);
            })
        });
    }

    if should_run(&format!(
        "stringzillas::SmithWatermanScores({}xCPU)",
        num_cores
    )) {
        group.throughput(Throughput::Elements(per_batch));
        group.bench_function(
            &format!("stringzillas::SmithWatermanScores({}xCPU)", num_cores),
            |bencher| {
                let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
                results.resize(batch_size, 0);
                let mut start_index = 0;
                bencher.iter(|| {
                    let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                        &tape_first_view,
                        &tape_second_view,
                        &mut start_index,
                        batch_size,
                        pairs_count,
                    );
                    sw_parallel
                        .compute_into(
                            &cpu_parallel,
                            AnyBytesTape::View64(batch_first_view),
                            AnyBytesTape::View64(batch_second_view),
                            &mut results[..actual_batch_size],
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "Failed to compute SmithWatermanScores on CPU (multi-threaded): {}",
                                error
                            );
                        });
                    std::hint::black_box(&results);
                })
            },
        );
    }

    if maybe_gpu.is_ok()
        && maybe_sw_gpu.is_some()
        && should_run("stringzillas::SmithWatermanScores(1xGPU)")
    {
        let gpu = maybe_gpu.as_ref().unwrap();
        let engine = maybe_sw_gpu.as_ref().unwrap();
        group.throughput(Throughput::Elements(per_batch));
        let mut results = UnifiedVec::<isize>::with_capacity_in(batch_size, UnifiedAlloc);
        results.resize(batch_size, 0);
        group.bench_function("stringzillas::SmithWatermanScores(1xGPU)", |bencher| {
            let mut start_index = 0;
            bencher.iter(|| {
                let (batch_first_view, batch_second_view, actual_batch_size) = bytes_tape_slice(
                    &tape_first_view,
                    &tape_second_view,
                    &mut start_index,
                    batch_size,
                    pairs_count,
                );
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_first_view),
                        AnyBytesTape::View64(batch_second_view),
                        &mut results[..actual_batch_size],
                    )
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
                std::hint::black_box(&results);
            })
        });
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();
    let mut criterion = configure_bench(CupsWallTime::default(), 5, 30);
    bench_similarities(&mut criterion);
    criterion.final_summary();
}
