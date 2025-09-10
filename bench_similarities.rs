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
use std::env;
use std::fs;

use criterion::{Criterion, Throughput};
use fork_union::count_logical_cores;

use rapidfuzz::distance::levenshtein;
use stringzilla::szs::{
    error_costs_256x256_unary, DeviceScope, LevenshteinDistances, LevenshteinDistancesUtf8,
    NeedlemanWunschScores, SmithWatermanScores,
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

fn make_batch_bytes<'a>(
    pairs: &'a [(&'a str, &'a str)],
    start_index: &mut usize,
    batch_size: usize,
) -> (Vec<&'a [u8]>, Vec<&'a [u8]>) {
    let mut a_out = Vec::with_capacity(batch_size);
    let mut b_out = Vec::with_capacity(batch_size);
    let len = pairs.len();
    let start = *start_index;
    for i in 0..batch_size {
        let idx = (start + i) % len;
        let (a, b) = pairs[idx];
        a_out.push(a.as_bytes());
        b_out.push(b.as_bytes());
    }
    *start_index = (start + batch_size) % len;
    (a_out, b_out)
}

fn make_batch_strs<'a>(
    pairs: &'a [(&'a str, &'a str)],
    start_index: &mut usize,
    batch_size: usize,
) -> (Vec<&'a str>, Vec<&'a str>) {
    let mut a_out = Vec::with_capacity(batch_size);
    let mut b_out = Vec::with_capacity(batch_size);
    let len = pairs.len();
    let start = *start_index;
    for i in 0..batch_size {
        let idx = (start + i) % len;
        let (a, b) = pairs[idx];
        a_out.push(a);
        b_out.push(b);
    }
    *start_index = (start + batch_size) % len;
    (a_out, b_out)
}

fn configure_bench() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(10))
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

    let mut pairs: Vec<(&str, &str)> = units
        .chunks(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                Some((chunk[0], chunk[1]))
            } else {
                None
            }
        })
        .collect();

    if pairs.is_empty() {
        panic!("No pairs could be formed from the dataset.");
    }

    if pairs.len() > max_pairs {
        pairs.truncate(max_pairs);
    }

    // Calculate average matrix sizes for throughput reporting
    // - bytes: number of bytes in each string product
    // - utf8: number of Unicode scalar values (code points) product
    let avg_cells_bytes: u64 = pairs
        .iter()
        .map(|(a, b)| (a.len() * b.len()) as u64)
        .sum::<u64>()
        / pairs.len() as u64;

    let avg_cells_utf8: u64 = pairs
        .iter()
        .map(|(a, b)| (a.chars().count() as u64) * (b.chars().count() as u64))
        .sum::<u64>()
        / pairs.len() as u64;

    // Uniform cost benchmarks (classic Levenshtein: match=0, mismatch=1, open=1, extend=1)
    let mut g = c.benchmark_group("uniform");
    perform_uniform_benchmarks(&mut g, &pairs, batch_size, avg_cells_bytes, avg_cells_utf8);
    g.finish();

    // Linear gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-2, extend=-2)
    let mut g = c.benchmark_group("linear");
    perform_linear_benchmarks(&mut g, &pairs, batch_size, avg_cells_bytes);
    g.finish();

    // Affine gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-5, extend=-1)
    let mut g = c.benchmark_group("affine");
    perform_affine_benchmarks(&mut g, &pairs, batch_size, avg_cells_bytes);
    g.finish();
}

/// Uniform cost benchmarks: Classic Levenshtein distance (match=0, mismatch=1, open=1, extend=1)
fn perform_uniform_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pairs: &[(&str, &str)],
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
    g.throughput(Throughput::Elements(per_pair_bytes));
    g.bench_function("rapidfuzz::levenshtein<Bytes>(1xCPU)", |b| {
        let mut pair_index = 0;
        b.iter(|| {
            let (a, b_str) = pairs[pair_index % pairs.len()];
            pair_index = (pair_index + 1) % pairs.len();
            levenshtein::distance(a.bytes(), b_str.bytes())
        })
    });

    g.throughput(Throughput::Elements(per_pair_utf8));
    g.bench_function("rapidfuzz::levenshtein<Chars>(1xCPU)", |b| {
        let mut pair_index = 0;
        b.iter(|| {
            let (a, b_str) = pairs[pair_index % pairs.len()];
            pair_index = (pair_index + 1) % pairs.len();
            levenshtein::distance(a.chars(), b_str.chars())
        })
    });

    // StringZilla Binary Levenshtein Distance (uniform costs: 0,1,1,1)
    g.throughput(Throughput::Elements(per_batch_bytes));
    g.bench_function("szs::LevenshteinDistances(1xCPU)", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
            lev_single.compute(&cpu_single, &batch_a, &batch_b).unwrap()
        })
    });

    g.throughput(Throughput::Elements(per_batch_bytes));
    g.bench_function(
        &format!("szs::LevenshteinDistances({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
                lev_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap()
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_lev_gpu.as_ref()) {
        g.throughput(Throughput::Elements(per_batch_bytes));
        g.bench_function("szs::LevenshteinDistances(1xGPU)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
                engine.compute(gpu, &batch_a, &batch_b).unwrap()
            })
        });
    }

    // StringZilla UTF-8 Levenshtein Distance (uniform costs: 0,1,1,1)
    g.throughput(Throughput::Elements(per_batch_utf8));
    g.bench_function("szs::LevenshteinDistancesUtf8(1xCPU)", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let (batch_a, batch_b) = make_batch_strs(pairs, &mut start_index, batch_size);
            lev_utf8_single
                .compute(&cpu_single, &batch_a, &batch_b)
                .unwrap()
        })
    });

    g.throughput(Throughput::Elements(per_batch_utf8));
    g.bench_function(
        &format!("szs::LevenshteinDistancesUtf8({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_strs(pairs, &mut start_index, batch_size);
                lev_utf8_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap()
            })
        },
    );
}

/// Linear gap cost benchmarks: NW/SW with linear penalties (match=2, mismatch=-1, open=-2, extend=-2)
fn perform_linear_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pairs: &[(&str, &str)],
    batch_size: usize,
    avg_cells_bytes: u64,
) {
    // No tapes needed - use simple string arrays

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
        let mut start_index = 0;
        b.iter(|| {
            let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
            nw_single.compute(&cpu_single, &batch_a, &batch_b).unwrap()
        })
    });

    g.throughput(Throughput::Elements(per_batch));
    g.bench_function(
        &format!("szs::NeedlemanWunschScores({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
                nw_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap()
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_nw_gpu.as_ref()) {
        g.throughput(Throughput::Elements(per_batch));
        g.bench_function("szs::NeedlemanWunschScores(1xGPU)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
                engine.compute(gpu, &batch_a, &batch_b).unwrap()
            })
        });
    }

    // Smith-Waterman (Local alignment)
    g.throughput(Throughput::Elements(per_batch));
    g.bench_function("szs::SmithWatermanScores(1xCPU)", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
            sw_single.compute(&cpu_single, &batch_a, &batch_b).unwrap()
        })
    });

    g.throughput(Throughput::Elements(per_batch));
    g.bench_function(
        &format!("szs::SmithWatermanScores({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
                sw_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap()
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_sw_gpu.as_ref()) {
        g.throughput(Throughput::Elements(per_batch));
        g.bench_function("szs::SmithWatermanScores(1xGPU)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
                engine.compute(gpu, &batch_a, &batch_b).unwrap()
            })
        });
    }
}

/// Affine gap cost benchmarks: NW/SW with affine penalties (match=2, mismatch=-1, open=-5, extend=-1)
fn perform_affine_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pairs: &[(&str, &str)],
    batch_size: usize,
    avg_cells_bytes: u64,
) {
    // No tapes needed - use simple string arrays

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
        let mut start_index = 0;
        b.iter(|| {
            let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
            nw_single.compute(&cpu_single, &batch_a, &batch_b).unwrap()
        })
    });

    g.throughput(Throughput::Elements(per_batch));
    g.bench_function(
        &format!("szs::NeedlemanWunschScores({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
                nw_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap()
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_nw_gpu.as_ref()) {
        g.throughput(Throughput::Elements(per_batch));
        g.bench_function("szs::NeedlemanWunschScores(1xGPU)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
                engine.compute(gpu, &batch_a, &batch_b).unwrap()
            })
        });
    }

    // Smith-Waterman (Local alignment)
    g.throughput(Throughput::Elements(per_batch));
    g.bench_function("szs::SmithWatermanScores(1xCPU)", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
            sw_single.compute(&cpu_single, &batch_a, &batch_b).unwrap()
        })
    });

    g.throughput(Throughput::Elements(per_batch));
    g.bench_function(
        &format!("szs::SmithWatermanScores({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
                sw_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap()
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_sw_gpu.as_ref()) {
        g.throughput(Throughput::Elements(per_batch));
        g.bench_function("szs::SmithWatermanScores(1xGPU)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let (batch_a, batch_b) = make_batch_bytes(pairs, &mut start_index, batch_size);
                engine.compute(gpu, &batch_a, &batch_b).unwrap()
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
