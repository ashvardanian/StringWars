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
- `STRINGWARS_BATCH`: Number of pairs to process in each batch (default: 1024).

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH=1024 \
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
use stringzilla::sz::{
    capabilities as sz_capabilities, dynamic_dispatch as sz_dynamic_dispatch, version as sz_version,
};

fn log_stringzilla_metadata() {
    let v = sz_version();
    println!("StringZilla v{}.{}.{}", v.major, v.minor, v.patch);
    println!("- uses dynamic dispatch: {}", sz_dynamic_dispatch());
    println!("- capabilities: {}", sz_capabilities().as_str());
}

fn log_stringzillas_metadata() {
    // Log information about the StringZilla szs module capabilities
    println!("StringZilla szs module:");
    println!("- DeviceScope support: CPU cores, GPU devices");
    println!("- Algorithms: LevenshteinDistances, LevenshteinDistancesUtf8");
    println!("- Scoring: NeedlemanWunschScores, SmithWatermanScores");
    println!("- Gap models: Uniform, Linear, Affine");
}

fn configure_bench() -> Criterion {
    Criterion::default()
        .sample_size(1000)
        .warm_up_time(std::time::Duration::from_secs(10))
        .measurement_time(std::time::Duration::from_secs(120))
}

fn bench_similarities(c: &mut Criterion) {
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_TOKENS").unwrap_or_else(|_| "lines".to_string());
    let content = fs::read_to_string(&dataset_path).expect("Could not read dataset");

    let batch_size = env::var("STRINGWARS_BATCH")
        .unwrap_or_else(|_| "1024".to_string())
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

    // Calculate average matrix size for throughput reporting
    let avg_matrix_size: u64 = pairs
        .iter()
        .map(|(a, b)| (a.len() * b.len()) as u64)
        .sum::<u64>()
        / pairs.len() as u64;

    // Uniform cost benchmarks (classic Levenshtein: match=0, mismatch=1, open=1, extend=1)
    let mut g = c.benchmark_group("uniform");
    g.throughput(Throughput::Elements(batch_size as u64 * avg_matrix_size));
    perform_uniform_benchmarks(&mut g, &pairs, batch_size);
    g.finish();

    // Linear gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-2, extend=-2)
    let mut g = c.benchmark_group("linear");
    g.throughput(Throughput::Elements(batch_size as u64 * avg_matrix_size));
    perform_linear_benchmarks(&mut g, &pairs, batch_size);
    g.finish();

    // Affine gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-5, extend=-1)
    let mut g = c.benchmark_group("affine");
    g.throughput(Throughput::Elements(batch_size as u64 * avg_matrix_size));
    perform_affine_benchmarks(&mut g, &pairs, batch_size);
    g.finish();
}

/// Uniform cost benchmarks: Classic Levenshtein distance (match=0, mismatch=1, open=1, extend=1)
fn perform_uniform_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pairs: &[(&str, &str)],
    batch_size: usize,
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
    let maybe_lev_utf8_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| LevenshteinDistancesUtf8::new(gpu, 0, 1, 1, 1).ok());

    // Note: Parallel RapidFuzz benchmarks removed for simplicity

    // RapidFuzz baselines
    g.bench_function("rapidfuzz::levenshtein<Bytes>(1xCPU)", |b| {
        let mut pair_index = 0;
        b.iter(|| {
            let mut results = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let (a, b_str) = pairs[pair_index % pairs.len()];
                results.push(levenshtein::distance(a.bytes(), b_str.bytes()));
                pair_index = (pair_index + 1) % pairs.len();
            }
            results
        })
    });

    g.bench_function("rapidfuzz::levenshtein<Chars>(1xCPU)", |b| {
        let mut pair_index = 0;
        b.iter(|| {
            let mut results = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let (a, b_str) = pairs[pair_index % pairs.len()];
                results.push(levenshtein::distance(a.chars(), b_str.chars()));
                pair_index = (pair_index + 1) % pairs.len();
            }
            results
        })
    });

    // StringZilla Binary Levenshtein Distance (uniform costs: 0,1,1,1)
    g.bench_function("szs::LevenshteinDistances(1xCPU)", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let end_index = (start_index + batch_size).min(pairs.len());
            let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                .iter()
                .map(|(a, _)| a.as_bytes())
                .collect();
            let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                .iter()
                .map(|(_, b)| b.as_bytes())
                .collect();
            let result = lev_single.compute(&cpu_single, &batch_a, &batch_b).unwrap();
            start_index = (start_index + batch_size) % pairs.len();
            result
        })
    });

    g.bench_function(
        &format!("szs::LevenshteinDistances({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_bytes())
                    .collect();
                let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_bytes())
                    .collect();
                let result = lev_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_lev_gpu.as_ref()) {
        g.bench_function("szs::LevenshteinDistances(1xGPU)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_bytes())
                    .collect();
                let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_bytes())
                    .collect();
                let result = engine.compute(gpu, &batch_a, &batch_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        });
    }

    // StringZilla UTF-8 Levenshtein Distance (uniform costs: 0,1,1,1)
    g.bench_function("szs::LevenshteinDistancesUtf8(1xCPU)", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let end_index = (start_index + batch_size).min(pairs.len());
            let batch_a: Vec<&str> = pairs[start_index..end_index]
                .iter()
                .map(|(a, _)| a.as_ref())
                .collect();
            let batch_b: Vec<&str> = pairs[start_index..end_index]
                .iter()
                .map(|(_, b)| b.as_ref())
                .collect();
            let result = lev_utf8_single
                .compute(&cpu_single, &batch_a, &batch_b)
                .unwrap();
            start_index = (start_index + batch_size) % pairs.len();
            result
        })
    });

    g.bench_function(
        &format!("szs::LevenshteinDistancesUtf8({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&str> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_ref())
                    .collect();
                let batch_b: Vec<&str> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_ref())
                    .collect();
                let result = lev_utf8_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_lev_utf8_gpu.as_ref()) {
        g.bench_function("szs::LevenshteinDistancesUtf8(1xGPU)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&str> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_ref())
                    .collect();
                let batch_b: Vec<&str> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_ref())
                    .collect();
                let result = engine.compute(gpu, &batch_a, &batch_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        });
    }
}

/// Linear gap cost benchmarks: NW/SW with linear penalties (match=2, mismatch=-1, open=-2, extend=-2)
fn perform_linear_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pairs: &[(&str, &str)],
    batch_size: usize,
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

    // Needleman-Wunsch (Global alignment)
    g.bench_function("szs::NeedlemanWunschScores(1xCPU)", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let end_index = (start_index + batch_size).min(pairs.len());
            let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                .iter()
                .map(|(a, _)| a.as_bytes())
                .collect();
            let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                .iter()
                .map(|(_, b)| b.as_bytes())
                .collect();
            let result = nw_single.compute(&cpu_single, &batch_a, &batch_b).unwrap();
            start_index = (start_index + batch_size) % pairs.len();
            result
        })
    });

    g.bench_function(
        &format!("szs::NeedlemanWunschScores({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_bytes())
                    .collect();
                let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_bytes())
                    .collect();
                let result = nw_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_nw_gpu.as_ref()) {
        g.bench_function("szs::NeedlemanWunschScores(1xGPU)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_bytes())
                    .collect();
                let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_bytes())
                    .collect();
                let result = engine.compute(gpu, &batch_a, &batch_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        });
    }

    // Smith-Waterman (Local alignment)
    g.bench_function("szs::SmithWatermanScores(1xCPU)", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let end_index = (start_index + batch_size).min(pairs.len());
            let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                .iter()
                .map(|(a, _)| a.as_bytes())
                .collect();
            let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                .iter()
                .map(|(_, b)| b.as_bytes())
                .collect();
            let result = sw_single.compute(&cpu_single, &batch_a, &batch_b).unwrap();
            start_index = (start_index + batch_size) % pairs.len();
            result
        })
    });

    g.bench_function(
        &format!("szs::SmithWatermanScores({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_bytes())
                    .collect();
                let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_bytes())
                    .collect();
                let result = sw_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_sw_gpu.as_ref()) {
        g.bench_function("szs::SmithWatermanScores(gpu)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_bytes())
                    .collect();
                let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_bytes())
                    .collect();
                let result = engine.compute(gpu, &batch_a, &batch_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        });
    }
}

/// Affine gap cost benchmarks: NW/SW with affine penalties (match=2, mismatch=-1, open=-5, extend=-1)
fn perform_affine_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pairs: &[(&str, &str)],
    batch_size: usize,
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

    // Needleman-Wunsch (Global alignment)
    g.bench_function("szs::NeedlemanWunschScores(1xCPU)", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let end_index = (start_index + batch_size).min(pairs.len());
            let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                .iter()
                .map(|(a, _)| a.as_bytes())
                .collect();
            let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                .iter()
                .map(|(_, b)| b.as_bytes())
                .collect();
            let result = nw_single.compute(&cpu_single, &batch_a, &batch_b).unwrap();
            start_index = (start_index + batch_size) % pairs.len();
            result
        })
    });

    g.bench_function(
        &format!("szs::NeedlemanWunschScores({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_bytes())
                    .collect();
                let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_bytes())
                    .collect();
                let result = nw_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_nw_gpu.as_ref()) {
        g.bench_function("szs::NeedlemanWunschScores(1xGPU)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_bytes())
                    .collect();
                let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_bytes())
                    .collect();
                let result = engine.compute(gpu, &batch_a, &batch_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        });
    }

    // Smith-Waterman (Local alignment)
    g.bench_function("szs::SmithWatermanScores(1xCPU)", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let end_index = (start_index + batch_size).min(pairs.len());
            let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                .iter()
                .map(|(a, _)| a.as_bytes())
                .collect();
            let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                .iter()
                .map(|(_, b)| b.as_bytes())
                .collect();
            let result = sw_single.compute(&cpu_single, &batch_a, &batch_b).unwrap();
            start_index = (start_index + batch_size) % pairs.len();
            result
        })
    });

    g.bench_function(
        &format!("szs::SmithWatermanScores({}xCPU)", num_cores),
        |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_bytes())
                    .collect();
                let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_bytes())
                    .collect();
                let result = sw_parallel
                    .compute(&cpu_parallel, &batch_a, &batch_b)
                    .unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        },
    );

    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_sw_gpu.as_ref()) {
        g.bench_function("szs::SmithWatermanScores(gpu)", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let batch_a: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(a, _)| a.as_bytes())
                    .collect();
                let batch_b: Vec<&[u8]> = pairs[start_index..end_index]
                    .iter()
                    .map(|(_, b)| b.as_bytes())
                    .collect();
                let result = engine.compute(gpu, &batch_a, &batch_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();
                result
            })
        });
    }
}

fn main() {
    log_stringzilla_metadata();
    log_stringzillas_metadata();
    let mut criterion = configure_bench();
    bench_similarities(&mut criterion);
    criterion.final_summary();
}
