#![doc = r#"# StringWa.rs: String Similarity Benchmarks

This file benchmarks different libraries implementing string alignment and edit
distance calculation, for both generic Levenshtein distances and the weighted
Needleman-Wunsch alignment scores used in Bioinformatics.

The input file is tokenized into lines or words and each consecutive pair of tokens
is evaluated for similarity. As most algorithms have quadratic complexity and use
Dynamic Programming techniques, their throughput is evaluate in the number of CUPS,
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
    cargo criterion --features bench_similarity bench_similarity --jobs 8
```
"#]
use std::env;
use std::fs;

use criterion::{Criterion, Throughput};
use fork_union::count_logical_cores;

use rapidfuzz::distance::levenshtein;
use stringzilla::szs::{
    DeviceScope, LevenshteinDistances, LevenshteinDistancesUtf8, NeedlemanWunschScores,
    SmithWatermanScores, UnifiedAlloc,
};
use stringtape::{BytesTape, StringTape};

use stringzilla::sz::{
    // Pull some metadata logging functionality
    capabilities as sz_capabilities,
    dynamic_dispatch as sz_dynamic_dispatch,
    error_costs_256x256_unary,
    version as sz_version,
};

fn log_stringzilla_metadata() {
    let v = sz_version();
    println!("StringZilla v{}.{}.{}", v.major, v.minor, v.patch);
    println!("- uses dynamic dispatch: {}", sz_dynamic_dispatch());
    println!("- capabilities: {}", sz_capabilities().as_str());
}

fn configure_bench() -> Criterion {
    Criterion::default()
        .sample_size(1000)
        .warm_up_time(std::time::Duration::from_secs(10))
        .measurement_time(std::time::Duration::from_secs(120))
}

fn bench_levenshtein(c: &mut Criterion) {
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

    // Binary benchmarks (byte-based edit distance)
    let mut g = c.benchmark_group("binary");
    g.throughput(Throughput::Elements(batch_size as u64 * avg_matrix_size));
    perform_binary_benchmarks(&mut g, &pairs, batch_size);
    g.finish();

    // UTF-8 benchmarks (Unicode-aware edit distance)
    let mut g = c.benchmark_group("utf8");
    g.throughput(Throughput::Elements(batch_size as u64 * avg_matrix_size));
    perform_utf8_benchmarks(&mut g, &pairs, batch_size);
    g.finish();
}

fn perform_binary_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pairs: &[(&str, &str)],
    batch_size: usize,
) {
    // Pre-populate BytesTape with first and second elements from pairs
    let mut tape_a = BytesTape::<i64, UnifiedAlloc>::new_in(UnifiedAlloc);
    let mut tape_b = BytesTape::<i64, UnifiedAlloc>::new_in(UnifiedAlloc);

    tape_a
        .extend(pairs.iter().map(|(a, _)| a.as_bytes()))
        .expect("Failed to extend tape_a");
    tape_b
        .extend(pairs.iter().map(|(_, b)| b.as_bytes()))
        .expect("Failed to extend tape_b");

    // Create engine and device scope before iteration
    let num_cores = count_logical_cores();
    let cpu = DeviceScope::cpu_cores(num_cores).expect("Failed to create device scope");
    let maybe_gpu = DeviceScope::gpu_device(0);

    // StringZilla Levenshtein Distance (CPU)
    {
        g.bench_function("stringzillas::LevenshteinDistances(CPUs)", |b| {
            let engine = LevenshteinDistances::new(&cpu, 0, 1, 1, 1)
                .expect("Failed to create StringZilla LevenshteinDistances");
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let subview_a = tape_a.subview(start_index, end_index).unwrap();
                let subview_b = tape_b.subview(start_index, end_index).unwrap();
                let result = engine.compute(&cpu, &subview_a, &subview_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();

                result
            })
        });
    }

    // StringZilla Levenshtein Distance (GPU)
    if let Ok(gpu) = &maybe_gpu {
        g.bench_function("stringzillas::LevenshteinDistances(GPUs)", |b| {
            let engine = LevenshteinDistances::new(gpu, 0, 1, 1, 1)
                .expect("Failed to create StringZilla LevenshteinDistances");
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let subview_a = tape_a.subview(start_index, end_index).unwrap();
                let subview_b = tape_b.subview(start_index, end_index).unwrap();
                let result = engine.compute(gpu, &subview_a, &subview_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();

                result
            })
        });
    }

    // StringZilla Needleman-Wunsch Scores (CPU)
    {
        let matrix = error_costs_256x256_unary();
        g.bench_function("stringzillas::NeedlemanWunschScores(CPUs)", |b| {
            let engine = NeedlemanWunschScores::new(&cpu, &matrix, -2, -1)
                .expect("Failed to create StringZilla NeedlemanWunschScores");
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let subview_a = tape_a.subview(start_index, end_index).unwrap();
                let subview_b = tape_b.subview(start_index, end_index).unwrap();
                let result = engine.compute(&cpu, &subview_a, &subview_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();

                result
            })
        });
    }

    // StringZilla Needleman-Wunsch Scores (GPU)
    if let Ok(gpu) = &maybe_gpu {
        let matrix = error_costs_256x256_unary();
        g.bench_function("stringzillas::NeedlemanWunschScores(GPUs)", |b| {
            let engine = NeedlemanWunschScores::new(gpu, &matrix, -2, -1)
                .expect("Failed to create StringZilla NeedlemanWunschScores");
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let subview_a = tape_a.subview(start_index, end_index).unwrap();
                let subview_b = tape_b.subview(start_index, end_index).unwrap();
                let result = engine.compute(gpu, &subview_a, &subview_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();

                result
            })
        });
    }

    // StringZilla Smith-Waterman Scores (CPU)
    {
        let matrix = error_costs_256x256_unary();
        g.bench_function("stringzillas::SmithWatermanScores(CPUs)", |b| {
            let engine = SmithWatermanScores::new(&cpu, &matrix, -2, -1)
                .expect("Failed to create StringZilla SmithWatermanScores");
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let subview_a = tape_a.subview(start_index, end_index).unwrap();
                let subview_b = tape_b.subview(start_index, end_index).unwrap();
                let result = engine.compute(&cpu, &subview_a, &subview_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();

                result
            })
        });
    }

    // StringZilla Smith-Waterman Scores (GPU)
    if let Ok(gpu) = &maybe_gpu {
        let matrix = error_costs_256x256_unary();
        g.bench_function("stringzillas::SmithWatermanScores(GPUs)", |b| {
            let engine = SmithWatermanScores::new(gpu, &matrix, -2, -1)
                .expect("Failed to create StringZilla SmithWatermanScores");
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let subview_a = tape_a.subview(start_index, end_index).unwrap();
                let subview_b = tape_b.subview(start_index, end_index).unwrap();
                let result = engine.compute(gpu, &subview_a, &subview_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();

                result
            })
        });
    }

    // RapidFuzz Levenshtein (parallelized with Fork Union)
    {
        let pool = fork_union::spawn(num_cores);
        let mut results = vec![0usize; batch_size];
        
        g.bench_function("rapidfuzz::levenshtein(binary)", |b| {
            let mut pair_index = 0;
            b.iter(|| {
                let mut batch_pairs = Vec::with_capacity(batch_size);
                for _ in 0..batch_size {
                    let (a, b_str) = pairs[pair_index % pairs.len()];
                    batch_pairs.push((a, b_str));
                    pair_index = (pair_index + 1) % pairs.len();
                }

                pool.for_n(batch_size, |i| {
                    let idx = i.task_index;
                    let (a, b_str) = batch_pairs[idx];
                    results[idx] = levenshtein::distance(a.bytes(), b_str.bytes());
                });
                &results
            })
        });
    }
}

fn perform_utf8_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pairs: &[(&str, &str)],
    batch_size: usize,
) {
    // Pre-populate StringTape with first and second elements from pairs
    let mut tape_a = StringTape::<i64, UnifiedAlloc>::new_in(UnifiedAlloc);
    let mut tape_b = StringTape::<i64, UnifiedAlloc>::new_in(UnifiedAlloc);

    tape_a
        .extend(pairs.iter().map(|(a, _)| *a))
        .expect("Failed to extend tape_a");
    tape_b
        .extend(pairs.iter().map(|(_, b)| *b))
        .expect("Failed to extend tape_b");

    // Create engine and device scope before iteration
    let num_cores = count_logical_cores();
    let cpu = DeviceScope::cpu_cores(num_cores).expect("Failed to create device scope");
    let maybe_gpu = DeviceScope::gpu_device(0);

    // StringZilla UTF-8 Levenshtein Distance (CPU)
    {
        g.bench_function("stringzillas::LevenshteinDistancesUtf8(CPUs)", |b| {
            let engine = LevenshteinDistancesUtf8::new(&cpu, 0, 1, 1, 1)
                .expect("Failed to create StringZilla LevenshteinDistancesUtf8");
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let subview_a = tape_a.subview(start_index, end_index).unwrap();
                let subview_b = tape_b.subview(start_index, end_index).unwrap();
                let result = engine.compute(&cpu, &subview_a, &subview_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();

                result
            })
        });
    }

    // StringZilla UTF-8 Levenshtein Distance (GPU)
    if let Ok(gpu) = &maybe_gpu {
        g.bench_function("stringzillas::LevenshteinDistancesUtf8(GPUs)", |b| {
            let engine = LevenshteinDistancesUtf8::new(gpu, 0, 1, 1, 1)
                .expect("Failed to create StringZilla LevenshteinDistancesUtf8");
            let mut start_index = 0;
            b.iter(|| {
                let end_index = (start_index + batch_size).min(pairs.len());
                let subview_a = tape_a.subview(start_index, end_index).unwrap();
                let subview_b = tape_b.subview(start_index, end_index).unwrap();
                let result = engine.compute(gpu, &subview_a, &subview_b).unwrap();
                start_index = (start_index + batch_size) % pairs.len();

                result
            })
        });
    }

    // RapidFuzz UTF-8 Levenshtein (parallelized with Fork Union)
    {
        let pool = fork_union::spawn(num_cores);
        let mut results = vec![0usize; batch_size];
        
        g.bench_function("rapidfuzz::levenshtein(utf8)", |b| {
            let mut pair_index = 0;
            b.iter(|| {
                let mut batch_pairs = Vec::with_capacity(batch_size);
                for _ in 0..batch_size {
                    let (a, b_str) = pairs[pair_index % pairs.len()];
                    batch_pairs.push((a, b_str));
                    pair_index = (pair_index + 1) % pairs.len();
                }

                pool.for_n(batch_size, |i| {
                    let idx = i.task_index;
                    let (a, b_str) = batch_pairs[idx];
                    results[idx] = levenshtein::distance(a.chars(), b_str.chars());
                });
                &results
            })
        });
    }
}

fn main() {
    log_stringzilla_metadata();
    let mut criterion = configure_bench();
    bench_levenshtein(&mut criterion);
    criterion.final_summary();
}
