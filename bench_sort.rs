//! # Sorting Benchmarks
//!
//! This file benchmarks the performance of three different sorting routines for
//! arrays of strings:
//!
//! - `sz::sort` from the StringZilla library
//! - The standard library’s `sort_unstable`
//! - Rayon’s parallel sort (`par_sort_unstable`)
//!
//! ## Environment Variables
//!
//! The benchmarks use two environment variables to control the input dataset and mode:
//!
//! - `STRINGWARS_DATASET`: Path to the input dataset file.
//! - `STRINGWARS_MODE`: Specifies how to interpret the input. Allowed values:
//!   - `lines`: Process the dataset line by line.
//!   - `words`: Process the dataset word by word.
//!   - `file`: Process the entire file as a single unit.
//!
//! ## Usage Example
//!
//! ```sh
//! STRINGWARS_MODE=lines STRINGWARS_DATASET=path/to/dataset cargo bench --bench bench_sort
//! ```

use std::env;
use std::fs;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

// Import the specialized sort from StringZilla. It is assumed that `sz::sort`
// sorts a mutable slice of `String` in place.
use stringzilla::sz::sort as sz_sort;

fn load_data() -> Vec<String> {
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_MODE").unwrap_or_else(|_| "lines".to_string());

    let content = fs::read_to_string(&dataset_path).expect("Could not read dataset");
    let data: Vec<String> = match mode.as_str() {
        "lines" => content.lines().map(|line| line.to_string()).collect(),
        "words" => content
            .split_whitespace()
            .map(|word| word.to_string())
            .collect(),
        "file" => vec![content],
        other => panic!(
            "Unknown STRINGWARS_MODE: {}. Use 'lines', 'words', or 'file'.",
            other
        ),
    };
    data
}

fn bench_sort(c: &mut Criterion) {
    // Load the dataset once; each benchmark iteration will clone this unsorted data.
    let unsorted = load_data();

    if unsorted.is_empty() {
        panic!("No data found in dataset for sorting benchmark.");
    }

    let mut group = c.benchmark_group("sorting");

    // Benchmark: Specialized sort from StringZilla.
    group.bench_function("sz::sort", |b| {
        b.iter(|| {
            // Clone to ensure each sort works on an unsorted vector.
            let mut data = unsorted.clone();
            // Perform the specialized sort.
            sz_sort(black_box(&mut data));
            black_box(&data);
        })
    });

    // Benchmark: Standard library sort_unstable.
    group.bench_function("std::sort_unstable", |b| {
        b.iter(|| {
            let mut data = unsorted.clone();
            data.sort_unstable();
            black_box(&data);
        })
    });

    // Benchmark: Rayon parallel sort_unstable.
    group.bench_function("rayon::par_sort_unstable", |b| {
        b.iter(|| {
            let mut data = unsorted.clone();
            // Parallel sort requires the `rayon` crate and the ParallelSliceMut trait.
            data.par_sort_unstable();
            black_box(&data);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_sort);
criterion_main!(benches);
