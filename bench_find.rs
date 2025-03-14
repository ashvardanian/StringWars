//! # StringWa.rs: Substring Search Benchmarks
//!
//! This file benchmarks the forward and reverse exact substring search functionality provided by
//! the StringZilla library and the memchr crate. The input file is treated as a haystack and all
//! of its tokens as needles. The throughput numbers are reported in Gigabytes per Second and for
//! any sampled token - all of its inclusions in a string are located.
//!
//! ## Usage Examples
//!
//! The benchmarks use two environment variables to control the input dataset and mode:
//!
//! - `STRINGWARS_DATASET`: Path to the input dataset file.
//! - `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
//!   - `lines`: Process the dataset line by line.
//!   - `words`: Process the dataset word by word.
//!
//! To run the benchmarks with the appropriate CPU features enabled, you can use the following commands:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" \
//!     STRINGWARS_DATASET=README.md \
//!     STRINGWARS_TOKENS=lines \
//!     cargo criterion --features bench_find bench_find --jobs 8
//! ```
use std::env;
use std::fs;
use std::time::Duration;

use criterion::{black_box, Criterion, Throughput};

use memchr::memmem;
use stringzilla::sz::{find as sz_find, rfind as sz_rfind};

use stringzilla::sz::{
    // Pull some metadata logging functionality
    capabilities as sz_capabilities,
    dynamic_dispatch as sz_dynamic_dispatch,
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
        .sample_size(1000) // Test this many needles.
        .warm_up_time(Duration::from_secs(10)) // Let the CPU frequencies settle.
        .measurement_time(Duration::from_secs(120)) // Actual measurement time.
}

fn bench_find(c: &mut Criterion) {
    // Get the haystack path from the environment variable.
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_TOKENS").unwrap_or_else(|_| "lines".to_string());
    let haystack_content = fs::read_to_string(&dataset_path).expect("Could not read haystack");

    // Tokenize the haystack content by white space or lines.
    let needles: Vec<&str> = match mode.as_str() {
        "lines" => haystack_content.lines().collect(),
        "words" => haystack_content.split_whitespace().collect(),
        other => panic!(
            "Unknown STRINGWARS_TOKENS: {}. Use 'lines' or 'words'.",
            other
        ),
    };

    if needles.is_empty() {
        panic!("No tokens found in the haystack.");
    }

    let haystack = haystack_content.as_bytes();
    let haystack_length = haystack.len();

    // Benchmarks for forward search
    let mut g = c.benchmark_group("search-forward");
    g.throughput(Throughput::Bytes(haystack_length as u64));
    perform_forward_benchmarks(&mut g, &needles, haystack);
    g.finish();

    // Benchmarks for reverse search
    let mut g = c.benchmark_group("search-reverse");
    g.throughput(Throughput::Bytes(haystack_length as u64));
    perform_reverse_benchmarks(&mut g, &needles, haystack);
    g.finish();
}

fn perform_forward_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    needles: &[&str],
    haystack: &[u8],
) {
    // Benchmark for StringZilla forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("sz::find", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let token_bytes = black_box(token.as_bytes());
            let mut pos: usize = 0;
            while let Some(found) = sz_find(&haystack[pos..], token_bytes) {
                pos += found + token_bytes.len();
            }
        })
    });

    // Benchmark for `memmem` forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("memmem::find", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let token_bytes = black_box(token.as_bytes());
            let mut pos: usize = 0;
            while let Some(found) = memmem::find(&haystack[pos..], token_bytes) {
                pos += found + token_bytes.len();
            }
        })
    });

    // Benchmark for default `std::str` forward search.
    let mut tokens = needles.iter().cycle();
    g.bench_function("std::str::find", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let mut pos = 0;
            while let Some(found) = haystack_str[pos..].find(token) {
                pos += found + token.len();
            }
        })
    });
}

fn perform_reverse_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    needles: &[&str],
    haystack: &[u8],
) {
    // Benchmark for StringZilla reverse search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("sz::rfind", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let token_bytes = black_box(token.as_bytes());
            let mut pos: Option<usize> = Some(haystack.len());
            while let Some(end) = pos {
                if let Some(found) = sz_rfind(&haystack[..end], token_bytes) {
                    pos = Some(found);
                } else {
                    break;
                }
            }
        })
    });

    // Benchmark for memmem reverse search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("memmem::rfind", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let token_bytes = black_box(token.as_bytes());
            let mut pos: Option<usize> = Some(haystack.len());
            while let Some(end) = pos {
                if let Some(found) = memmem::rfind(&haystack[..end], token_bytes) {
                    pos = Some(found);
                } else {
                    break;
                }
            }
        })
    });

    // Benchmark for default `std::str` reverse search.
    let mut tokens = needles.iter().cycle();
    g.bench_function("std::str::rfind", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let mut pos: Option<usize> = Some(haystack_str.len());
            while let Some(end) = pos {
                if let Some(found) = haystack_str[..end].rfind(token) {
                    pos = Some(found);
                } else {
                    break;
                }
            }
        })
    });
}

fn main() {
    log_stringzilla_metadata();
    let mut criterion = configure_bench();
    bench_find(&mut criterion);
    criterion.final_summary();
}
