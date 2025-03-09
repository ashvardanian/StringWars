//! # StringWa.rs Search Benchmarks
//!
//! This file benchmarks the forward and reverse search functionality provided by
//! the StringZilla library and the memchr crate. The benchmarks read an input file
//! (specified by the `STRINGWARS_DATASET` environment variable), tokenize its contents
//! by whitespace into search needles, and then run forward and reverse search benchmarks.
//!
//! ## Usage
//!
//! Set the environment variable `STRINGWARS_DATASET` to the path of your input file.
//! Then run the benchmarks with:
//!
//! ```sh
//! STRINGWARS_DATASET=<path_to_dataset> cargo bench --features bench_search
//! ```
//!
//! ## Library Metadata
//!
//! Before running the benchmarks, this binary logs the StringZilla metadata (version,
//! dynamic dispatch status, and capabilities) so that you can verify that the library
//! is configured correctly for your CPU.
//!
use std::env;
use std::fs;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use memchr::memmem;
use stringzilla::sz::{
    capabilities as sz_capabilities, //
    dynamic_dispatch as sz_dynamic_dispatch,
    version as sz_version,
};

fn log_stringzilla_metadata() {
    let sz_v = sz_version();
    println!(
        "StringZilla version: {}.{}.{}",
        sz_v.major, sz_v.minor, sz_v.patch
    );
    println!(
        "StringZilla uses dynamic dispatch: {}",
        sz_dynamic_dispatch()
    );
    println!("StringZilla capabilities: {}", sz_capabilities().as_str());
}

fn configure_bench() -> Criterion {
    Criterion::default()
        .sample_size(1000) // Test this many needles.
        .warm_up_time(std::time::Duration::from_secs(10)) // Let the CPU frequencies settle.
        .measurement_time(std::time::Duration::from_secs(120)) // Actual measurement time.
}

fn bench_find(c: &mut Criterion) {
    // Get the haystack path from the environment variable.
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let haystack_content = fs::read_to_string(&dataset_path).expect("Could not read haystack");

    // Tokenize the haystack content by white space.
    let needles: Vec<&str> = haystack_content.split_whitespace().collect();
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
    // Benchmark for StringZilla forward search
    let mut token_index: usize = 0;
    g.bench_function("stringzilla::find", |b| {
        b.iter(|| {
            let token = needles[token_index];
            let token_bytes = token.as_bytes();
            let mut pos: usize = 0;
            while let Some(found) = (&haystack[pos..]).sz_find(token_bytes) {
                pos += found + token_bytes.len();
            }
            token_index = (token_index + 1) % needles.len();
        })
    });

    // Benchmark for memchr (forward search)
    let mut token_index: usize = 0; // Reset token index for the next benchmark
    g.bench_function("memmem::find", |b| {
        b.iter(|| {
            let token = needles[token_index];
            let token_bytes = token.as_bytes();
            let mut pos: usize = 0;
            while let Some(found) = memmem::find(&haystack[pos..], token_bytes) {
                pos += found + token_bytes.len();
            }
            token_index = (token_index + 1) % needles.len();
        })
    });
}

fn perform_reverse_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    needles: &[&str],
    haystack: &[u8],
) {
    // Benchmark for StringZilla reverse search
    let mut token_index: usize = 0;
    g.bench_function("stringzilla::rfind", |b| {
        b.iter(|| {
            let token = needles[token_index];
            let token_bytes = token.as_bytes();
            let mut pos: Option<usize> = Some(haystack.len());
            while let Some(end) = pos {
                if let Some(found) = (&haystack[..end]).sz_rfind(token_bytes) {
                    pos = Some(found); // Update position to the start of the found token for the next search.
                } else {
                    break; // No more occurrences found.
                }
            }
            token_index = (token_index + 1) % needles.len();
        })
    });

    // Benchmark for memchr reverse search
    let mut token_index: usize = 0;
    g.bench_function("memmem::rfind", |b| {
        b.iter(|| {
            let token = needles[token_index];
            let token_bytes = token.as_bytes();
            let mut pos: Option<usize> = Some(haystack.len());
            while let Some(end) = pos {
                if let Some(found) = memmem::rfind(&haystack[..end], token_bytes) {
                    pos = Some(found); // Update position to the start of the found token for the next search.
                } else {
                    break; // No more occurrences found.
                }
            }
            token_index = (token_index + 1) % needles.len();
        })
    });
}

fn main() {
    log_stringzilla_metadata();
    let mut criterion = configure_bench();
    bench_find(&mut criterion);
    criterion.final_summary();
}
