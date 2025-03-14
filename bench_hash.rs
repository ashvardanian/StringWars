//! # StringWa.rs: String Hashing Benchmarks
//!
//! This file contains benchmarks for various Rust hashing libraries using Criterion.
//!
//! The benchmarks compare the performance of different hash functions including:
//!
//! - Standard `Hash` implementation
//! - StringZilla (`bytesum`, `hash`, and incremental `hash` function variants)
//! - aHash (both incremental and single-entry variants)
//! - xxHash (xxh3) through the third-party `xxhash-rust` crate
//! - gxhash (gxhash64)
//! - Blake3 (the only cryptographic hash in the comparison, for reference)
//!
//! ## Usage Examples
//!
//! The benchmarks use two environment variables to control the input dataset and mode:
//!
//! - `STRINGWARS_DATASET`: Path to the input dataset file.
//! - `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
//!   - `lines`: Process the dataset line by line.
//!   - `words`: Process the dataset word by word.
//!   - `file`: Process the entire file as a single unit.
//!
//! To run the benchmarks with the appropriate CPU features enabled, you can use the following commands:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" \
//!     STRINGWARS_DATASET=README.md \
//!     STRINGWARS_TOKENS=lines \
//!     cargo criterion --features bench_hash bench_hash --jobs 8
//! ```
//!
//! ## Notes
//!
//! - Ensure your CPU supports the required AES and SSE2 instructions when using `gxhash`.
use std::env;
use std::fs;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use ahash::RandomState;
use blake3;
use gxhash;
use std::hash::{BuildHasher, Hasher};
use stringzilla::sz::{bytesum as sz_bytesum, hash as sz_hash};
use xxhash_rust::xxh3::xxh3_64;

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
        .warm_up_time(std::time::Duration::from_secs(5)) // Let CPU frequencies settle.
        .measurement_time(std::time::Duration::from_secs(10)) // Actual measurement time.
}

fn bench_hash(c: &mut Criterion) {
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_TOKENS").unwrap_or_else(|_| "lines".to_string());

    let content = fs::read_to_string(&dataset_path).expect("Could not read dataset");
    let units: Vec<&str> = match mode.as_str() {
        "lines" => content.lines().collect(),
        "words" => content.split_whitespace().collect(),
        "file" => {
            // In "file" mode, treat the entire content as a single unit.
            vec![&content]
        }
        other => panic!(
            "Unknown STRINGWARS_TOKENS: {}. Use 'lines', 'words', or 'file'.",
            other
        ),
    };

    if units.is_empty() {
        panic!("No data found for hashing in the provided dataset.");
    }

    // Calculate total bytes processed for throughput reporting.
    let total_bytes: usize = units.iter().map(|u| u.len()).sum();
    let mut g = c.benchmark_group("stateless");
    g.throughput(Throughput::Bytes(total_bytes as u64));
    stateless_benchmarks(&mut g, &units);
    g.finish();
}

fn stateless_benchmarks(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    units: &[&str],
) {
    // Benchmark: StringZilla `bytesum`
    group.bench_function("stringzilla::bytesum", |b| {
        b.iter(|| {
            for unit in units {
                // Using black_box to prevent compiler optimizations.
                let _hash = sz_bytesum(black_box(unit.as_bytes()));
            }
        })
    });

    // Benchmark: StringZilla `hash`
    group.bench_function("stringzilla::hash", |b| {
        b.iter(|| {
            for unit in units {
                let _hash = sz_hash(black_box(unit.as_bytes()));
            }
        })
    });

    // Benchmark: SipHash via `std::hash::BuildHasher`
    group.bench_function("std::hash::BuildHasher", |b| {
        let std_builder = std::collections::hash_map::RandomState::new();
        b.iter(|| {
            for unit in units {
                let mut hasher = std_builder.build_hasher();
                hasher.write(unit.as_bytes());
                let _hash = black_box(hasher.finish());
            }
        })
    });

    // Benchmark: aHash (`hash_one`)
    group.bench_function("aHash::hash_one", |b| {
        let hash_builder = RandomState::with_seed(42);
        b.iter(|| {
            for unit in units {
                let _hash = black_box(hash_builder.hash_one(unit.as_bytes()));
            }
        })
    });

    // Benchmark: xxHash (`xxh3`)
    group.bench_function("xxh3::xxh3_64", |b| {
        b.iter(|| {
            for unit in units {
                let _hash = black_box(xxh3_64(unit.as_bytes()));
            }
        })
    });

    // Benchmark: gxhash
    group.bench_function("gxhash::gxhash64", |b| {
        b.iter(|| {
            for unit in units {
                let _hash = black_box(gxhash::gxhash64(unit.as_bytes(), 42));
            }
        })
    });

    // Benchmark: Blake3 - should be by far the slowest, as it's a cryptographic hash.
    group.bench_function("blake3", |b| {
        b.iter(|| {
            for unit in units {
                let _hash = black_box(blake3::hash(unit.as_bytes()));
            }
        })
    });
}

fn main() {
    log_stringzilla_metadata();
    let mut criterion = Criterion::default().configure_from_args();
    bench_hash(&mut criterion);
    criterion.final_summary();
}
