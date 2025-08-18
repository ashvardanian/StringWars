#![doc = r#"
# StringWa.rs: String Hashing Benchmarks

This file contains benchmarks for various Rust hashing libraries using Criterion,
treating the inputs as binary strings without any UTF-8 validity constrains.
For accurate stats aggregation, on each iteration, the whole file is scanned.
Be warned, for large files, it may take a while!

The benchmarks compare the performance of different hash functions including:

- Standard `Hash` implementation
- StringZilla (`bytesum`, `hash`, and incremental `hash` function variants)
- aHash (both incremental and single-entry variants)
- xxHash (xxh3) through the third-party `xxhash-rust` crate
- gxhash (gxhash64)
- Blake3 (the only cryptographic hash in the comparison, for reference)

## Usage Examples

The benchmarks use two environment variables to control the input dataset and mode:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line.
  - `words`: Process the dataset word by word.
  - `file`: Process the entire file as a single token.

To run the benchmarks with the appropriate CPU features enabled, you can use the following commands:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_hash bench_hash --jobs 8
```

For `gxhash`, ensure that your CPU supports the required AES and SSE2 instructions.
"#]
use std::env;
use std::error::Error;
use std::fs;
use std::hint::black_box;

use criterion::{Criterion, Throughput};

use ahash::{AHasher, RandomState};
use blake3;
use gxhash;
use std::hash::{BuildHasher, Hasher};
use stringzilla::sz;
use xxhash_rust::xxh3::xxh3_64;

/// Loads the dataset from the file specified by the `STRINGWARS_DATASET` environment variable.
pub fn load_dataset() -> Result<Vec<u8>, Box<dyn Error>> {
    let dataset_path = env::var("STRINGWARS_DATASET")
        .map_err(|_| "STRINGWARS_DATASET environment variable not set")?;
    let content = fs::read(&dataset_path)?;
    Ok(content)
}

/// Tokenizes the given haystack based on the `STRINGWARS_TOKENS` environment variable.
/// Supported modes: "lines", "words", and "file".
pub fn tokenize<'a>(haystack: &'a [u8]) -> Result<Vec<&'a [u8]>, Box<dyn Error>> {
    let mode = env::var("STRINGWARS_TOKENS").unwrap_or_else(|_| "lines".to_string());
    let tokens = match mode.as_str() {
        "lines" => haystack.split(|&c| c == b'\n').collect(),
        "words" => haystack.split(|&c| c == b'\n' || c == b' ').collect(),
        "file" => vec![haystack],
        other => {
            return Err(format!(
                "Unknown STRINGWARS_TOKENS: {}. Use 'lines', 'words', or 'file'.",
                other
            )
            .into())
        }
    };
    Ok(tokens)
}

/// Benchmarks stateless hashes seeing the whole input at once
fn bench_stateless(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &[&[u8]],
) {
    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = tokens.iter().map(|u| u.len()).sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Benchmark: StringZilla `bytesum`
    group.bench_function("stringzilla::bytesum", |b| {
        b.iter(|| {
            for token in tokens {
                // Using black_box to prevent compiler optimizations.
                let _hash = sz::bytesum(black_box(token));
            }
        })
    });

    // Benchmark: StringZilla `hash`
    group.bench_function("stringzilla::hash", |b| {
        b.iter(|| {
            for token in tokens {
                let _hash = sz::hash(black_box(token));
            }
        })
    });

    // Benchmark: SipHash via `std::hash::BuildHasher`
    group.bench_function("std::hash::BuildHasher", |b| {
        let std_builder = std::collections::hash_map::RandomState::new();
        b.iter(|| {
            for token in tokens {
                let mut hasher = std_builder.build_hasher();
                hasher.write(token);
                let _hash = black_box(hasher.finish());
            }
        })
    });

    // Benchmark: aHash (`hash_one`)
    group.bench_function("aHash::hash_one", |b| {
        let hash_builder = RandomState::with_seed(42);
        b.iter(|| {
            for token in tokens {
                let _hash = black_box(hash_builder.hash_one(token));
            }
        })
    });

    // Benchmark: xxHash (`xxh3`)
    group.bench_function("xxh3::xxh3_64", |b| {
        b.iter(|| {
            for token in tokens {
                let _hash = black_box(xxh3_64(token));
            }
        })
    });

    // Benchmark: gxhash
    group.bench_function("gxhash::gxhash64", |b| {
        b.iter(|| {
            for token in tokens {
                let _hash = black_box(gxhash::gxhash64(token, 42));
            }
        })
    });

    // Benchmark: Blake3 - should be by far the slowest, as it's a cryptographic hash.
    group.bench_function("blake3", |b| {
        b.iter(|| {
            for token in tokens {
                let _hash = black_box(blake3::hash(token));
            }
        })
    });
}

/// Benchmarks stateful hashes seeing one slice at a time
fn bench_stateful(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &[&[u8]],
) {
    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = tokens.iter().map(|u| u.len()).sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Benchmark: StringZilla `bytesum`
    group.bench_function("stringzilla::bytesum", |b| {
        b.iter(|| {
            let mut aggregate = 0u64;
            for token in tokens {
                aggregate += sz::bytesum(token);
            }
            black_box(aggregate);
        })
    });

    // Benchmark: StringZilla `hash`
    group.bench_function("stringzilla::HashState", |b| {
        b.iter(|| {
            let mut aggregate = sz::HashState::new(0);
            for token in tokens {
                aggregate.stream(token);
            }
            black_box(aggregate.fold());
        })
    });

    // Benchmark: SipHash via `std::hash::BuildHasher`
    group.bench_function("std::hash::BuildHasher", |b| {
        let std_builder = std::collections::hash_map::RandomState::new();
        b.iter(|| {
            let mut aggregate = std_builder.build_hasher();
            for token in tokens {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
        })
    });

    // Benchmark: aHash (`hash_one`)
    group.bench_function("aHash::AHasher", |b| {
        b.iter(|| {
            let mut aggregate = AHasher::default();
            for token in tokens {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
        })
    });
}

fn main() {
    // Log StringZilla metadata
    let v = sz::version();
    println!("StringZilla v{}.{}.{}", v.major, v.minor, v.patch);
    println!("- uses dynamic dispatch: {}", sz::dynamic_dispatch());
    println!("- capabilities: {}", sz::capabilities().as_str());

    // Load the dataset defined by the environment variables, and panic if the content is missing
    let dataset = load_dataset().unwrap();
    let tokens = tokenize(&dataset).unwrap();
    if tokens.is_empty() {
        panic!("No tokens found in the dataset.");
    }

    let mut criterion = Criterion::default()
        .configure_from_args()
        .sample_size(30) // Number of samples to collect.
        .warm_up_time(std::time::Duration::from_secs(5)) // Let CPU frequencies settle.
        .measurement_time(std::time::Duration::from_secs(10)); // Actual measurement time.

    // Profile hash functions that see the whole input at once
    let mut group = criterion.benchmark_group("stateful");
    bench_stateful(&mut group, &tokens);
    group.finish();

    // Profile incremental hash functions that see only a slice of data at a time
    let mut group = criterion.benchmark_group("stateless");
    bench_stateless(&mut group, &tokens);
    group.finish();

    criterion.final_summary();
}
