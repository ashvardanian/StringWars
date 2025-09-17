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
- CRC32 (IEEE) via `crc32fast`
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
    cargo criterion --features bench_hash bench_hash --jobs $(nproc)
```

For `gxhash`, ensure that your CPU supports the required AES and SSE2 instructions.
"#]
use std::env;
use std::error::Error;
use std::fs;
use std::hash::{BuildHasher, Hasher};
use std::hint::black_box;
use std::io::Cursor;

use criterion::{Criterion, Throughput};

use ahash::RandomState;
use blake3;
use cityhash;
use crc32fast;
use gxhash;
use murmur3;
use stringzilla::sz;
use xxhash_rust::xxh3::xxh3_64;

type AHashState = RandomState;

fn log_stringzilla_metadata() {
    let v = sz::version();
    println!("StringZilla v{}.{}.{}", v.major, v.minor, v.patch);
    println!("- uses dynamic dispatch: {}", sz::dynamic_dispatch());
    println!("- capabilities: {}", sz::capabilities().as_str());
}

fn configure_bench() -> Criterion {
    Criterion::default()
        .configure_from_args()
        .sample_size(30) // Number of samples to collect.
        .warm_up_time(std::time::Duration::from_secs(5)) // Let CPU frequencies settle.
        .measurement_time(std::time::Duration::from_secs(10)) // Actual measurement time.
}

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
    let tokens: Vec<&[u8]> = match mode.as_str() {
        "lines" => haystack
            .split(|&c| c == b'\n')
            .filter(|t| !t.is_empty())
            .collect(),
        "words" => haystack
            .split(|&c| c == b'\n' || c == b' ')
            .filter(|t| !t.is_empty())
            .collect(),
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

    // Benchmark: StringZilla `bytesum` reference
    group.bench_function("stringzilla::bytesum", |b| {
        b.iter(|| {
            for token in tokens {
                let t = black_box(*token);
                let _ = black_box(sz::bytesum(t));
            }
        })
    });

    // Benchmark: StringZilla
    group.bench_function("stringzilla::hash", |b| {
        b.iter(|| {
            for token in tokens {
                let t = black_box(*token);
                let _ = black_box(sz::hash(t));
            }
        })
    });

    // Benchmark: SipHash via `std::DefaultHasher`
    group.bench_function("std::DefaultHasher::hash_one", |b| {
        let std_builder = std::collections::hash_map::RandomState::new();
        b.iter(|| {
            for token in tokens {
                let t = black_box(*token);
                let _ = black_box(std_builder.hash_one(t));
            }
        })
    });

    // Benchmark: aHash
    group.bench_function("aHash::hash_one", |b| {
        let hash_builder = RandomState::with_seed(42);
        b.iter(|| {
            for token in tokens {
                let t = black_box(*token);
                let _ = black_box(hash_builder.hash_one(t));
            }
        })
    });

    // Benchmark: xxHash
    group.bench_function("xxh3::xxh3_64", |b| {
        b.iter(|| {
            for token in tokens {
                let t = black_box(*token);
                let _ = black_box(xxh3_64(t));
            }
        })
    });

    // Benchmark: gxhash
    group.bench_function("gxhash::gxhash64", |b| {
        b.iter(|| {
            for token in tokens {
                let t = black_box(*token);
                let _ = black_box(gxhash::gxhash64(t, 42));
            }
        })
    });

    // Benchmark: CRC32
    group.bench_function("crc32fast::hash", |b| {
        b.iter(|| {
            for token in tokens {
                let t = black_box(*token);
                let _ = black_box(crc32fast::hash(t));
            }
        })
    });

    // Benchmark: MurmurHash3 (x64_128) via `murmur3` (stateless)
    group.bench_function("murmur3::x64_128", |b| {
        b.iter(|| {
            for token in tokens {
                let t = black_box(*token);
                let mut cursor = Cursor::new(t);
                let _ = black_box(murmur3::murmur3_x64_128(&mut cursor, 0).unwrap());
            }
        })
    });

    // Benchmark: CityHash64 via `cityhash` (stateless)
    group.bench_function("cityhash::city_hash_64", |b| {
        b.iter(|| {
            for token in tokens {
                let t = black_box(*token);
                let _ = black_box(cityhash::city_hash_64(t));
            }
        })
    });

    // Benchmark: Blake3 - should be by far the slowest, as it's a cryptographic hash.
    group.bench_function("blake3", |b| {
        b.iter(|| {
            for token in tokens {
                let t = black_box(*token);
                let _ = black_box(blake3::hash(t));
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

    // Benchmark: StringZilla `hash`
    group.bench_function("stringzilla::Hasher", |b| {
        b.iter(|| {
            let mut hasher = sz::Hasher::new(0);
            for token in tokens {
                hasher.write(token);
            }
            black_box(hasher.finish());
        })
    });

    // Benchmark: SipHash via `std::DefaultHasher`
    group.bench_function("std::DefaultHasher", |b| {
        let std_builder = std::collections::hash_map::RandomState::new();
        b.iter(|| {
            let mut aggregate = std_builder.build_hasher();
            for token in tokens {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
        })
    });

    // Benchmark: aHash
    group.bench_function("aHash::AHasher", |b| {
        let state: AHashState = RandomState::with_seed(42);
        b.iter(|| {
            let mut aggregate = state.build_hasher();
            for token in tokens {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
        })
    });

    // Benchmark: CRC32
    group.bench_function("crc32fast::Hasher", |b| {
        b.iter(|| {
            let mut hasher = crc32fast::Hasher::new();
            for token in tokens {
                hasher.update(token);
            }
            black_box(hasher.finalize());
        })
    });
}

fn main() {
    log_stringzilla_metadata();

    // Load the dataset defined by the environment variables, and panic if the content is missing
    let dataset = load_dataset().unwrap();
    let tokens = tokenize(&dataset).unwrap();
    if tokens.is_empty() {
        panic!("No tokens found in the dataset.");
    }

    let mut criterion = configure_bench();

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
