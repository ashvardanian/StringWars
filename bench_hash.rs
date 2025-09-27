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
use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs;
use std::hash::{BuildHasher, Hasher};
use std::hint::black_box;
use std::io::Cursor;

use bit_set::BitSet;
use criterion::{Criterion, Throughput};

use ahash::RandomState as AHashState;
use blake3;
use cityhash;
use crc32fast;
use gxhash;
use murmur3;
use stringzilla::sz;
use xxhash_rust::xxh3::xxh3_64;

mod utils;
use utils::should_run;

/// Counts collisions for a given hash function using a bitset sized to the number of unique tokens
fn count_collisions<F>(unique_tokens: &[&[u8]], hash_fn: F) -> usize
where
    F: Fn(&[u8]) -> u64,
{
    if unique_tokens.is_empty() {
        return 0;
    }

    let table_size = unique_tokens.len();
    let mut bitset = BitSet::with_capacity(table_size);
    let mut collisions = 0;

    for token in unique_tokens {
        let hash = hash_fn(token);
        let bit_pos = (hash as usize) % table_size;
        collisions += !bitset.insert(bit_pos) as usize;
    }

    collisions
}

/// Calculate and print collision rate for a hash function using a bitset matching the unique token count
fn print_collision_rate<F>(unique_tokens: &[&[u8]], hash_fn: F)
where
    F: Fn(&[u8]) -> u64,
{
    let n_unique = unique_tokens.len();
    if n_unique == 0 {
        return;
    }

    let collisions = count_collisions(unique_tokens, hash_fn);
    let rate = (collisions as f64 / n_unique as f64) * 100.0;

    println!(
        "                        collisions: {:.2}% ({} collisions across {} buckets)",
        rate, collisions, n_unique
    );
}

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

    // Precompute unique tokens once for collision detection
    let unique_set: HashSet<&[u8]> = tokens.iter().copied().collect();
    let unique_tokens: Vec<&[u8]> = unique_set.into_iter().collect();

    println!(
        "\nCollision statistics for {} unique tokens (from {} total):",
        unique_tokens.len(),
        tokens.len()
    );

    // Benchmark: StringZilla `bytesum` reference
    if should_run("stateless/stringzilla::bytesum") {
        group.bench_function("stringzilla::bytesum", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(*token);
                    let _ = black_box(sz::bytesum(t));
                }
            })
        });
    }

    // Benchmark: StringZilla
    if should_run("stateless/stringzilla::hash") {
        group.bench_function("stringzilla::hash", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(*token);
                    let _ = black_box(sz::hash(t));
                }
            })
        });
        print_collision_rate(&unique_tokens, |t| sz::hash(t));
    }

    // Benchmark: SipHash via `std::DefaultHasher`
    if should_run("stateless/std::DefaultHasher::hash_one") {
        let std_builder = std::collections::hash_map::RandomState::new();
        group.bench_function("std::DefaultHasher::hash_one", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(*token);
                    let _ = black_box(std_builder.hash_one(t));
                }
            })
        });
        print_collision_rate(&unique_tokens, |t| std_builder.hash_one(t));
    }

    // Benchmark: aHash
    if should_run("stateless/aHash::hash_one") {
        let hash_builder = AHashState::with_seed(42);
        group.bench_function("aHash::hash_one", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(*token);
                    let _ = black_box(hash_builder.hash_one(t));
                }
            })
        });
        print_collision_rate(&unique_tokens, |t| hash_builder.hash_one(t));
    }

    // Benchmark: xxHash
    if should_run("stateless/xxh3::xxh3_64") {
        group.bench_function("xxh3::xxh3_64", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(*token);
                    let _ = black_box(xxh3_64(t));
                }
            })
        });
        print_collision_rate(&unique_tokens, |t| xxh3_64(t));
    }

    // Benchmark: gxhash
    if should_run("stateless/gxhash::gxhash64") {
        group.bench_function("gxhash::gxhash64", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(*token);
                    let _ = black_box(gxhash::gxhash64(t, 42));
                }
            })
        });
        print_collision_rate(&unique_tokens, |t| gxhash::gxhash64(t, 42));
    }

    // Benchmark: CRC32
    if should_run("stateless/crc32fast::hash") {
        group.bench_function("crc32fast::hash", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(*token);
                    let _ = black_box(crc32fast::hash(t));
                }
            })
        });
        print_collision_rate(&unique_tokens, |t| crc32fast::hash(t) as u64);
    }

    // Benchmark: MurmurHash3 (x64_128) via `murmur3` (stateless)
    if should_run("stateless/murmur3::x64_128") {
        group.bench_function("murmur3::x64_128", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(*token);
                    let mut cursor = Cursor::new(t);
                    let _ = black_box(murmur3::murmur3_x64_128(&mut cursor, 0).unwrap());
                }
            })
        });
        print_collision_rate(&unique_tokens, |t| {
            let mut cursor = Cursor::new(t);
            murmur3::murmur3_x64_128(&mut cursor, 0).unwrap() as u64
        });
    }

    // Benchmark: CityHash64 via `cityhash` (stateless)
    if should_run("stateless/cityhash::city_hash_64") {
        group.bench_function("cityhash::city_hash_64", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(*token);
                    let _ = black_box(cityhash::city_hash_64(t));
                }
            })
        });
        print_collision_rate(&unique_tokens, |t| cityhash::city_hash_64(t));
    }

    // Benchmark: Blake3 - should be by far the slowest, as it's a cryptographic hash.
    if should_run("stateless/blake3") {
        group.bench_function("blake3", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(*token);
                    let _ = black_box(blake3::hash(t));
                }
            })
        });
        print_collision_rate(&unique_tokens, |t| {
            let hash = blake3::hash(t);
            let bytes = hash.as_bytes();
            u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        });
    }

    println!();
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
    if should_run("stateful/stringzilla::Hasher") {
        group.bench_function("stringzilla::Hasher", |b| {
            b.iter(|| {
                let mut hasher = sz::Hasher::new(0);
                for token in tokens {
                    hasher.write(token);
                }
                black_box(hasher.finish());
            })
        });
    }

    // Benchmark: SipHash via `std::DefaultHasher`
    if should_run("stateful/std::DefaultHasher") {
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
    }

    // Benchmark: aHash
    if should_run("stateful/aHash::AHasher") {
        group.bench_function("aHash::AHasher", |b| {
            let state = AHashState::with_seed(42);
            b.iter(|| {
                let mut aggregate = state.build_hasher();
                for token in tokens {
                    aggregate.write(token);
                }
                black_box(aggregate.finish());
            })
        });
    }

    // Benchmark: CRC32
    if should_run("stateful/crc32fast::Hasher") {
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
