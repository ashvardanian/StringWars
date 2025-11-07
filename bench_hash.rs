#![doc = r#"
# StringWars: String Hashing Benchmarks

This file contains benchmarks for various Rust hashing libraries using Criterion,
treating the inputs as binary strings without any UTF-8 validity constrains.
For accurate stats aggregation, on each iteration, the whole file is scanned.
Be warned, for large files, it may take a while!

The benchmarks are organized into three categories:

**Stateless Hashes** (hash each input independently):
- StringZilla `hash`
- Standard `Hash` implementation (SipHash)
- aHash
- xxHash (xxh3)
- gxhash (x86_64 only)
- FoldHash
- CRC32 (IEEE) via `crc32fast`
- MurmurHash32 via `murmurhash32`
- CityHash64 via `cityhash` (x86_64 & Clang only)

**Stateful Hashes** (incremental/streaming):
- StringZilla `Hasher`
- Standard `DefaultHasher` (SipHash)
- aHash `AHasher`
- FoldHash `FoldHasher`
- CRC32 via `crc32fast::Hasher`

**Checksum Hashes** (cryptographic and reference bounds):
- StringZilla `bytesum` (reference lower bound)
- Blake3 (cryptographic)
- SHA256 via `sha2` (cryptographic)
- SHA256 via `ring` (cryptographic)
- SHA256 via `stringzilla` (cryptographic, stateless and stateful)

## System Dependencies

Before running these benchmarks, ensure the following system packages are installed:

```sh
sudo apt install -y build-essential llvm-18-dev libclang-18-dev clang-18 # for Ubuntu/Debian
sudo dnf install -y gcc llvm-devel clang-devel # for RHEL/Fedora
brew install llvm clang # for macOS
```

## Usage Examples

The benchmarks use environment variables to control the input dataset and mode:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line.
  - `words`: Process the dataset word by word.
  - `file`: Process the entire file as a single token.
- `STRINGWARS_COLLISIONS`: Set to `1` or `true` to enable collision detection (disabled by default to avoid OOM on large datasets).
- `STRINGWARS_FILTER`: Regex pattern to filter which benchmarks to run (e.g., `sha` for SHA benchmarks, `stateless/.*hash` for stateless hashes).

To run the benchmarks with the appropriate CPU features enabled, you can use the following commands:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_hash bench_hash --jobs $(nproc)
```

Note: `gxhash` and `cityhash` are only compiled on x86_64 targets as they require x86-specific instructions.
"#]
use std::collections::HashSet;
use std::env;
use std::hash::{BuildHasher, Hasher};
use std::hint::black_box;

use bit_set::BitSet;
use criterion::{Criterion, Throughput};
use stringtape::{BytesCowsAuto, BytesTape};

use ahash::RandomState as AHashState;
use blake3;
use crc32fast;
use foldhash;
use murmurhash32;
use ring::digest as ring_digest;
use sha2::{Digest, Sha256};
use stringzilla::sz;
use xxhash_rust::xxh3::xxh3_64;

#[cfg(target_arch = "x86_64")]
use cityhash;
#[cfg(target_arch = "x86_64")]
use gxhash;

mod utils;
use utils::{load_dataset, should_run};

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
        .sample_size(10) // 10 is the lowest supported by Criterion
        .warm_up_time(std::time::Duration::from_secs(5)) // Let CPU frequencies settle.
        .measurement_time(std::time::Duration::from_secs(10)) // Actual measurement time.
}

/// Benchmarks stateless hashes seeing the whole input at once
fn bench_stateless(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &BytesCowsAuto,
) {
    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = tokens.iter().map(|t| t.len()).sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Collision detection is opt-in via STRINGWARS_COLLISIONS environment variable
    // This avoids OOM on large datasets (can use GBs of RAM for deduplication)
    let enable_collision_detection = env::var("STRINGWARS_COLLISIONS")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    let unique_tokens: Vec<&[u8]> = if enable_collision_detection {
        println!("\nComputing unique tokens for collision detection...");
        let unique_set: HashSet<&[u8]> = tokens.iter().collect();
        let unique: Vec<&[u8]> = unique_set.into_iter().collect();
        println!(
            "Collision statistics for {} unique tokens (from {} total):",
            unique.len(),
            tokens.len()
        );
        unique
    } else {
        Vec::new()
    };

    // Use BytesTape to colocate strings and reduce memory access overhead
    let mut tokens_tape = BytesTape::<u64>::new();
    tokens_tape
        .extend(tokens.iter())
        .expect("Failed to create BytesTape");
    let tokens = tokens_tape.view();

    // Benchmark: StringZilla
    if should_run("stateless/stringzilla::hash") {
        group.bench_function("stringzilla::hash", |b| {
            b.iter(|| {
                for token in &tokens {
                    let _ = black_box(sz::hash(black_box(token)));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| sz::hash(t));
        }
    }

    // Benchmark: SipHash via `std::DefaultHasher`
    if should_run("stateless/std::DefaultHasher::hash_one") {
        let std_builder = std::collections::hash_map::RandomState::new();
        group.bench_function("std::DefaultHasher::hash_one", |b| {
            b.iter(|| {
                for token in &tokens {
                    let t = black_box(token);
                    let _ = black_box(std_builder.hash_one(t));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| std_builder.hash_one(t));
        }
    }

    // Benchmark: aHash
    if should_run("stateless/aHash::hash_one") {
        let hash_builder = AHashState::with_seed(42);
        group.bench_function("aHash::hash_one", |b| {
            b.iter(|| {
                for token in &tokens {
                    let t = black_box(token);
                    let _ = black_box(hash_builder.hash_one(t));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| hash_builder.hash_one(t));
        }
    }

    // Benchmark: xxHash
    if should_run("stateless/xxh3::xxh3_64") {
        group.bench_function("xxh3::xxh3_64", |b| {
            b.iter(|| {
                for token in &tokens {
                    let t = black_box(token);
                    let _ = black_box(xxh3_64(t));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| xxh3_64(t));
        }
    }

    // Benchmark: gxhash (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    if should_run("stateless/gxhash::gxhash64") {
        group.bench_function("gxhash::gxhash64", |b| {
            b.iter(|| {
                for token in &tokens {
                    let t = black_box(token);
                    let _ = black_box(gxhash::gxhash64(t, 42));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| gxhash::gxhash64(t, 42));
        }
    }

    // Benchmark: FoldHash
    if should_run("stateless/foldhash::hash_one") {
        let foldhash_builder = foldhash::fast::RandomState::default();
        group.bench_function("foldhash::hash_one", |b| {
            b.iter(|| {
                for token in &tokens {
                    let t = black_box(token);
                    let _ = black_box(foldhash_builder.hash_one(t));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| foldhash_builder.hash_one(t));
        }
    }

    // Benchmark: CRC32
    if should_run("stateless/crc32fast::hash") {
        group.bench_function("crc32fast::hash", |b| {
            b.iter(|| {
                for token in &tokens {
                    let t = black_box(token);
                    let _ = black_box(crc32fast::hash(t));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| crc32fast::hash(t) as u64);
        }
    }

    // Benchmark: MurmurHash32 via `murmurhash32` (stateless)
    if should_run("stateless/murmurhash32") {
        group.bench_function("murmurhash32", |b| {
            b.iter(|| {
                for token in &tokens {
                    let t = black_box(token);
                    let _ = black_box(murmurhash32::murmurhash3(t) as u64);
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| murmurhash32::murmurhash3(t) as u64);
        }
    }

    // Benchmark: CityHash64 via `cityhash` (stateless, x86_64 only)
    #[cfg(target_arch = "x86_64")]
    if should_run("stateless/cityhash::city_hash_64") {
        group.bench_function("cityhash::city_hash_64", |b| {
            b.iter(|| {
                for token in &tokens {
                    let t = black_box(token);
                    let _ = black_box(cityhash::city_hash_64(t));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| cityhash::city_hash_64(t));
        }
    }

    if !unique_tokens.is_empty() {
        println!();
    }
}

/// Benchmarks checksum hashes including cryptographic hashes and reference bounds
fn bench_checksum(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &BytesCowsAuto,
) {
    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = tokens.iter().map(|t| t.len()).sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Collision detection is opt-in via STRINGWARS_COLLISIONS environment variable
    let enable_collision_detection = env::var("STRINGWARS_COLLISIONS")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    let unique_tokens: Vec<&[u8]> = if enable_collision_detection {
        let unique_set: HashSet<&[u8]> = tokens.iter().collect();
        let unique: Vec<&[u8]> = unique_set.into_iter().collect();
        println!(
            "Collision statistics for {} unique tokens (from {} total):",
            unique.len(),
            tokens.len()
        );
        unique
    } else {
        Vec::new()
    };

    // Benchmark: StringZilla `bytesum` reference lower bound
    if should_run("checksum/stringzilla::bytesum") {
        group.bench_function("stringzilla::bytesum", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(token);
                    let _ = black_box(sz::bytesum(t));
                }
            })
        });
    }

    // Benchmark: Blake3 - cryptographic hash
    if should_run("checksum/blake3::hash") {
        group.bench_function("blake3::hash", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(token);
                    let _ = black_box(blake3::hash(t));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| {
                let hash = blake3::hash(t);
                let bytes = hash.as_bytes();
                u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ])
            });
        }
    }

    // Benchmark: SHA256 via sha2
    if should_run("checksum/sha2::Sha256") {
        group.bench_function("sha2::Sha256", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(token);
                    let mut hasher = Sha256::new();
                    hasher.update(t);
                    let _ = black_box(hasher.finalize());
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| {
                let mut hasher = Sha256::new();
                hasher.update(t);
                let result = hasher.finalize();
                u64::from_le_bytes([
                    result[0], result[1], result[2], result[3], result[4], result[5], result[6],
                    result[7],
                ])
            });
        }
    }

    // Benchmark: SHA256 via ring
    if should_run("checksum/ring::SHA256") {
        group.bench_function("ring::SHA256", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(token);
                    let _ = black_box(ring_digest::digest(&ring_digest::SHA256, t));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| {
                let digest = ring_digest::digest(&ring_digest::SHA256, t);
                let bytes = digest.as_ref();
                u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ])
            });
        }
    }

    // Benchmark: SHA256 via stringzilla
    if should_run("checksum/stringzilla::Sha256") {
        group.bench_function("stringzilla::Sha256", |b| {
            b.iter(|| {
                for token in tokens {
                    let t = black_box(token);
                    let _ = black_box(sz::Sha256::hash(t));
                }
            })
        });
        if !unique_tokens.is_empty() {
            print_collision_rate(&unique_tokens, |t| {
                let digest = sz::Sha256::hash(t);
                u64::from_le_bytes([
                    digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6],
                    digest[7],
                ])
            });
        }
    }

    if !unique_tokens.is_empty() {
        println!();
    }
}

/// Benchmarks stateful hashes seeing one slice at a time
fn bench_stateful(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &BytesCowsAuto,
) {
    // Use BytesTape to colocate strings and reduce memory access overhead
    let mut tokens_tape = BytesTape::<u64>::new();
    tokens_tape
        .extend(tokens.iter())
        .expect("Failed to create BytesTape");
    let tokens = tokens_tape.view();

    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = (&tokens).into_iter().map(|t| t.len()).sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Benchmark: StringZilla `hash`
    if should_run("stateful/stringzilla::Hasher") {
        group.bench_function("stringzilla::Hasher", |b| {
            b.iter(|| {
                let mut hasher = sz::Hasher::new(0);
                for token in &tokens {
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
                for token in &tokens {
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
                for token in &tokens {
                    aggregate.write(token);
                }
                black_box(aggregate.finish());
            })
        });
    }

    // Benchmark: FoldHash
    if should_run("stateful/foldhash::FoldHasher") {
        group.bench_function("foldhash::FoldHasher", |b| {
            let state = foldhash::fast::RandomState::default();
            b.iter(|| {
                let mut aggregate = state.build_hasher();
                for token in &tokens {
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
                for token in &tokens {
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
    let tape = load_dataset();
    if tape.is_empty() {
        panic!("No tokens found in the dataset.");
    }

    let mut criterion = configure_bench();

    // Profile stateless hash functions that see the whole input at once
    let mut group = criterion.benchmark_group("stateless");
    bench_stateless(&mut group, &tape);
    group.finish();

    // Profile stateful/incremental hash functions that see only a slice of data at a time
    let mut group = criterion.benchmark_group("stateful");
    bench_stateful(&mut group, &tape);
    group.finish();

    // Profile checksum and cryptographic hash functions
    let mut group = criterion.benchmark_group("checksum");
    bench_checksum(&mut group, &tape);
    group.finish();

    criterion.final_summary();
}
