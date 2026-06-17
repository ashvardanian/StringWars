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

Note: `cityhash` is only compiled on x86_64 targets as it requires x86-specific instructions.
"#]
use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};
use std::hint::black_box;

use bit_set::BitSet;
use stringtape::{BytesCowsAuto, BytesTape};

use ahash::RandomState as AHashState;
use blake3;
use crc32fast;
use foldhash;
use murmurhash32;
use ring::digest as ring_digest;
use sha2::{Digest, Sha256};
use stringzilla::sz;
use wyhash::wyhash;
use xxhash_rust::xxh3::xxh3_64;

#[cfg(target_arch = "x86_64")]
use cityhash;

#[path = "../utils.rs"]
mod utils;
use utils::{
    get_env_bool, install_panic_hook, load_dataset_with_default_mode, log_stringzilla_metadata,
    measure_throughput, should_run, BenchBudget, ReportAs, ResultExt, WorkUnits,
};

/// Time one stateless hash over the dataset by cycling tokens for the budget. The kernel
/// `hash_one` is called on one token per iteration; throughput is reported as bytes/s.
fn bench_each_token<HashOne: FnMut(&[u8])>(
    name: &str,
    budget: &BenchBudget,
    tokens: &[&[u8]],
    mut hash_one: HashOne,
) {
    let mut cursor = 0usize;
    measure_throughput(name, ReportAs::Bytes, budget, || {
        let token = tokens[cursor % tokens.len()];
        cursor += 1;
        hash_one(black_box(token));
        WorkUnits::new(1, token.len() as u64)
    });
}

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

/// Benchmarks stateless hashes, hashing one token per call and cycling the dataset.
fn bench_stateless(budget: &BenchBudget, tokens: &BytesCowsAuto) {
    // Collision detection is opt-in via STRINGWARS_COLLISIONS environment variable.
    // This avoids OOM on large datasets (can use GBs of RAM for deduplication).
    let enable_collision_detection = get_env_bool("STRINGWARS_COLLISIONS");
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

    // Use BytesTape to colocate strings and reduce memory access overhead.
    let mut tokens_tape = BytesTape::<u64>::new();
    tokens_tape
        .extend(tokens.iter())
        .expect("Failed to create BytesTape");
    let view = tokens_tape.view();
    let slices: Vec<&[u8]> = (&view).into_iter().collect();

    // Benchmark: StringZilla
    bench_each_token("stateless/stringzilla/hash()", budget, &slices, |token| {
        let _ = black_box(sz::hash(token));
    });
    if !unique_tokens.is_empty() && should_run("stateless/stringzilla/hash()") {
        print_collision_rate(&unique_tokens, |token_bytes| sz::hash(token_bytes));
    }

    // Benchmark: SipHash via `std::DefaultHasher`
    let std_builder = std::collections::hash_map::RandomState::new();
    bench_each_token(
        "stateless/std/DefaultHasher.hash_one()",
        budget,
        &slices,
        |token| {
            let _ = black_box(std_builder.hash_one(token));
        },
    );
    if !unique_tokens.is_empty() && should_run("stateless/std/DefaultHasher.hash_one()") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            std_builder.hash_one(token_bytes)
        });
    }

    // Benchmark: aHash
    let hash_builder = AHashState::with_seed(42);
    bench_each_token("stateless/ahash/hash_one()", budget, &slices, |token| {
        let _ = black_box(hash_builder.hash_one(token));
    });
    if !unique_tokens.is_empty() && should_run("stateless/ahash/hash_one()") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            hash_builder.hash_one(token_bytes)
        });
    }

    // Benchmark: xxHash
    bench_each_token("stateless/xxh3/xxh3_64()", budget, &slices, |token| {
        let _ = black_box(xxh3_64(token));
    });
    if !unique_tokens.is_empty() && should_run("stateless/xxh3/xxh3_64()") {
        print_collision_rate(&unique_tokens, |token_bytes| xxh3_64(token_bytes));
    }

    // Benchmark: wyhash
    bench_each_token("stateless/wyhash/wyhash()", budget, &slices, |token| {
        let _ = black_box(wyhash(token, 42));
    });
    if !unique_tokens.is_empty() && should_run("stateless/wyhash/wyhash()") {
        print_collision_rate(&unique_tokens, |token_bytes| wyhash(token_bytes, 42));
    }

    // Benchmark: FoldHash
    let foldhash_builder = foldhash::fast::RandomState::default();
    bench_each_token("stateless/foldhash/hash_one()", budget, &slices, |token| {
        let _ = black_box(foldhash_builder.hash_one(token));
    });
    if !unique_tokens.is_empty() && should_run("stateless/foldhash/hash_one()") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            foldhash_builder.hash_one(token_bytes)
        });
    }

    // Benchmark: CRC32
    bench_each_token("stateless/crc32fast/hash()", budget, &slices, |token| {
        let _ = black_box(crc32fast::hash(token));
    });
    if !unique_tokens.is_empty() && should_run("stateless/crc32fast/hash()") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            crc32fast::hash(token_bytes) as u64
        });
    }

    // Benchmark: MurmurHash32 via `murmurhash32` (stateless)
    bench_each_token(
        "stateless/murmurhash32/murmurhash3()",
        budget,
        &slices,
        |token| {
            let _ = black_box(murmurhash32::murmurhash3(token) as u64);
        },
    );
    if !unique_tokens.is_empty() && should_run("stateless/murmurhash32/murmurhash3()") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            murmurhash32::murmurhash3(token_bytes) as u64
        });
    }

    // Benchmark: CityHash64 via `cityhash` (stateless, x86_64 only)
    #[cfg(target_arch = "x86_64")]
    {
        bench_each_token(
            "stateless/cityhash/city_hash_64()",
            budget,
            &slices,
            |token| {
                let _ = black_box(cityhash::city_hash_64(token));
            },
        );
        if !unique_tokens.is_empty() && should_run("stateless/cityhash/city_hash_64()") {
            print_collision_rate(&unique_tokens, |token_bytes| {
                cityhash::city_hash_64(token_bytes)
            });
        }
    }

    if !unique_tokens.is_empty() {
        println!();
    }
}

/// Benchmarks checksum hashes including cryptographic hashes and reference bounds.
fn bench_checksum(budget: &BenchBudget, tokens: &BytesCowsAuto) {
    let enable_collision_detection = get_env_bool("STRINGWARS_COLLISIONS");
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

    let mut tokens_tape = BytesTape::<u64>::new();
    tokens_tape
        .extend(tokens.iter())
        .expect("Failed to create BytesTape");
    let view = tokens_tape.view();
    let slices: Vec<&[u8]> = (&view).into_iter().collect();

    // Benchmark: StringZilla `bytesum` reference lower bound
    bench_each_token("checksum/stringzilla/bytesum()", budget, &slices, |token| {
        let _ = black_box(sz::bytesum(token));
    });

    // Benchmark: Blake3 - cryptographic hash
    bench_each_token("checksum/blake3/hash()", budget, &slices, |token| {
        let _ = black_box(blake3::hash(token));
    });
    if !unique_tokens.is_empty() && should_run("checksum/blake3/hash()") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let hash = blake3::hash(token_bytes);
            let bytes = hash.as_bytes();
            u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        });
    }

    // Benchmark: SHA256 via sha2
    bench_each_token("checksum/sha2/Sha256()", budget, &slices, |token| {
        let mut hasher = Sha256::new();
        hasher.update(token);
        let _ = black_box(hasher.finalize());
    });
    if !unique_tokens.is_empty() && should_run("checksum/sha2/Sha256()") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let mut hasher = Sha256::new();
            hasher.update(token_bytes);
            let result = hasher.finalize();
            u64::from_le_bytes([
                result[0], result[1], result[2], result[3], result[4], result[5], result[6],
                result[7],
            ])
        });
    }

    // Benchmark: SHA256 via ring
    bench_each_token("checksum/ring/SHA256()", budget, &slices, |token| {
        let _ = black_box(ring_digest::digest(&ring_digest::SHA256, token));
    });
    if !unique_tokens.is_empty() && should_run("checksum/ring/SHA256()") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let digest = ring_digest::digest(&ring_digest::SHA256, token_bytes);
            let bytes = digest.as_ref();
            u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        });
    }

    // Benchmark: SHA256 via stringzilla
    bench_each_token("checksum/stringzilla/Sha256()", budget, &slices, |token| {
        let _ = black_box(sz::Sha256::hash(token));
    });
    if !unique_tokens.is_empty() && should_run("checksum/stringzilla/Sha256()") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let digest = sz::Sha256::hash(token_bytes);
            u64::from_le_bytes([
                digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6],
                digest[7],
            ])
        });
    }

    if !unique_tokens.is_empty() {
        println!();
    }
}

/// Benchmarks stateful hashes, streaming the whole dataset through one hasher per pass and
/// cycling passes for the budget. The per-call unit is one full streaming pass, so the deadline
/// check after each call bounds overshoot to a single pass.
fn bench_stateful(budget: &BenchBudget, tokens: &BytesCowsAuto) {
    let mut tokens_tape = BytesTape::<u64>::new();
    tokens_tape
        .extend(tokens.iter())
        .expect("Failed to create BytesTape");
    let view = tokens_tape.view();
    let total_bytes: u64 = (&view).into_iter().map(|token| token.len() as u64).sum();

    // Benchmark: StringZilla `Hasher`
    measure_throughput(
        "stateful/stringzilla/Hasher()",
        ReportAs::Bytes,
        budget,
        || {
            let mut hasher = sz::Hasher::new(0);
            for token in &view {
                hasher.write(token);
            }
            black_box(hasher.finish());
            WorkUnits::bytes(total_bytes)
        },
    );

    // Benchmark: SipHash via `std::DefaultHasher`
    let std_builder = std::collections::hash_map::RandomState::new();
    measure_throughput(
        "stateful/std/DefaultHasher()",
        ReportAs::Bytes,
        budget,
        || {
            let mut aggregate = std_builder.build_hasher();
            for token in &view {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
            WorkUnits::bytes(total_bytes)
        },
    );

    // Benchmark: aHash
    let ahash_state = AHashState::with_seed(42);
    measure_throughput("stateful/ahash/AHasher()", ReportAs::Bytes, budget, || {
        let mut aggregate = ahash_state.build_hasher();
        for token in &view {
            aggregate.write(token);
        }
        black_box(aggregate.finish());
        WorkUnits::bytes(total_bytes)
    });

    // Benchmark: FoldHash
    let foldhash_state = foldhash::fast::RandomState::default();
    measure_throughput(
        "stateful/foldhash/FoldHasher()",
        ReportAs::Bytes,
        budget,
        || {
            let mut aggregate = foldhash_state.build_hasher();
            for token in &view {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
            WorkUnits::bytes(total_bytes)
        },
    );

    // Benchmark: CRC32
    measure_throughput(
        "stateful/crc32fast/Hasher()",
        ReportAs::Bytes,
        budget,
        || {
            let mut hasher = crc32fast::Hasher::new();
            for token in &view {
                hasher.update(token);
            }
            black_box(hasher.finalize());
            WorkUnits::bytes(total_bytes)
        },
    );
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    // Load the dataset defined by the environment variables.
    let tape = load_dataset_with_default_mode("words").unwrap_nice();

    let budget = BenchBudget::from_env(2.0, 10.0);

    println!("# stateless");
    bench_stateless(&budget, &tape);

    println!("# stateful");
    bench_stateful(&budget, &tape);

    println!("# checksum");
    bench_checksum(&budget, &tape);
}
