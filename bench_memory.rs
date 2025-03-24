#![doc = r#"
# StringWa.rs: Low-level Memory-related Benchmarks

This file benchmarks low-level memory operations. The input file is treated as a collection
of size-representative tokens and for every token the following operations are benchmarked:

- case inversion using Lookup Table Transforms (LUT), common in image processing
- memory obfuscation using Pseudo-Random Number Generators (PRNG), common in sensitive apps

## Usage Examples

The benchmarks use two environment variables to control the input dataset and mode:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line.
  - `words`: Process the dataset word by word.

To run the benchmarks with the appropriate CPU features enabled, you can use the following commands:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_memory bench_memory --jobs 8
```
"#]
use std::env;
use std::error::Error;
use std::fs;
use std::time::Duration;

use criterion::{black_box, Criterion, Throughput};

use getrandom;
use rand;
use rand::{RngCore, SeedableRng};
use rand_chacha;
use rand_xoshiro;
use stringzilla::sz;
use zeroize::Zeroize;

/// Loads the dataset from the file specified by the `STRINGWARS_DATASET` environment variable.
pub fn load_dataset() -> Result<Vec<u8>, Box<dyn Error>> {
    let dataset_path = env::var("STRINGWARS_DATASET")
        .map_err(|_| "STRINGWARS_DATASET environment variable not set")?;
    let content = fs::read(&dataset_path)?;
    Ok(content)
}

/// Tokenizes the given haystack based on the `STRINGWARS_TOKENS` environment variable.
/// Supported modes: "lines", "words", and "file".
pub fn tokenize<'a>(haystack: &'a mut [u8]) -> Result<Vec<&'a mut [u8]>, Box<dyn Error>> {
    let mode = env::var("STRINGWARS_TOKENS").unwrap_or_else(|_| "lines".to_string());
    let tokens = match mode.as_str() {
        "lines" => haystack.split_mut(|&c| c == b'\n').collect(),
        "words" => haystack.split_mut(|&c| c == b'\n' || c == b' ').collect(),
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

fn bench_lookup_table(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &mut [&mut [u8]],
) {
    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = tokens.iter().map(|u| u.len()).sum();
    g.throughput(Throughput::Bytes(total_bytes as u64));

    // Benchmark for StringZilla forward search using a cycle iterator.
    let mut lookup_invert_case: [u8; 256] = core::array::from_fn(|i| i as u8);
    for (upper, lower) in ('A'..='Z').zip('a'..='z') {
        lookup_invert_case[upper as usize] = lower as u8;
    }
    for (upper, lower) in ('A'..='Z').zip('a'..='z') {
        lookup_invert_case[lower as usize] = upper as u8;
    }

    // Benchmark using StringZilla's `lookup_inplace`.
    g.bench_function("sz::lookup_inplace", |b| {
        b.iter(|| {
            for token in tokens.iter_mut() {
                sz::lookup_inplace(&mut *token, lookup_invert_case);
                black_box(token);
            }
        })
    });

    // Benchmark a plain serial mapping using the same lookup table.
    g.bench_function("serial", |b| {
        b.iter(|| {
            for token in tokens.iter_mut() {
                for byte in token.iter_mut() {
                    *byte = lookup_invert_case[*byte as usize];
                }
                black_box(&token);
            }
        })
    });
}

fn bench_generate_random(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &mut [&mut [u8]],
) {
    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = tokens.iter().map(|u| u.len()).sum();
    g.throughput(Throughput::Bytes(total_bytes as u64));

    // Benchmark for StringZilla AES-based PRNG
    g.bench_function("sz::fill_random", |b| {
        b.iter(|| {
            for token in tokens.iter_mut() {
                sz::fill_random(&mut *token, 0)
            }
        })
    });

    // Benchmark using zeroize to obfuscate (zero out) the buffer.
    g.bench_function("zeroize::zeroize", |b| {
        b.iter(|| {
            for token in tokens.iter_mut() {
                token.zeroize();
                black_box(&token);
            }
        })
    });

    // Benchmark using `getrandom` to randomize the buffer via the OS.
    g.bench_function("getrandom::fill", |b| {
        b.iter(|| {
            for token in tokens.iter_mut() {
                getrandom::fill(&mut *token).expect("getrandom failed");
                black_box(&token);
            }
        })
    });

    // Benchmark using `rand_chacha::ChaCha20Rng`.
    g.bench_function("rand_chacha::ChaCha20Rng", |b| {
        b.iter(|| {
            for token in tokens.iter_mut() {
                let mut rng = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
                rng.fill_bytes(&mut *token);
                black_box(&token);
            }
        })
    });

    // Benchmark using `rand_xoshiro::Xoshiro128Plus`.
    g.bench_function("rand_xoshiro::Xoshiro128Plus", |b| {
        b.iter(|| {
            for token in tokens.iter_mut() {
                let mut rng = rand_xoshiro::Xoshiro128Plus::from_seed([0u8; 16]);
                rng.fill_bytes(&mut *token);
                black_box(&token);
            }
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
    let mut dataset = load_dataset().unwrap();
    let mut tokens = tokenize(&mut dataset).unwrap();
    if tokens.is_empty() {
        panic!("No tokens found in the dataset.");
    }

    // Setup the default durations
    let mut criterion = Criterion::default()
        .sample_size(10) // Each loop scans the whole dataset.
        .warm_up_time(Duration::from_secs(1)) // Let the CPU frequencies settle.
        .measurement_time(Duration::from_secs(20)); // Actual measurement time.

    // Benchmarks for lookup table transform
    let mut group = criterion.benchmark_group("lookup-table");
    bench_lookup_table(&mut group, &mut tokens[..]);
    group.finish();

    // Benchmarks for random string generation
    let mut group = criterion.benchmark_group("generate-random");
    bench_generate_random(&mut group, &mut tokens[..]);
    group.finish();

    criterion.final_summary();
}
