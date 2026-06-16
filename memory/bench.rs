#![doc = r#"
# StringWars: Low-level Memory-related Benchmarks

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
    cargo criterion --features bench_memory bench_memory --jobs $(nproc)
```
"#]
use std::env;
use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::ptr;
use std::slice;

use criterion::{Criterion, Throughput};

use getrandom;
use rand;
use rand::{RngCore, SeedableRng};
use rand_chacha;
use rand_xoshiro;
use stringzilla::sz;
use zeroize::Zeroize;

#[path = "../utils.rs"]
mod utils;
use criterion::measurement::WallTime;
use utils::{configure_bench, install_panic_hook, log_stringzilla_metadata, should_run, ResultExt};

/// Reads the raw dataset bytes named by `STRINGWARS_DATASET`.
///
/// The in-place LUT/translate/PRNG benchmarks mutate their tokens, so they need owned,
/// mutable bytes and mutable token slices; the shared `utils::load_dataset` returns an
/// immutable, leaked tape and cannot be used here.
pub fn load_dataset_bytes() -> Result<Vec<u8>, Box<dyn Error>> {
    let dataset_path = env::var("STRINGWARS_DATASET")
        .map_err(|_| "STRINGWARS_DATASET environment variable not set")?;
    let content = fs::read(&dataset_path)?;
    Ok(content)
}

/// Tokenizes the haystack into mutable slices based on `STRINGWARS_TOKENS`.
/// Supported modes: "lines", "words", and "file".
pub fn tokenize_mut<'a>(haystack: &'a mut [u8]) -> Result<Vec<&'a mut [u8]>, Box<dyn Error>> {
    let mode = env::var("STRINGWARS_TOKENS").unwrap_or_else(|_| "lines".to_string());
    let tokens = match mode.as_str() {
        "lines" => haystack
            .split_mut(|&byte| byte == b'\n')
            .filter(|token| !token.is_empty())
            .collect(),
        "words" => haystack
            .split_mut(|&byte| byte == b'\n' || byte == b' ')
            .filter(|token| !token.is_empty())
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

fn bench_lookup_table(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &mut [&mut [u8]],
) {
    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = tokens.iter().map(|token| token.len()).sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Benchmark for StringZilla forward search using a cycle iterator.
    let mut lookup_invert_case: [u8; 256] = core::array::from_fn(|index| index as u8);
    for (upper, lower) in ('A'..='Z').zip('a'..='z') {
        lookup_invert_case[upper as usize] = lower as u8;
    }
    for (upper, lower) in ('A'..='Z').zip('a'..='z') {
        lookup_invert_case[lower as usize] = upper as u8;
    }

    // Benchmark using StringZilla's `lookup_inplace`.
    if should_run("lookup-table/stringzilla::lookup_inplace") {
        group.bench_function("stringzilla::lookup_inplace", |bencher| {
            bencher.iter(|| {
                for token in tokens.iter_mut() {
                    sz::lookup_inplace(&mut *token, lookup_invert_case);
                    black_box(token);
                }
            })
        });
    }

    // Benchmark a plain serial mapping using the same lookup table.
    if should_run("lookup-table/serial") {
        group.bench_function("serial", |bencher| {
            bencher.iter(|| {
                for token in tokens.iter_mut() {
                    for byte in token.iter_mut() {
                        *byte = lookup_invert_case[*byte as usize];
                    }
                    black_box(&token);
                }
            })
        });
    }
}

fn bench_generate_random(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &mut [&mut [u8]],
) {
    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = tokens.iter().map(|token| token.len()).sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Benchmark for StringZilla AES-based PRNG
    if should_run("generate-random/stringzilla::fill_random") {
        group.bench_function("stringzilla::fill_random", |bencher| {
            bencher.iter(|| {
                for token in tokens.iter_mut() {
                    sz::fill_random(&mut *token, 0)
                }
            })
        });
    }

    // Benchmark using zeroize to obfuscate (zero out) the buffer.
    if should_run("generate-random/zeroize::zeroize") {
        group.bench_function("zeroize::zeroize", |bencher| {
            bencher.iter(|| {
                for token in tokens.iter_mut() {
                    token.zeroize();
                    black_box(&token);
                }
            })
        });
    }

    // Benchmark using `getrandom` to randomize the buffer via the OS.
    if should_run("generate-random/getrandom::fill") {
        group.bench_function("getrandom::fill", |bencher| {
            bencher.iter(|| {
                for token in tokens.iter_mut() {
                    getrandom::fill(&mut *token).expect("getrandom failed");
                    black_box(&token);
                }
            })
        });
    }

    // Benchmark using `rand_chacha::ChaCha20Rng`.
    if should_run("generate-random/rand_chacha::ChaCha20Rng") {
        group.bench_function("rand_chacha::ChaCha20Rng", |bencher| {
            bencher.iter(|| {
                for token in tokens.iter_mut() {
                    let mut random_generator = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
                    random_generator.fill_bytes(&mut *token);
                    black_box(&token);
                }
            })
        });
    }

    // Benchmark using `rand_xoshiro::Xoshiro128Plus`.
    if should_run("generate-random/rand_xoshiro::Xoshiro128Plus") {
        group.bench_function("rand_xoshiro::Xoshiro128Plus", |bencher| {
            bencher.iter(|| {
                for token in tokens.iter_mut() {
                    let mut random_generator = rand_xoshiro::Xoshiro128Plus::from_seed([0u8; 16]);
                    random_generator.fill_bytes(&mut *token);
                    black_box(&token);
                }
            })
        });
    }
}

fn bench_memset(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &mut [&mut [u8]],
) {
    const FILL_VALUE: u8 = 0xAA;
    let templates: Vec<Vec<u8>> = tokens.iter().map(|token| (**token).to_vec()).collect();
    let total_bytes: usize = templates.iter().map(|buffer| buffer.len()).sum();
    if total_bytes == 0 {
        return;
    }
    group.throughput(Throughput::Bytes(total_bytes as u64));

    if should_run("memset/stringzilla::fill") {
        let mut buffers = templates.clone();
        group.bench_function("stringzilla::fill", |bencher| {
            bencher.iter(|| {
                for buffer in buffers.iter_mut() {
                    sz::fill(buffer, FILL_VALUE);
                    black_box(&buffer);
                }
            })
        });
    }

    if should_run("memset/std::ptr::write_bytes") {
        let mut buffers = templates.clone();
        group.bench_function("std::ptr::write_bytes", |bencher| {
            bencher.iter(|| {
                for buffer in buffers.iter_mut() {
                    unsafe {
                        ptr::write_bytes(buffer.as_mut_ptr(), FILL_VALUE, buffer.len());
                    }
                    black_box(&buffer);
                }
            })
        });
    }

    if should_run("memset/slice::fill") {
        let mut buffers = templates.clone();
        group.bench_function("slice::fill", |bencher| {
            bencher.iter(|| {
                for buffer in buffers.iter_mut() {
                    buffer.fill(FILL_VALUE);
                    black_box(&buffer);
                }
            })
        });
    }
}

fn bench_memcpy(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &mut [&mut [u8]],
) {
    let sources: Vec<Vec<u8>> = tokens.iter().map(|token| (**token).to_vec()).collect();
    let dest_template: Vec<Vec<u8>> = sources.iter().map(|src| vec![0u8; src.len()]).collect();
    let total_bytes: usize = sources.iter().map(|buffer| buffer.len()).sum();
    if total_bytes == 0 {
        return;
    }
    group.throughput(Throughput::Bytes(total_bytes as u64));

    if should_run("memcpy/stringzilla::copy") {
        let mut dests = dest_template.clone();
        group.bench_function("stringzilla::copy", |bencher| {
            bencher.iter(|| {
                for (src, dst) in sources.iter().zip(dests.iter_mut()) {
                    sz::copy(dst, src);
                    black_box(&dst);
                }
            })
        });
    }

    if should_run("memcpy/slice::copy_from_slice") {
        let mut dests = dest_template.clone();
        group.bench_function("slice::copy_from_slice", |bencher| {
            bencher.iter(|| {
                for (src, dst) in sources.iter().zip(dests.iter_mut()) {
                    dst.copy_from_slice(src);
                    black_box(&dst);
                }
            })
        });
    }

    if should_run("memcpy/std::ptr::copy_nonoverlapping") {
        let mut dests = dest_template.clone();
        group.bench_function("std::ptr::copy_nonoverlapping", |bencher| {
            bencher.iter(|| {
                for (src, dst) in sources.iter().zip(dests.iter_mut()) {
                    unsafe {
                        ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), src.len());
                    }
                    black_box(&dst);
                }
            })
        });
    }
}

fn bench_memmove(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &mut [&mut [u8]],
) {
    const SHIFT: usize = 8;
    let templates: Vec<Vec<u8>> = tokens
        .iter()
        .filter_map(|token| {
            let slice = &**token;
            if slice.len() <= SHIFT {
                None
            } else {
                Some(slice.to_vec())
            }
        })
        .collect();
    if templates.is_empty() {
        return;
    }
    let total_bytes: usize = templates.iter().map(|buffer| buffer.len() - SHIFT).sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    if should_run("memmove/stringzilla::move_") {
        let mut buffers = templates.clone();
        group.bench_function("stringzilla::move_", |bencher| {
            bencher.iter(|| {
                for buffer in buffers.iter_mut() {
                    let move_len = buffer.len() - SHIFT;
                    unsafe {
                        let src = slice::from_raw_parts(buffer.as_ptr(), move_len);
                        let dst =
                            slice::from_raw_parts_mut(buffer.as_mut_ptr().add(SHIFT), move_len);
                        sz::move_(dst, &src);
                    }
                    black_box(&buffer);
                }
            })
        });
    }

    if should_run("memmove/std::ptr::copy") {
        let mut buffers = templates.clone();
        group.bench_function("std::ptr::copy", |bencher| {
            bencher.iter(|| {
                for buffer in buffers.iter_mut() {
                    let move_len = buffer.len() - SHIFT;
                    unsafe {
                        ptr::copy(buffer.as_ptr(), buffer.as_mut_ptr().add(SHIFT), move_len);
                    }
                    black_box(&buffer);
                }
            })
        });
    }

    if should_run("memmove/slice::copy_within") {
        let mut buffers = templates.clone();
        group.bench_function("slice::copy_within", |bencher| {
            bencher.iter(|| {
                for buffer in buffers.iter_mut() {
                    let move_len = buffer.len() - SHIFT;
                    buffer.copy_within(0..move_len, SHIFT);
                    black_box(&buffer);
                }
            })
        });
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    // Load the dataset defined by the environment variables
    let mut dataset = load_dataset_bytes().unwrap_nice();
    let mut tokens = tokenize_mut(&mut dataset).unwrap_nice();
    if tokens.is_empty() {
        panic!("No tokens found in the dataset.");
    }

    let mut criterion = configure_bench(WallTime, 1, 20);

    // Benchmarks for lookup table transform
    let mut group = criterion.benchmark_group("lookup-table");
    bench_lookup_table(&mut group, &mut tokens[..]);
    group.finish();

    // Benchmarks for random string generation
    let mut group = criterion.benchmark_group("generate-random");
    bench_generate_random(&mut group, &mut tokens[..]);
    group.finish();

    // Benchmarks for memory fill operations
    let mut group = criterion.benchmark_group("memset");
    bench_memset(&mut group, &mut tokens[..]);
    group.finish();

    // Benchmarks for memory copy operations
    let mut group = criterion.benchmark_group("memcpy");
    bench_memcpy(&mut group, &mut tokens[..]);
    group.finish();

    // Benchmarks for memory move operations
    let mut group = criterion.benchmark_group("memmove");
    bench_memmove(&mut group, &mut tokens[..]);
    group.finish();

    criterion.final_summary();
}
