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

use getrandom;
use rand;
use rand::{RngCore, SeedableRng};
use rand_chacha;
use rand_xoshiro;
use stringzilla::sz;
use zeroize::Zeroize;

#[path = "../utils.rs"]
mod utils;
use utils::{
    install_panic_hook, log_stringzilla_metadata, measure_throughput, BenchBudget, ReportAs,
    ResultExt, WorkUnits,
};

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

/// Benchmarks in-place lookup-table transforms, transforming one token per call and cycling the
/// dataset. Throughput is reported as bytes/s, matching the original `Throughput::Bytes` over the
/// sum of token lengths.
fn bench_lookup_table(budget: &BenchBudget, tokens: &mut [&mut [u8]]) {
    // Build the case-inverting lookup table.
    let mut lookup_invert_case: [u8; 256] = core::array::from_fn(|index| index as u8);
    for (upper, lower) in ('A'..='Z').zip('a'..='z') {
        lookup_invert_case[upper as usize] = lower as u8;
    }
    for (upper, lower) in ('A'..='Z').zip('a'..='z') {
        lookup_invert_case[lower as usize] = upper as u8;
    }

    let token_count = tokens.len();

    // Benchmark using StringZilla's `lookup_inplace`.
    {
        let mut cursor = 0usize;
        measure_throughput(
            "lookup-table/stringzilla::lookup_inplace",
            ReportAs::Bytes,
            budget,
            || {
                let token = &mut tokens[cursor % token_count];
                cursor += 1;
                let token_bytes = token.len() as u64;
                sz::lookup_inplace(&mut **token, lookup_invert_case);
                black_box(token);
                WorkUnits::new(1, token_bytes)
            },
        );
    }

    // Benchmark a plain serial mapping using the same lookup table.
    {
        let mut cursor = 0usize;
        measure_throughput("lookup-table/serial", ReportAs::Bytes, budget, || {
            let token = &mut tokens[cursor % token_count];
            cursor += 1;
            let token_bytes = token.len() as u64;
            for byte in token.iter_mut() {
                *byte = lookup_invert_case[*byte as usize];
            }
            black_box(&token);
            WorkUnits::new(1, token_bytes)
        });
    }
}

/// Benchmarks random-string generation, filling one token per call and cycling the dataset.
/// Throughput is reported as bytes/s, matching the original `Throughput::Bytes` over the sum of
/// token lengths.
fn bench_generate_random(budget: &BenchBudget, tokens: &mut [&mut [u8]]) {
    let token_count = tokens.len();

    // Benchmark for StringZilla AES-based PRNG.
    {
        let mut cursor = 0usize;
        measure_throughput(
            "generate-random/stringzilla::fill_random",
            ReportAs::Bytes,
            budget,
            || {
                let token = &mut tokens[cursor % token_count];
                cursor += 1;
                let token_bytes = token.len() as u64;
                sz::fill_random(&mut **token, 0);
                WorkUnits::new(1, token_bytes)
            },
        );
    }

    // Benchmark using zeroize to obfuscate (zero out) the buffer.
    {
        let mut cursor = 0usize;
        measure_throughput(
            "generate-random/zeroize::zeroize",
            ReportAs::Bytes,
            budget,
            || {
                let token = &mut tokens[cursor % token_count];
                cursor += 1;
                let token_bytes = token.len() as u64;
                token.zeroize();
                black_box(&token);
                WorkUnits::new(1, token_bytes)
            },
        );
    }

    // Benchmark using `getrandom` to randomize the buffer via the OS.
    {
        let mut cursor = 0usize;
        measure_throughput(
            "generate-random/getrandom::fill",
            ReportAs::Bytes,
            budget,
            || {
                let token = &mut tokens[cursor % token_count];
                cursor += 1;
                let token_bytes = token.len() as u64;
                getrandom::fill(&mut **token).expect("getrandom failed");
                black_box(&token);
                WorkUnits::new(1, token_bytes)
            },
        );
    }

    // Benchmark using `rand_chacha::ChaCha20Rng`.
    {
        let mut cursor = 0usize;
        measure_throughput(
            "generate-random/rand_chacha::ChaCha20Rng",
            ReportAs::Bytes,
            budget,
            || {
                let token = &mut tokens[cursor % token_count];
                cursor += 1;
                let token_bytes = token.len() as u64;
                let mut random_generator = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
                random_generator.fill_bytes(&mut **token);
                black_box(&token);
                WorkUnits::new(1, token_bytes)
            },
        );
    }

    // Benchmark using `rand_xoshiro::Xoshiro128Plus`.
    {
        let mut cursor = 0usize;
        measure_throughput(
            "generate-random/rand_xoshiro::Xoshiro128Plus",
            ReportAs::Bytes,
            budget,
            || {
                let token = &mut tokens[cursor % token_count];
                cursor += 1;
                let token_bytes = token.len() as u64;
                let mut random_generator = rand_xoshiro::Xoshiro128Plus::from_seed([0u8; 16]);
                random_generator.fill_bytes(&mut **token);
                black_box(&token);
                WorkUnits::new(1, token_bytes)
            },
        );
    }
}

/// Benchmarks memory-fill operations, filling one buffer per call and cycling the dataset.
fn bench_memset(budget: &BenchBudget, tokens: &mut [&mut [u8]]) {
    const FILL_VALUE: u8 = 0xAA;
    let templates: Vec<Vec<u8>> = tokens.iter().map(|token| (**token).to_vec()).collect();
    let total_bytes: usize = templates.iter().map(|buffer| buffer.len()).sum();
    if total_bytes == 0 {
        return;
    }
    let buffer_count = templates.len();

    {
        let mut buffers = templates.clone();
        let mut cursor = 0usize;
        measure_throughput("memset/stringzilla::fill", ReportAs::Bytes, budget, || {
            let buffer = &mut buffers[cursor % buffer_count];
            cursor += 1;
            let buffer_bytes = buffer.len() as u64;
            sz::fill(buffer, FILL_VALUE);
            black_box(&buffer);
            WorkUnits::new(1, buffer_bytes)
        });
    }

    {
        let mut buffers = templates.clone();
        let mut cursor = 0usize;
        measure_throughput(
            "memset/std::ptr::write_bytes",
            ReportAs::Bytes,
            budget,
            || {
                let buffer = &mut buffers[cursor % buffer_count];
                cursor += 1;
                let buffer_bytes = buffer.len() as u64;
                unsafe {
                    ptr::write_bytes(buffer.as_mut_ptr(), FILL_VALUE, buffer.len());
                }
                black_box(&buffer);
                WorkUnits::new(1, buffer_bytes)
            },
        );
    }

    {
        let mut buffers = templates.clone();
        let mut cursor = 0usize;
        measure_throughput("memset/slice::fill", ReportAs::Bytes, budget, || {
            let buffer = &mut buffers[cursor % buffer_count];
            cursor += 1;
            let buffer_bytes = buffer.len() as u64;
            buffer.fill(FILL_VALUE);
            black_box(&buffer);
            WorkUnits::new(1, buffer_bytes)
        });
    }
}

/// Benchmarks memory-copy operations, copying one buffer per call and cycling the dataset.
fn bench_memcpy(budget: &BenchBudget, tokens: &mut [&mut [u8]]) {
    let sources: Vec<Vec<u8>> = tokens.iter().map(|token| (**token).to_vec()).collect();
    let dest_template: Vec<Vec<u8>> = sources.iter().map(|src| vec![0u8; src.len()]).collect();
    let total_bytes: usize = sources.iter().map(|buffer| buffer.len()).sum();
    if total_bytes == 0 {
        return;
    }
    let buffer_count = sources.len();

    {
        let mut dests = dest_template.clone();
        let mut cursor = 0usize;
        measure_throughput("memcpy/stringzilla::copy", ReportAs::Bytes, budget, || {
            let index = cursor % buffer_count;
            cursor += 1;
            let source = &sources[index];
            let dest = &mut dests[index];
            let buffer_bytes = source.len() as u64;
            sz::copy(dest, source);
            black_box(&dest);
            WorkUnits::new(1, buffer_bytes)
        });
    }

    {
        let mut dests = dest_template.clone();
        let mut cursor = 0usize;
        measure_throughput(
            "memcpy/slice::copy_from_slice",
            ReportAs::Bytes,
            budget,
            || {
                let index = cursor % buffer_count;
                cursor += 1;
                let source = &sources[index];
                let dest = &mut dests[index];
                let buffer_bytes = source.len() as u64;
                dest.copy_from_slice(source);
                black_box(&dest);
                WorkUnits::new(1, buffer_bytes)
            },
        );
    }

    {
        let mut dests = dest_template.clone();
        let mut cursor = 0usize;
        measure_throughput(
            "memcpy/std::ptr::copy_nonoverlapping",
            ReportAs::Bytes,
            budget,
            || {
                let index = cursor % buffer_count;
                cursor += 1;
                let source = &sources[index];
                let dest = &mut dests[index];
                let buffer_bytes = source.len() as u64;
                unsafe {
                    ptr::copy_nonoverlapping(source.as_ptr(), dest.as_mut_ptr(), source.len());
                }
                black_box(&dest);
                WorkUnits::new(1, buffer_bytes)
            },
        );
    }
}

/// Benchmarks memory-move operations, shifting one buffer per call and cycling the dataset.
/// Only tokens longer than `SHIFT` participate, and the per-call byte count is `len - SHIFT`,
/// matching the original `Throughput::Bytes(sum(len - SHIFT))` accounting.
fn bench_memmove(budget: &BenchBudget, tokens: &mut [&mut [u8]]) {
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
    let buffer_count = templates.len();

    {
        let mut buffers = templates.clone();
        let mut cursor = 0usize;
        measure_throughput(
            "memmove/stringzilla::move_",
            ReportAs::Bytes,
            budget,
            || {
                let buffer = &mut buffers[cursor % buffer_count];
                cursor += 1;
                let move_len = buffer.len() - SHIFT;
                unsafe {
                    let source = slice::from_raw_parts(buffer.as_ptr(), move_len);
                    let dest = slice::from_raw_parts_mut(buffer.as_mut_ptr().add(SHIFT), move_len);
                    sz::move_(dest, &source);
                }
                black_box(&buffer);
                WorkUnits::new(1, move_len as u64)
            },
        );
    }

    {
        let mut buffers = templates.clone();
        let mut cursor = 0usize;
        measure_throughput("memmove/std::ptr::copy", ReportAs::Bytes, budget, || {
            let buffer = &mut buffers[cursor % buffer_count];
            cursor += 1;
            let move_len = buffer.len() - SHIFT;
            unsafe {
                ptr::copy(buffer.as_ptr(), buffer.as_mut_ptr().add(SHIFT), move_len);
            }
            black_box(&buffer);
            WorkUnits::new(1, move_len as u64)
        });
    }

    {
        let mut buffers = templates.clone();
        let mut cursor = 0usize;
        measure_throughput(
            "memmove/slice::copy_within",
            ReportAs::Bytes,
            budget,
            || {
                let buffer = &mut buffers[cursor % buffer_count];
                cursor += 1;
                let move_len = buffer.len() - SHIFT;
                buffer.copy_within(0..move_len, SHIFT);
                black_box(&buffer);
                WorkUnits::new(1, move_len as u64)
            },
        );
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

    let budget = BenchBudget::from_env(1.0, 20.0);

    println!("# lookup-table");
    bench_lookup_table(&budget, &mut tokens[..]);

    println!("# generate-random");
    bench_generate_random(&budget, &mut tokens[..]);

    println!("# memset");
    bench_memset(&budget, &mut tokens[..]);

    println!("# memcpy");
    bench_memcpy(&budget, &mut tokens[..]);

    println!("# memmove");
    bench_memmove(&budget, &mut tokens[..]);
}
