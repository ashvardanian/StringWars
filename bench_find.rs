#![doc = r#"
# StringWa.rs: Substring & Character-Set Search Benchmarks

This file benchmarks the forward and backward exact substring search functionality provided by
the StringZilla library and the memchr crate. The input file is treated as a haystack and all
of its tokens as needles. The throughput numbers are reported in Gigabytes per Second and for
any sampled token - all of its inclusions in a string are located.
Be warned, for large files, it may take a while!

The input file is treated as a haystack and all of its tokens as needles. For substring searches,
each occurrence is located. For byteset searches, three separate operations are performed per token,
looking for:

- any of "\n\r\v\f" - the 4 tabulation characters
- any of "</>&'\"=[]" - the 9 HTML-related characters
- any of "0123456789" - the 10 numeric characters

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
    cargo criterion --features bench_find bench_find --jobs $(nproc)
```
"#]
use std::env;
use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput};

use aho_corasick::AhoCorasick;
use bstr::ByteSlice;
use memchr::memmem;
use regex::bytes::Regex;
use stringzilla::sz;

fn log_stringzilla_metadata() {
    let v = sz::version();
    println!("StringZilla v{}.{}.{}", v.major, v.minor, v.patch);
    println!("- uses dynamic dispatch: {}", sz::dynamic_dispatch());
    println!("- capabilities: {}", sz::capabilities().as_str());
}

fn configure_bench() -> Criterion {
    Criterion::default()
        .sample_size(10) // Each loop scans the whole dataset, but this can't be under 10
        .warm_up_time(Duration::from_secs(3)) // Let the CPU frequencies settle.
        .measurement_time(Duration::from_secs(20)) // Actual measurement time.
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
    let tokens = match mode.as_str() {
        "lines" => haystack
            .split(|&c| c == b'\n')
            .filter(|token| !token.is_empty())
            .collect(),
        "words" => haystack
            .split(|&c| c == b'\n' || c == b' ')
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

/// Benchmarks forward substring search using "StringZilla", "MemMem", and standard strings.
fn bench_substring_forward(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    needles: &[&[u8]],
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for StringZilla forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("sz::find", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let mut pos: usize = 0;
            while let Some(found) = sz::find(&haystack[pos..], token) {
                pos += found + token.len();
            }
        })
    });

    // Benchmark for `memmem::find` forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("memmem::find", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let mut pos: usize = 0;
            while let Some(found) = memmem::find(&haystack[pos..], token) {
                pos += found + token.len();
            }
        })
    });

    // Benchmark for default `std::str::find` forward search.
    let mut tokens = needles.iter().cycle();
    g.bench_function("std::str::find", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let mut pos = 0;
            while let Some(found) = haystack[pos..].find(token) {
                pos += found + token.len();
            }
        })
    });

    // Benchmark for `memmem::find_iter` forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("memmem::find_iter", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            for match_ in memmem::find_iter(haystack, token) {
                black_box(match_);
            }
        })
    });
}

/// Benchmarks backward substring search using "StringZilla", "MemMem", and standard strings.
fn bench_substring_backward(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    needles: &[&[u8]],
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for StringZilla backward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("sz::rfind", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let mut pos: Option<usize> = Some(haystack.len());
            while let Some(end) = pos {
                if let Some(found) = sz::rfind(&haystack[..end], token) {
                    pos = Some(found);
                } else {
                    break;
                }
            }
        })
    });

    // Benchmark for `memmem::rfind` backward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("memmem::rfind", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let mut pos: Option<usize> = Some(haystack.len());
            while let Some(end) = pos {
                if let Some(found) = memmem::rfind(&haystack[..end], token) {
                    pos = Some(found);
                } else {
                    break;
                }
            }
        })
    });

    // Benchmark for default `std::str::rfind` backward search.
    let mut tokens = needles.iter().cycle();
    g.bench_function("std::str::rfind", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let mut pos: Option<usize> = Some(haystack.len());
            while let Some(end) = pos {
                if let Some(found) = haystack[..end].rfind(token) {
                    pos = Some(found);
                } else {
                    break;
                }
            }
        })
    });

    // Benchmark for `memmem::rfind_iter` forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("memmem::rfind_iter", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            for match_ in memmem::rfind_iter(haystack, token) {
                black_box(match_);
            }
        })
    });
}

/// Benchmarks byteset search using "StringZilla", "bstr", "RegEx", and "AhoCorasick"
fn bench_byteset_forward(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    needles: &[&[u8]],
) {
    g.throughput(Throughput::Bytes(3 * haystack.len() as u64));

    // Define the three bytesets we will analyze.
    const BYTES_TABS: &[u8] = b"\n\r\x0B\x0C";
    const BYTES_HTML: &[u8] = b"</>&'\"=[]";
    const BYTES_DIGITS: &[u8] = b"0123456789";

    // Benchmark for StringZilla forward search using a cycle iterator.
    let sz_tabs = sz::Byteset::from(BYTES_TABS);
    let sz_html = sz::Byteset::from(BYTES_HTML);
    let sz_digits = sz::Byteset::from(BYTES_DIGITS);
    g.bench_function("sz::find_byteset", |b| {
        b.iter(|| {
            for token in needles.iter() {
                let mut pos: usize = 0;
                while let Some(found) = sz::find_byteset(&token[pos..], sz_tabs) {
                    pos += found + 1;
                }
                pos = 0;
                while let Some(found) = sz::find_byteset(&token[pos..], sz_html) {
                    pos += found + 1;
                }
                pos = 0;
                while let Some(found) = sz::find_byteset(&token[pos..], sz_digits) {
                    pos += found + 1;
                }
            }
        })
    });

    // Benchmark for bstr's byteset search.
    g.bench_function("bstr::iter", |b| {
        b.iter(|| {
            for token in needles.iter() {
                let mut pos: usize = 0;
                // Inline search for `BYTES_TABS`.
                while let Some(found) = token[pos..].iter().position(|&c| BYTES_TABS.contains(&c)) {
                    pos += found + 1;
                }
                pos = 0;
                // Inline search for `BYTES_HTML`.
                while let Some(found) = token[pos..].iter().position(|&c| BYTES_HTML.contains(&c)) {
                    pos += found + 1;
                }
                pos = 0;
                // Inline search for `BYTES_DIGITS`.
                while let Some(found) = token[pos..].iter().position(|&c| BYTES_DIGITS.contains(&c))
                {
                    pos += found + 1;
                }
            }
        })
    });

    // Benchmark for Regex-based byteset search.
    let re_tabs = Regex::new("[\n\r\x0B\x0C]").unwrap();
    let re_html = Regex::new("[</>&'\"=\\[\\]]").unwrap();
    let re_digits = Regex::new("[0-9]").unwrap();
    g.bench_function("regex::find_iter", |b| {
        b.iter(|| {
            for token in needles.iter() {
                black_box(re_tabs.find_iter(token.as_bytes()).count());
                black_box(re_html.find_iter(token.as_bytes()).count());
                black_box(re_digits.find_iter(token.as_bytes()).count());
            }
        })
    });

    // Benchmark for Aho–Corasick-based byteset search.
    let ac_tabs = AhoCorasick::new(
        &BYTES_TABS
            .iter()
            .map(|&b| (b as char).to_string())
            .collect::<Vec<_>>(),
    )
    .expect("failed to create AhoCorasick FSA");
    let ac_html = AhoCorasick::new(
        &BYTES_HTML
            .iter()
            .map(|&b| (b as char).to_string())
            .collect::<Vec<_>>(),
    )
    .expect("failed to create AhoCorasick FSA");
    let ac_digits = AhoCorasick::new(
        &BYTES_DIGITS
            .iter()
            .map(|&b| (b as char).to_string())
            .collect::<Vec<_>>(),
    )
    .expect("failed to create AhoCorasick FSA");
    g.bench_function("aho_corasick::find_iter", |b| {
        b.iter(|| {
            for token in needles.iter() {
                black_box(ac_tabs.find_iter(token).count());
                black_box(ac_html.find_iter(token).count());
                black_box(ac_digits.find_iter(token).count());
            }
        })
    });
}

fn main() {
    log_stringzilla_metadata();

    // Load the dataset defined by the environment variables, and panic if the content is missing
    let haystack = load_dataset().unwrap();
    let needles = tokenize(&haystack).unwrap();
    if needles.is_empty() {
        panic!("No tokens found in the dataset.");
    }

    let mut criterion = configure_bench();

    // Benchmarks for forward search
    let mut group = criterion.benchmark_group("substring-forward");
    bench_substring_forward(&mut group, &haystack, &needles);
    group.finish();

    // Benchmarks for backward search
    let mut group = criterion.benchmark_group("substring-backward");
    bench_substring_backward(&mut group, &haystack, &needles);
    group.finish();

    // Benchmarks for byteset search
    let mut group = criterion.benchmark_group("byteset-forward");
    bench_byteset_forward(&mut group, &haystack, &needles);
    group.finish();

    criterion.final_summary();
}
