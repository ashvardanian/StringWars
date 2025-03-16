#![doc = r#"
# StringWa.rs: Substring Search Benchmarks

This file benchmarks the forward and backward exact substring search functionality provided by
the StringZilla library and the memchr crate. The input file is treated as a haystack and all
of its tokens as needles. The throughput numbers are reported in Gigabytes per Second and for
any sampled token - all of its inclusions in a string are located.

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
    cargo criterion --features bench_find bench_find --jobs 8
```
"#]
use std::env;
use std::fs;
use std::time::Duration;

use criterion::{black_box, Criterion, Throughput};

use aho_corasick::AhoCorasick;
use bstr::ByteSlice;
use memchr::memmem;
use regex::bytes::Regex;
use stringzilla::sz::{
    find as sz_find,
    find_byteset as sz_find_byteset, //
    rfind as sz_rfind,
    Byteset,
};

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
        .sample_size(10) // Each loop scans the whole dataset.
        .warm_up_time(Duration::from_secs(10)) // Let the CPU frequencies settle.
        .measurement_time(Duration::from_secs(120)) // Actual measurement time.
}

fn bench_find(c: &mut Criterion) {
    // Get the haystack path from the environment variable.
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_TOKENS").unwrap_or_else(|_| "lines".to_string());
    let haystack_content = fs::read_to_string(&dataset_path).expect("Could not read haystack");

    // Tokenize the haystack content by white space or lines.
    let needles: Vec<&str> = match mode.as_str() {
        "lines" => haystack_content.lines().collect(),
        "words" => haystack_content.split_whitespace().collect(),
        other => panic!(
            "Unknown STRINGWARS_TOKENS: {}. Use 'lines' or 'words'.",
            other
        ),
    };

    if needles.is_empty() {
        panic!("No tokens found in the haystack.");
    }

    let haystack = haystack_content.as_bytes();
    let haystack_length = haystack.len();

    // Benchmarks for forward search
    let mut g = c.benchmark_group("substring-forward");
    g.throughput(Throughput::Bytes(haystack_length as u64));
    bench_substring_forward(&mut g, &needles, haystack);
    g.finish();

    // Benchmarks for backward search
    let mut g = c.benchmark_group("substring-backward");
    g.throughput(Throughput::Bytes(haystack_length as u64));
    bench_substring_backward(&mut g, &needles, haystack);
    g.finish();

    // Benchmarks for byteset search
    let mut g = c.benchmark_group("byteset-forward");
    g.throughput(Throughput::Bytes(3 * haystack_length as u64));
    bench_byteset_forward(&mut g, &needles);
    g.finish();
}

fn bench_substring_forward(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    needles: &[&str],
    haystack: &[u8],
) {
    // Benchmark for StringZilla forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("sz::find", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let token_bytes = black_box(token.as_bytes());
            let mut pos: usize = 0;
            while let Some(found) = sz_find(&haystack[pos..], token_bytes) {
                pos += found + token_bytes.len();
            }
        })
    });

    // Benchmark for `memmem` forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("memmem::find", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let token_bytes = black_box(token.as_bytes());
            let mut pos: usize = 0;
            while let Some(found) = memmem::find(&haystack[pos..], token_bytes) {
                pos += found + token_bytes.len();
            }
        })
    });

    // Benchmark for default `std::str` forward search.
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
}

fn bench_substring_backward(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    needles: &[&str],
    haystack: &[u8],
) {
    // Benchmark for StringZilla backward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("sz::rfind", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let token_bytes = black_box(token.as_bytes());
            let mut pos: Option<usize> = Some(haystack.len());
            while let Some(end) = pos {
                if let Some(found) = sz_rfind(&haystack[..end], token_bytes) {
                    pos = Some(found);
                } else {
                    break;
                }
            }
        })
    });

    // Benchmark for memmem backward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    g.bench_function("memmem::rfind", |b| {
        b.iter(|| {
            let token = black_box(*tokens.next().unwrap());
            let token_bytes = black_box(token.as_bytes());
            let mut pos: Option<usize> = Some(haystack.len());
            while let Some(end) = pos {
                if let Some(found) = memmem::rfind(&haystack[..end], token_bytes) {
                    pos = Some(found);
                } else {
                    break;
                }
            }
        })
    });

    // Benchmark for default `std::str` backward search.
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
}

fn bench_byteset_forward(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    needles: &[&str],
) {
    // Define the three bytesets we will analyze.
    const BYTES_TABS: &[u8] = b"\n\r\x0B\x0C";
    const BYTES_HTML: &[u8] = b"</>&'\"=[]";
    const BYTES_DIGITS: &[u8] = b"0123456789";

    // Benchmark for StringZilla forward search using a cycle iterator.
    let sz_tabs = Byteset::from(BYTES_TABS);
    let sz_html = Byteset::from(BYTES_HTML);
    let sz_digits = Byteset::from(BYTES_DIGITS);
    g.bench_function("sz::find_byteset", |b| {
        b.iter(|| {
            for token in needles.iter() {
                let token_bytes = black_box(token.as_bytes());
                let mut pos: usize = 0;
                while let Some(found) = sz_find_byteset(&token_bytes[pos..], sz_tabs) {
                    pos += found + 1;
                }
                pos = 0;
                while let Some(found) = sz_find_byteset(&token_bytes[pos..], sz_html) {
                    pos += found + 1;
                }
                pos = 0;
                while let Some(found) = sz_find_byteset(&token_bytes[pos..], sz_digits) {
                    pos += found + 1;
                }
            }
        })
    });

    // Benchmark for bstr's byteset search.
    g.bench_function("bstr::iter", |b| {
        b.iter(|| {
            for token in needles.iter() {
                let token_bytes = black_box(token.as_bytes());
                let mut pos: usize = 0;
                // Inline search for `BYTES_TABS`.
                while let Some(found) = token_bytes[pos..]
                    .iter()
                    .position(|&c| BYTES_TABS.contains(&c))
                {
                    pos += found + 1;
                }
                pos = 0;
                // Inline search for `BYTES_HTML`.
                while let Some(found) = token_bytes[pos..]
                    .iter()
                    .position(|&c| BYTES_HTML.contains(&c))
                {
                    pos += found + 1;
                }
                pos = 0;
                // Inline search for `BYTES_DIGITS`.
                while let Some(found) = token_bytes[pos..]
                    .iter()
                    .position(|&c| BYTES_DIGITS.contains(&c))
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
    let mut criterion = configure_bench();
    bench_find(&mut criterion);
    criterion.final_summary();
}
