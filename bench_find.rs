#![doc = r#"
# StringWars: Substring & Character-Set Search Benchmarks

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
use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput};
use stringtape::BytesCowsAuto;

use aho_corasick::AhoCorasick;
use bstr::ByteSlice;
use icu::properties::props::WhiteSpace;
use icu::properties::CodePointSetData;
use memchr::memmem;
use regex::bytes::Regex;
use stringzilla::sz;

mod utils;
use utils::{load_dataset, should_run};

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

/// Benchmarks forward substring search using "StringZilla", "MemMem", and standard strings.
fn bench_substring_forward(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    needles: &BytesCowsAuto,
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for StringZilla forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    if should_run("substring-forward/stringzilla::find") {
        g.bench_function("stringzilla::find", |b| {
            b.iter(|| {
                let token = black_box(tokens.next().unwrap());
                let mut pos: usize = 0;
                while let Some(found) = sz::find(&haystack[pos..], token) {
                    pos += found + token.len();
                }
            })
        });
    }

    // Benchmark for `memmem::find` forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    if should_run("substring-forward/memmem::find") {
        g.bench_function("memmem::find", |b| {
            b.iter(|| {
                let token = black_box(tokens.next().unwrap());
                let mut pos: usize = 0;
                while let Some(found) = memmem::find(&haystack[pos..], token) {
                    pos += found + token.len();
                }
            })
        });
    }

    // Benchmark for `memmem::Finder` forward search with pre-constructed matcher.
    let mut tokens = needles.iter().cycle();
    if should_run("substring-forward/memmem::Finder") {
        g.bench_function("memmem::Finder", |b| {
            b.iter(|| {
                let token = black_box(tokens.next().unwrap());
                let finder = memmem::Finder::new(token);
                let mut pos: usize = 0;
                while let Some(found) = finder.find(&haystack[pos..]) {
                    pos += found + token.len();
                }
            })
        });
    }

    // Benchmark for default `std::str::find` forward search.
    let mut tokens = needles.iter().cycle();
    if should_run("substring-forward/std::str::find") {
        g.bench_function("std::str::find", |b| {
            b.iter(|| {
                let token = black_box(tokens.next().unwrap());
                let mut pos = 0;
                while let Some(found) = haystack[pos..].find(token) {
                    pos += found + token.len();
                }
            })
        });
    }
}

/// Benchmarks backward substring search using "StringZilla", "MemMem", and standard strings.
fn bench_substring_backward(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    needles: &BytesCowsAuto,
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for StringZilla backward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    if should_run("substring-backward/stringzilla::rfind") {
        g.bench_function("stringzilla::rfind", |b| {
            b.iter(|| {
                let token = black_box(tokens.next().unwrap());
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
    }

    // Benchmark for `memmem::rfind` backward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    if should_run("substring-backward/memmem::rfind") {
        g.bench_function("memmem::rfind", |b| {
            b.iter(|| {
                let token = black_box(tokens.next().unwrap());
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
    }

    // Benchmark for `memmem::FinderRev` backward search with pre-constructed matcher.
    let mut tokens = needles.iter().cycle();
    if should_run("substring-backward/memmem::FinderRev") {
        g.bench_function("memmem::FinderRev", |b| {
            b.iter(|| {
                let token = black_box(tokens.next().unwrap());
                let finder = memmem::FinderRev::new(token);
                let mut pos: Option<usize> = Some(haystack.len());
                while let Some(end) = pos {
                    if let Some(found) = finder.rfind(&haystack[..end]) {
                        pos = Some(found);
                    } else {
                        break;
                    }
                }
            })
        });
    }

    // Benchmark for default `std::str::rfind` backward search.
    let mut tokens = needles.iter().cycle();
    if should_run("substring-backward/std::str::rfind") {
        g.bench_function("std::str::rfind", |b| {
            b.iter(|| {
                let token = black_box(tokens.next().unwrap());
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
}

/// Benchmarks byteset search using "StringZilla", "bstr", "RegEx", and "AhoCorasick"
fn bench_byteset_forward(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    needles: &BytesCowsAuto,
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
    if should_run("byteset-forward/stringzilla::find_byteset") {
        g.bench_function("stringzilla::find_byteset", |b| {
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
    }

    // Benchmark for bstr's byteset search.
    if should_run("byteset-forward/bstr::iter") {
        g.bench_function("bstr::iter", |b| {
            b.iter(|| {
                for token in needles.iter() {
                    let mut pos: usize = 0;
                    // Inline search for `BYTES_TABS`.
                    while let Some(found) =
                        token[pos..].iter().position(|&c| BYTES_TABS.contains(&c))
                    {
                        pos += found + 1;
                    }
                    pos = 0;
                    // Inline search for `BYTES_HTML`.
                    while let Some(found) =
                        token[pos..].iter().position(|&c| BYTES_HTML.contains(&c))
                    {
                        pos += found + 1;
                    }
                    pos = 0;
                    // Inline search for `BYTES_DIGITS`.
                    while let Some(found) =
                        token[pos..].iter().position(|&c| BYTES_DIGITS.contains(&c))
                    {
                        pos += found + 1;
                    }
                }
            })
        });
    }

    // Benchmark for Regex-based byteset search.
    let re_tabs = Regex::new("[\n\r\x0B\x0C]").unwrap();
    let re_html = Regex::new("[</>&'\"=\\[\\]]").unwrap();
    let re_digits = Regex::new("[0-9]").unwrap();
    if should_run("byteset-forward/regex::find_iter") {
        g.bench_function("regex::find_iter", |b| {
            b.iter(|| {
                for token in needles.iter() {
                    black_box(re_tabs.find_iter(token.as_bytes()).count());
                    black_box(re_html.find_iter(token.as_bytes()).count());
                    black_box(re_digits.find_iter(token.as_bytes()).count());
                }
            })
        });
    }

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
    if should_run("byteset-forward/aho_corasick::find_iter") {
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
}

/// Benchmarks Unicode whitespace splitting using ICU, stdlib, and StringZilla.
fn bench_utf8_whitespaces(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    _needles: &BytesCowsAuto,
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for StringZilla whitespace splits.
    if should_run("utf8-whitespaces/stringzilla::utf8_whitespace_splits().count()") {
        use sz::StringZillableUnary;
        g.bench_function("stringzilla::utf8_whitespace_splits().count()", |b| {
            b.iter(|| {
                let haystack_bytes = black_box(haystack);
                let count: usize = haystack_bytes.sz_utf8_whitespace_splits().count();
                black_box(count);
            })
        });
    }

    // Benchmark for Rust stdlib char::is_whitespace.
    if should_run("utf8-whitespaces/stdlib::split(char::is_whitespace).count()") {
        g.bench_function("stdlib::split(char::is_whitespace).count()", |b| {
            b.iter(|| {
                let haystack_str = black_box(std::str::from_utf8(haystack).unwrap());
                let count: usize = haystack_str
                    .split(char::is_whitespace)
                    .filter(|s| !s.is_empty())
                    .count();
                black_box(count);
            })
        });
    }

    // Benchmark for ICU4X WhiteSpace property.
    if should_run("utf8-whitespaces/icu::WhiteSpace.split().count()") {
        let white_space = CodePointSetData::new::<WhiteSpace>();
        g.bench_function("icu::WhiteSpace.split().count()", |b| {
            b.iter(|| {
                let haystack_str = black_box(std::str::from_utf8(haystack).unwrap());
                let count: usize = haystack_str
                    .split(|c: char| white_space.contains(c))
                    .filter(|s: &&str| !s.is_empty())
                    .count();
                black_box(count);
            })
        });
    }
}

/// Benchmarks Unicode newline splitting using custom predicates and StringZilla.
fn bench_utf8_newlines(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    _needles: &BytesCowsAuto,
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Custom newline predicate matching StringZilla's 7 newline characters.
    fn is_unicode_newline(c: char) -> bool {
        matches!(
            c,
            '\n' | '\r' | '\x0B' | '\x0C' | '\u{0085}' | '\u{2028}' | '\u{2029}'
        )
    }

    // Benchmark for StringZilla newline splits.
    if should_run("utf8-newlines/stringzilla::utf8_newline_splits().count()") {
        use sz::StringZillableUnary;
        g.bench_function("stringzilla::utf8_newline_splits().count()", |b| {
            b.iter(|| {
                let haystack_bytes = black_box(haystack);
                let count: usize = haystack_bytes.sz_utf8_newline_splits().count();
                black_box(count);
            })
        });
    }

    // Benchmark for custom newline predicate.
    if should_run("utf8-newlines/custom::split(is_unicode_newline).count()") {
        g.bench_function("custom::split(is_unicode_newline).count()", |b| {
            b.iter(|| {
                let haystack_str = black_box(std::str::from_utf8(haystack).unwrap());
                let count: usize = haystack_str
                    .split(is_unicode_newline)
                    .filter(|s| !s.is_empty())
                    .count();
                black_box(count);
            })
        });
    }
}

/// Benchmarks UTF-8 character counting using StringZilla, simdutf, and stdlib.
fn bench_utf8_length(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    _needles: &BytesCowsAuto,
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for StringZilla UTF-8 character counting.
    if should_run("utf8-length/stringzilla::utf8_chars().len()") {
        use sz::StringZillableUnary;
        g.bench_function("stringzilla::utf8_chars().len()", |b| {
            b.iter(|| {
                let haystack_bytes = black_box(haystack);
                let count: usize = haystack_bytes.sz_utf8_chars().len();
                black_box(count);
            })
        });
    }

    // Benchmark for simdutf UTF-8 character counting.
    if should_run("utf8-length/simdutf::count_utf8()") {
        g.bench_function("simdutf::count_utf8()", |b| {
            b.iter(|| {
                let haystack_bytes = black_box(haystack);
                let count: usize = simdutf::count_utf8(haystack_bytes);
                black_box(count);
            })
        });
    }

    // Benchmark for stdlib UTF-8 character counting.
    if should_run("utf8-length/stdlib::chars().count()") {
        g.bench_function("stdlib::chars().count()", |b| {
            b.iter(|| {
                let haystack_str = black_box(std::str::from_utf8(haystack).unwrap());
                let count: usize = haystack_str.chars().count();
                black_box(count);
            })
        });
    }
}

/// Benchmarks UTF-8 to UTF-32 decoding using StringZilla, simdutf, and stdlib.
fn bench_utf8_iterator(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    _needles: &BytesCowsAuto,
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for StringZilla UTF-8 character iteration.
    if should_run("utf8-iterator/stringzilla::utf8_chars().iter()") {
        use sz::StringZillableUnary;
        g.bench_function("stringzilla::utf8_chars().iter()", |b| {
            b.iter(|| {
                let haystack_bytes = black_box(haystack);
                let mut sum: u32 = 0;
                for ch in haystack_bytes.sz_utf8_chars().iter() {
                    sum = sum.wrapping_add(ch as u32);
                }
                black_box(sum);
            })
        });
    }

    // Benchmark for simdutf UTF-8 to UTF-32 conversion.
    if should_run("utf8-iterator/simdutf::convert_utf8_to_utf32()") {
        // Pre-allocate buffer for UTF-32 output (worst case: same number of codepoints as bytes)
        let mut utf32_buffer = vec![0u32; haystack.len()];
        g.bench_function("simdutf::convert_utf8_to_utf32()", |b| {
            b.iter(|| {
                let haystack_bytes = black_box(haystack);
                let len = unsafe {
                    simdutf::convert_utf8_to_utf32(
                        haystack_bytes.as_ptr(),
                        haystack_bytes.len(),
                        utf32_buffer.as_mut_ptr(),
                    )
                };
                let mut sum: u32 = 0;
                for i in 0..len {
                    sum = sum.wrapping_add(utf32_buffer[i]);
                }
                black_box(sum);
            })
        });
    }

    // Benchmark for stdlib UTF-8 character iteration.
    if should_run("utf8-iterator/stdlib::chars()") {
        g.bench_function("stdlib::chars()", |b| {
            b.iter(|| {
                let haystack_str = black_box(unsafe { std::str::from_utf8_unchecked(haystack) });
                let mut sum: u32 = 0;
                for ch in haystack_str.chars() {
                    sum = sum.wrapping_add(ch as u32);
                }
                black_box(sum);
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

    // Get the parent data directly from the tape (zero-copy)
    let haystack = tape.parent();
    let needles = &tape;

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

    // Benchmarks for Unicode whitespace splitting
    let mut group = criterion.benchmark_group("utf8-whitespaces");
    bench_utf8_whitespaces(&mut group, &haystack, &needles);
    group.finish();

    // Benchmarks for Unicode newline splitting
    let mut group = criterion.benchmark_group("utf8-newlines");
    bench_utf8_newlines(&mut group, &haystack, &needles);
    group.finish();

    // Benchmarks for UTF-8 character counting
    let mut group = criterion.benchmark_group("utf8-length");
    bench_utf8_length(&mut group, &haystack, &needles);
    group.finish();

    // Benchmarks for UTF-8 character iteration
    let mut group = criterion.benchmark_group("utf8-iterator");
    bench_utf8_iterator(&mut group, &haystack, &needles);
    group.finish();

    criterion.final_summary();
}
