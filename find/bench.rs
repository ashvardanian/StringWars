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

use stringtape::BytesCowsAuto;

use aho_corasick::AhoCorasick;
use bstr::ByteSlice;
use memchr::memmem;
use regex::bytes::Regex;
use stringzilla::sz;

#[path = "../utils.rs"]
mod utils;
use utils::{
    install_panic_hook, load_dataset, log_stringzilla_metadata, measure_throughput, BenchBudget,
    ReportAs, ResultExt, WorkUnits,
};

/// Benchmarks forward substring search using "StringZilla", "MemMem", and standard strings.
///
/// Each call cycles to the next needle and scans the whole haystack for every occurrence of it,
/// so the per-call work is one full haystack pass (`haystack.len()` bytes), matching the original
/// `Throughput::Bytes(haystack.len())` accounting.
fn bench_substring_forward(budget: &BenchBudget, haystack: &[u8], needles: &BytesCowsAuto) {
    let haystack_bytes = haystack.len() as u64;

    // Benchmark for StringZilla forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "substring-forward/stringzilla::find",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            let mut position: usize = 0;
            while let Some(found) = sz::find(&haystack[position..], token) {
                position += found + token.len();
            }
            WorkUnits::new(1, haystack_bytes)
        },
    );

    // Benchmark for `memmem::find` forward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "substring-forward/memmem::find",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            let mut position: usize = 0;
            while let Some(found) = memmem::find(&haystack[position..], token) {
                position += found + token.len();
            }
            WorkUnits::new(1, haystack_bytes)
        },
    );

    // Benchmark for `memmem::Finder` forward search with pre-constructed matcher.
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "substring-forward/memmem::Finder",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            let finder = memmem::Finder::new(token);
            let mut position: usize = 0;
            while let Some(found) = finder.find(&haystack[position..]) {
                position += found + token.len();
            }
            WorkUnits::new(1, haystack_bytes)
        },
    );

    // Benchmark for default `std::str::find` forward search.
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "substring-forward/std::str::find",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            let mut position = 0;
            while let Some(found) = haystack[position..].find(token) {
                position += found + token.len();
            }
            WorkUnits::new(1, haystack_bytes)
        },
    );
}

/// Benchmarks backward substring search using "StringZilla", "MemMem", and standard strings.
///
/// Each call cycles to the next needle and scans the whole haystack backward, so the per-call
/// work is one full haystack pass, matching the original `Throughput::Bytes(haystack.len())`.
fn bench_substring_backward(budget: &BenchBudget, haystack: &[u8], needles: &BytesCowsAuto) {
    let haystack_bytes = haystack.len() as u64;

    // Benchmark for StringZilla backward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "substring-backward/stringzilla::rfind",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            let mut position: Option<usize> = Some(haystack.len());
            while let Some(end) = position {
                if let Some(found) = sz::rfind(&haystack[..end], token) {
                    position = Some(found);
                } else {
                    break;
                }
            }
            WorkUnits::new(1, haystack_bytes)
        },
    );

    // Benchmark for `memmem::rfind` backward search using a cycle iterator.
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "substring-backward/memmem::rfind",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            let mut position: Option<usize> = Some(haystack.len());
            while let Some(end) = position {
                if let Some(found) = memmem::rfind(&haystack[..end], token) {
                    position = Some(found);
                } else {
                    break;
                }
            }
            WorkUnits::new(1, haystack_bytes)
        },
    );

    // Benchmark for `memmem::FinderRev` backward search with pre-constructed matcher.
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "substring-backward/memmem::FinderRev",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            let finder = memmem::FinderRev::new(token);
            let mut position: Option<usize> = Some(haystack.len());
            while let Some(end) = position {
                if let Some(found) = finder.rfind(&haystack[..end]) {
                    position = Some(found);
                } else {
                    break;
                }
            }
            WorkUnits::new(1, haystack_bytes)
        },
    );

    // Benchmark for default `std::str::rfind` backward search.
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "substring-backward/std::str::rfind",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            let mut position: Option<usize> = Some(haystack.len());
            while let Some(end) = position {
                if let Some(found) = haystack[..end].rfind(token) {
                    position = Some(found);
                } else {
                    break;
                }
            }
            WorkUnits::new(1, haystack_bytes)
        },
    );
}

/// Benchmarks byteset search using "StringZilla", "bstr", "RegEx", and "AhoCorasick".
///
/// Each call cycles to the next needle token and runs all three bytesets over it. The original
/// looped over every needle in one iteration with `Throughput::Bytes(3 * haystack.len())`; since
/// the needles collectively span the haystack, the per-token equivalent is `3 * token.len()`.
fn bench_byteset_forward(budget: &BenchBudget, _haystack: &[u8], needles: &BytesCowsAuto) {
    // Define the three bytesets we will analyze.
    const BYTES_TABS: &[u8] = b"\n\r\x0B\x0C";
    const BYTES_HTML: &[u8] = b"</>&'\"=[]";
    const BYTES_DIGITS: &[u8] = b"0123456789";

    // Benchmark for StringZilla forward search using a cycle iterator.
    let sz_tabs = sz::Byteset::from(BYTES_TABS);
    let sz_html = sz::Byteset::from(BYTES_HTML);
    let sz_digits = sz::Byteset::from(BYTES_DIGITS);
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "byteset-forward/stringzilla::find_byteset",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            let mut position: usize = 0;
            while let Some(found) = sz::find_byteset(&token[position..], sz_tabs) {
                position += found + 1;
            }
            position = 0;
            while let Some(found) = sz::find_byteset(&token[position..], sz_html) {
                position += found + 1;
            }
            position = 0;
            while let Some(found) = sz::find_byteset(&token[position..], sz_digits) {
                position += found + 1;
            }
            WorkUnits::new(1, 3 * token.len() as u64)
        },
    );

    // Benchmark for bstr's byteset search.
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "byteset-forward/bstr::iter",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            let mut position: usize = 0;
            // Inline search for `BYTES_TABS`.
            while let Some(found) = token[position..]
                .iter()
                .position(|&byte| BYTES_TABS.contains(&byte))
            {
                position += found + 1;
            }
            position = 0;
            // Inline search for `BYTES_HTML`.
            while let Some(found) = token[position..]
                .iter()
                .position(|&byte| BYTES_HTML.contains(&byte))
            {
                position += found + 1;
            }
            position = 0;
            // Inline search for `BYTES_DIGITS`.
            while let Some(found) = token[position..]
                .iter()
                .position(|&byte| BYTES_DIGITS.contains(&byte))
            {
                position += found + 1;
            }
            WorkUnits::new(1, 3 * token.len() as u64)
        },
    );

    // Benchmark for Regex-based byteset search.
    let re_tabs = Regex::new("[\n\r\x0B\x0C]").unwrap();
    let re_html = Regex::new("[</>&'\"=\\[\\]]").unwrap();
    let re_digits = Regex::new("[0-9]").unwrap();
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "byteset-forward/regex::find_iter",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            black_box(re_tabs.find_iter(token.as_bytes()).count());
            black_box(re_html.find_iter(token.as_bytes()).count());
            black_box(re_digits.find_iter(token.as_bytes()).count());
            WorkUnits::new(1, 3 * token.len() as u64)
        },
    );

    // Benchmark for Aho–Corasick-based byteset search.
    let ac_tabs = AhoCorasick::new(
        &BYTES_TABS
            .iter()
            .map(|&byte| (byte as char).to_string())
            .collect::<Vec<_>>(),
    )
    .expect("failed to create AhoCorasick FSA");
    let ac_html = AhoCorasick::new(
        &BYTES_HTML
            .iter()
            .map(|&byte| (byte as char).to_string())
            .collect::<Vec<_>>(),
    )
    .expect("failed to create AhoCorasick FSA");
    let ac_digits = AhoCorasick::new(
        &BYTES_DIGITS
            .iter()
            .map(|&byte| (byte as char).to_string())
            .collect::<Vec<_>>(),
    )
    .expect("failed to create AhoCorasick FSA");
    let mut tokens = needles.iter().cycle();
    measure_throughput(
        "byteset-forward/aho_corasick::find_iter",
        ReportAs::Bytes,
        budget,
        || {
            let token = black_box(tokens.next().unwrap());
            black_box(ac_tabs.find_iter(token).count());
            black_box(ac_html.find_iter(token).count());
            black_box(ac_digits.find_iter(token).count());
            WorkUnits::new(1, 3 * token.len() as u64)
        },
    );
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    // Load the dataset defined by the environment variables
    let tape = load_dataset().unwrap_nice();

    // Get the parent data directly from the tape (zero-copy)
    let haystack = tape.parent();
    let needles = &tape;

    let budget = BenchBudget::from_env(3.0, 20.0);

    println!("# substring-forward");
    bench_substring_forward(&budget, &haystack, &needles);

    println!("# substring-backward");
    bench_substring_backward(&budget, &haystack, &needles);

    println!("# byteset-forward");
    bench_byteset_forward(&budget, &haystack, &needles);
}
