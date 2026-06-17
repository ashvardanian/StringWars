#![doc = r#"
# StringWars: UTF-8 Tokenization & Iteration Benchmarks

This file benchmarks UTF-8 segmentation and codepoint iteration:
- Unicode whitespace and newline splitting
- Unicode TR29 (UAX#29) word segmentation
- UTF-8 character counting and UTF-8 to UTF-32 decoding
- Locating the byte offset of the Nth UTF-8 codepoint

## Benchmark Groups

- `tokenize-whitespace`: Unicode whitespace splitting
- `tokenize-newlines`: Unicode newline splitting
- `tokenize-words-tr29`: Unicode TR29 word boundary segmentation
- `utf8-length`: UTF-8 character counting
- `utf8-iterate`: UTF-8 to UTF-32 decoding
- `find-nth-utf8`: byte offset of the Nth UTF-8 codepoint

## Usage Examples

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_tokenization bench_tokenization --jobs $(nproc)
```
"#]
use std::hint::black_box;

use stringtape::BytesCowsAuto;

use icu::properties::props::WhiteSpace;
use icu::properties::CodePointSetData;
use icu::segmenter::WordSegmenter;
use stringzilla::sz;
use stringzilla::sz::StringZillableUnary;
use unicode_segmentation::UnicodeSegmentation;

#[path = "../utils.rs"]
mod utils;
use utils::{
    install_panic_hook, load_dataset, log_stringzilla_metadata, measure_throughput, BenchBudget,
    ReportAs, ResultExt, WorkUnits,
};

/// Benchmarks Unicode whitespace splitting using ICU, stdlib, and StringZilla.
fn bench_tokenize_whitespace(budget: &BenchBudget, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_length = haystack.len() as u64;

    // Validate UTF-8 once, outside the timed closures. The byte-based StringZilla variant
    // does not need this; the `str`-based baselines bind it before their loop.
    let haystack_str = std::str::from_utf8(haystack).ok();

    // Benchmark for StringZilla whitespace splits.
    measure_throughput(
        "tokenize-whitespace/stringzilla/utf8_whitespace_splits().count()",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = haystack_bytes.sz_utf8_whitespace_splits().count();
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for Rust stdlib char::is_whitespace.
    {
        let text = haystack_str.expect("UTF-8 text required for the stdlib whitespace baseline");
        measure_throughput(
            "tokenize-whitespace/std/split(char::is_whitespace).count()",
            ReportAs::Bytes,
            budget,
            || {
                let count: usize = black_box(text)
                    .split(char::is_whitespace)
                    .filter(|segment| !segment.is_empty())
                    .count();
                black_box(count);
                WorkUnits::bytes(haystack_length)
            },
        );
    }

    // Benchmark for ICU4X WhiteSpace property.
    {
        let text = haystack_str.expect("UTF-8 text required for the ICU whitespace baseline");
        let white_space = CodePointSetData::new::<WhiteSpace>();
        measure_throughput(
            "tokenize-whitespace/icu/WhiteSpace.split().count()",
            ReportAs::Bytes,
            budget,
            || {
                let count: usize = black_box(text)
                    .split(|character: char| white_space.contains(character))
                    .filter(|segment: &&str| !segment.is_empty())
                    .count();
                black_box(count);
                WorkUnits::bytes(haystack_length)
            },
        );
    }
}
/// Benchmarks Unicode newline splitting using custom predicates and StringZilla.
fn bench_tokenize_newlines(budget: &BenchBudget, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_length = haystack.len() as u64;

    // Custom newline predicate matching StringZilla's 7 newline characters.
    fn is_unicode_newline(character: char) -> bool {
        matches!(
            character,
            '\n' | '\r' | '\x0B' | '\x0C' | '\u{0085}' | '\u{2028}' | '\u{2029}'
        )
    }

    // Validate UTF-8 once, outside the timed closures (only the custom `str` baseline needs it).
    let haystack_str = std::str::from_utf8(haystack).ok();

    // Benchmark for StringZilla newline splits.
    measure_throughput(
        "tokenize-newlines/stringzilla/utf8_newline_splits().count()",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = haystack_bytes.sz_utf8_newline_splits().count();
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for custom newline predicate.
    {
        let text = haystack_str.expect("UTF-8 text required for the custom newline baseline");
        measure_throughput(
            "tokenize-newlines/custom/split(is_unicode_newline).count()",
            ReportAs::Bytes,
            budget,
            || {
                let count: usize = black_box(text)
                    .split(is_unicode_newline)
                    .filter(|segment| !segment.is_empty())
                    .count();
                black_box(count);
                WorkUnits::bytes(haystack_length)
            },
        );
    }
}
/// Benchmarks Unicode TR29 (UAX#29) word segmentation.
///
/// TR29 defines linguistically-aware word boundaries that handle complex cases like
/// contractions ("can't"), numeric sequences ("3.14"), and scripts without spaces.
/// - `unicode-segmentation::unicode_words()`: Filters to word-like segments only
/// - `unicode-segmentation::split_word_bounds()`: All boundary segments including punctuation
/// - `icu::segmenter::WordSegmenter`: ICU4X implementation with LSTM/dictionary models
fn bench_tokenize_words_tr29(budget: &BenchBudget, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_str = match std::str::from_utf8(haystack) {
        Ok(text) => text,
        Err(_) => {
            eprintln!("Warning: Haystack is not valid UTF-8, skipping TR29 word benchmarks");
            return;
        }
    };

    let haystack_length = haystack.len() as u64;

    // Benchmark for StringZilla's single-pass TR29 word iterator. `.count()` consumes the
    // iterator without materializing the segments, so no allocation taints the measurement.
    measure_throughput(
        "tokenize-words-tr29/stringzilla/utf8_word_splits().count()",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = haystack_bytes.sz_utf8_word_splits().count();
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for unicode-segmentation: unicode_words() - only word-like segments
    measure_throughput(
        "tokenize-words-tr29/unicode-segmentation/unicode_words().count()",
        ReportAs::Bytes,
        budget,
        || {
            let text = black_box(haystack_str);
            let count: usize = text.unicode_words().count();
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for unicode-segmentation: split_word_bounds() - all segments
    measure_throughput(
        "tokenize-words-tr29/unicode-segmentation/split_word_bounds().count()",
        ReportAs::Bytes,
        budget,
        || {
            let text = black_box(haystack_str);
            let count: usize = text.split_word_bounds().count();
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for ICU4X WordSegmenter with dictionary model
    {
        let segmenter = WordSegmenter::new_dictionary(Default::default());
        measure_throughput(
            "tokenize-words-tr29/icu/WordSegmenter::new_dictionary().segment_str()",
            ReportAs::Bytes,
            budget,
            || {
                let text = black_box(haystack_str);
                // WordSegmenter returns boundary indices; count segments = boundaries - 1
                let boundaries: usize = segmenter.segment_str(text).count();
                black_box(boundaries);
                WorkUnits::bytes(haystack_length)
            },
        );
    }

    // Benchmark for stdlib split_whitespace as baseline comparison
    measure_throughput(
        "tokenize-words-tr29/std/split_whitespace().count()",
        ReportAs::Bytes,
        budget,
        || {
            let text = black_box(haystack_str);
            let count: usize = text.split_whitespace().count();
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );
}
/// Benchmarks UTF-8 character counting using StringZilla, simdutf, and stdlib.
fn bench_utf8_length(budget: &BenchBudget, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_length = haystack.len() as u64;

    // Validate UTF-8 once, outside the timed closures (only the stdlib baseline needs it; the
    // StringZilla and simdutf counters operate directly on bytes).
    let haystack_str = std::str::from_utf8(haystack).ok();

    // Benchmark for StringZilla UTF-8 character counting via the lazy view.
    measure_throughput(
        "utf8-length/stringzilla/utf8_chars().len()",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = haystack_bytes.sz_utf8_chars().len();
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for StringZilla's dedicated `count_utf8()` free function (direct SIMD scan,
    // without constructing a view object).
    measure_throughput(
        "utf8-length/stringzilla/count_utf8()",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = sz::count_utf8(haystack_bytes);
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for simdutf UTF-8 character counting.
    measure_throughput(
        "utf8-length/simdutf/count_utf8()",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = simdutf::count_utf8(haystack_bytes);
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for stdlib UTF-8 character counting.
    {
        let text = haystack_str.expect("UTF-8 text required for the stdlib codepoint counter");
        measure_throughput(
            "utf8-length/std/chars().count()",
            ReportAs::Bytes,
            budget,
            || {
                let count: usize = black_box(text).chars().count();
                black_box(count);
                WorkUnits::bytes(haystack_length)
            },
        );
    }
}
/// Benchmarks UTF-8 to UTF-32 decoding using StringZilla, simdutf, and stdlib.
fn bench_utf8_iterate(budget: &BenchBudget, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_length = haystack.len() as u64;

    // Benchmark for StringZilla UTF-8 character iteration.
    measure_throughput(
        "utf8-iterate/stringzilla/utf8_chars().iter()",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let mut sum: u32 = 0;
            for character in haystack_bytes.sz_utf8_chars().iter() {
                sum = sum.wrapping_add(character as u32);
            }
            black_box(sum);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for simdutf UTF-8 to UTF-32 conversion.
    {
        // Pre-allocate buffer for UTF-32 output (worst case: same number of codepoints as bytes)
        let mut utf32_buffer = vec![0u32; haystack.len()];
        measure_throughput(
            "utf8-iterate/simdutf/convert_utf8_to_utf32()",
            ReportAs::Bytes,
            budget,
            || {
                let haystack_bytes = black_box(haystack);
                let len = unsafe {
                    simdutf::convert_utf8_to_utf32(
                        haystack_bytes.as_ptr(),
                        haystack_bytes.len(),
                        utf32_buffer.as_mut_ptr(),
                    )
                };
                let mut sum: u32 = 0;
                for index in 0..len {
                    sum = sum.wrapping_add(utf32_buffer[index]);
                }
                black_box(sum);
                WorkUnits::bytes(haystack_length)
            },
        );
    }

    // Benchmark for stdlib UTF-8 character iteration.
    measure_throughput("utf8-iterate/std/chars()", ReportAs::Bytes, budget, || {
        let haystack_str = black_box(unsafe { std::str::from_utf8_unchecked(haystack) });
        let mut sum: u32 = 0;
        for character in haystack_str.chars() {
            sum = sum.wrapping_add(character as u32);
        }
        black_box(sum);
        WorkUnits::bytes(haystack_length)
    });
}
/// Benchmarks locating the byte offset of the Nth UTF-8 codepoint.
///
/// - `stringzilla::find_nth_utf8()`: SIMD-accelerated scan to the Nth codepoint
/// - `std::str::char_indices().nth()`: scalar decode-and-count baseline
///
/// We target the *last* codepoint, so every implementation scans the whole buffer once —
/// a fair workload whose throughput is simply the input size.
fn bench_find_nth_utf8(budget: &BenchBudget, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_str = match std::str::from_utf8(haystack) {
        Ok(text) => text,
        Err(_) => {
            eprintln!("Warning: Haystack is not valid UTF-8, skipping find-nth-utf8 benchmarks");
            return;
        }
    };

    let codepoint_count = sz::count_utf8(haystack);
    if codepoint_count == 0 {
        return;
    }
    let last_index = codepoint_count - 1;

    let haystack_length = haystack.len() as u64;

    // Benchmark for StringZilla's SIMD `find_nth_utf8`.
    measure_throughput(
        "find-nth-utf8/stringzilla/find_nth_utf8()",
        ReportAs::Bytes,
        budget,
        || {
            let offset = sz::find_nth_utf8(black_box(haystack), last_index);
            black_box(offset);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for the stdlib scalar baseline.
    measure_throughput(
        "find-nth-utf8/std/char_indices().nth()",
        ReportAs::Bytes,
        budget,
        || {
            let offset = black_box(haystack_str)
                .char_indices()
                .nth(last_index)
                .map(|(byte_offset, _)| byte_offset);
            black_box(offset);
            WorkUnits::bytes(haystack_length)
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

    println!("# tokenize-whitespace");
    bench_tokenize_whitespace(&budget, &haystack, needles);

    println!("# tokenize-newlines");
    bench_tokenize_newlines(&budget, &haystack, needles);

    println!("# tokenize-words-tr29");
    bench_tokenize_words_tr29(&budget, &haystack, needles);

    println!("# utf8-length");
    bench_utf8_length(&budget, &haystack, needles);

    println!("# utf8-iterate");
    bench_utf8_iterate(&budget, &haystack, needles);

    println!("# find-nth-utf8");
    bench_find_nth_utf8(&budget, &haystack, needles);
}
