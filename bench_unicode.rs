#![doc = r#"
# StringWars: Unicode Text Processing Benchmarks

This file benchmarks Unicode text processing operations including:
- UTF-8 character counting and iteration
- Unicode whitespace and newline splitting (tokenization)
- Unicode TR29 (UAX#29) word segmentation
- Case folding transformation
- Case-insensitive string comparison
- Case-insensitive substring search

## Benchmark Groups

- `tokenize-whitespace`: Unicode whitespace splitting
- `tokenize-newlines`: Unicode newline splitting
- `tokenize-words-tr29`: Unicode TR29 word boundary segmentation
- `utf8-length`: UTF-8 character counting
- `utf8-iterate`: UTF-8 to UTF-32 decoding
- `case-fold`: Case folding transformation
- `case-insensitive-compare`: Case-insensitive equality
- `case-insensitive-find`: Case-insensitive substring search

## Usage Examples

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_unicode bench_unicode --jobs $(nproc)
```

Filter specific operations:

```sh
STRINGWARS_FILTER="case-insensitive" cargo criterion ...
STRINGWARS_FILTER="tokenize" cargo criterion ...
STRINGWARS_FILTER="tokenize-words-tr29" cargo criterion ...
```
"#]
use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput};
use stringtape::BytesCowsAuto;

use focaccia::unicode_full_case_eq;
use icu::properties::props::WhiteSpace;
use icu::properties::CodePointSetData;
use icu::segmenter::WordSegmenter;
use pcre2::bytes::RegexBuilder;
use stringzilla::sz;
use unicase::UniCase;
use unicode_segmentation::UnicodeSegmentation;

mod utils;
use utils::{install_panic_hook, load_dataset, should_run, ResultExt};

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

/// Benchmarks Unicode whitespace splitting using ICU, stdlib, and StringZilla.
fn bench_tokenize_whitespace(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    _needles: &BytesCowsAuto,
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for StringZilla whitespace splits.
    if should_run("tokenize-whitespace/stringzilla/utf8_whitespace_splits().count()") {
        use sz::StringZillableUnary;
        g.bench_function("stringzilla/utf8_whitespace_splits().count()", |b| {
            b.iter(|| {
                let haystack_bytes = black_box(haystack);
                let count: usize = haystack_bytes.sz_utf8_whitespace_splits().count();
                black_box(count);
            })
        });
    }

    // Benchmark for Rust stdlib char::is_whitespace.
    if should_run("tokenize-whitespace/std/split(char::is_whitespace).count()") {
        g.bench_function("std/split(char::is_whitespace).count()", |b| {
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
    if should_run("tokenize-whitespace/icu/WhiteSpace.split().count()") {
        let white_space = CodePointSetData::new::<WhiteSpace>();
        g.bench_function("icu/WhiteSpace.split().count()", |b| {
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
fn bench_tokenize_newlines(
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
    if should_run("tokenize-newlines/stringzilla/utf8_newline_splits().count()") {
        use sz::StringZillableUnary;
        g.bench_function("stringzilla/utf8_newline_splits().count()", |b| {
            b.iter(|| {
                let haystack_bytes = black_box(haystack);
                let count: usize = haystack_bytes.sz_utf8_newline_splits().count();
                black_box(count);
            })
        });
    }

    // Benchmark for custom newline predicate.
    if should_run("tokenize-newlines/custom/split(is_unicode_newline).count()") {
        g.bench_function("custom/split(is_unicode_newline).count()", |b| {
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

/// Benchmarks Unicode TR29 (UAX#29) word segmentation.
///
/// TR29 defines linguistically-aware word boundaries that handle complex cases like
/// contractions ("can't"), numeric sequences ("3.14"), and scripts without spaces.
/// - `unicode-segmentation::unicode_words()`: Filters to word-like segments only
/// - `unicode-segmentation::split_word_bounds()`: All boundary segments including punctuation
/// - `icu::segmenter::WordSegmenter`: ICU4X implementation with LSTM/dictionary models
fn bench_tokenize_words_tr29(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    _needles: &BytesCowsAuto,
) {
    let haystack_str = match std::str::from_utf8(haystack) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Warning: Haystack is not valid UTF-8, skipping TR29 word benchmarks");
            return;
        }
    };

    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for unicode-segmentation: unicode_words() - only word-like segments
    if should_run("tokenize-words-tr29/unicode-segmentation/unicode_words().count()") {
        g.bench_function("unicode-segmentation/unicode_words().count()", |b| {
            b.iter(|| {
                let text = black_box(haystack_str);
                let count: usize = text.unicode_words().count();
                black_box(count);
            })
        });
    }

    // Benchmark for unicode-segmentation: split_word_bounds() - all segments
    if should_run("tokenize-words-tr29/unicode-segmentation/split_word_bounds().count()") {
        g.bench_function("unicode-segmentation/split_word_bounds().count()", |b| {
            b.iter(|| {
                let text = black_box(haystack_str);
                let count: usize = text.split_word_bounds().count();
                black_box(count);
            })
        });
    }

    // Benchmark for ICU4X WordSegmenter with dictionary model
    if should_run("tokenize-words-tr29/icu/WordSegmenter::new_dictionary().segment_str()") {
        let segmenter = WordSegmenter::new_dictionary(Default::default());
        g.bench_function("icu/WordSegmenter::new_dictionary().segment_str()", |b| {
            b.iter(|| {
                let text = black_box(haystack_str);
                // WordSegmenter returns boundary indices; count segments = boundaries - 1
                let boundaries: usize = segmenter.segment_str(text).count();
                black_box(boundaries);
            })
        });
    }

    // Benchmark for stdlib split_whitespace as baseline comparison
    if should_run("tokenize-words-tr29/std/split_whitespace().count()") {
        g.bench_function("std/split_whitespace().count()", |b| {
            b.iter(|| {
                let text = black_box(haystack_str);
                let count: usize = text.split_whitespace().count();
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
    if should_run("utf8-length/stringzilla/utf8_chars().len()") {
        use sz::StringZillableUnary;
        g.bench_function("stringzilla/utf8_chars().len()", |b| {
            b.iter(|| {
                let haystack_bytes = black_box(haystack);
                let count: usize = haystack_bytes.sz_utf8_chars().len();
                black_box(count);
            })
        });
    }

    // Benchmark for simdutf UTF-8 character counting.
    if should_run("utf8-length/simdutf/count_utf8()") {
        g.bench_function("simdutf/count_utf8()", |b| {
            b.iter(|| {
                let haystack_bytes = black_box(haystack);
                let count: usize = simdutf::count_utf8(haystack_bytes);
                black_box(count);
            })
        });
    }

    // Benchmark for stdlib UTF-8 character counting.
    if should_run("utf8-length/std/chars().count()") {
        g.bench_function("std/chars().count()", |b| {
            b.iter(|| {
                let haystack_str = black_box(std::str::from_utf8(haystack).unwrap());
                let count: usize = haystack_str.chars().count();
                black_box(count);
            })
        });
    }
}

/// Benchmarks UTF-8 to UTF-32 decoding using StringZilla, simdutf, and stdlib.
fn bench_utf8_iterate(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    _needles: &BytesCowsAuto,
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for StringZilla UTF-8 character iteration.
    if should_run("utf8-iterate/stringzilla/utf8_chars().iter()") {
        use sz::StringZillableUnary;
        g.bench_function("stringzilla/utf8_chars().iter()", |b| {
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
    if should_run("utf8-iterate/simdutf/convert_utf8_to_utf32()") {
        // Pre-allocate buffer for UTF-32 output (worst case: same number of codepoints as bytes)
        let mut utf32_buffer = vec![0u32; haystack.len()];
        g.bench_function("simdutf/convert_utf8_to_utf32()", |b| {
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
    if should_run("utf8-iterate/std/chars()") {
        g.bench_function("std/chars()", |b| {
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

/// Benchmarks case folding transformation throughput.
///
/// Unicode case folding may expand characters (e.g., German ß -> ss).
/// - `stringzilla::utf8_case_fold()`: Full Unicode case folding per Unicode Standard
/// - `stdlib::to_lowercase()`: Full Unicode lowercasing (locale-independent, allocates)
fn bench_case_fold(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    _needles: &BytesCowsAuto,
) {
    g.throughput(Throughput::Bytes(haystack.len() as u64));

    let haystack_str = std::str::from_utf8(haystack).unwrap();

    // Pre-allocate buffer for StringZilla case folding (3x for worst-case expansion)
    let mut fold_buffer = vec![0u8; haystack.len() * 3];

    // Benchmark for StringZilla case folding (full Unicode).
    if should_run("case-fold/stringzilla/utf8_case_fold()") {
        g.bench_function("stringzilla/utf8_case_fold()", |b| {
            b.iter(|| {
                let input = black_box(haystack);
                let len = sz::utf8_case_fold(input, &mut fold_buffer);
                black_box(len);
            })
        });
    }

    // Benchmark for stdlib to_lowercase (full Unicode, allocates).
    if should_run("case-fold/std/to_lowercase()") {
        g.bench_function("std/to_lowercase()", |b| {
            b.iter(|| {
                let input = black_box(haystack_str);
                let lowered = input.to_lowercase();
                black_box(lowered);
            })
        });
    }
}

/// Benchmarks case-insensitive string equality comparison.
fn bench_case_insensitive_compare(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    _haystack: &[u8],
    needles: &BytesCowsAuto,
) {
    // We compare each pair of adjacent tokens
    let pairs: Vec<(&[u8], &[u8])> = needles
        .iter()
        .zip(needles.iter().skip(1))
        .take(1000) // Limit to avoid excessive runtime
        .collect();

    if pairs.is_empty() {
        eprintln!("Warning: Not enough tokens for case-insensitive comparison benchmarks");
        return;
    }

    // Total bytes compared (both sides of each pair)
    let total_bytes: usize = pairs.iter().map(|(a, b)| a.len() + b.len()).sum();
    g.throughput(Throughput::Bytes(total_bytes as u64));

    // Benchmark for StringZilla case-insensitive comparison.
    if should_run("case-insensitive-compare/stringzilla/utf8_case_insensitive_order()") {
        g.bench_function("stringzilla/utf8_case_insensitive_order()", |b| {
            b.iter(|| {
                let mut matches = 0usize;
                for (left, right) in &pairs {
                    if sz::utf8_case_insensitive_order(left, right) == std::cmp::Ordering::Equal {
                        matches += 1;
                    }
                }
                black_box(matches);
            })
        });
    }

    // Benchmark for unicase equality.
    if should_run("case-insensitive-compare/unicase/eq()") {
        g.bench_function("unicase/eq()", |b| {
            b.iter(|| {
                let mut matches = 0usize;
                for (left, right) in &pairs {
                    let left_str = std::str::from_utf8(left).unwrap_or("");
                    let right_str = std::str::from_utf8(right).unwrap_or("");
                    if UniCase::new(left_str) == UniCase::new(right_str) {
                        matches += 1;
                    }
                }
                black_box(matches);
            })
        });
    }

    // Benchmark for focaccia case-insensitive equality (full Unicode).
    if should_run("case-insensitive-compare/focaccia/unicode_full_case_eq()") {
        g.bench_function("focaccia/unicode_full_case_eq()", |b| {
            b.iter(|| {
                let mut matches = 0usize;
                for (left, right) in &pairs {
                    let left_str = std::str::from_utf8(left).unwrap_or("");
                    let right_str = std::str::from_utf8(right).unwrap_or("");
                    if unicode_full_case_eq(left_str, right_str) {
                        matches += 1;
                    }
                }
                black_box(matches);
            })
        });
    }

    // Benchmark for stdlib lowercase + equality (baseline).
    if should_run("case-insensitive-compare/std/to_lowercase().eq()") {
        g.bench_function("std/to_lowercase().eq()", |b| {
            b.iter(|| {
                let mut matches = 0usize;
                for (left, right) in &pairs {
                    let left_str = std::str::from_utf8(left).unwrap_or("");
                    let right_str = std::str::from_utf8(right).unwrap_or("");
                    if left_str.to_lowercase() == right_str.to_lowercase() {
                        matches += 1;
                    }
                }
                black_box(matches);
            })
        });
    }
}

/// Benchmarks case-insensitive substring search.
fn bench_case_insensitive_find(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    needles: &BytesCowsAuto,
) {
    use rand::prelude::IndexedRandom;
    use rand::SeedableRng;

    // Limit haystack size to 10MB for case-insensitive search (it's O(n*m) with Unicode folding)
    const MAX_HAYSTACK_SIZE: usize = 10 * 1024 * 1024;
    let haystack = if haystack.len() > MAX_HAYSTACK_SIZE {
        // Find a valid UTF-8 boundary near the limit
        let mut end = MAX_HAYSTACK_SIZE;
        while end > 0 && (haystack[end] & 0xC0) == 0x80 {
            end -= 1;
        }
        &haystack[..end]
    } else {
        haystack
    };

    let haystack_str = std::str::from_utf8(haystack).unwrap();

    // Collect candidate needles (valid UTF-8, length >= 3)
    let candidates: Vec<&str> = needles
        .iter()
        .filter_map(|n| std::str::from_utf8(n).ok())
        .filter(|s| s.len() >= 3)
        .collect();

    if candidates.is_empty() {
        eprintln!("Warning: No suitable needles for case-insensitive find benchmarks");
        return;
    }

    // Random-sample 100 needles with fixed seed for reproducibility
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let search_needles: Vec<&str> = candidates
        .choose_multiple(&mut rng, 100.min(candidates.len()))
        .copied()
        .collect();

    // Throughput: each iteration searches for first match of each needle
    g.throughput(Throughput::Bytes(
        (haystack.len() * search_needles.len()) as u64,
    ));

    // Benchmark for StringZilla case-insensitive find (all matches).
    if should_run("case-insensitive-find/stringzilla/utf8_case_insensitive_find()") {
        g.bench_function("stringzilla/utf8_case_insensitive_find()", |b| {
            b.iter(|| {
                let hay = black_box(haystack);
                let mut total_matches = 0usize;
                for needle in &search_needles {
                    let mut remaining = hay;
                    while let Some((offset, len)) =
                        sz::utf8_case_insensitive_find(remaining, needle)
                    {
                        total_matches += 1;
                        remaining = &remaining[offset + len.max(1)..];
                    }
                }
                black_box(total_matches);
            })
        });
    }

    // Benchmark for PCRE2 case-insensitive search (all matches).
    // We use PCRE2 instead of Rust's `regex` crate because `regex` only supports
    // Unicode "simple" case folding (1:1 character mappings). PCRE2 with `.utf(true)`
    // supports full Unicode case folding, including expansions like ß→ss, İ→i̇, ﬁ→fi.
    // This makes it a fair comparison against StringZilla's full case folding.
    // See: https://github.com/rust-lang/regex/blob/master/UNICODE.md
    if should_run("case-insensitive-find/pcre2/find_iter(caseless+utf)") {
        // Pre-compile regexes for fair comparison
        let regexes: Vec<_> = search_needles
            .iter()
            .filter_map(|needle| {
                RegexBuilder::new()
                    .caseless(true)
                    .utf(true)
                    .build(&pcre2::escape(needle))
                    .ok()
            })
            .collect();

        g.bench_function("pcre2/find_iter(caseless+utf)", |b| {
            b.iter(|| {
                let hay: &[u8] = black_box(haystack);
                let mut total_matches = 0usize;
                for re in &regexes {
                    total_matches += re.find_iter(hay).count();
                }
                black_box(total_matches);
            })
        });
    }

    // Benchmark for stdlib: lowercase haystack + needle for each search, find all matches.
    // This is the fair comparison - includes full allocation and case folding cost per search.
    if should_run("case-insensitive-find/std/to_lowercase().find()") {
        g.bench_function("std/to_lowercase().find()", |b| {
            b.iter(|| {
                let haystack_str = black_box(haystack_str);
                let mut total_matches = 0usize;
                for needle in &search_needles {
                    let hay = haystack_str.to_lowercase();
                    let needle_lower = needle.to_lowercase();
                    // Work on the lowercased string slice to avoid UTF-8 boundary issues
                    let mut remaining: &str = &hay;
                    while let Some(pos) = remaining.find(&needle_lower) {
                        total_matches += 1;
                        // Advance past the match (at least 1 byte to avoid infinite loop)
                        remaining = &remaining[pos + needle_lower.len().max(1)..];
                    }
                }
                black_box(total_matches);
            })
        });
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    // Load the dataset defined by the environment variables
    let tape = load_dataset().unwrap_nice();

    // Get the parent data directly from the tape (zero-copy)
    let haystack = tape.parent();
    let needles = &tape;

    let mut criterion = configure_bench();

    // Tokenization benchmarks
    let mut group = criterion.benchmark_group("tokenize-whitespace");
    bench_tokenize_whitespace(&mut group, &haystack, needles);
    group.finish();

    let mut group = criterion.benchmark_group("tokenize-newlines");
    bench_tokenize_newlines(&mut group, &haystack, needles);
    group.finish();

    let mut group = criterion.benchmark_group("tokenize-words-tr29");
    bench_tokenize_words_tr29(&mut group, &haystack, needles);
    group.finish();

    // UTF-8 processing benchmarks
    let mut group = criterion.benchmark_group("utf8-length");
    bench_utf8_length(&mut group, &haystack, needles);
    group.finish();

    let mut group = criterion.benchmark_group("utf8-iterate");
    bench_utf8_iterate(&mut group, &haystack, needles);
    group.finish();

    // Case folding benchmarks
    let mut group = criterion.benchmark_group("case-fold");
    bench_case_fold(&mut group, &haystack, needles);
    group.finish();

    // Case-insensitive comparison benchmarks
    let mut group = criterion.benchmark_group("case-insensitive-compare");
    bench_case_insensitive_compare(&mut group, &haystack, needles);
    group.finish();

    // Case-insensitive find benchmarks
    let mut group = criterion.benchmark_group("case-insensitive-find");
    bench_case_insensitive_find(&mut group, &haystack, needles);
    group.finish();

    criterion.final_summary();
}
