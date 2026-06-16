#![doc = r#"
# StringWars: Case Folding & Normalization Benchmarks

This file benchmarks Unicode case-insensitive operations and normalization:
- Case folding transformation
- Case-insensitive string comparison
- Case-insensitive substring search
- Unicode normalization (NFC / NFD / NFKC / NFKD)

## Benchmark Groups

- `case-fold`: Case folding transformation
- `normalize`: Unicode normalization (NFC / NFD / NFKC / NFKD)
- `case-insensitive-compare`: Case-insensitive equality
- `case-insensitive-find`: Case-insensitive substring search

## Usage Examples

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_normalization bench_normalization --jobs $(nproc)
```
"#]
use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::Throughput;
use rand::prelude::IndexedRandom;
use rand::SeedableRng;
use stringtape::BytesCowsAuto;

use icu::casemap::CaseMapper;
use icu::normalizer::{ComposingNormalizerBorrowed, DecomposingNormalizerBorrowed};
use memchr::memmem;
use pcre2::bytes::RegexBuilder;
use stringzilla::sz;
use stringzilla::sz::Utf8NormalForm;
use unicase::UniCase;
use unicode_normalization::UnicodeNormalization;

#[path = "../utils.rs"]
mod utils;
use utils::{
    configure_bench, install_panic_hook, load_dataset, log_stringzilla_metadata, should_run,
    ResultExt,
};

fn log_pcre2_metadata() {
    let (major, minor) = pcre2::version();
    println!("PCRE2 v{}.{}", major, minor);
    println!("- JIT available: {}", pcre2::is_jit_available());
}
/// Benchmarks case folding transformation throughput.
///
/// Unicode case folding may expand characters (e.group., German ß -> ss).
/// - `stringzilla::utf8_uncased_fold()`: Full Unicode case folding per Unicode Standard
/// - `stdlib::to_lowercase()`: Full Unicode lowercasing (locale-independent, allocates)
fn bench_case_fold(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    _needles: &BytesCowsAuto,
) {
    group.throughput(Throughput::Bytes(haystack.len() as u64));

    let haystack_str = std::str::from_utf8(haystack).unwrap();

    // Pre-allocate buffer for StringZilla case folding (3x for worst-case expansion)
    let mut fold_buffer = vec![0u8; haystack.len() * 3];

    // Benchmark for StringZilla case folding (full Unicode).
    if should_run("case-fold/stringzilla/utf8_uncased_fold()") {
        group.bench_function("stringzilla/utf8_uncased_fold()", |bencher| {
            bencher.iter(|| {
                let input = black_box(haystack);
                let len = sz::utf8_uncased_fold(input, &mut fold_buffer);
                black_box(len);
            })
        });
    }

    // Benchmark for stdlib to_lowercase (full Unicode, allocates).
    if should_run("case-fold/std/to_lowercase()") {
        group.bench_function("std/to_lowercase()", |bencher| {
            bencher.iter(|| {
                let input = black_box(haystack_str);
                let lowered = input.to_lowercase();
                black_box(lowered);
            })
        });
    }
}
/// Benchmarks Unicode normalization (NFC / NFD / NFKC / NFKD) throughput.
///
/// - `stringzilla::utf8_norm()`: single-pass SIMD normalization into a caller buffer
/// - `unicode-normalization`: the de-facto Rust crate, iterator of normalized `char`s
/// - `icu::normalizer`: ICU4X Composing/Decomposing normalizers over `&str`
///
/// Normalization is most meaningful on Indic / Arabic / Vietnamese / Korean corpora; on
/// ASCII-heavy inputs every implementation degenerates to a near-passthrough copy.
fn bench_normalize(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    _needles: &BytesCowsAuto,
) {
    let haystack_str = match std::str::from_utf8(haystack) {
        Ok(text) => text,
        Err(_) => {
            eprintln!("Warning: Haystack is not valid UTF-8, skipping normalization benchmarks");
            return;
        }
    };

    group.throughput(Throughput::Bytes(haystack.len() as u64));

    // Benchmark for StringZilla normalization across all four forms. The form is a
    // runtime enum value, so a single loop covers every form without duplication.
    let mut normalization_buffer = vec![0u8; haystack.len() * 3];
    let stringzilla_forms = [
        ("NFC", Utf8NormalForm::Nfc),
        ("NFD", Utf8NormalForm::Nfd),
        ("NFKC", Utf8NormalForm::Nfkc),
        ("NFKD", Utf8NormalForm::Nfkd),
    ];
    for (form_name, form) in stringzilla_forms {
        let identifier = format!(
            "normalize-{}/stringzilla/utf8_norm()",
            form_name.to_lowercase()
        );
        if should_run(&identifier) {
            group.bench_function(format!("stringzilla/utf8_norm({form_name})"), |bencher| {
                bencher.iter(|| {
                    let length =
                        sz::utf8_norm(black_box(haystack), form, &mut normalization_buffer);
                    black_box(length);
                })
            });
        }
    }

    // A single reusable UTF-8 output buffer shared by the baseline implementations, so
    // every benchmarked path writes into pre-allocated capacity exactly like StringZilla's
    // `normalization_buffer` above. `clear()` keeps the allocation; `extend`/`normalize_to`
    // refill it without touching the heap. This keeps the comparison apples-to-apples:
    // we measure normalization work, not allocator throughput.
    let mut string_buffer = String::with_capacity(haystack.len() * 3);

    // Benchmark for the `unicode-normalization` crate. Each form returns a distinct
    // iterator type (Decompositions vs Recompositions), so the four cases are spelled out.
    // `String::extend` consumes the iterator into the reused buffer without reallocating.
    if should_run("normalize-nfc/unicode-normalization/nfc()") {
        group.bench_function("unicode-normalization/nfc()", |bencher| {
            bencher.iter(|| {
                string_buffer.clear();
                string_buffer.extend(black_box(haystack_str).nfc());
                black_box(string_buffer.len());
            })
        });
    }
    if should_run("normalize-nfd/unicode-normalization/nfd()") {
        group.bench_function("unicode-normalization/nfd()", |bencher| {
            bencher.iter(|| {
                string_buffer.clear();
                string_buffer.extend(black_box(haystack_str).nfd());
                black_box(string_buffer.len());
            })
        });
    }
    if should_run("normalize-nfkc/unicode-normalization/nfkc()") {
        group.bench_function("unicode-normalization/nfkc()", |bencher| {
            bencher.iter(|| {
                string_buffer.clear();
                string_buffer.extend(black_box(haystack_str).nfkc());
                black_box(string_buffer.len());
            })
        });
    }
    if should_run("normalize-nfkd/unicode-normalization/nfkd()") {
        group.bench_function("unicode-normalization/nfkd()", |bencher| {
            bencher.iter(|| {
                string_buffer.clear();
                string_buffer.extend(black_box(haystack_str).nfkd());
                black_box(string_buffer.len());
            })
        });
    }

    // Benchmark for ICU4X normalizers. Composing forms (NFC/NFKC) and decomposing forms
    // (NFD/NFKD) are different types; both expose `normalize_to(&str, &mut impl fmt::Write)`,
    // which streams into the reused `string_buffer` instead of allocating a fresh `Cow`.
    let icu_nfc = ComposingNormalizerBorrowed::new_nfc();
    let icu_nfkc = ComposingNormalizerBorrowed::new_nfkc();
    let icu_nfd = DecomposingNormalizerBorrowed::new_nfd();
    let icu_nfkd = DecomposingNormalizerBorrowed::new_nfkd();
    if should_run("normalize-nfc/icu/ComposingNormalizer::normalize_to()") {
        group.bench_function("icu/Normalizer(NFC)", |bencher| {
            bencher.iter(|| {
                string_buffer.clear();
                icu_nfc
                    .normalize_to(black_box(haystack_str), &mut string_buffer)
                    .unwrap();
                black_box(string_buffer.len());
            })
        });
    }
    if should_run("normalize-nfd/icu/DecomposingNormalizer::normalize_to()") {
        group.bench_function("icu/Normalizer(NFD)", |bencher| {
            bencher.iter(|| {
                string_buffer.clear();
                icu_nfd
                    .normalize_to(black_box(haystack_str), &mut string_buffer)
                    .unwrap();
                black_box(string_buffer.len());
            })
        });
    }
    if should_run("normalize-nfkc/icu/ComposingNormalizer::normalize_to()") {
        group.bench_function("icu/Normalizer(NFKC)", |bencher| {
            bencher.iter(|| {
                string_buffer.clear();
                icu_nfkc
                    .normalize_to(black_box(haystack_str), &mut string_buffer)
                    .unwrap();
                black_box(string_buffer.len());
            })
        });
    }
    if should_run("normalize-nfkd/icu/DecomposingNormalizer::normalize_to()") {
        group.bench_function("icu/Normalizer(NFKD)", |bencher| {
            bencher.iter(|| {
                string_buffer.clear();
                icu_nfkd
                    .normalize_to(black_box(haystack_str), &mut string_buffer)
                    .unwrap();
                black_box(string_buffer.len());
            })
        });
    }
}
/// Benchmarks case-insensitive string equality comparison.
fn bench_case_insensitive_compare(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
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
    let total_bytes: usize = pairs
        .iter()
        .map(|(first, second)| first.len() + second.len())
        .sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Decode each pair to `&str` once, outside the timed closures, so the string-based baselines
    // do not re-validate UTF-8 on every iteration. StringZilla compares the raw bytes directly.
    let pairs_str: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(left, right)| {
            (
                std::str::from_utf8(left).unwrap_or(""),
                std::str::from_utf8(right).unwrap_or(""),
            )
        })
        .collect();

    // Benchmark for StringZilla case-insensitive comparison.
    if should_run("case-insensitive-compare/stringzilla/utf8_uncased_order()") {
        group.bench_function("stringzilla/utf8_uncased_order()", |bencher| {
            bencher.iter(|| {
                let mut matches = 0usize;
                for (left, right) in &pairs {
                    if sz::utf8_uncased_order(left, right) == std::cmp::Ordering::Equal {
                        matches += 1;
                    }
                }
                black_box(matches);
            })
        });
    }

    // Benchmark for unicase equality.
    if should_run("case-insensitive-compare/unicase/eq()") {
        group.bench_function("unicase/eq()", |bencher| {
            bencher.iter(|| {
                let mut matches = 0usize;
                for (left_str, right_str) in &pairs_str {
                    if UniCase::new(left_str) == UniCase::new(right_str) {
                        matches += 1;
                    }
                }
                black_box(matches);
            })
        });
    }

    // Benchmark for stdlib lowercase + equality (baseline).
    if should_run("case-insensitive-compare/std/to_lowercase().eq()") {
        group.bench_function("std/to_lowercase().eq()", |bencher| {
            bencher.iter(|| {
                let mut matches = 0usize;
                for (left_str, right_str) in &pairs_str {
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
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    haystack: &[u8],
    needles: &BytesCowsAuto,
) {
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
        .filter_map(|needle_bytes| std::str::from_utf8(needle_bytes).ok())
        .filter(|candidate| candidate.len() >= 3)
        .collect();

    if candidates.is_empty() {
        eprintln!("Warning: No suitable needles for case-insensitive find benchmarks");
        return;
    }

    // Random-sample 100 needles with fixed seed for reproducibility
    let mut random_generator = rand::rngs::StdRng::seed_from_u64(42);
    let search_needles: Vec<&str> = candidates
        .choose_multiple(&mut random_generator, 100.min(candidates.len()))
        .copied()
        .collect();

    // Throughput: each iteration searches haystack once with a single needle
    group.throughput(Throughput::Bytes(haystack.len() as u64));

    // Rotate through needles across iterations with a plain counter. criterion runs the
    // measured closure serially, so a captured `FnMut` counter suffices and avoids the
    // atomic read-modify-write overhead on the hot path.
    // Benchmark for StringZilla case-insensitive find (all matches for one needle).
    if should_run("case-insensitive-find/stringzilla/utf8_uncased_find()") {
        let mut needle_index = 0usize;
        group.bench_function("stringzilla/utf8_uncased_find()", |bencher| {
            bencher.iter(|| {
                let haystack_bytes = black_box(haystack);
                let index = needle_index % search_needles.len();
                needle_index += 1;
                let needle = sz::Utf8UncasedNeedle::new(search_needles[index].as_bytes());
                let mut matches = 0usize;
                let mut remaining = haystack_bytes;
                while let Some((offset, len)) = sz::utf8_uncased_find(remaining, &needle) {
                    matches += 1;
                    remaining = &remaining[offset + len.max(1)..];
                }
                black_box(matches)
            })
        });
    }

    // PCRE2 benchmarks for case-insensitive search with full Unicode case folding.
    // NOTE: We use PCRE2 instead of Rust's `regex` crate because `regex` only supports
    // Unicode "simple" case folding (1:1 character mappings). PCRE2 with `.utf(true)`
    // supports full Unicode case folding, including expansions like ß→ss, İ→i̇, ﬁ→fi.
    // This makes it a fair comparison against StringZilla's full case folding.
    // See: https://github.com/rust-lang/regex/blob/master/UNICODE.md

    // Variant 1: Pre-compiled with JIT (compilation cost excluded from benchmark)
    if should_run("case-insensitive-find/pcre2/pre-jit") {
        let regexes: Vec<_> = search_needles
            .iter()
            .filter_map(|needle| {
                RegexBuilder::new()
                    .caseless(true)
                    .utf(true)
                    .jit_if_available(true)
                    .build(&pcre2::escape(needle))
                    .ok()
            })
            .collect();

        let mut needle_index = 0usize;
        group.bench_function("pcre2/pre-jit", |bencher| {
            bencher.iter(|| {
                let haystack_bytes: &[u8] = black_box(haystack);
                let index = needle_index % regexes.len();
                needle_index += 1;
                black_box(regexes[index].find_iter(haystack_bytes).count())
            })
        });
    }

    // Variant 2: JIT compilation included in benchmark (compile + search per iteration)
    if should_run("case-insensitive-find/pcre2/jit-on-fly") {
        let mut needle_index = 0usize;
        group.bench_function("pcre2/jit-on-fly", |bencher| {
            bencher.iter(|| {
                let haystack_bytes: &[u8] = black_box(haystack);
                let index = needle_index % search_needles.len();
                needle_index += 1;
                let needle = search_needles[index];
                let regex = RegexBuilder::new()
                    .caseless(true)
                    .utf(true)
                    .jit_if_available(true)
                    .build(&pcre2::escape(needle))
                    .unwrap();
                black_box(regex.find_iter(haystack_bytes).count())
            })
        });
    }

    // Variant 3: No JIT (interpreter mode only)
    if should_run("case-insensitive-find/pcre2/no-jit") {
        let regexes: Vec<_> = search_needles
            .iter()
            .filter_map(|needle| {
                RegexBuilder::new()
                    .caseless(true)
                    .utf(true)
                    // No .jit_if_available() - uses interpreter
                    .build(&pcre2::escape(needle))
                    .ok()
            })
            .collect();

        let mut needle_index = 0usize;
        group.bench_function("pcre2/no-jit", |bencher| {
            bencher.iter(|| {
                let haystack_bytes: &[u8] = black_box(haystack);
                let index = needle_index % regexes.len();
                needle_index += 1;
                black_box(regexes[index].find_iter(haystack_bytes).count())
            })
        });
    }

    // Benchmark for ICU case-fold + memchr SIMD search.
    // Full Unicode case folding (ß→ss) + fast byte search.
    // Folding happens inside the loop for fair comparison.
    if should_run("case-insensitive-find/icu+memchr/fold+find") {
        let case_mapper = CaseMapper::new();
        let mut needle_index = 0usize;
        group.bench_function("icu+memchr/fold+find", |bencher| {
            bencher.iter(|| {
                let haystack_text = black_box(haystack_str);
                let index = needle_index % search_needles.len();
                needle_index += 1;
                let needle = search_needles[index];
                // Fold both haystack and needle inside the benchmark
                let folded_haystack = case_mapper.fold_string(haystack_text);
                let folded_needle = case_mapper.fold_string(needle);
                let finder = memmem::Finder::new(folded_needle.as_bytes());
                black_box(finder.find_iter(folded_haystack.as_bytes()).count())
            })
        });
    }
}
fn main() {
    install_panic_hook();
    log_stringzilla_metadata();
    log_pcre2_metadata();

    // Load the dataset defined by the environment variables
    let tape = load_dataset().unwrap_nice();

    // Get the parent data directly from the tape (zero-copy)
    let haystack = tape.parent();
    let needles = &tape;

    let mut criterion = configure_bench(WallTime, 3, 20);

    let mut group = criterion.benchmark_group("case-fold");
    bench_case_fold(&mut group, &haystack, needles);
    group.finish();

    let mut group = criterion.benchmark_group("normalize");
    bench_normalize(&mut group, &haystack, needles);
    group.finish();

    let mut group = criterion.benchmark_group("case-insensitive-compare");
    bench_case_insensitive_compare(&mut group, &haystack, needles);
    group.finish();

    let mut group = criterion.benchmark_group("case-insensitive-find");
    bench_case_insensitive_find(&mut group, &haystack, needles);
    group.finish();

    criterion.final_summary();
}
