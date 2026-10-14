#![doc = r#"# StringWars: Memory

Low-level memory benchmarks: lookup-table transforms, PRNG fills, memset, memcpy, memmove, and
the column verbs that map or gather a whole column of tokens in one call.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_memory --bench bench_memory
```
"#]
use std::alloc::Global;
use std::hint::black_box;
use std::process::ExitCode;
use std::ptr;

use arrow::array::BinaryArray;
use rand::{Rng, SeedableRng};
use stringzilla::sz;
use stringzilla::sz::{Capabilities, Stream, Strings, StringsView};
use zeroize::Zeroize;

use stringwars::{
    expect_ok, finish, install_panic_hook, log_timing_overhead, measure, note_unavailable,
    note_working_set, print_machine, read_dataset, working_set_ranges, Bytes, MeasureSpec,
    ResultExt, Settings, Unit, WorkUnits,
};

/// Cycles `tokens` by a local cursor, calls `work` on the current mutable token, and
/// reports throughput as bytes equal to the token length.  This avoids repeating the
/// `{ let mut cursor = 0usize; measure_throughput(…) }` block for every single-buffer
/// variant that transforms one token in place.
fn measure_mut_token<Work: FnMut(&mut [u8])>(
    settings: &Settings,
    name: &str,
    tokens: &mut [&mut [u8]],
    mut work: Work,
) {
    let pass = WorkUnits::new(
        tokens.len() as u64,
        Bytes(tokens.iter().map(|token| token.len() as u64).sum()),
    );
    measure(settings, name, MeasureSpec::new(Unit::Bytes, pass), || {
        for token in tokens.iter_mut() {
            work(token);
        }
    });
}

/// Borrows the working set as mutable token slices.
///
/// The split and deduplication rules live in `stringwars::working_set_ranges`; this only
/// turns the ranges into disjoint `&mut` slices. The suite used to carry its own copy of the
/// rule, and the copy had drifted. The in-place benchmarks mutate their tokens, so they
/// need these owned, mutable slices rather than the shared immutable tape.
pub fn tokenize_mut<'a>(settings: &Settings, haystack: &'a mut [u8]) -> Vec<&'a mut [u8]> {
    let ranges = working_set_ranges(haystack, settings);

    // Hand out one disjoint `&mut` per range by splitting the tail repeatedly.
    let mut rest = haystack;
    let mut consumed = 0usize;
    let mut tokens = Vec::with_capacity(ranges.len());
    for range in ranges {
        let (_, tail) = rest.split_at_mut(range.start - consumed);
        let (token, tail) = tail.split_at_mut(range.end - range.start);
        tokens.push(token);
        consumed = range.end;
        rest = tail;
    }
    tokens
}

/// Benchmarks in-place lookup-table transforms, transforming one token per call and cycling the
/// dataset. Throughput is reported as bytes/s, matching the original `Throughput::Bytes` over the
/// sum of token lengths.
fn bench_lookup_table(settings: &Settings, tokens: &mut [&mut [u8]]) {
    let lookup_invert_case: [u8; 256] = core::array::from_fn(|index| match index as u8 {
        byte @ b'A'..=b'Z' => byte + 32,
        byte @ b'a'..=b'z' => byte - 32,
        byte => byte,
    });

    measure_mut_token(
        settings,
        "lookup-table/stringzilla::lookup_inplace",
        tokens,
        |token| {
            sz::lookup_inplace(token, lookup_invert_case);
            black_box(token);
        },
    );

    measure_mut_token(settings, "lookup-table/serial", tokens, |token| {
        for byte in token.iter_mut() {
            *byte = lookup_invert_case[*byte as usize];
        }
        black_box(&token);
    });

    // The column verb maps every token in one call, writing a new tape rather than in place.
    let sources: Vec<&[u8]> = tokens.iter().map(|token| &**token).collect();
    let pass = WorkUnits::new(
        sources.len() as u64,
        Bytes(sources.iter().map(|token| token.len() as u64).sum()),
    );
    let cpu = Stream::default(Capabilities::cpu_enabled());
    measure(
        settings,
        "lookup-table/stringzilla::StringsView::lookup_in",
        MeasureSpec::new(Unit::Bytes, pass),
        || {
            let column = StringsView::from_slices(&sources);
            black_box(expect_ok(column.lookup_in(&lookup_invert_case, Global, &cpu)));
        },
    );
}

/// Benchmarks random-string generation, filling one token per call and cycling the dataset.
/// Throughput is reported as bytes/s, matching the original `Throughput::Bytes` over the sum of
/// token lengths.
fn bench_generate_random(settings: &Settings, tokens: &mut [&mut [u8]]) {
    measure_mut_token(
        settings,
        "generate-random/stringzilla::fill_random",
        tokens,
        |token| {
            sz::fill_random(token, 0);
        },
    );

    measure_mut_token(
        settings,
        "generate-random/getrandom::fill",
        tokens,
        |token| {
            getrandom::fill(token).expect("getrandom failed");
            black_box(&token);
        },
    );

    // Benchmark using `rand_chacha::ChaCha20Rng`. The generator is seeded once, outside
    // the timed region: `sz::fill_random` constructs nothing per token, so seeding per
    // token would charge these rows for setup their contender never pays.
    {
        let mut random_generator = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
        measure_mut_token(
            settings,
            "generate-random/rand_chacha::ChaCha20Rng",
            tokens,
            |token| {
                random_generator.fill_bytes(token);
                black_box(&token);
            },
        );
    }

    {
        let mut random_generator = rand_xoshiro::Xoshiro128Plus::from_seed([0u8; 16]);
        measure_mut_token(
            settings,
            "generate-random/rand_xoshiro::Xoshiro128Plus",
            tokens,
            |token| {
                random_generator.fill_bytes(token);
                black_box(&token);
            },
        );
    }
}

/// Benchmarks memory-fill operations, filling one buffer per call and cycling the dataset.
///
/// Fills the tokens themselves rather than per-row `Vec<Vec<u8>>` copies: the generators
/// above have already overwritten every byte, and the copies cost one allocation per token
/// plus a corpus-sized clone for each row.
fn bench_memset(settings: &Settings, tokens: &mut [&mut [u8]]) {
    const FILL_VALUE: u8 = 0xAA;

    if settings.selects("memset/stringzilla::fill") {
        note_unavailable("memset/stringzilla::fill", "StringZilla v6 has no memset");
    }

    measure_mut_token(settings, "memset/std::ptr::write_bytes", tokens, |buffer| {
        unsafe {
            ptr::write_bytes(buffer.as_mut_ptr(), FILL_VALUE, buffer.len());
        }
        black_box(&buffer);
    });

    measure_mut_token(settings, "memset/slice::fill", tokens, |buffer| {
        buffer.fill(FILL_VALUE);
        black_box(&buffer);
    });

    // `zeroize` writes zeros through volatile stores, so it is a memset with a fixed value
    // rather than a generator; it used to sit in the random-generation group.
    measure_mut_token(settings, "memset/zeroize::zeroize", tokens, |buffer| {
        buffer.zeroize();
        black_box(&buffer);
    });
}

/// Benchmarks memory-copy operations, copying one buffer per call and cycling the dataset.
///
/// Sources are the tokens and destinations one arena split into matching slices. The previous
/// shape held three corpus-sized `Vec<Vec<u8>>` at once — a separate allocation per token.
fn bench_memcpy(settings: &Settings, tokens: &[&mut [u8]]) {
    let total_bytes: usize = tokens.iter().map(|token| token.len()).sum();
    if total_bytes == 0 {
        return;
    }
    let mut arena = vec![0u8; total_bytes];
    let mut destinations: Vec<&mut [u8]> = Vec::with_capacity(tokens.len());
    let mut rest = &mut arena[..];
    for token in tokens.iter() {
        let (head, tail) = rest.split_at_mut(token.len());
        destinations.push(head);
        rest = tail;
    }
    let pass = WorkUnits::new(tokens.len() as u64, Bytes(total_bytes as u64));

    if settings.selects("memcpy/stringzilla::copy") {
        note_unavailable("memcpy/stringzilla::copy", "StringZilla v6 has no memcpy");
    }

    measure(
        settings,
        "memcpy/slice::copy_from_slice",
        MeasureSpec::new(Unit::Bytes, pass),
        || {
            for (source, dest) in tokens.iter().zip(destinations.iter_mut()) {
                dest.copy_from_slice(source);
                black_box(&dest);
            }
        },
    );

    measure(
        settings,
        "memcpy/std::ptr::copy_nonoverlapping",
        MeasureSpec::new(Unit::Bytes, pass),
        || {
            for (source, dest) in tokens.iter().zip(destinations.iter_mut()) {
                unsafe {
                    ptr::copy_nonoverlapping(source.as_ptr(), dest.as_mut_ptr(), source.len());
                }
                black_box(&dest);
            }
        },
    );

    // Gathering a whole column into one tape of bytes and offsets, the column-level copy.
    let sources: Vec<&[u8]> = tokens.iter().map(|token| &**token).collect();
    let cpu = Stream::default(Capabilities::cpu_enabled());
    measure(
        settings,
        "memcpy/stringzilla::Strings::copy_in",
        MeasureSpec::new(Unit::Bytes, pass),
        || {
            let column = StringsView::from_slices(&sources);
            black_box(expect_ok(Strings::copy_in(column, Global, &cpu)));
        },
    );

    measure(
        settings,
        "memcpy/arrow::BinaryArray::from_iter_values",
        MeasureSpec::new(Unit::Bytes, pass),
        || {
            black_box(BinaryArray::from_iter_values(sources.iter().copied()));
        },
    );
}

/// Benchmarks memory-move operations, shifting one buffer per call and cycling the dataset.
/// Only tokens longer than `SHIFT` participate, and the per-call byte count is `len - SHIFT`,
/// matching the original `Throughput::Bytes(sum(len - SHIFT))` accounting.
fn bench_memmove(settings: &Settings, tokens: &mut [&mut [u8]]) {
    const SHIFT: usize = 8;
    // The tokens themselves, shifted in place: the shorter ones are filtered out here so the
    // timed loop carries no length test.
    let mut movable: Vec<&mut [u8]> = tokens
        .iter_mut()
        .filter(|token| token.len() > SHIFT)
        .map(|token| &mut **token)
        .collect();
    if movable.is_empty() {
        return;
    }
    let pass = WorkUnits::new(
        movable.len() as u64,
        Bytes(movable.iter().map(|b| (b.len() - SHIFT) as u64).sum()),
    );

    if settings.selects("memmove/stringzilla::move_") {
        note_unavailable("memmove/stringzilla::move_", "StringZilla v6 has no memmove");
    }

    measure(
        settings,
        "memmove/std::ptr::copy",
        MeasureSpec::new(Unit::Bytes, pass),
        || {
            for buffer in movable.iter_mut() {
                let move_len = buffer.len() - SHIFT;
                unsafe {
                    ptr::copy(buffer.as_ptr(), buffer.as_mut_ptr().add(SHIFT), move_len);
                }
                black_box(&buffer);
            }
        },
    );

    measure(
        settings,
        "memmove/slice::copy_within",
        MeasureSpec::new(Unit::Bytes, pass),
        || {
            for buffer in movable.iter_mut() {
                let move_len = buffer.len() - SHIFT;
                buffer.copy_within(0..move_len, SHIFT);
                black_box(&buffer);
            }
        },
    );
}

fn main() -> ExitCode {
    install_panic_hook();
    print_machine();
    let settings = Settings::read("memory");
    settings.print();

    let mut dataset = read_dataset(&settings).unwrap_nice();
    let mut tokens = tokenize_mut(&settings, &mut dataset);
    if tokens.is_empty() {
        panic!("No tokens found in the dataset.");
    }
    note_working_set(&settings, &tokens);
    log_timing_overhead(&settings);

    println!("# lookup-table");
    bench_lookup_table(&settings, &mut tokens[..]);

    println!("# generate-random");
    bench_generate_random(&settings, &mut tokens[..]);

    println!("# memset");
    bench_memset(&settings, &mut tokens[..]);

    println!("# memcpy");
    bench_memcpy(&settings, &tokens[..]);

    println!("# memmove");
    bench_memmove(&settings, &mut tokens[..]);

    finish()
}
