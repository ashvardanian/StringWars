#![doc = r#"# StringWars: Fingerprints

MinHash fingerprinting benchmarks, and the exact window overlap MinHash estimates.

- `STRINGWARS_DIMS` sets the sketch widths, one count or a comma list like `64,128`; the
  width moves throughput by ~8x, so it is recorded with every published number.
- `STRINGWARS_BATCH_PER_CORE` sets how many documents the overlap rows take as queries.
- `STRINGWARS_THREADS` sets how many overlap engines the multi-core row runs, one per pool thread.
- `STRINGWARS_DEVICES` names the GPUs that carry rows, in a `cuda` build.

StringZilla v6 has no MinHash engine, so the MinHash groups note its row as unavailable and
`OverlapEngine` answers the question MinHash approximates: how many of one text's n-grams the
other spells.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_fingerprints --bench bench_fingerprints
```
"#]
#![allow(clippy::needless_range_loop)]

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::process::ExitCode;

use probabilistic_collections::similarity::{ByteGrams, MinHash};
use stringzilla::sz::{
    Capabilities, OverlapEngine, Stream, Strings, StringsView, UnifiedAllocator,
};

use stringwars::{
    batch_size, expect_ok, finish, gpu_ordinals, gpu_stream, install_panic_hook,
    log_timing_overhead, measure, measure_per_thread, note_unavailable, print_machine,
    resolve_dataset, Bytes, MeasureSpec, ResultExt, Settings, Unit, WorkUnits,
};

// Fixed n-gram widths for multi-scale fingerprinting
const NGRAM_WIDTHS: [usize; 4] = [5, 9, 17, 33];

/// Calculate bit entropy (how well distributed the bits are) over a flattened
/// `documents × dimensions` sketch matrix, normalized to [0, 1].
fn bit_entropy<T>(hash_values: &[T]) -> f64
where
    T: Copy + Into<u64>,
{
    if hash_values.is_empty() {
        return 0.0;
    }

    let bits_per_hash = std::mem::size_of::<T>() * 8;
    let mut bit_ones_count = vec![0usize; bits_per_hash];
    for &hash_value in hash_values {
        let hash_as_u64: u64 = hash_value.into();
        for bit_position in 0..bits_per_hash {
            if (hash_as_u64 >> bit_position) & 1 == 1 {
                bit_ones_count[bit_position] += 1;
            }
        }
    }

    let mut total_entropy = 0.0;
    for ones_count in bit_ones_count {
        let probability_of_one = ones_count as f64 / hash_values.len() as f64;
        if probability_of_one > 0.0 && probability_of_one < 1.0 {
            total_entropy -= probability_of_one * probability_of_one.log2()
                + (1.0 - probability_of_one) * (1.0 - probability_of_one).log2();
        }
    }

    total_entropy / bits_per_hash as f64
}

/// Calculate collision rate (duplicate hash values) over a flattened sketch matrix. This is a
/// birthday statistic, so it is meaningful only over the whole sample it is reported for.
fn collision_rate<T>(hash_values: &[T]) -> f64
where
    T: Copy + std::hash::Hash + Eq,
{
    if hash_values.is_empty() {
        return 0.0;
    }

    let unique_hash_values: HashSet<T> = hash_values.iter().copied().collect();
    1.0 - (unique_hash_values.len() as f64 / hash_values.len() as f64)
}

fn bench_minhash(settings: &Settings, documents: &[&[u8]], total_bytes: Bytes) {
    let total_documents = documents.len();
    for &dimensions in &settings.dims {
        println!("\nBenchmark configuration:");
        println!("- Fingerprint dimensions: {}", dimensions);
        println!("- N-gram widths: {:?} bytes", NGRAM_WIDTHS);
        println!(
            "- Hashes per width: {} (total NDIM: {})",
            dimensions / NGRAM_WIDTHS.len(),
            dimensions
        );

        // Quality-analysis matrices, flattened to one contiguous `N_docs * N_dims` buffer each:
        // a Vec-of-Vec rebuilt per NDIM scale cost ~120k allocations against a 1 MB working set.
        let mut serial_matrix = vec![0u64; total_documents * dimensions];
        let mut pc_matrix = vec![0u64; total_documents * dimensions];
        let mut serial_document_index = 0usize;
        let mut pc_document_index = 0usize;

        println!("# minhash-{dimensions}");

        let stringzilla_name = format!("minhash-{dimensions}/stringzillas::Fingerprints");
        if settings.selects(&stringzilla_name) {
            note_unavailable(&stringzilla_name, "StringZilla v6 has no MinHash engine");
        }

        let hashes_per_width = dimensions / NGRAM_WIDTHS.len();
        // One MinHash per n-gram width, on the stack (no heap Vec).
        let minhashers: [MinHash<ByteGrams, _>; NGRAM_WIDTHS.len()] =
            core::array::from_fn(|_| MinHash::new(hashes_per_width));
        let work = || {
            MeasureSpec::new(
                Unit::Hashes,
                WorkUnits::new(dimensions as u64 * total_bytes.0, total_bytes),
            )
        };

        // pc::MinHash baseline (single-threaded scalar).
        {
            // Reused across documents and passes, like the serial baseline below.
            let mut combined_signature = Vec::with_capacity(dimensions);
            measure(
                settings,
                &format!("minhash-{dimensions}/pc::MinHash<ByteGrams>"),
                work(),
                || {
                    for line_bytes in documents.iter().filter(|line| !line.is_empty()) {
                        combined_signature.clear();
                        for (width_index, &width) in NGRAM_WIDTHS.iter().enumerate() {
                            let iter = ByteGrams::new(line_bytes, width);
                            let partial_sig = minhashers[width_index].get_min_hashes(iter);
                            combined_signature.extend(partial_sig);
                        }
                        if pc_document_index < total_documents
                            && combined_signature.len() == dimensions
                        {
                            pc_matrix[pc_document_index * dimensions..][..dimensions]
                                .copy_from_slice(&combined_signature);
                            pc_document_index += 1;
                        }
                        std::hint::black_box(&combined_signature);
                    }
                },
            );
        }

        // Serial MinHash baseline with independent universal hash functions per dimension.
        {
            // Each hash function uses: hash_i(x) = (a_i * hash(x) + b_i) mod mersenne_prime
            const MERSENNE_PRIME: u64 = (1u64 << 61) - 1;
            let hash_params: Vec<(u64, u64)> = (0..dimensions)
                .map(|dimension_index| (2 * dimension_index as u64 + 1, dimension_index as u64))
                .collect();

            // Reused across documents and passes, so no allocation lands in the timed loop.
            let mut min_hashes = vec![u64::MAX; dimensions];
            measure(
                settings,
                &format!("minhash-{dimensions}/serial::MinHash<ByteGrams>"),
                work(),
                || {
                    for line_bytes in documents.iter().filter(|line| !line.is_empty()) {
                        min_hashes.fill(u64::MAX);
                        for &width in &NGRAM_WIDTHS {
                            for window in line_bytes.windows(width) {
                                let mut hasher = DefaultHasher::new();
                                window.hash(&mut hasher);
                                let base_hash = hasher.finish();
                                for (hash_index, &(multiplier, offset)) in
                                    hash_params.iter().enumerate()
                                {
                                    let independent_hash = (multiplier
                                        .wrapping_mul(base_hash)
                                        .wrapping_add(offset))
                                        % MERSENNE_PRIME;
                                    min_hashes[hash_index] =
                                        min_hashes[hash_index].min(independent_hash);
                                }
                            }
                        }
                        if serial_document_index < total_documents {
                            serial_matrix[serial_document_index * dimensions..][..dimensions]
                                .copy_from_slice(&min_hashes);
                            serial_document_index += 1;
                        }
                        std::hint::black_box(&min_hashes);
                    }
                },
            );
        }

        println!("\nHash Quality Analysis");
        println!(
            "Documents processed: PC={}, Serial={}",
            pc_document_index, serial_document_index
        );

        if pc_document_index > 0 {
            let pc_slice = &pc_matrix[..pc_document_index * dimensions];
            println!("\npc::MinHash<ByteGrams>:");
            println!("  Bit Entropy:  {:.4}", bit_entropy(pc_slice));
            println!("  Collision:    {:.4}%", collision_rate(pc_slice) * 100.0);
        }

        if serial_document_index > 0 {
            let serial_slice = &serial_matrix[..serial_document_index * dimensions];
            println!("\nserial::MinHash<ByteGrams>:");
            println!("  Bit Entropy:  {:.4}", bit_entropy(serial_slice));
            println!(
                "  Collision:    {:.4}%",
                collision_rate(serial_slice) * 100.0
            );
        }
    }
}

/// The candidates one of `threads` engines scores, contiguous so its block of scores is too.
fn share(count: usize, thread: usize, threads: usize) -> std::ops::Range<usize> {
    let chunk = count.div_ceil(threads);
    let first = (thread * chunk).min(count);
    first..(first + chunk).min(count)
}

/// Scores the first `batch_per_core` documents as queries against every document as a candidate,
/// at every n-gram width, the exact overlap a MinHash sketch estimates. One pass streams every
/// candidate past every query, so the byte rate is candidate bytes per query batch.
fn bench_overlap(settings: &Settings, documents: &[&[u8]], total_bytes: Bytes) {
    let queries = &documents[..batch_size(settings, 1).min(documents.len())];
    let widths = NGRAM_WIDTHS.len();
    let work = WorkUnits::new((queries.len() * documents.len()) as u64, total_bytes);
    let call = "overlap/stringzilla::OverlapEngine::scores";
    println!("\n# overlap - {} queries, {} candidates", queries.len(), documents.len());

    // One engine, every candidate, on this core.
    let cpu = Stream::default(Capabilities::cpu_enabled());
    let name = format!("{call}<1cpu>");
    let candidate_stride = widths;
    let query_stride = documents.len() * widths;
    if settings.selects(&name) {
        let mut engine = expect_ok(OverlapEngine::new(queries, &NGRAM_WIDTHS, 0, &cpu));
        let mut scores = vec![0f32; queries.len() * query_stride];
        measure(settings, &name, MeasureSpec::new(Unit::Bytes, work), || {
            let candidates = StringsView::from_slices(documents);
            expect_ok(cpu.scope(|scope| {
                engine.scores(scope, candidates, &mut scores, query_stride, candidate_stride)
            }));
        });
    }

    // One engine per pool thread, each holding every query against its share of the candidates.
    let threads = settings.threads;
    let count = threads.0.get();
    measure_per_thread(
        settings,
        &format!("{call}<{threads}cpu>"),
        MeasureSpec::new(Unit::Bytes, work).with_concurrency(threads),
        |thread| {
            let candidates = share(documents.len(), thread, count);
            let stream = Stream::default(Capabilities::cpu_enabled());
            let engine = expect_ok(OverlapEngine::new(queries, &NGRAM_WIDTHS, 0, &stream));
            let scores = vec![0f32; queries.len() * candidates.len() * widths];
            (stream, engine, scores, candidates)
        },
        |(stream, engine, scores, candidates), _| {
            let query_stride = candidates.len() * widths;
            let view = StringsView::from_slices(&documents[candidates.clone()]);
            expect_ok(stream.scope(|scope| {
                engine.scores(scope, view, scores, query_stride.max(widths), widths)
            }));
        },
    );

    // One engine per GPU, its first round checked against the host engine.
    for ordinal in gpu_ordinals(settings) {
        let name = format!("{call}<cuda:{ordinal}>");
        if !settings.selects(&name) {
            continue;
        }
        let Some(stream) = gpu_stream(settings, ordinal) else {
            continue;
        };
        let mut engine =
            match OverlapEngine::new(queries, &NGRAM_WIDTHS, documents.len(), &stream) {
                Ok(engine) => engine,
                Err(error) => {
                    note_unavailable(&name, &error.to_string());
                    continue;
                }
            };
        let unified = UnifiedAllocator::new(&stream);
        let candidates = expect_ok(Strings::copy_in(
            StringsView::from_slices(documents),
            unified,
            &cpu,
        ));
        let mut scores = Vec::new_in(unified);
        scores.resize(queries.len() * query_stride, 0f32);
        expect_ok(stream.scope(|scope| {
            engine.scores(scope, candidates.view(), &mut scores, query_stride, candidate_stride)
        }));
        let mut host = expect_ok(OverlapEngine::new(queries, &NGRAM_WIDTHS, 0, &cpu));
        let mut expected = vec![0f32; queries.len() * query_stride];
        let host_candidates = StringsView::from_slices(documents);
        expect_ok(cpu.scope(|scope| {
            host.scores(scope, host_candidates, &mut expected, query_stride, candidate_stride)
        }));
        assert!(
            scores.iter().zip(expected.iter()).all(|(device, host)| (device - host).abs() <= 1e-4),
            "{name} disagrees with the host engine on its first round"
        );
        measure(settings, &name, MeasureSpec::new(Unit::Bytes, work), || {
            let view = candidates.view();
            expect_ok(stream.scope(|scope| {
                engine.scores(scope, view, &mut scores, query_stride, candidate_stride)
            }));
        });
    }

    // The same shares computed exactly with one set of distinct windows per query and width, the
    // way an index-free deduplication pass would.
    let name = "overlap/serial::window_overlap<1cpu>";
    if settings.selects(name) {
        let query_windows: Vec<Vec<HashSet<&[u8]>>> = queries
            .iter()
            .map(|query| {
                NGRAM_WIDTHS
                    .iter()
                    .map(|&width| query.windows(width).collect())
                    .collect()
            })
            .collect();
        let mut scores = vec![0f32; queries.len() * query_stride];
        measure(settings, name, MeasureSpec::new(Unit::Bytes, work), || {
            for (query_index, (query, windows_by_width)) in
                queries.iter().zip(&query_windows).enumerate()
            {
                for (candidate_index, candidate) in documents.iter().enumerate() {
                    for (width_index, &width) in NGRAM_WIDTHS.iter().enumerate() {
                        let distinct = &windows_by_width[width_index];
                        let windows = candidate.len().saturating_sub(width - 1);
                        let hits = candidate
                            .windows(width)
                            .filter(|window| distinct.contains(window))
                            .count();
                        // The engine divides by every window of the longer side, repeats included.
                        let total = query.len().saturating_sub(width - 1).max(windows).max(1);
                        scores[query_index * query_stride
                            + candidate_index * candidate_stride
                            + width_index] = hits as f32 / total as f32;
                    }
                }
            }
            std::hint::black_box(&scores);
        });
    }
}

fn main() -> ExitCode {
    install_panic_hook();
    print_machine();
    let settings = Settings::read("fingerprints");
    settings.print();
    println!("Text Fingerprinting Benchmarks");
    println!("- pc::MinHash<ByteGrams>: MinHash with ByteGrams iterator");
    println!("- stringzilla::OverlapEngine: exact n-gram overlap, the quantity MinHash estimates");

    let tape = resolve_dataset(&settings).unwrap_nice();
    log_timing_overhead(&settings);
    let documents: Vec<&[u8]> = tape.iter().collect();
    let total_bytes = Bytes(documents.iter().map(|token| token.len() as u64).sum());

    bench_minhash(&settings, &documents, total_bytes);
    bench_overlap(&settings, &documents, total_bytes);

    finish()
}
