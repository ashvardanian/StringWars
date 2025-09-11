#![doc = r#"# StringWa.rs: Text Fingerprinting Benchmarks

This benchmark compares fingerprinting implementations using multi-width byte n-grams
across CPU (1x, Nx) and GPU architectures.

It evaluates:
- StringZilla Fingerprints (`szs::Fingerprints`) with configurable window widths [3,5,7,15] bytes  
- Custom MinHash with byte n-grams (baseline comparison)

Both use NDIM hash functions distributed evenly across the 4 window widths.
Work is reported in bytes/sec (average bytes per line × batch size).

## Usage Examples

Environment variables control dataset and processing:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_BATCH`: Number of lines per batch for StringZilla (default: 1024).
- `STRINGWARS_MAX_TOKENS`: Optional cap for total lines to process.
- `STRINGWARS_NDIM`: Total hash functions distributed across n-gram widths (default: 256).

N-gram configuration:
- Fixed window widths: [15, 33, 65, 129] bytes for multi-scale fingerprinting
- Hash distribution: NDIM/4 hash functions per window width
- Example: NDIM=256 → 64 hashes each for 3-byte, 5-byte, 7-byte, 15-byte n-grams

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH=1024 \
    STRINGWARS_NDIM=256 \
    cargo criterion --features bench_fingerprints bench_fingerprints --jobs 8
```
"#]

use core::convert::TryInto;
use std::env;
use std::fs;

use criterion::{Criterion, Throughput};
use fork_union::count_logical_cores;

use probabilistic_collections::similarity::MinHash;
use stringtape::{BytesTape, BytesTapeView, CharsTapeView};
use stringzilla::sz::dynamic_dispatch as sz_dynamic_dispatch;
use stringzilla::szs::{capabilities as szs_capabilities, version as szs_version};
use stringzilla::szs::{AnyBytesTape, DeviceScope, Fingerprints, UnifiedAlloc, UnifiedVec};

// Fixed n-gram widths for multi-scale fingerprinting
const NGRAM_WIDTHS: [usize; 4] = [15, 33, 65, 129];

/// Simple iterator that generates n-byte-grams of a single specific width
/// Compatible with `probabilistic_collections::MinHash` API
pub struct ByteGrams<'a> {
    data: &'a [u8],
    width: usize,
    pos: usize,
}

impl<'a> ByteGrams<'a> {
    pub fn new(data: &'a [u8], width: usize) -> Self {
        Self {
            data,
            width,
            pos: 0,
        }
    }
}

impl<'a> Iterator for ByteGrams<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + self.width <= self.data.len() {
            let gram = &self.data[self.pos..self.pos + self.width];
            self.pos += 1;
            Some(gram)
        } else {
            None
        }
    }
}

fn tokens_tape_slice<'a>(
    bytes_view: &'a BytesTapeView<u64>,
    chars_view: &'a CharsTapeView<u64>,
    start_idx: &mut usize,
    batch_size: usize,
    tokens_count: usize,
) -> (BytesTapeView<'a, u64>, CharsTapeView<'a, u64>, usize) {
    let current_start = *start_idx;
    let end = std::cmp::min(current_start + batch_size, tokens_count);
    let actual_batch_size = end - current_start;

    let batch_bytes_view = bytes_view.subview(current_start, end).unwrap_or_else(|e| {
        panic!(
            "Failed to create BytesTape subview from {} to {} (total tokens: {}): {}",
            current_start, end, tokens_count, e
        )
    });

    let batch_chars_view = chars_view.subview(current_start, end).unwrap_or_else(|e| {
        panic!(
            "Failed to create CharsTape subview from {} to {} (total tokens: {}): {}",
            current_start, end, tokens_count, e
        )
    });

    *start_idx = (current_start + actual_batch_size) % tokens_count;
    (batch_bytes_view, batch_chars_view, actual_batch_size)
}

fn log_stringzilla_metadata() {
    let v = szs_version();
    println!("StringZilla v{}.{}.{}", v.major, v.minor, v.patch);
    println!("- uses dynamic dispatch: {}", sz_dynamic_dispatch());
    println!("- capabilities: {}", szs_capabilities().as_str());
}

fn configure_bench() -> Criterion {
    // Align with bench_similarities defaults
    Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(10))
}

fn bench_fingerprints(c: &mut Criterion) {
    let dataset_path = env::var("STRINGWARS_DATASET").unwrap_or_else(|_| {
        panic!("STRINGWARS_DATASET environment variable must be set for fingerprinting benchmarks")
    });
    let content = fs::read_to_string(&dataset_path)
        .unwrap_or_else(|e| panic!("Failed to read dataset file '{}': {}", dataset_path, e));

    let batch_size = env::var("STRINGWARS_BATCH")
        .unwrap_or_else(|_| "1024".to_string())
        .parse::<usize>()
        .unwrap_or_else(|e| {
            panic!(
                "STRINGWARS_BATCH must be a valid number for fingerprinting benchmarks: {}",
                e
            )
        });

    let max_tokens = env::var("STRINGWARS_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    let ndim = env::var("STRINGWARS_NDIM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256);

    // Tokenize into documents according to mode (zero-copy slices of `content`).
    let units: Vec<&str> = content.lines().collect();

    if units.is_empty() {
        panic!("Dataset must contain at least one token for fingerprinting.");
    }

    if batch_size == 0 {
        panic!("STRINGWARS_BATCH must be greater than zero for fingerprinting benchmarks.");
    }

    if ndim == 0 {
        panic!("STRINGWARS_NDIM must be greater than zero for fingerprinting benchmarks.");
    }

    // Log dataset statistics
    let total_units = units.len();
    let total_bytes: usize = units.iter().map(|s| s.len()).sum();
    let total_chars: usize = units.iter().map(|s| s.chars().count()).sum();
    let avg_bytes_per_unit = total_bytes as f64 / total_units as f64;
    let avg_chars_per_unit = total_chars as f64 / total_units as f64;

    println!("Dataset statistics:");
    println!("- Source: {}", dataset_path);
    println!("- Processing mode: lines");
    println!("- Total lines: {}", total_units);
    println!(
        "- Average line length: {:.1} bytes, {:.1} chars",
        avg_bytes_per_unit, avg_chars_per_unit
    );
    println!(
        "- Total dataset size: {} bytes, {} chars",
        total_bytes, total_chars
    );
    println!("- Batch size (for StringZilla): {}", batch_size);
    println!("- Fingerprint dimensions: {}", ndim);

    // Log n-gram configuration
    println!("- N-gram widths: {:?} bytes", NGRAM_WIDTHS);
    println!(
        "- Hashes per width: {} (total NDIM: {})",
        ndim / NGRAM_WIDTHS.len(),
        ndim
    );

    // Truncate if max_tokens is specified
    let mut truncated_units = units.clone();
    if let Some(max_t) = max_tokens {
        if max_t < units.len() {
            truncated_units.truncate(max_t);
            println!(
                "- Max tokens limit: {} (truncated from {})",
                max_t,
                units.len()
            );
        }
    }

    // Create single BytesTape and populate it with all units (following bench_similarities.rs)
    let mut units_tape: BytesTape<u64, UnifiedAlloc> = BytesTape::new_in(UnifiedAlloc);
    units_tape
        .extend(truncated_units.iter().map(|s| s.as_bytes()))
        .unwrap_or_else(|e| panic!("Failed to extend BytesTape for fingerprinting: {}", e));

    // Create both byte and char views from single tape (zero-copy casting)
    let bytes_view = units_tape.view();
    let chars_view: CharsTapeView<u64> = units_tape
        .view()
        .try_into()
        .unwrap_or_else(|e| panic!("Failed to convert BytesTapeView to CharsTapeView: {}", e));

    // Average bytes per token for throughput reporting
    let avg_token_bytes: u64 = total_bytes as u64 / truncated_units.len() as u64;
    let per_batch_bytes = (batch_size as u64) * avg_token_bytes;

    let mut g = c.benchmark_group("fingerprinting");

    // StringZilla engines and device scopes (1xCPU, NxCPU, GPU?)
    let num_cores = count_logical_cores();
    let cpu_single = DeviceScope::cpu_cores(1).unwrap_or_else(|e| {
        panic!(
            "Failed to create single-core CPU device scope for fingerprinting: {}",
            e
        )
    });
    let cpu_parallel = DeviceScope::cpu_cores(num_cores).unwrap_or_else(|e| {
        panic!(
            "Failed to create {}-core CPU device scope for fingerprinting: {}",
            num_cores, e
        )
    });
    let maybe_gpu = DeviceScope::gpu_device(0);

    let sz_single = Fingerprints::builder()
        .ascii()
        .window_widths(&NGRAM_WIDTHS)
        .dimensions(ndim)
        .build(&cpu_single)
        .unwrap_or_else(|e| {
            panic!(
                "Failed to create single-core StringZilla fingerprinting engine: {}",
                e
            )
        });
    let sz_parallel = Fingerprints::builder()
        .ascii()
        .window_widths(&NGRAM_WIDTHS)
        .dimensions(ndim)
        .build(&cpu_parallel)
        .unwrap_or_else(|e| {
            panic!(
                "Failed to create {}-core StringZilla fingerprinting engine: {}",
                num_cores, e
            )
        });
    let maybe_sz_gpu = maybe_gpu.as_ref().ok().and_then(|gpu| {
        Fingerprints::builder()
            .ascii()
            .window_widths(&NGRAM_WIDTHS)
            .dimensions(ndim)
            .build(gpu)
            .ok()
    });

    let mut start_idx = 0usize;
    let tokens_count = truncated_units.len();

    // Pre-allocate result buffers for StringZilla compute_into in unified memory (reused across iterations)
    let mut min_hashes = UnifiedVec::<u32>::with_capacity_in(batch_size * ndim, UnifiedAlloc);
    min_hashes.resize(batch_size * ndim, 0);
    let mut min_counts = UnifiedVec::<u32>::with_capacity_in(batch_size * ndim, UnifiedAlloc);
    min_counts.resize(batch_size * ndim, 0);

    // StringZilla: 1x CPU
    g.throughput(Throughput::Bytes(per_batch_bytes));
    g.bench_function("szs::Fingerprints(1xCPU)", |b| {
        start_idx = 0;
        b.iter(|| {
            let (batch_bytes_view, _batch_chars_view, actual) = tokens_tape_slice(
                &bytes_view,
                &chars_view,
                &mut start_idx,
                batch_size,
                tokens_count,
            );

            // Use compute_into for zero-allocation processing
            sz_single
                .compute_into(
                    &cpu_single,
                    AnyBytesTape::View64(batch_bytes_view),
                    ndim,
                    &mut min_hashes[..actual * ndim],
                    &mut min_counts[..actual * ndim],
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to compute StringZilla fingerprints on single CPU core: {}",
                        e
                    )
                });
            std::hint::black_box((&min_hashes[..actual * ndim], &min_counts[..actual * ndim]));
        })
    });

    // StringZilla: Nx CPU
    g.throughput(Throughput::Bytes(per_batch_bytes));
    g.bench_function(&format!("szs::Fingerprints({}xCPU)", num_cores), |b| {
        start_idx = 0;
        b.iter(|| {
            let (batch_bytes_view, _batch_chars_view, actual) = tokens_tape_slice(
                &bytes_view,
                &chars_view,
                &mut start_idx,
                batch_size,
                tokens_count,
            );

            // Use compute_into for zero-allocation processing
            sz_parallel
                .compute_into(
                    &cpu_parallel,
                    AnyBytesTape::View64(batch_bytes_view),
                    ndim,
                    &mut min_hashes[..actual * ndim],
                    &mut min_counts[..actual * ndim],
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to compute StringZilla fingerprints on {} CPU cores: {}",
                        num_cores, e
                    )
                });
            std::hint::black_box((&min_hashes[..actual * ndim], &min_counts[..actual * ndim]));
        })
    });

    // StringZilla: 1x GPU (if available)
    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_sz_gpu.as_ref()) {
        g.throughput(Throughput::Bytes(per_batch_bytes));
        g.bench_function("szs::Fingerprints(1xGPU)", |b| {
            start_idx = 0;
            b.iter(|| {
                let (batch_bytes_view, _batch_chars_view, actual) = tokens_tape_slice(
                    &bytes_view,
                    &chars_view,
                    &mut start_idx,
                    batch_size,
                    tokens_count,
                );

                // Use compute_into for zero-allocation GPU processing
                engine
                    .compute_into(
                        gpu,
                        AnyBytesTape::View64(batch_bytes_view),
                        ndim,
                        &mut min_hashes[..actual * ndim],
                        &mut min_counts[..actual * ndim],
                    )
                    .unwrap_or_else(|e| {
                        panic!("Failed to compute StringZilla fingerprints on GPU: {}", e)
                    });
                std::hint::black_box((&min_hashes[..actual * ndim], &min_counts[..actual * ndim]));
            })
        });
    }

    // Probabilistic-Collections MinHash baseline with multi-width byte n-grams
    // Create separate MinHash instances for each n-gram width
    let hashes_per_width = ndim / NGRAM_WIDTHS.len();
    let minhashers: Vec<MinHash<ByteGrams, _>> = (0..NGRAM_WIDTHS.len())
        .map(|_| MinHash::new(hashes_per_width))
        .collect();

    // Pre-allocate output buffer outside benchmark loop (reused across iterations)
    let mut out = Vec::with_capacity(batch_size);
    let mut combined_signature = Vec::with_capacity(ndim);

    g.throughput(Throughput::Bytes(per_batch_bytes));
    g.bench_function("probabilistic_collections::MinHash<ByteGrams>", |b| {
        start_idx = 0;
        b.iter(|| {
            let (batch_bytes_view, _batch_chars_view, actual) = tokens_tape_slice(
                &bytes_view,
                &chars_view,
                &mut start_idx,
                batch_size,
                tokens_count,
            );

            // Reuse output buffer (clear and reserve)
            out.clear();
            out.reserve(actual);

            // Process each line with separate MinHash per width
            for i in 0..batch_bytes_view.len() {
                let line_bytes: &[u8] = &batch_bytes_view[i];
                if !line_bytes.is_empty() {
                    // Clear and reuse combined signature buffer
                    combined_signature.clear();
                    combined_signature.reserve(ndim);

                    // Process each n-gram width separately and concatenate results
                    for (width_idx, &width) in NGRAM_WIDTHS.iter().enumerate() {
                        let iter = ByteGrams::new(line_bytes, width);
                        let partial_sig = minhashers[width_idx].get_min_hashes(iter);
                        combined_signature.extend(partial_sig);
                    }

                    out.push(combined_signature.clone());
                }
            }
            std::hint::black_box(&out);
        })
    });

    g.finish();
}

fn main() {
    log_stringzilla_metadata();
    println!("Text Fingerprinting Benchmarks");
    println!("- szs::Fingerprints: CPU/GPU fingerprints with multi-width byte n-grams");
    println!("- probabilistic_collections::MinHash<ByteGrams>: Probabilistic collections MinHash with byte n-gram iterator");

    let mut criterion = configure_bench();
    bench_fingerprints(&mut criterion);
    criterion.final_summary();
}
