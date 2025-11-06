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
- Fixed window widths: [5, 9, 17, 33] bytes for multi-scale fingerprinting
- Hash distribution: NDIM/4 hash functions per window width
- Example: NDIM=256 → 64 hashes each for 5-byte, 9-byte, 17-byte, 33-byte n-grams

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH=1024 \
    STRINGWARS_NDIM=256 \
    cargo criterion --features bench_fingerprints bench_fingerprints --jobs 1
```

To run on a GPU-capable machine, enable the CUDA feature and consider larger batches:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH=32768 \
    STRINGWARS_NDIM=256 \
    STRINGWARS_FILTER=1xGPU \
    cargo criterion --features "cuda bench_fingerprints" bench_fingerprints --jobs 1
```
"#]

use core::convert::TryInto;
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::hash::{Hash, Hasher};

use criterion::{Criterion, Throughput};
use fork_union::count_logical_cores;
use stringtape::{BytesTape, BytesTapeView, CharsTapeView};

use probabilistic_collections::similarity::{ByteGrams, MinHash};
use stringzilla::sz::dynamic_dispatch as sz_dynamic_dispatch;
use stringzilla::szs::{capabilities as szs_capabilities, version as szs_version};
use stringzilla::szs::{AnyBytesTape, DeviceScope, Fingerprints, UnifiedAlloc, UnifiedVec};

mod utils;
use utils::{load_dataset, set_fingerprints_bytes_per_hash, should_run, HashesWallTime};

// Fixed n-gram widths for multi-scale fingerprinting
const NGRAM_WIDTHS: [usize; 4] = [5, 9, 17, 33];

/// Calculate bit entropy (how well distributed the bits are) - generic
fn bit_entropy<T>(hash_matrix: &[Vec<T>]) -> f64
where
    T: Copy + Into<u64>,
{
    if hash_matrix.is_empty() || hash_matrix[0].is_empty() {
        return 0.0;
    }

    // Determine bit width based on type
    let bits_per_hash = std::mem::size_of::<T>() * 8;
    let mut bit_ones_count = vec![0usize; bits_per_hash]; // Count of 1s for each bit position
    let total_hash_values = hash_matrix.len() * hash_matrix[0].len();

    for document_hashes in hash_matrix {
        for &hash_value in document_hashes {
            let hash_as_u64: u64 = hash_value.into();
            for bit_position in 0..bits_per_hash {
                if (hash_as_u64 >> bit_position) & 1 == 1 {
                    bit_ones_count[bit_position] += 1;
                }
            }
        }
    }

    // Calculate entropy
    let mut total_entropy = 0.0;
    for ones_count in bit_ones_count {
        let probability_of_one = ones_count as f64 / total_hash_values as f64;
        if probability_of_one > 0.0 && probability_of_one < 1.0 {
            total_entropy -= probability_of_one * probability_of_one.log2()
                + (1.0 - probability_of_one) * (1.0 - probability_of_one).log2();
        }
    }

    total_entropy / bits_per_hash as f64 // Normalize to [0, 1]
}

/// Calculate collision rate (duplicate hash values) - generic
fn collision_rate<T>(hash_matrix: &[Vec<T>]) -> f64
where
    T: Copy + std::hash::Hash + Eq,
{
    use std::collections::HashSet;

    if hash_matrix.is_empty() || hash_matrix[0].is_empty() {
        return 0.0;
    }

    let mut unique_hash_values = HashSet::new();
    let mut total_hash_count = 0;

    for document_hashes in hash_matrix {
        for &hash_value in document_hashes {
            unique_hash_values.insert(hash_value);
            total_hash_count += 1;
        }
    }

    1.0 - (unique_hash_values.len() as f64 / total_hash_count as f64)
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

fn configure_bench() -> Criterion<HashesWallTime> {
    // Align with bench_similarities defaults
    Criterion::default()
        .with_measurement(HashesWallTime::default())
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(30))
}

fn bench_fingerprints(c: &mut Criterion<HashesWallTime>) {
    // Load dataset using unified loader
    let tape_bytes = load_dataset();
    let tape = tape_bytes
        .as_chars()
        .expect("Dataset must be valid UTF-8 for fingerprinting");

    let batch_size = env::var("STRINGWARS_BATCH")
        .unwrap_or_else(|_| "1024".to_string())
        .parse::<usize>()
        .unwrap_or_else(|e| {
            panic!(
                "STRINGWARS_BATCH must be a valid number for fingerprinting benchmarks: {}",
                e
            )
        });

    let ndim = env::var("STRINGWARS_NDIM")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256);

    if tape.is_empty() {
        panic!("Dataset must contain at least one token for fingerprinting.");
    }

    if batch_size == 0 {
        panic!("STRINGWARS_BATCH must be greater than zero for fingerprinting benchmarks.");
    }

    if ndim == 0 {
        panic!("STRINGWARS_NDIM must be greater than zero for fingerprinting benchmarks.");
    }

    // Log benchmark-specific configuration
    println!("Benchmark configuration:");
    println!("- Batch size (for StringZilla): {}", batch_size);
    println!("- Fingerprint dimensions: {}", ndim);
    println!("- N-gram widths: {:?} bytes", NGRAM_WIDTHS);
    println!(
        "- Hashes per width: {} (total NDIM: {})",
        ndim / NGRAM_WIDTHS.len(),
        ndim
    );

    // Create single BytesTape and populate it with all units
    let mut units_tape: BytesTape<u64, UnifiedAlloc> = BytesTape::new_in(UnifiedAlloc);
    units_tape
        .extend(tape.iter().map(|s| s.as_bytes()))
        .unwrap_or_else(|e| panic!("Failed to extend BytesTape for fingerprinting: {}", e));

    // Create both byte and char views from single tape (zero-copy casting)
    let bytes_view = units_tape.view();
    let chars_view: CharsTapeView<u64> = units_tape
        .view()
        .try_into()
        .unwrap_or_else(|e| panic!("Failed to convert BytesTapeView to CharsTapeView: {}", e));

    // Calculate total bytes for throughput reporting
    let total_documents = units_tape.len();
    let total_bytes: usize = tape.iter().map(|s| s.len()).sum();
    let avg_token_bytes: u64 = total_bytes as u64 / total_documents as u64;
    let per_batch_bytes = (batch_size as u64) * avg_token_bytes;

    // Count hash operations as: NDIM hashes per token times average token length (approx)
    let per_batch_hash_ops: u64 = (batch_size as u64)
        .saturating_mul(ndim as u64)
        .saturating_mul(avg_token_bytes as u64);

    // Configure formatter to derive GB/s from hashes/s using average bytes-per-hash-op
    if per_batch_hash_ops > 0 {
        let bytes_per_hash_op = (per_batch_bytes as f64) / (per_batch_hash_ops as f64);
        set_fingerprints_bytes_per_hash(bytes_per_hash_op);
    }

    // Pre-allocated matrices for quality analysis: N_docs × N_dims (algorithm-specific types)
    let mut serial_matrix = vec![vec![0u64; ndim]; total_documents]; // u64 for serial MinHash
    let mut pc_matrix = vec![vec![0u64; ndim]; total_documents]; // u64 for probabilistic_collections
    let mut sz_matrix = vec![vec![0u32; ndim]; total_documents]; // u32 for StringZilla fingerprints

    // Track which documents have been processed for each implementation
    let mut serial_document_index = 0usize;
    let mut pc_document_index = 0usize;
    let mut sz_document_index = 0usize;

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
    let tokens_count = total_documents;

    // Pre-allocate result buffers for StringZilla compute_into in unified memory (reused across iterations)
    let mut min_hashes = UnifiedVec::<u32>::with_capacity_in(batch_size * ndim, UnifiedAlloc);
    min_hashes.resize(batch_size * ndim, 0);
    let mut min_counts = UnifiedVec::<u32>::with_capacity_in(batch_size * ndim, UnifiedAlloc);
    min_counts.resize(batch_size * ndim, 0);

    // StringZilla: 1x CPU
    if should_run("fingerprinting/stringzillas::Fingerprints(1xCPU)") {
        g.throughput(Throughput::Elements(per_batch_hash_ops));
        g.bench_function("stringzillas::Fingerprints(1xCPU)", |b| {
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
    }

    // StringZilla: Nx CPU
    if should_run(&format!(
        "fingerprinting/stringzillas::Fingerprints({}xCPU)",
        num_cores
    )) {
        g.throughput(Throughput::Elements(per_batch_hash_ops));
        g.bench_function(
            &format!("stringzillas::Fingerprints({}xCPU)", num_cores),
            |b| {
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

                    // Fill quality matrix - direct copy for StringZilla (u32 values)
                    for document_idx in 0..actual {
                        if sz_document_index < total_documents {
                            let start = document_idx * ndim;
                            let end = start + ndim;
                            for (dim_idx, &hash_value) in min_hashes[start..end].iter().enumerate()
                            {
                                sz_matrix[sz_document_index][dim_idx] = hash_value;
                            }
                            sz_document_index += 1;
                        }
                    }

                    std::hint::black_box((
                        &min_hashes[..actual * ndim],
                        &min_counts[..actual * ndim],
                    ));
                })
            },
        );
    }

    // StringZilla: 1x GPU (if available)
    if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_sz_gpu.as_ref()) {
        if should_run("fingerprinting/stringzillas::Fingerprints(1xGPU)") {
            g.throughput(Throughput::Elements(per_batch_hash_ops));
            g.bench_function("stringzillas::Fingerprints(1xGPU)", |b| {
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
                    std::hint::black_box((
                        &min_hashes[..actual * ndim],
                        &min_counts[..actual * ndim],
                    ));
                })
            });
        }
    }

    // Create separate MinHash instances for each n-gram width
    let hashes_per_width = ndim / NGRAM_WIDTHS.len();
    let minhashers: Vec<MinHash<ByteGrams, _>> = (0..NGRAM_WIDTHS.len())
        .map(|_| MinHash::new(hashes_per_width))
        .collect();

    // Pre-allocate output buffer outside benchmark loop (reused across iterations)
    let mut out = Vec::with_capacity(batch_size);
    let mut combined_signature = Vec::with_capacity(ndim);

    if should_run("fingerprinting/pc::MinHash<ByteGrams>") {
        g.throughput(Throughput::Elements(per_batch_hash_ops));
        g.bench_function("pc::MinHash<ByteGrams>", |b| {
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

                        // Fill quality matrix - direct memcpy
                        if pc_document_index < total_documents && combined_signature.len() == ndim {
                            pc_matrix[pc_document_index].copy_from_slice(&combined_signature);
                            pc_document_index += 1;
                        }
                    }
                }

                std::hint::black_box(&out);
            })
        });
    }

    // Serial MinHash baseline implementing correct independent hash functions
    // This addresses the flaw in probabilistic_collections where hash function index is ignored
    if should_run("fingerprinting/serial::MinHash<ByteGrams>") {
        g.throughput(Throughput::Elements(per_batch_hash_ops));
        g.bench_function("serial::MinHash<ByteGrams>", |b| {
            // Pre-construct hash parameters for independent universal hash functions
            // Each hash function uses: hash_i(x) = (a_i * hash(x) + b_i) mod mersenne_prime

            const MERSENNE_PRIME: u64 = (1u64 << 61) - 1; // Large prime for universal hashing

            // Generate independent hash function parameters
            let hash_params: Vec<(u64, u64)> = (0..ndim)
                .map(|i| {
                    let a = 2 * i as u64 + 1; // Odd number for universal hashing
                    let b = i as u64;
                    (a, b)
                })
                .collect();

            let mut start_idx = 0;
            b.iter(|| {
                let (batch_bytes_view, _batch_chars_view, actual) = tokens_tape_slice(
                    &bytes_view,
                    &chars_view,
                    &mut start_idx,
                    batch_size,
                    tokens_count,
                );

                let mut out = Vec::with_capacity(actual);

                // Process each line with serial MinHash
                for i in 0..batch_bytes_view.len() {
                    let line_bytes: &[u8] = &batch_bytes_view[i];
                    if !line_bytes.is_empty() {
                        // Initialize minimum hash values for each hash function
                        let mut min_hashes = vec![u64::MAX; ndim];

                        // Process all n-gram widths
                        for &width in &NGRAM_WIDTHS {
                            if line_bytes.len() >= width {
                                // Generate n-grams of this width
                                for window in line_bytes.windows(width) {
                                    // Compute base hash of the n-gram
                                    let mut hasher = DefaultHasher::new();
                                    window.hash(&mut hasher);
                                    let base_hash = hasher.finish();

                                    // Apply each independent hash function
                                    for (hash_idx, &(a, b)) in hash_params.iter().enumerate() {
                                        let independent_hash =
                                            (a.wrapping_mul(base_hash).wrapping_add(b))
                                                % MERSENNE_PRIME;
                                        min_hashes[hash_idx] =
                                            min_hashes[hash_idx].min(independent_hash);
                                    }
                                }
                            }
                        }

                        out.push(min_hashes.clone());

                        // Fill quality matrix - direct memcpy
                        if serial_document_index < total_documents && min_hashes.len() == ndim {
                            serial_matrix[serial_document_index].copy_from_slice(&min_hashes);
                            serial_document_index += 1;
                        }
                    }
                }

                std::hint::black_box(&out);
            })
        });
    }

    g.finish();

    // Compute and display quality metrics
    println!("\n=== Hash Quality Analysis ===");
    println!(
        "Documents processed: PC={}, Serial={}, StringZilla={}",
        pc_document_index, serial_document_index, sz_document_index
    );

    if pc_document_index > 0 {
        let pc_slice = &pc_matrix[..pc_document_index];
        println!("\npc::MinHash<ByteGrams>:");
        println!("  Bit Entropy:  {:.4}", bit_entropy(pc_slice));
        println!("  Collision:    {:.4}%", collision_rate(pc_slice) * 100.0);
    }

    if serial_document_index > 0 {
        let serial_slice = &serial_matrix[..serial_document_index];
        println!("\nserial::MinHash<ByteGrams>:");
        println!("  Bit Entropy:  {:.4}", bit_entropy(serial_slice));
        println!(
            "  Collision:    {:.4}%",
            collision_rate(serial_slice) * 100.0
        );
    }

    if sz_document_index > 0 {
        let sz_slice = &sz_matrix[..sz_document_index];
        println!("\nszs::Fingerprints:");
        println!("  Bit Entropy:  {:.4}", bit_entropy(sz_slice));
        println!("  Collision:    {:.4}%", collision_rate(sz_slice) * 100.0);
    }
}

fn main() {
    log_stringzilla_metadata();
    println!("Text Fingerprinting Benchmarks");
    println!("- szs::Fingerprints: CPU/GPU fingerprints with multi-width byte n-grams");
    println!("- pc::MinHash<ByteGrams>: MinHash with ByteGrams iterator");

    let mut criterion = configure_bench();
    bench_fingerprints(&mut criterion);
    criterion.final_summary();
}
