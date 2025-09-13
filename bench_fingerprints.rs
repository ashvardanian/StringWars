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
"#]

use core::convert::TryInto;
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};

use criterion::{Criterion, Throughput};
use fork_union::count_logical_cores;

use probabilistic_collections::similarity::{ByteGrams, MinHash};
use stringtape::{BytesTape, BytesTapeView, CharsTapeView};
use stringzilla::sz::dynamic_dispatch as sz_dynamic_dispatch;
use stringzilla::szs::{capabilities as szs_capabilities, version as szs_version};
use stringzilla::szs::{AnyBytesTape, DeviceScope, Fingerprints, UnifiedAlloc, UnifiedVec};

// Fixed n-gram widths for multi-scale fingerprinting
const NGRAM_WIDTHS: [usize; 4] = [5, 9, 17, 33];

/// Calculate variance of hash values across dimensions (generic, no allocations)
fn calculate_variance<T>(hash_matrix: &[Vec<T>]) -> f64
where
    T: Copy + ToF64,
{
    if hash_matrix.is_empty() || hash_matrix[0].is_empty() {
        return 0.0;
    }

    let num_documents = hash_matrix.len();
    let num_dimensions = hash_matrix[0].len();
    let mut total_variance = 0.0;

    // Calculate variance for each dimension without allocations
    for dimension_index in 0..num_dimensions {
        // Calculate mean for this dimension
        let dimension_mean = hash_matrix
            .iter()
            .map(|document_hashes| document_hashes[dimension_index].to_f64())
            .sum::<f64>()
            / num_documents as f64;

        // Calculate variance for this dimension
        let dimension_variance = hash_matrix
            .iter()
            .map(|document_hashes| {
                let hash_value = document_hashes[dimension_index].to_f64();
                (hash_value - dimension_mean).powi(2)
            })
            .sum::<f64>()
            / num_documents as f64;

        total_variance += dimension_variance;
    }

    // Return average variance across all dimensions
    total_variance / num_dimensions as f64
}

/// Trait for converting hash values to f64
trait ToF64 {
    fn to_f64(self) -> f64;
}

impl ToF64 for u32 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToF64 for u64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// Calculate bit entropy (how well distributed the bits are) - generic
fn calculate_bit_entropy<T>(hash_matrix: &[Vec<T>]) -> f64
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
fn calculate_collision_rate<T>(hash_matrix: &[Vec<T>]) -> f64
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

/// Calculate inter-dimension correlation (independence measure) - generic
fn calculate_correlation<T>(hash_matrix: &[Vec<T>]) -> f64
where
    T: Copy + ToF64,
{
    if hash_matrix.is_empty() || hash_matrix[0].len() < 2 {
        return 0.0;
    }

    let num_documents = hash_matrix.len();
    let num_dimensions = hash_matrix[0].len();
    let mut correlation_sum = 0.0;
    let mut correlation_count = 0;

    // Calculate correlation between adjacent hash functions (no allocations)
    for dimension_index in 0..num_dimensions.min(64) {
        // Limit to first 64 dimensions for performance
        if dimension_index + 1 >= num_dimensions {
            break;
        }

        // Calculate means for both dimensions
        let dimension_x_mean = hash_matrix
            .iter()
            .map(|document_hashes| document_hashes[dimension_index].to_f64())
            .sum::<f64>()
            / num_documents as f64;
        let dimension_y_mean = hash_matrix
            .iter()
            .map(|document_hashes| document_hashes[dimension_index + 1].to_f64())
            .sum::<f64>()
            / num_documents as f64;

        // Calculate covariance and variances
        let mut covariance = 0.0;
        let mut variance_x = 0.0;
        let mut variance_y = 0.0;

        for document_hashes in hash_matrix {
            let x_value = document_hashes[dimension_index].to_f64();
            let y_value = document_hashes[dimension_index + 1].to_f64();
            let x_deviation = x_value - dimension_x_mean;
            let y_deviation = y_value - dimension_y_mean;

            covariance += x_deviation * y_deviation;
            variance_x += x_deviation * x_deviation;
            variance_y += y_deviation * y_deviation;
        }

        if variance_x > 0.0 && variance_y > 0.0 {
            let correlation = (covariance / (variance_x * variance_y).sqrt()).abs();
            correlation_sum += correlation;
            correlation_count += 1;
        }
    }

    if correlation_count == 0 {
        0.0
    } else {
        correlation_sum / correlation_count as f64
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

    // Pre-allocated matrices for quality analysis: N_docs × N_dims (algorithm-specific types)
    let total_documents = truncated_units.len();
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

            // Fill quality matrix - direct copy for StringZilla (u32 values)
            for document_idx in 0..actual {
                if sz_document_index < total_documents {
                    let start = document_idx * ndim;
                    let end = start + ndim;
                    for (dim_idx, &hash_value) in min_hashes[start..end].iter().enumerate() {
                        sz_matrix[sz_document_index][dim_idx] = hash_value;
                    }
                    sz_document_index += 1;
                }
            }

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

    // Serial MinHash baseline implementing correct independent hash functions
    // This addresses the flaw in probabilistic_collections where hash function index is ignored
    g.throughput(Throughput::Bytes(per_batch_bytes));
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

    g.finish();

    // Compute and display quality metrics
    println!("\n=== Hash Quality Analysis ===");
    println!(
        "Documents processed: PC={}, Serial={}, StringZilla={}",
        pc_document_index, serial_document_index, sz_document_index
    );

    if pc_document_index > 0 {
        let pc_slice = &pc_matrix[..pc_document_index];
        println!("\nProbabilistic Collections MinHash (u64 double-hashing):");
        println!("  Variance:     {:.2e}", calculate_variance(pc_slice));
        println!("  Bit Entropy:  {:.4}", calculate_bit_entropy(pc_slice));
        println!(
            "  Collision:    {:.4}%",
            calculate_collision_rate(pc_slice) * 100.0
        );
        println!("  Correlation:  {:.4}", calculate_correlation(pc_slice));
    }

    if serial_document_index > 0 {
        let serial_slice = &serial_matrix[..serial_document_index];
        println!("\nSerial MinHash (u64 universal hashing):");
        println!("  Variance:     {:.2e}", calculate_variance(serial_slice));
        println!("  Bit Entropy:  {:.4}", calculate_bit_entropy(serial_slice));
        println!(
            "  Collision:    {:.4}%",
            calculate_collision_rate(serial_slice) * 100.0
        );
        println!("  Correlation:  {:.4}", calculate_correlation(serial_slice));
    }

    if sz_document_index > 0 {
        let sz_slice = &sz_matrix[..sz_document_index];
        println!("\nStringZilla Multi-core (u32 rolling hashes):");
        println!("  Variance:     {:.2e}", calculate_variance(sz_slice));
        println!("  Bit Entropy:  {:.4}", calculate_bit_entropy(sz_slice));
        println!(
            "  Collision:    {:.4}%",
            calculate_collision_rate(sz_slice) * 100.0
        );
        println!("  Correlation:  {:.4}", calculate_correlation(sz_slice));
    }
}

fn main() {
    log_stringzilla_metadata();
    println!("Text Fingerprinting Benchmarks");
    println!("- szs::Fingerprints: CPU/GPU fingerprints with multi-width byte n-grams");
    println!("- probabilistic_collections::MinHash<ByteGrams>: MinHash with ByteGrams iterator");

    let mut criterion = configure_bench();
    bench_fingerprints(&mut criterion);
    criterion.final_summary();
}
