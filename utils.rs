use std::borrow::Cow;
use std::env;
use std::fs;
use stringtape::{BytesCowsAuto, CharsCowsAuto};

/// Forces the allocator to release memory back to the OS.
/// This is particularly useful after dropping large allocations in benchmarks.
#[cfg(target_os = "linux")]
#[inline]
pub fn reclaim_memory() {
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(not(target_os = "linux"))]
#[inline]
pub fn reclaim_memory() {
    // No-op on non-Linux platforms
}

/// Loads binary data from the file specified by the `STRINGWARS_DATASET` environment variable.
/// Uses StringTape to avoid allocating separate byte vectors for each token.
/// Returns BytesCowsAuto for memory-efficient byte slice storage.
/// Can be cast to CharsCowsAuto for UTF-8 string benchmarks (StringTape 2.2+).
/// Supports `STRINGWARS_MAX_TOKENS` to limit the number of tokens loaded.
/// Logs dataset statistics to stderr.
pub fn load_dataset() -> BytesCowsAuto<'static> {
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_TOKENS").unwrap_or_else(|_| "lines".to_string());
    let max_tokens = env::var("STRINGWARS_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    if let Some(max) = max_tokens {
        eprintln!("STRINGWARS_MAX_TOKENS: limiting to {} tokens", max);
    }

    // Read and leak the content to get 'static lifetime
    let content = fs::read(&dataset_path).expect("Could not read dataset");
    let content_static: &'static [u8] = Box::leak(content.into_boxed_slice());
    let content_bytes = Cow::Borrowed(content_static);

    // Build BytesCowsAuto directly from iterator - it will own references to the leaked bytes
    let tape = match mode.as_str() {
        "lines" => {
            let iter = content_static
                .split(|&b| b == b'\n')
                .filter(|s| !s.is_empty());
            if let Some(max) = max_tokens {
                BytesCowsAuto::from_iter_and_data(iter.take(max), content_bytes)
            } else {
                BytesCowsAuto::from_iter_and_data(iter, content_bytes)
            }
        }
        "words" => {
            let iter = content_static
                .split(|&b| b == b' ' || b == b'\n')
                .filter(|s| !s.is_empty());
            if let Some(max) = max_tokens {
                BytesCowsAuto::from_iter_and_data(iter.take(max), content_bytes)
            } else {
                BytesCowsAuto::from_iter_and_data(iter, content_bytes)
            }
        }
        "file" => {
            let iter = std::iter::once(content_static);
            BytesCowsAuto::from_iter_and_data(iter, content_bytes)
        }
        other => panic!(
            "Unknown STRINGWARS_TOKENS: {}. Use 'lines', 'words', or 'file'.",
            other
        ),
    };

    let tape = tape.expect("Failed to create BytesTape");

    // Log dataset statistics
    let count = tape.len();
    let total_bytes: usize = tape.iter().map(|s| s.len()).sum();
    let mean_len = if count > 0 {
        total_bytes as f64 / count as f64
    } else {
        0.0
    };
    let variance: f64 = tape
        .iter()
        .map(|s| {
            let diff = s.len() as f64 - mean_len;
            diff * diff
        })
        .sum::<f64>()
        / count as f64;
    let std_dev = variance.sqrt();

    eprintln!(
        "Dataset: {} tokens, {} bytes ({:.2} GB), mean {:.1} ± {:.1} bytes",
        format_number(count as u64),
        format_number(total_bytes as u64),
        total_bytes as f64 / 1e9,
        mean_len,
        std_dev
    );

    tape
}

/// Format large numbers with thousand separators for readability
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

// Perf profiling utilities
#[cfg(target_os = "linux")]
use perf_event::{
    events::CacheOp, events::CacheResult, events::Hardware, events::WhichCache, Builder, Counter,
};

#[cfg(any(
    feature = "bench_similarities",
    feature = "bench_fingerprints",
    feature = "bench_sequence"
))]
use criterion::measurement::{Measurement, ValueFormatter};
#[cfg(any(
    feature = "bench_similarities",
    feature = "bench_fingerprints",
    feature = "bench_sequence"
))]
use criterion::Throughput;
#[cfg(any(
    feature = "bench_similarities",
    feature = "bench_fingerprints",
    feature = "bench_sequence"
))]
use std::time::Instant;

/// Filter helper function to check if a benchmark should run based on STRINGWARS_FILTER env var
#[allow(dead_code)]
pub fn should_run(name: &str) -> bool {
    use std::sync::Once;
    static FILTER_INIT: Once = Once::new();

    if let Ok(filter) = env::var("STRINGWARS_FILTER") {
        FILTER_INIT.call_once(|| {
            eprintln!("STRINGWARS_FILTER active: '{}'", filter);
        });

        if let Ok(re) = regex::Regex::new(&filter) {
            let matches = re.is_match(name);
            if !matches {
                eprintln!("  Skipping: {}", name);
            }
            matches
        } else {
            // Fallback to substring match if regex is invalid
            eprintln!(
                "Warning: Invalid regex pattern '{}', falling back to substring match",
                filter
            );
            name.contains(&filter)
        }
    } else {
        true
    }
}

#[cfg(feature = "bench_fingerprints")]
use std::sync::atomic::{AtomicU64, Ordering};

// Simple SI scaling helper
#[cfg(any(
    feature = "bench_similarities",
    feature = "bench_fingerprints",
    feature = "bench_sequence"
))]
fn scale_si(mut v: f64) -> (f64, &'static str) {
    if v >= 1_000_000_000.0 {
        v /= 1_000_000_000.0;
        (v, "G")
    } else if v >= 1_000_000.0 {
        v /= 1_000_000.0;
        (v, "M")
    } else if v >= 1_000.0 {
        v /= 1_000.0;
        (v, "k")
    } else {
        (v, "")
    }
}

#[cfg(any(
    feature = "bench_similarities",
    feature = "bench_fingerprints",
    feature = "bench_sequence"
))]
fn format_seconds(value: f64) -> String {
    // value is seconds
    if value < 1e-6 {
        format!("{:.2} ns", value * 1e9)
    } else if value < 1e-3 {
        format!("{:.2} µs", value * 1e6)
    } else if value < 1.0 {
        format!("{:.2} ms", value * 1e3)
    } else {
        format!("{:.2} s", value)
    }
}

#[cfg(feature = "bench_similarities")]
pub struct CupsFormatter;
#[cfg(feature = "bench_similarities")]
impl ValueFormatter for CupsFormatter {
    fn format_value(&self, value: f64) -> String {
        // Format raw times
        format_seconds(value)
    }

    fn format_throughput(&self, throughput: &Throughput, secs: f64) -> String {
        match throughput {
            Throughput::Bytes(bytes) | Throughput::BytesDecimal(bytes) => {
                let rate = (*bytes as f64) / secs; // bytes/s
                let (v, unit) = if rate >= 1e9 {
                    (rate / 1e9, "GB/s")
                } else if rate >= 1e6 {
                    (rate / 1e6, "MB/s")
                } else if rate >= 1e3 {
                    (rate / 1e3, "kB/s")
                } else {
                    (rate, "B/s")
                };
                format!("{:.2} {}", v, unit)
            }
            Throughput::Elements(elems) => {
                let cups = (*elems as f64) / secs; // elements/s
                let (v, p) = scale_si(cups);
                let unit = match p {
                    "G" => "GCUPS",
                    "M" => "MCUPS",
                    "k" => "kCUPS",
                    _ => "CUPS",
                };
                format!("{:.2} {}", v, unit)
            }
            Throughput::Bits(bits) => {
                let rate = (*bits as f64) / secs; // bits/s
                let (v, p) = scale_si(rate);
                let unit = match p {
                    "G" => "Gb/s",
                    "M" => "Mb/s",
                    "k" => "kb/s",
                    _ => "b/s",
                };
                format!("{:.2} {}", v, unit)
            }
        }
    }

    fn scale_values(&self, _typical_value: f64, _values: &mut [f64]) -> &'static str {
        "s"
    }
    fn scale_throughputs(
        &self,
        _typical_value: f64,
        _throughput: &Throughput,
        _values: &mut [f64],
    ) -> &'static str {
        "s"
    }
    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "s"
    }
}

#[cfg(feature = "bench_fingerprints")]
pub struct HashesFormatter;
#[cfg(feature = "bench_fingerprints")]
impl ValueFormatter for HashesFormatter {
    fn format_value(&self, value: f64) -> String {
        format_seconds(value)
    }

    fn format_throughput(&self, throughput: &Throughput, secs: f64) -> String {
        match throughput {
            Throughput::Bytes(bytes) | Throughput::BytesDecimal(bytes) => {
                let bytes_per_sec = (*bytes as f64) / secs;
                let (bv, bunit) = if bytes_per_sec >= 1e9 {
                    (bytes_per_sec / 1e9, "GB/s")
                } else if bytes_per_sec >= 1e6 {
                    (bytes_per_sec / 1e6, "MB/s")
                } else if bytes_per_sec >= 1e3 {
                    (bytes_per_sec / 1e3, "kB/s")
                } else {
                    (bytes_per_sec, "B/s")
                };
                // If a bytes-per-hash ratio is set, also render hashes/s
                let bph = get_bytes_per_hash();
                if bph > 0.0 {
                    let hashes_per_sec = bytes_per_sec / bph;
                    let (hv, hp) = scale_si(hashes_per_sec);
                    let hunit = match hp {
                        "G" => "G hashes/s",
                        "M" => "M hashes/s",
                        "k" => "k hashes/s",
                        _ => "hashes/s",
                    };
                    format!("{:.2} {} | {:.2} {}", bv, bunit, hv, hunit)
                } else {
                    format!("{:.2} {}", bv, bunit)
                }
            }
            Throughput::Elements(elems) => {
                let hashes_per_sec = (*elems as f64) / secs;
                let (hv, hp) = scale_si(hashes_per_sec);
                let hunit = match hp {
                    "G" => "G hashes/s",
                    "M" => "M hashes/s",
                    "k" => "k hashes/s",
                    _ => "hashes/s",
                };
                // Also compute bytes/s if ratio present
                let bph = get_bytes_per_hash();
                if bph > 0.0 {
                    let bytes_per_sec = hashes_per_sec * bph;
                    let (bv, bunit) = if bytes_per_sec >= 1e9 {
                        (bytes_per_sec / 1e9, "GB/s")
                    } else if bytes_per_sec >= 1e6 {
                        (bytes_per_sec / 1e6, "MB/s")
                    } else if bytes_per_sec >= 1e3 {
                        (bytes_per_sec / 1e3, "kB/s")
                    } else {
                        (bytes_per_sec, "B/s")
                    };
                    format!("{:.2} {} | {:.2} {}", hv, hunit, bv, bunit)
                } else {
                    format!("{:.2} {}", hv, hunit)
                }
            }
            Throughput::Bits(bits) => {
                let rate = (*bits as f64) / secs; // bits/s
                let (v, p) = scale_si(rate);
                let unit = match p {
                    "G" => "Gb/s",
                    "M" => "Mb/s",
                    "k" => "kb/s",
                    _ => "b/s",
                };
                format!("{:.2} {}", v, unit)
            }
        }
    }

    fn scale_values(&self, _typical_value: f64, _values: &mut [f64]) -> &'static str {
        "s"
    }
    fn scale_throughputs(
        &self,
        _typical_value: f64,
        _throughput: &Throughput,
        _values: &mut [f64],
    ) -> &'static str {
        "s"
    }
    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "s"
    }
}

// Measurement wrappers that mirror WallTime but override formatting.

#[cfg(feature = "bench_similarities")]
#[derive(Clone, Default)]
pub struct CupsWallTime;

#[cfg(feature = "bench_similarities")]
impl Measurement for CupsWallTime {
    type Intermediate = Instant;
    type Value = f64; // seconds

    fn start(&self) -> Self::Intermediate {
        Instant::now()
    }

    fn end(&self, i: Self::Intermediate) -> Self::Value {
        i.elapsed().as_secs_f64()
    }

    fn add(&self, v: &Self::Value, a: &Self::Value) -> Self::Value {
        v + a
    }

    fn zero(&self) -> Self::Value {
        0.0
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &CupsFormatter
    }
}

#[cfg(feature = "bench_fingerprints")]
#[derive(Clone, Default)]
pub struct HashesWallTime;

#[cfg(feature = "bench_fingerprints")]
impl Measurement for HashesWallTime {
    type Intermediate = Instant;
    type Value = f64; // seconds

    fn start(&self) -> Self::Intermediate {
        Instant::now()
    }

    fn end(&self, i: Self::Intermediate) -> Self::Value {
        i.elapsed().as_secs_f64()
    }

    fn add(&self, v: &Self::Value, a: &Self::Value) -> Self::Value {
        v + a
    }

    fn zero(&self) -> Self::Value {
        0.0
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &HashesFormatter
    }
}

// Global ratio to let the formatter print both hashes/s and bytes/s
#[cfg(feature = "bench_fingerprints")]
static FINGERPRINTS_BYTES_PER_HASH_BITS: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "bench_fingerprints")]
pub fn set_fingerprints_bytes_per_hash(v: f64) {
    FINGERPRINTS_BYTES_PER_HASH_BITS.store(v.to_bits(), Ordering::Relaxed);
}

#[cfg(feature = "bench_fingerprints")]
fn get_bytes_per_hash() -> f64 {
    let bits = FINGERPRINTS_BYTES_PER_HASH_BITS.load(Ordering::Relaxed);
    f64::from_bits(bits)
}

// Comparisons/sec formatter: k/M/G cmp/s
#[cfg(feature = "bench_sequence")]
pub struct ComparisonsFormatter;

#[cfg(feature = "bench_sequence")]
impl ValueFormatter for ComparisonsFormatter {
    fn format_value(&self, value: f64) -> String {
        format_seconds(value)
    }

    fn format_throughput(&self, throughput: &Throughput, secs: f64) -> String {
        match throughput {
            Throughput::Bytes(bytes_per_iter) | Throughput::BytesDecimal(bytes_per_iter) => {
                let rate = (*bytes_per_iter as f64) / secs; // bytes/s
                let (v, unit) = if rate >= 1e9 {
                    (rate / 1e9, "GB/s")
                } else if rate >= 1e6 {
                    (rate / 1e6, "MB/s")
                } else if rate >= 1e3 {
                    (rate / 1e3, "kB/s")
                } else {
                    (rate, "B/s")
                };
                format!("{:.2} {}", v, unit)
            }
            Throughput::Elements(elems_per_iter) => {
                let cmps_per_sec = (*elems_per_iter as f64) / secs;
                let (v, p) = scale_si(cmps_per_sec);
                let unit = match p {
                    "G" => "G cmp/s",
                    "M" => "M cmp/s",
                    "k" => "k cmp/s",
                    _ => "cmp/s",
                };
                format!("{:.2} {}", v, unit)
            }
            Throughput::Bits(bits) => {
                let rate = (*bits as f64) / secs;
                let (v, p) = scale_si(rate);
                let unit = match p {
                    "G" => "Gb/s",
                    "M" => "Mb/s",
                    "k" => "kb/s",
                    _ => "b/s",
                };
                format!("{:.2} {}", v, unit)
            }
        }
    }

    fn scale_values(&self, _typical_value: f64, _values: &mut [f64]) -> &'static str {
        "s"
    }
    fn scale_throughputs(
        &self,
        _typical_value: f64,
        _throughput: &Throughput,
        _values: &mut [f64],
    ) -> &'static str {
        "s"
    }
    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "s"
    }
}

#[cfg(feature = "bench_sequence")]
#[derive(Clone, Default)]
pub struct ComparisonsWallTime;

#[cfg(feature = "bench_sequence")]
impl Measurement for ComparisonsWallTime {
    type Intermediate = Instant;
    type Value = f64; // seconds

    fn start(&self) -> Self::Intermediate {
        Instant::now()
    }
    fn end(&self, i: Self::Intermediate) -> Self::Value {
        i.elapsed().as_secs_f64()
    }
    fn add(&self, v: &Self::Value, a: &Self::Value) -> Self::Value {
        v + a
    }
    fn zero(&self) -> Self::Value {
        0.0
    }
    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value
    }
    fn formatter(&self) -> &dyn ValueFormatter {
        &ComparisonsFormatter
    }
}

// ============================================================================
// Perf profiling utilities for comprehensive hardware counter collection
// ============================================================================

/// RAII guard that profiles a code section using perf events on Linux.
/// On non-Linux platforms, this is a zero-cost abstraction.
#[allow(dead_code)]
#[cfg(target_os = "linux")]
pub struct PerfSection {
    name: &'static str,
    // Core performance
    cycles: Option<Counter>,
    instructions: Option<Counter>,
    // Stall analysis (ARM-specific)
    stall_frontend: Option<Counter>,
    stall_backend: Option<Counter>,
    stall_backend_mem: Option<Counter>,
    // Memory hierarchy
    l1d_cache_misses: Option<Counter>,
    l1d_cache_accesses: Option<Counter>,
    dtlb_misses: Option<Counter>,
    l2_cache_misses: Option<Counter>,
}

#[cfg(target_os = "linux")]
impl PerfSection {
    /// Creates a new perf section with comprehensive counter collection.
    /// Collects: cycles, instructions, stalls (frontend/backend), cache, and TLB metrics.
    pub fn new(name: &'static str) -> Self {
        // Core performance counters
        let cycles = Self::build_counter(Hardware::CPU_CYCLES);
        let instructions = Self::build_counter(Hardware::INSTRUCTIONS);

        // Stall counters (standard hardware events)
        let stall_frontend = Self::build_counter(Hardware::STALLED_CYCLES_FRONTEND);
        let stall_backend = Self::build_counter(Hardware::STALLED_CYCLES_BACKEND);

        // L1 data cache
        let l1d_cache_misses =
            Self::build_cache_counter(WhichCache::L1D, CacheOp::READ, CacheResult::MISS);
        let l1d_cache_accesses =
            Self::build_cache_counter(WhichCache::L1D, CacheOp::READ, CacheResult::ACCESS);

        // DTLB (data TLB)
        let dtlb_misses =
            Self::build_cache_counter(WhichCache::DTLB, CacheOp::READ, CacheResult::MISS);

        // L2/LLC cache
        let l2_cache_misses =
            Self::build_cache_counter(WhichCache::LL, CacheOp::READ, CacheResult::MISS);

        Self {
            name,
            cycles,
            instructions,
            stall_frontend,
            stall_backend,
            stall_backend_mem: None, // Not available via standard events
            l1d_cache_misses,
            l1d_cache_accesses,
            dtlb_misses,
            l2_cache_misses,
        }
    }

    /// Creates a minimal perf section that only tracks core performance (cycles + instructions).
    /// Use this for lower overhead when detailed metrics aren't needed.
    pub fn minimal(name: &'static str) -> Self {
        let cycles = Self::build_counter(Hardware::CPU_CYCLES);
        let instructions = Self::build_counter(Hardware::INSTRUCTIONS);

        Self {
            name,
            cycles,
            instructions,
            stall_frontend: None,
            stall_backend: None,
            stall_backend_mem: None,
            l1d_cache_misses: None,
            l1d_cache_accesses: None,
            dtlb_misses: None,
            l2_cache_misses: None,
        }
    }

    /// Helper to build and enable a hardware counter
    fn build_counter(kind: Hardware) -> Option<Counter> {
        Builder::new().kind(kind).build().ok().and_then(|mut c| {
            if c.enable().is_err() {
                eprintln!("Warning: Failed to enable counter {:?}", kind);
            }
            Some(c)
        })
    }

    /// Helper to build and enable a cache counter
    fn build_cache_counter(cache: WhichCache, op: CacheOp, result: CacheResult) -> Option<Counter> {
        use perf_event::events::Cache;
        Builder::new()
            .kind(Cache {
                which: cache,
                operation: op,
                result,
            })
            .build()
            .ok()
            .and_then(|mut c| {
                c.enable().ok()?;
                Some(c)
            })
    }

    /// Disable and read a counter
    fn read_counter(counter: &mut Option<Counter>) -> Option<u64> {
        counter.as_mut().and_then(|c| {
            c.disable().ok()?;
            c.read().ok()
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for PerfSection {
    fn drop(&mut self) {
        // Read all counters
        let cycles = Self::read_counter(&mut self.cycles);
        let instructions = Self::read_counter(&mut self.instructions);
        let stall_frontend = Self::read_counter(&mut self.stall_frontend);
        let stall_backend = Self::read_counter(&mut self.stall_backend);
        let l1d_cache_misses = Self::read_counter(&mut self.l1d_cache_misses);
        let l1d_cache_accesses = Self::read_counter(&mut self.l1d_cache_accesses);
        let dtlb_misses = Self::read_counter(&mut self.dtlb_misses);
        let l2_cache_misses = Self::read_counter(&mut self.l2_cache_misses);

        // Only print if we have any counters
        let has_counters = cycles.is_some()
            || instructions.is_some()
            || stall_frontend.is_some()
            || stall_backend.is_some()
            || l1d_cache_misses.is_some()
            || dtlb_misses.is_some()
            || l2_cache_misses.is_some();

        if !has_counters {
            return;
        }

        eprintln!("[perf] {}", self.name);

        // Core performance
        if let (Some(c), Some(i)) = (cycles, instructions) {
            let ipc = i as f64 / c as f64;
            eprintln!(
                "  cycles: {}, instructions: {}, IPC: {:.2}",
                format_perf_number(c),
                format_perf_number(i),
                ipc
            );
        } else if let Some(c) = cycles {
            eprintln!("  cycles: {}", format_perf_number(c));
        } else if let Some(i) = instructions {
            eprintln!("  instructions: {}", format_perf_number(i));
        }

        // Stalls
        if let (Some(sf), Some(c)) = (stall_frontend, cycles) {
            let pct = (sf as f64 / c as f64) * 100.0;
            eprintln!(
                "  frontend stalls: {} ({:.1}%)",
                format_perf_number(sf),
                pct
            );
        }
        if let (Some(sb), Some(c)) = (stall_backend, cycles) {
            let pct = (sb as f64 / c as f64) * 100.0;
            eprintln!("  backend stalls: {} ({:.1}%)", format_perf_number(sb), pct);
        }

        // Memory hierarchy
        if let Some(misses) = l1d_cache_misses {
            if let Some(accesses) = l1d_cache_accesses {
                let rate = (misses as f64 / accesses as f64) * 100.0;
                eprintln!(
                    "  L1D misses: {} ({:.1}%)",
                    format_perf_number(misses),
                    rate
                );
            } else {
                eprintln!("  L1D misses: {}", format_perf_number(misses));
            }
        }
        if let Some(misses) = l2_cache_misses {
            eprintln!("  L2 misses: {}", format_perf_number(misses));
        }
        if let Some(misses) = dtlb_misses {
            eprintln!("  DTLB misses: {}", format_perf_number(misses));
        }
    }
}

/// Format large numbers with thousand separators for readability
#[allow(dead_code)]
#[cfg(target_os = "linux")]
fn format_perf_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

// No-op implementation for non-Linux platforms
#[allow(dead_code)]
#[cfg(not(target_os = "linux"))]
pub struct PerfSection;

#[cfg(not(target_os = "linux"))]
impl PerfSection {
    #[inline(always)]
    pub fn new(_name: &'static str) -> Self {
        Self
    }

    #[inline(always)]
    pub fn minimal(_name: &'static str) -> Self {
        Self
    }
}
