use criterion::measurement::{Measurement, ValueFormatter};
use criterion::Throughput;
#[cfg(feature = "bench_fingerprints")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// Simple SI scaling helper
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
