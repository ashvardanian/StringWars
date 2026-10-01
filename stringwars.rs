//! Shared harness for the StringWars Rust suites, mirrored by `stringwars.py`.
//!
//! Every suite reads these once, at the top of `main`; defaults come from `stringwars.toml`.
//!
//! Variable                     Default                      Meaning
//! `STRINGWARS_SEED`            `42`                         Seed of the drawn inputs, an integer or `random`
//! `STRINGWARS_FILTER`          none                         Regex over row names, falling back to a substring match
//! `STRINGWARS_WARMUP`          `1s`                         Warm-up cap per row, like `1s` or `200ms`
//! `STRINGWARS_TIME_LIMIT`      `10s`                        Measurement cap per row, like `10s` or `500ms`
//! `STRINGWARS_BYTES`           per suite, see the manifest  Bytes read from the dataset, like `256MB`
//! `STRINGWARS_BATCH_PER_CORE`  per suite, see the manifest  Items per core, or per GPU multiprocessor
//! `STRINGWARS_THREADS`         all cores                    Cores for multi-core rows, `0` for all
//! `STRINGWARS_DIMS`            per suite, see the manifest  MinHash widths, like `128` or `64,128,256`
//! `STRINGWARS_DATASET`         per suite, see the manifest  Path to the textual dataset
//! `STRINGWARS_TOKENS`          per suite, see the manifest  `lines`, `words` or `file`
//! `STRINGWARS_UNIQUE`          `false`                      Drops repeated tokens
//! `STRINGWARS_MIN_SAMPLES`     `10`                         Fewest samples a row needs before it may converge
//! `STRINGWARS_TARGET_SPREAD`   `0.025`                      Relative half-width at which a row converges
//! `STRINGWARS_RESULTS_DIR`     none                         Directory for per-row NDJSON records
//! `STRINGWARS_COUNTERS`        `false`                      Adds cycles-per-byte and IPC columns, Linux only
//! `STRINGWARS_COLLISIONS`      `false`                      Adds collision rates to `hash`
#![allow(dead_code)] // Ten benches each use a subset of this harness.
use std::borrow::Cow;
use std::collections::HashSet;
use std::env;
use std::fmt;
use std::fs;
use std::hint::black_box;
use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::panic;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use stringtape::BytesCowsAuto;

/// Reads `name`, or `None` when it is unset or empty.
pub fn env_text(name: &str) -> Option<String> {
    env::var(name).ok().filter(|text| !text.is_empty())
}

/// Reads `name` through `parse`, or `fallback` when unset or empty; exits with status 1 if it does not parse.
pub fn env_parsed<T>(
    name: &str,
    fallback: T,
    parse: impl FnOnce(&str) -> Option<T>,
    expected: &str,
) -> T {
    let Some(text) = env_text(name) else {
        return fallback;
    };
    parse(&text).unwrap_or_else(|| {
        eprintln!("{name}=\"{text}\" does not parse, expected {expected}");
        std::process::exit(1)
    })
}

/// Reads a positive count like `128`, or `fallback` when unset or empty; exits if it does not parse.
pub fn env_count(name: &str, fallback: usize) -> usize {
    env_parsed(name, fallback, parse_count, "a positive count")
}

/// Reads a duration like `200ms` or `10s`, or `fallback` when unset or empty; exits if it does not parse.
pub fn env_duration(name: &str, fallback: Duration) -> Duration {
    env_parsed(
        name,
        fallback,
        parse_duration,
        "a duration like 200ms or 10s",
    )
}

/// Reads a size like `256MB`, or `fallback` when unset or empty; exits if it does not parse.
pub fn env_size(name: &str, fallback: Bytes) -> Bytes {
    env_parsed(name, fallback, parse_size, "a size like 4096, 64KB or 1GB")
}

/// Reads `0`, `1`, `true` or `false`, or `fallback` when unset or empty; exits if it does not parse.
pub fn env_flag(name: &str, fallback: bool) -> bool {
    let parse = |text: &str| match text {
        "0" | "false" => Some(false),
        "1" | "true" => Some(true),
        _ => None,
    };
    env_parsed(name, fallback, parse, "0, 1, true or false")
}

/// Reads a 32-bit seed or `random`, or `fallback` when unset or empty; exits if it does not parse.
pub fn env_seed(name: &str, fallback: Seed) -> Seed {
    env_parsed(name, fallback, parse_seed, "an unsigned integer or random")
}

/// Parses a 32-bit unsigned integer, or `random` as 32 bits from the OS entropy source.
pub fn parse_seed(text: &str) -> Option<Seed> {
    if text == "random" {
        use std::hash::{BuildHasher, Hasher};

        return Some(Seed(
            std::hash::RandomState::new().build_hasher().finish() as u32
        ));
    }
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    digits.then(|| text.parse().ok().map(Seed)).flatten()
}

/// Parses a thread count like `8`, or `0` as `all_cores`.
pub fn parse_threads(text: &str, all_cores: NonZeroUsize) -> Option<Threads> {
    match text {
        "0" => Some(Threads(all_cores)),
        _ => parse_count(text).and_then(NonZeroUsize::new).map(Threads),
    }
}

/// Parses a positive whole number in ASCII digits, like `128`; zero is `None`.
pub fn parse_count(text: &str) -> Option<usize> {
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    digits
        .then(|| text.parse().ok())
        .flatten()
        .filter(|&count| count != 0)
}

/// Parses one count like `128` or a comma list like `64,128,256,512`.
pub fn parse_dims(text: &str) -> Option<Vec<usize>> {
    text.split(',').map(parse_count).collect()
}

/// Parses a duration like `200ms` or `10s`; a bare number, a fraction or zero is `None`.
pub fn parse_duration(text: &str) -> Option<Duration> {
    match text.strip_suffix("ms") {
        Some(count) => parse_count(count).map(|count| Duration::from_millis(count as u64)),
        None => parse_count(text.strip_suffix('s')?).map(|count| Duration::from_secs(count as u64)),
    }
}

/// Parses a size like `256MB`: whole bytes, or `KB`, `MB`, `GB` or `TB` in any case; zero is `None`.
pub fn parse_size(text: &str) -> Option<Bytes> {
    let lowered = text.to_ascii_lowercase();
    let digits = lowered
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(lowered.len());
    let shift = match &lowered[digits..] {
        "" => 0,
        "kb" => 10,
        "mb" => 20,
        "gb" => 30,
        "tb" => 40,
        _ => return None,
    };
    let count: u64 = lowered[..digits].parse().ok()?;
    count
        .checked_mul(1 << shift)
        .filter(|&bytes| bytes != 0)
        .map(Bytes)
}

/// Spells a duration the way `parse_duration` reads it: `1s`, `1500ms`.
pub fn spell_duration(duration: Duration) -> String {
    let milliseconds = duration.as_millis();
    match milliseconds % 1000 {
        0 => format!("{}s", milliseconds / 1000),
        _ => format!("{milliseconds}ms"),
    }
}

/// Spells a size the way `parse_size` reads it: `256MB`, `1000`.
pub fn spell_size(bytes: u64) -> String {
    let (mut count, mut unit) = (bytes, "");
    for larger in ["KB", "MB", "GB", "TB"] {
        if count == 0 || count % 1024 != 0 {
            break;
        }
        count /= 1024;
        unit = larger;
    }
    format!("{count}{unit}")
}

/// A 32-bit run seed, an integer or drawn from the OS for `random`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seed(pub u32);

impl From<Seed> for u64 {
    fn from(seed: Seed) -> u64 {
        u64::from(seed.0)
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// A thread count; `0` in the variable resolves to every core when read, so it is never zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Threads(pub NonZeroUsize);

impl Threads {
    pub const ONE: Threads = Threads(NonZeroUsize::MIN);
}

impl fmt::Display for Threads {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// A size in bytes, never an element count; prints the way `parse_size` reads it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(pub u64);

impl fmt::Display for Bytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&spell_size(self.0))
    }
}

/// SplitMix64's increment, the golden ratio in 64 bits.
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64's finalizer, a bijection that spreads every input bit over the whole output.
pub fn mix(value: u64) -> u64 {
    let value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The key of stream `index` of `name`, two mixes away from `seed`, so neighboring seeds and indices never meet.
pub fn stream_key(seed: Seed, name: &str, index: u64) -> u64 {
    let hashed = name.bytes().fold(0xCBF2_9CE4_8422_2325, |hash: u64, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
    });
    mix(mix(u64::from(seed) ^ hashed).wrapping_add(index))
}

/// A SplitMix64 stream seeded with a `stream_key`, bit-identical to the C++ `splitmix64_t`.
pub struct SplitMix64 {
    pub state: u64,
}

impl SplitMix64 {
    /// The next 64 random bits.
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
        mix(self.state)
    }

    /// A draw below `bound`: the high half of a draw times `bound`, with a bias of at most `bound` over 2^64.
    pub fn below(&mut self, bound: u64) -> u64 {
        ((u128::from(self.next()) * u128::from(bound)) >> 64) as u64
    }
}

/// Installs a custom panic hook that formats errors cleanly for CLI usage.
/// Call this at the start of main() before any potential panics.
pub fn install_panic_hook() {
    panic::set_hook(Box::new(|info| {
        let message = if let Some(payload_text) = info.payload().downcast_ref::<&str>() {
            payload_text.to_string()
        } else if let Some(payload_text) = info.payload().downcast_ref::<String>() {
            payload_text.clone()
        } else {
            "Unknown error".to_string()
        };

        eprintln!("\nError: {}", message);

        // Location only in debug/RUST_BACKTRACE mode, not for CLI users.
        if cfg!(debug_assertions) || env_text("RUST_BACKTRACE").is_some() {
            if let Some(location) = info.location() {
                eprintln!("  at {}:{}", location.file(), location.line());
            }
        }
    }));
}

/// Prints the version line, then the dispatch mode and capabilities as "- Dynamic dispatch:" and "- This machine:".
pub fn print_machine() {
    let version = stringzilla::sz::version();
    println!(
        "StringZilla {}.{}.{}",
        version.major, version.minor, version.patch
    );
    println!(
        "- Dynamic dispatch: {}",
        stringzilla::sz::dynamic_dispatch()
    );
    println!(
        "- This machine: {}",
        stringzilla::sz::capabilities().as_str()
    );
}

/// Extension trait for Result that panics with the error's `Display` text, which the
/// custom panic hook prints as one clean line.
pub trait ResultExt<T> {
    /// Unwraps the result, or panics with the `Display`-formatted error.
    fn unwrap_nice(self) -> T;

    /// Unwraps the result, or panics with `message` and the `Display`-formatted error.
    fn expect_display(self, message: &str) -> T;
}

impl<T, E: fmt::Display> ResultExt<T> for Result<T, E> {
    #[track_caller]
    fn unwrap_nice(self) -> T {
        match self {
            Ok(value) => value,
            Err(error) => panic!("{}", error),
        }
    }

    #[track_caller]
    fn expect_display(self, message: &str) -> T {
        match self {
            Ok(value) => value,
            Err(error) => panic!("{}: {}", message, error),
        }
    }
}

/// Errors that can occur when loading a dataset.
#[derive(Debug)]
pub enum DatasetError {
    /// The dataset file does not exist.
    FileNotFound { path: String },
    /// Failed to read the dataset file.
    ReadError {
        path: String,
        source: std::io::Error,
    },
    /// The dataset file is empty.
    EmptyFile { path: String },
    /// No tokens were extracted from the dataset.
    NoTokens {
        path: String,
        tokenization: Tokenization,
    },
    /// Failed to create the token tape.
    TapeCreationFailed { path: String },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatasetError::FileNotFound { path } => {
                write!(
                    f,
                    "Dataset file not found: {}\n\n\
                     Please ensure the file exists. For Leipzig corpora, download with:\n  \
                       curl -fL https://downloads.wortschatz-leipzig.de/corpora/<corpus>.tar.gz \\\n    \
                         | tar --wildcards -xzf - --to-stdout '*-sentences.txt' | cut -f2 > {}",
                    path, path
                )
            }
            DatasetError::ReadError { path, source } => {
                write!(f, "Failed to read dataset '{}': {}", path, source)
            }
            DatasetError::EmptyFile { path } => {
                write!(
                    f,
                    "Dataset file is empty: {}\n\n\
                     Please provide a non-empty file.",
                    path
                )
            }
            DatasetError::NoTokens { path, tokenization } => {
                write!(
                    f,
                    "No tokens found in dataset '{}' with mode '{}'.\n\n\
                     The file exists but contains no valid tokens for this mode.\n\
                     Try a different STRINGWARS_TOKENS mode (lines, words, or file).",
                    path, tokenization
                )
            }
            DatasetError::TapeCreationFailed { path } => {
                write!(f, "Failed to create token tape from '{}'", path)
            }
        }
    }
}

impl std::error::Error for DatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DatasetError::ReadError { source, .. } => Some(source),
            _ => None,
        }
    }
}

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

/// Width of the left-aligned row-name column, matching `stringwars.py` and the C++ harnesses.
pub const REPORT_NAME_WIDTH: usize = 56;

/// Fewest samples a dispersion estimate needs; also the warm-up agreement window.
const DISPERSION_SAMPLES: u32 = 3;

/// Cap on passes per sample, so calibrating a sub-nanosecond pass cannot overflow.
const MAX_PASSES_PER_SAMPLE: u64 = 1 << 40;

/// Multiprocessors assumed when sizing a GPU batch without a GPU to ask.
pub const FALLBACK_GPU_MULTIPROCESSORS: usize = 64;

/// The ASCII whitespace set used by `words`, spelled out so it cannot drift from
/// the Python side. Python's bare `str.split()` also splits on Unicode spaces.
const ASCII_WHITESPACE: &[u8] = b" \n\t\r\x0b\x0c";

/// Derived from `ASCII_WHITESPACE` so the normative set stays the single source of truth.
/// `u8::is_ascii_whitespace` is *not* a substitute: it excludes U+000B, which Python's
/// bare `bytes.split()` does split on.
static IS_ASCII_WHITESPACE: [bool; 256] = {
    let mut table = [false; 256];
    let mut index = 0;
    while index < ASCII_WHITESPACE.len() {
        table[ASCII_WHITESPACE[index] as usize] = true;
        index += 1;
    }
    table
};

/// Length of `bytes` without an incomplete UTF-8 sequence at its end.
///
/// Asking the validator rather than walking back over continuation bytes: a hand-rolled
/// walk once guarded on `end < bytes.len()`, which is never true for a read cut at exactly
/// `STRINGWARS_BYTES`, so a buffer ending mid-sequence went through whole. Rust panicked on
/// the first multi-byte corpus, and Python quietly dropped the partial character.
fn char_boundary_floor(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Err(error) if error.error_len().is_none() => error.valid_up_to(),
        _ => bytes.len(),
    }
}

/// `stringwars.toml` — the single source of defaults shared with `stringwars.py`,
/// resolved relative to the crate root rather than the working directory.
pub fn manifest() -> &'static toml::Value {
    static MANIFEST: OnceLock<toml::Value> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        fs::read_to_string(root.join("stringwars.toml"))
            .or_else(|_| fs::read_to_string("stringwars.toml"))
            .expect_display(
                "stringwars.toml not found; it is the shared manifest for both harnesses",
            )
            .parse::<toml::Value>()
            .expect_display("stringwars.toml is not valid TOML")
    })
}

/// How a dataset splits into tokens; spelled `lines`, `words` or `file`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tokenization {
    Lines,
    Words,
    File,
}

impl Tokenization {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "lines" => Some(Self::Lines),
            "words" => Some(Self::Words),
            "file" => Some(Self::File),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lines => "lines",
            Self::Words => "words",
            Self::File => "file",
        }
    }
}

impl fmt::Display for Tokenization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Whether repeated tokens stay in the working set; `STRINGWARS_UNIQUE=1` drops them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Duplicates {
    Keep,
    Drop,
}

/// Whether rows also read the cycle and instruction counters, Linux only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counters {
    Off,
    CyclesAndInstructions,
}

/// Whether `hash` also reports each function's collision rate over the unique tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collisions {
    Skip,
    Report,
}

/// Every setting of one suite, read once at the top of `main` and passed down. Defaults come
/// from `stringwars.toml`, the suite's `[suite]` entry first and `[limits]` second, so neither
/// harness carries its own. Mirrors the Python `Settings` field for field, plus `counters`
/// and `collisions`.
pub struct Settings {
    pub suite: String,
    pub seed: Seed,
    pub filter: String,
    pub filter_pattern: Option<regex::Regex>,
    pub warmup: Duration,
    pub time_limit: Duration,
    pub dataset_bytes: Bytes,
    pub batch_per_core: Option<usize>,
    pub threads: Threads,
    pub dims: Vec<usize>,
    pub dataset_path: PathBuf,
    pub tokenization: Tokenization,
    pub duplicates: Duplicates,
    pub min_measure_time: Duration,
    pub min_sample_time: Duration,
    pub min_samples: usize,
    pub target_spread: f64,
    pub results_dir: Option<PathBuf>,
    pub counters: Counters,
    pub collisions: Collisions,
}

impl Settings {
    /// Reads every `STRINGWARS_*` variable `suite` uses, exiting with status 1 on the first that
    /// does not parse. A malformed `stringwars.toml` is a bug in the repository, so it panics.
    pub fn read(suite: &str) -> Self {
        let manifest = manifest();
        let entry = manifest.get("suite").and_then(|suites| suites.get(suite));
        let value = |key: &str| {
            let from_suite = entry.and_then(|entry| entry.get(key));
            from_suite.or_else(|| manifest["limits"].get(key))
        };
        let text = |key: &str| value(key).and_then(toml::Value::as_str);
        let duration = |key: &str| {
            text(key)
                .and_then(parse_duration)
                .expect("stringwars.toml: a duration is not like 10s")
        };
        let count = |key: &str| {
            value(key)
                .and_then(toml::Value::as_integer)
                .map(|count| count as usize)
        };
        let all_cores = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
        let filter = env_text("STRINGWARS_FILTER").unwrap_or_default();
        Settings {
            suite: suite.to_string(),
            seed: env_seed(
                "STRINGWARS_SEED",
                count("seed")
                    .and_then(|seed| u32::try_from(seed).ok())
                    .map(Seed)
                    .expect("stringwars.toml: `seed` is not a 32-bit integer"),
            ),
            filter_pattern: (!filter.is_empty())
                .then(|| regex::Regex::new(&filter).ok())
                .flatten(),
            filter,
            warmup: env_duration("STRINGWARS_WARMUP", duration("warmup")),
            time_limit: env_duration("STRINGWARS_TIME_LIMIT", duration("time_limit")),
            dataset_bytes: env_size(
                "STRINGWARS_BYTES",
                text("bytes")
                    .and_then(parse_size)
                    .expect("stringwars.toml: `bytes` is not a size like 256MB"),
            ),
            batch_per_core: env_parsed(
                "STRINGWARS_BATCH_PER_CORE",
                count("batch_per_core"),
                |text| parse_count(text).map(Some),
                "a positive count",
            ),
            threads: env_parsed(
                "STRINGWARS_THREADS",
                Threads(all_cores),
                |text| parse_threads(text, all_cores),
                "a count, 0 for all cores",
            ),
            dims: env_parsed(
                "STRINGWARS_DIMS",
                text("dims").map_or_else(Vec::new, |dims| {
                    parse_dims(dims).expect("stringwars.toml: `dims` is not a count list")
                }),
                parse_dims,
                "positive counts like 64,128",
            ),
            dataset_path: env_text("STRINGWARS_DATASET")
                .or_else(|| text("dataset").map(str::to_string))
                .map(PathBuf::from)
                .expect("stringwars.toml names no dataset for this suite"),
            tokenization: env_parsed(
                "STRINGWARS_TOKENS",
                text("tokens")
                    .map_or(Some(Tokenization::Lines), Tokenization::parse)
                    .expect("stringwars.toml: `tokens` is not lines, words or file"),
                Tokenization::parse,
                "lines, words or file",
            ),
            duplicates: match env_flag("STRINGWARS_UNIQUE", false) {
                true => Duplicates::Drop,
                false => Duplicates::Keep,
            },
            min_measure_time: duration("min_measure_time"),
            min_sample_time: duration("min_sample_time"),
            min_samples: env_count(
                "STRINGWARS_MIN_SAMPLES",
                count("min_samples").expect("stringwars.toml: `min_samples` is not a count"),
            ),
            target_spread: env_parsed(
                "STRINGWARS_TARGET_SPREAD",
                value("target_spread")
                    .and_then(toml::Value::as_float)
                    .expect("stringwars.toml: `target_spread` is not a fraction"),
                |text| {
                    text.parse()
                        .ok()
                        .filter(|&spread: &f64| spread > 0.0 && spread.is_finite())
                },
                "a fraction like 0.025",
            ),
            results_dir: env_text("STRINGWARS_RESULTS_DIR").map(PathBuf::from),
            counters: match env_flag("STRINGWARS_COUNTERS", false) {
                true => Counters::CyclesAndInstructions,
                false => Counters::Off,
            },
            collisions: match env_flag("STRINGWARS_COLLISIONS", false) {
                true => Collisions::Report,
                false => Collisions::Skip,
            },
        }
    }

    /// Prints every setting as "- Name: value", in the grammar it is read in.
    pub fn print(&self) {
        println!("- Seed: {}", self.seed);
        println!(
            "- Filter: {}",
            if self.filter.is_empty() {
                "none"
            } else {
                &self.filter
            }
        );
        println!("- Warm-up: {}", spell_duration(self.warmup));
        println!("- Time limit: {}", spell_duration(self.time_limit));
        println!("- Bytes: {}", self.dataset_bytes);
        if let Some(batch_per_core) = self.batch_per_core {
            println!("- Batch per core: {batch_per_core}");
        }
        println!("- Threads: {}", self.threads);
        if !self.dims.is_empty() {
            let dims: Vec<String> = self.dims.iter().map(usize::to_string).collect();
            println!("- Dims: {}", dims.join(","));
        }
        println!("- Dataset: {}", self.dataset_path.display());
        println!("- Tokens: {}", self.tokenization);
        println!("- Unique: {}", self.duplicates == Duplicates::Drop);
        println!("- Min samples: {}", self.min_samples);
        println!("- Target spread: {}", self.target_spread);
        println!(
            "- Results dir: {}",
            self.results_dir
                .as_deref()
                .map_or_else(|| "none".into(), Path::to_string_lossy)
        );
        println!(
            "- Counters: {}",
            self.counters == Counters::CyclesAndInstructions
        );
        println!("- Collisions: {}", self.collisions == Collisions::Report);
    }

    /// Whether `STRINGWARS_FILTER` selects the row `name`, as a regex or else as a substring.
    ///
    /// Records a filtered row as it decides, because several suites consult the filter
    /// themselves and return early; those rows used to vanish from the tally entirely.
    pub fn selects(&self, name: &str) -> bool {
        let selected = match &self.filter_pattern {
            Some(pattern) => pattern.is_match(name),
            None => name.contains(&self.filter),
        };
        if !selected {
            note_row(name, RowStatus::Filtered); // StringWars-only: the roster tallies filtered rows.
        }
        selected
    }
}

/// Identity of a resolved working set: CRC32 over the tokens joined by a NUL.
/// Digesting the tokens rather than the file means capping the working set does
/// not require re-hashing gigabytes. Mirrored exactly in `stringwars.py`.
pub fn fingerprint_tokens<'a>(tokens: impl Iterator<Item = &'a [u8]>) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    let mut first = true;
    for token in tokens {
        if !first {
            hasher.update(&[0u8]);
        }
        hasher.update(token);
        first = false;
    }
    hasher.finalize()
}

/// Resolves a suite's working set from its settings and registers its identity.
pub fn resolve_dataset(settings: &Settings) -> Result<BytesCowsAuto<'static>, DatasetError> {
    let tape = load_working_set(settings)?;
    let _ = RUN.set(RunIdentity {
        suite: settings.suite.clone(),
        dataset: settings.dataset_path.display().to_string(),
        tokenization: settings.tokenization,
        tokens: tape.len() as u64,
        token_bytes: Bytes(tape.iter().map(|token| token.len() as u64).sum()),
        crc: fingerprint_tokens(tape.iter()),
    });
    Ok(tape)
}

/// Unwraps inside a measured region, so a failing kernel stops the run.
///
/// `let _ = black_box(fallible())` measures the throughput of *returning an error*.
/// `encryption` did exactly that: every OpenSSL row was silently erroring on a null
/// IV and would have published the speed of failing as a cipher benchmark.
#[inline(always)]
pub fn expect_ok<T, E: fmt::Debug>(result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("benchmarked call failed: {error:?}"),
    }
}

/// What became of one row the run reached, for the roster `finish` tallies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowStatus {
    Measured,
    Refused,
    TooSlow,
    Skipped,
    Filtered,
}

/// Every row the run reached, with what became of it.
static ROSTER: Mutex<Vec<(String, RowStatus)>> = Mutex::new(Vec::new());

fn note_row(name: &str, status: RowStatus) {
    if let Ok(mut roster) = ROSTER.lock() {
        roster.push((name.to_string(), status));
    }
}

/// Prints one result line: the name in the shared 56-column field, then `text`.
fn print_row(name: &str, text: &str) {
    println!("{name:<width$} {text}", width = REPORT_NAME_WIDTH);
}

/// Records a contender that could not run at all — a missing optional dependency, a
/// gated backend. Prints a line so the absence is in the output rather than implied
/// by a gap in the table.
pub fn note_unavailable(name: &str, reason: &str) {
    print_row(name, &format!("SKIPPED: {reason}"));
    note_row(name, RowStatus::Skipped);
}

/// Prints the roster tally and returns the exit status: a failure if any row that was asked
/// to run failed to produce a number. Return it from `main`.
pub fn finish() -> ExitCode {
    let roster = ROSTER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let count = |want: RowStatus| roster.iter().filter(|(_, status)| *status == want).count();
    let (measured, refused) = (count(RowStatus::Measured), count(RowStatus::Refused));
    let (too_slow, skipped) = (count(RowStatus::TooSlow), count(RowStatus::Skipped));
    let filtered = count(RowStatus::Filtered);

    println!(
        "\nRoster: {measured} measured, {refused} refused, {too_slow} too slow, \
         {skipped} skipped, {filtered} filtered"
    );
    for (name, _) in roster
        .iter()
        .filter(|(_, status)| *status == RowStatus::Refused)
    {
        println!("  refused: {name}");
    }
    match refused {
        0 => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    }
}

/// Registers a working set built by a suite that needs its own mutable tape.
pub fn note_working_set(settings: &Settings, tokens: &[impl AsRef<[u8]>]) {
    let bytes: u64 = tokens.iter().map(|t| t.as_ref().len() as u64).sum();
    let crc = fingerprint_tokens(tokens.iter().map(|t| t.as_ref()));
    eprintln!(
        "Dataset: {} tokens, {} bytes ({:.2} GB)\n  Identity: mode {} crc 0x{:08x}",
        format_number(tokens.len() as u64),
        format_number(bytes),
        bytes as f64 / (1u64 << 30) as f64,
        settings.tokenization,
        crc
    );
    let _ = RUN.set(RunIdentity {
        suite: settings.suite.clone(),
        dataset: settings.dataset_path.display().to_string(),
        tokenization: settings.tokenization,
        tokens: tokens.len() as u64,
        token_bytes: Bytes(bytes),
        crc,
    });
}

/// Stamped onto every record; a static because `measure` sees only its own row.
struct RunIdentity {
    suite: String,
    dataset: String,
    tokenization: Tokenization,
    tokens: u64,
    token_bytes: Bytes,
    crc: u32,
}

static RUN: OnceLock<RunIdentity> = OnceLock::new();

/// Appends one NDJSON record per row when `STRINGWARS_RESULTS_DIR` is set.
fn record_outcome(
    settings: &Settings,
    outcome: &Outcome,
    spec: &MeasureSpec,
    bytes_per_second: f64,
) {
    let Some(dir) = &settings.results_dir else {
        return;
    };
    let Some(run) = RUN.get() else { return };
    if fs::create_dir_all(dir).is_err() {
        return;
    }
    let status = match &outcome.status {
        Status::Converged => "converged".to_string(),
        Status::Unconverged => format!("unconverged:{:.4}", outcome.spread),
        Status::TooFewSamples => format!("too_few_samples:{}", outcome.samples),
        Status::TooSlow { estimated_rate, .. } => format!("too_slow:{estimated_rate:.3e}"),
        Status::NonStationary { mean_median_gap } => format!("non_stationary:{mean_median_gap:.4}"),
        Status::Filtered => return,
    };
    let line = format!(
        "{{\"lang\":\"rust\",\"suite\":\"{}\",\"dataset\":\"{}\",\"mode\":\"{}\",\
         \"tokens\":{},\"token_bytes\":{},\"crc\":\"0x{:08x}\",\"row\":\"{}\",\
         \"unit\":\"{}\",\"rate\":{:.6e},\"bytes_per_second\":{:.6e},\"spread\":{:.6},\
         \"samples\":{},\"passes_per_sample\":{},\"concurrency\":{},\"status\":\"{}\"}}\n",
        run.suite,
        run.dataset,
        run.tokenization,
        run.tokens,
        run.token_bytes.0,
        run.crc,
        outcome.name.replace('"', "'"),
        spec.unit.record_name(),
        outcome.median_rate,
        bytes_per_second,
        outcome.spread,
        outcome.samples,
        outcome.passes_per_sample,
        spec.concurrency,
        status,
    );
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(format!("{}.ndjson", run.suite)))
    {
        let _ = file.write_all(line.as_bytes());
    }
}

/// Splits a haystack into token ranges under the normative rule, mirrored in `stringwars.py`:
/// `lines` splits on `\n`, `words` on ASCII whitespace, both dropping empty tokens, and
/// `file` is one token.
///
/// Returned as ranges rather than slices so a caller that needs `&mut` tokens can
/// walk them without the rule being written twice: `memory` used to carry its own
/// copy, and the copy had already drifted.
pub fn token_ranges(haystack: &[u8], tokenization: Tokenization) -> Vec<std::ops::Range<usize>> {
    let is_separator: fn(&u8) -> bool = match tokenization {
        Tokenization::File => return vec![0..haystack.len()],
        Tokenization::Lines => |byte| *byte == b'\n',
        Tokenization::Words => |byte| IS_ASCII_WHITESPACE[*byte as usize],
    };
    let (mut ranges, mut offset) = (Vec::new(), 0usize);
    for field in haystack.split(is_separator) {
        let start = offset;
        offset += field.len() + 1; // `split` consumed exactly one separator
        if !field.is_empty() {
            ranges.push(start..start + field.len());
        }
    }
    ranges
}

/// The token ranges of the working set: `token_ranges` under `STRINGWARS_TOKENS`, keeping only
/// the first of repeated tokens under `STRINGWARS_UNIQUE`.
pub fn working_set_ranges(haystack: &[u8], settings: &Settings) -> Vec<std::ops::Range<usize>> {
    let mut seen = HashSet::new();
    token_ranges(haystack, settings.tokenization)
        .into_iter()
        .filter(|range| {
            settings.duplicates == Duplicates::Keep || seen.insert(&haystack[range.clone()])
        })
        .collect()
}

/// Reads the first `STRINGWARS_BYTES` of the dataset, rounded down to a power of two as
/// StringZilla does, then back to a UTF-8 boundary so no character is torn.
pub fn read_dataset(settings: &Settings) -> std::io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    fs::File::open(&settings.dataset_path)?
        .take(settings.dataset_bytes.0)
        .read_to_end(&mut buffer)?;
    if !buffer.is_empty() {
        buffer.truncate(1 << buffer.len().ilog2());
    }
    buffer.truncate(char_boundary_floor(&buffer));
    Ok(buffer)
}

/// Resolves the working set for one suite: every token in the dataset's first
/// `STRINGWARS_BYTES`, deduplicated under `STRINGWARS_UNIQUE`.
fn load_working_set(settings: &Settings) -> Result<BytesCowsAuto<'static>, DatasetError> {
    let dataset_path = settings.dataset_path.display().to_string();
    if settings.duplicates == Duplicates::Drop {
        eprintln!("STRINGWARS_UNIQUE: deduplicating tokens");
    }

    // Check if file exists before attempting to read
    if !Path::new(&dataset_path).exists() {
        return Err(DatasetError::FileNotFound { path: dataset_path });
    }

    let content = read_dataset(settings).map_err(|error| DatasetError::ReadError {
        path: dataset_path.clone(),
        source: error,
    })?;

    // Check for empty file
    if content.is_empty() {
        return Err(DatasetError::EmptyFile { path: dataset_path });
    }

    // Leak the content to get 'static lifetime
    let content_static: &'static [u8] = Box::leak(content.into_boxed_slice());

    let kept: Vec<&'static [u8]> = working_set_ranges(content_static, settings)
        .into_iter()
        .map(|range| &content_static[range])
        .collect();
    let tape = BytesCowsAuto::from_iter_and_data(kept, Cow::Borrowed(content_static));

    let tape = tape.map_err(|_error| DatasetError::TapeCreationFailed {
        path: dataset_path.clone(),
    })?;

    // Check if we got any tokens
    if tape.is_empty() {
        return Err(DatasetError::NoTokens {
            path: dataset_path,
            tokenization: settings.tokenization,
        });
    }

    // Streaming statistics with log-scale histogram (O(1) memory)
    let count = tape.len();
    let total_bytes: usize = tape.iter().map(|slice: &[u8]| slice.len()).sum();
    let mean_len = total_bytes as f64 / count as f64;

    // Log-scale buckets: 0, 1, 2-3, 4-7, 8-15, 16-31, ... 32K-64K, 64K+
    let mut buckets = [0u64; 18];
    let mut min_len = usize::MAX;
    let mut max_len = 0;
    let mut variance_sum = 0.0;

    for token in tape.iter() {
        let len: usize = token.len();
        min_len = min_len.min(len);
        max_len = max_len.max(len);

        // Variance calculation
        let diff = len as f64 - mean_len;
        variance_sum += diff * diff;

        // Log-scale bucketing (powers of 2)
        let bucket = if len == 0 {
            0
        } else if len == 1 {
            1
        } else {
            // For len >= 2: bucket = log2(len) + 1
            // E.g., len=2-3 -> bucket 2, len=4-7 -> bucket 3, etc.
            ((len.ilog2() + 1) as usize).min(17)
        };
        buckets[bucket] += 1;
    }

    let std_dev: f64 = (variance_sum / count as f64).sqrt();

    // The identity line is what makes the two harnesses checkable against each
    // other: same mode, count, bytes and CRC means they resolved the same working
    // set. They silently diverged for months (Rust split words on ASCII bytes,
    // Python's `str.split()` also split U+3000 in the CJK corpora), and nothing in
    // the output would have shown it.
    eprintln!(
        "Dataset: {} tokens, {} bytes ({:.2} GB)\n  \
         Identity: mode {} crc 0x{:08x}\n  \
         Length: min {}, max {}, mean {:.1}, std {:.1}",
        format_number(count as u64),
        format_number(total_bytes as u64),
        total_bytes as f64 / (1u64 << 30) as f64,
        settings.tokenization,
        fingerprint_tokens(tape.iter()),
        min_len,
        max_len,
        mean_len,
        std_dev
    );

    // Show distribution (only non-empty buckets)
    eprintln!("  Distribution:");
    let bucket_ranges = [
        "0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1K",
        "1K-2K", "2K-4K", "4K-8K", "8K-16K", "16K-32K", "32K-64K", "64K+",
    ];
    for (index, &bucket_count) in buckets.iter().enumerate() {
        if bucket_count > 0 {
            let percent = (bucket_count as f64 / count as f64) * 100.0;
            let label = if index < bucket_ranges.len() {
                bucket_ranges[index]
            } else {
                "64K+"
            };
            eprintln!("    {:>10} bytes: {:>6.2}%", label, percent);
        }
    }

    Ok(tape)
}

/// Format large numbers with thousand separators for readability.
fn format_number(n: u64) -> String {
    let digits = n.to_string();
    let head = match digits.len() % 3 {
        0 => 3,
        rest => rest,
    };
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    out.push_str(&digits[..head]);
    for start in (head..digits.len()).step_by(3) {
        out.push(',');
        out.push_str(&digits[start..start + 3]);
    }
    out
}

#[cfg(target_os = "linux")]
use perf_event::{events::Hardware, Builder, Counter};

/// Cycles and instructions across the measured span, when `STRINGWARS_COUNTERS=1`.
///
/// Read twice per row, at the same points as the clock, so the overhead bound is
/// the one already argued for timing. Linux only; elsewhere this is a no-op and the
/// columns simply do not appear.
struct HardwareCounters {
    #[cfg(target_os = "linux")]
    inner: Option<(Option<Counter>, Option<Counter>)>,
}

impl HardwareCounters {
    #[cfg(target_os = "linux")]
    fn start(counters: Counters) -> Self {
        if counters == Counters::Off {
            return Self { inner: None };
        }
        let build = |kind: Hardware| -> Option<Counter> {
            let mut counter = Builder::new().kind(kind).build().ok()?;
            counter.enable().ok()?;
            Some(counter)
        };
        Self {
            inner: Some((build(Hardware::CPU_CYCLES), build(Hardware::INSTRUCTIONS))),
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn start(_counters: Counters) -> Self {
        Self {}
    }

    /// Cycles and instructions over the span, or `None` when disabled.
    #[cfg(target_os = "linux")]
    fn stop(self) -> Option<(u64, u64)> {
        let (mut cycles, mut instructions) = self.inner?;
        let read = |counter: &mut Option<Counter>| -> Option<u64> {
            let handle = counter.as_mut()?;
            handle.disable().ok()?;
            handle.read().ok()
        };
        Some((read(&mut cycles)?, read(&mut instructions)?))
    }

    #[cfg(not(target_os = "linux"))]
    fn stop(self) -> Option<(u64, u64)> {
        None
    }
}

/// What one routine call accomplished, for dual-metric reporting.
/// `elements` counts pairs / hashes / comparisons / tokens (0 when not applicable);
/// `bytes` counts the bytes touched.
#[derive(Clone, Copy, Default)]
pub struct WorkUnits {
    pub elements: u64,
    pub bytes: Bytes,
}

impl WorkUnits {
    /// Byte-only work (whole-buffer scans, transforms): `elements` stays 0.
    pub fn bytes(bytes: Bytes) -> Self {
        Self { elements: 0, bytes }
    }

    /// Both an element count and the bytes it spanned.
    pub fn new(elements: u64, bytes: Bytes) -> Self {
        Self { elements, bytes }
    }
}

/// Which primary unit a benchmark reports; bytes/s is always shown as the secondary metric.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// Bytes per second is the primary (and only) rate.
    Bytes,
    /// Cell updates per second (Needleman-Wunsch / Smith-Waterman / Levenshtein).
    Cups,
    /// Hashes per second (fingerprinting, hashing).
    Hashes,
    /// Hash-digest bits per second (multi-hash generation).
    Bits,
    /// Comparisons per second (sorting / sequence operations).
    Comparisons,
}

impl Unit {
    /// The unit as NDJSON records name it, shared with `stringwars.py`.
    fn record_name(self) -> &'static str {
        match self {
            Unit::Bytes => "bytes/s",
            Unit::Cups => "CUPS",
            Unit::Hashes => "hashes/s",
            Unit::Bits => "bits/s",
            Unit::Comparisons => "cmp/s",
        }
    }
}

/// Spells `per_second` with the significant figures a `relative_halfwidth` earns, so a nonzero
/// rate never prints as 0: "47.6 cmp/s", "1.2 GCUPS", "812 MB/s". Byte rates take binary prefixes.
fn spell_rate(per_second: f64, relative_halfwidth: f64, unit: Unit) -> String {
    let (step, prefixes, suffix) = match unit {
        Unit::Bytes => (1024.0, ["", "K", "M", "G", "T"], "B/s"),
        Unit::Cups => (1000.0, ["", "k", "M", "G", "T"], "CUPS"),
        Unit::Hashes => (1000.0, ["", "k", "M", "G", "T"], "hashes/s"),
        Unit::Bits => (1000.0, ["", "k", "M", "G", "T"], "bits/s"),
        Unit::Comparisons => (1000.0, ["", "k", "M", "G", "T"], "cmp/s"),
    };
    let (mut value, mut prefix) = (per_second, prefixes[0]);
    for larger in &prefixes[1..] {
        if value < step {
            break;
        }
        value /= step;
        prefix = larger;
    }
    let shown = round_to_earned_digits(value, relative_halfwidth);
    let decimals = match shown == 0.0 || !shown.is_finite() {
        true => 0,
        false => {
            (earned_digits(relative_halfwidth) - 1 - shown.abs().log10().floor() as i32).max(0)
        }
    };
    format!(
        "{shown:.decimals$} {prefix}{suffix}",
        decimals = decimals as usize
    )
}

/// What one pass over the pinned working set costs, declared up front.
///
/// The work is declared rather than accumulated because accumulating it costs two
/// `+=` per call inside the hot loop. At `hash`'s ~15 ns/call that bookkeeping was
/// a measurable fraction of the kernel it was supposed to be measuring.
pub struct MeasureSpec {
    pub unit: Unit,
    /// Work performed by a single pass.
    pub work: WorkUnits,
    /// Threads the variant is expected to use; one for single-threaded rows.
    pub concurrency: Threads,
}

impl MeasureSpec {
    pub fn new(unit: Unit, work: WorkUnits) -> Self {
        Self {
            unit,
            work,
            concurrency: Threads::ONE,
        }
    }
    pub fn with_concurrency(mut self, threads: Threads) -> Self {
        self.concurrency = threads;
        self
    }
}

/// Why a row printed no number. Reporting something plausible-looking is worse
/// than reporting nothing, so each of these is a hard stop rather than a caveat.
#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Converged,
    /// Hit `time_limit` before the spread came inside `target_spread`.
    Unconverged,
    /// Fewer samples than `min_samples` fit inside the cap.
    TooFewSamples,
    /// One pass alone rules out the three samples a dispersion estimate needs, so the
    /// row was never run at full scale. Carries a one-figure ceiling, not a
    /// measurement: `-` in a table, an estimate in the console and the record.
    TooSlow {
        estimated_rate: f64,
        projected_pass_seconds: f64,
    },
    /// Per-sample rates are not stationary — thermal drift, a competing process,
    /// or a leak. A free contamination detector: a stationary distribution has
    /// mean and median within a couple of spreads of each other.
    NonStationary {
        mean_median_gap: f64,
    },
    Filtered,
}

pub struct Outcome {
    pub name: String,
    pub median_rate: f64,
    pub spread: f64,
    pub samples: usize,
    pub passes_per_sample: u64,
    pub status: Status,
}

/// Cost of one `Instant::now()`, measured. The harness reads the clock exactly
/// twice per sample, so this over `min_sample_time` is the entire timing overhead.
pub fn clock_overhead_nanoseconds() -> f64 {
    let rounds = 10_000u32;
    let start = Instant::now();
    for _ in 0..rounds {
        black_box(Instant::now());
    }
    start.elapsed().as_nanos() as f64 / rounds as f64
}

/// Prints the measured timing overhead. This is the proof obligation behind
/// "the harness does not perturb the measurement" — an assertion otherwise.
pub fn log_timing_overhead(settings: &Settings) {
    let per_clock = clock_overhead_nanoseconds();
    let floor_nanos = settings.min_sample_time.as_nanos() as f64;
    println!(
        "Timing: {:.0} ns/clock, {:.1} ms sample floor -> harness overhead <= {:.4}%",
        per_clock,
        floor_nanos / 1e6,
        200.0 * per_clock / floor_nanos
    );
}

/// Selects a sample by rank. Half-up is written out, not delegated: `f64::round`
/// and Python's `round` break ties differently and picked different medians.
fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (fraction * (sorted.len() as f64 - 1.0) + 0.5).floor() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Significant figures a relative half-width earns: the most, up to 4, with `10^-(digits-1)` still above it.
fn earned_digits(relative_halfwidth: f64) -> i32 {
    (1..=4)
        .filter(|&digits| 10f64.powi(-(digits - 1)) >= relative_halfwidth)
        .last()
        .unwrap_or(1)
}

/// Rounds to the number of significant figures the measured dispersion justifies.
/// Printing `239,350 MCUPS` for a number good to +-35% is lying with typography.
fn round_to_earned_digits(value: f64, relative_halfwidth: f64) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    let magnitude = value.abs().log10().floor() as i32;
    let scale = 10f64.powi(earned_digits(relative_halfwidth) - 1 - magnitude);
    // Half-up, spelled out, for the same reason as `quantile`: rates are positive,
    // so this is the tie rule both harnesses can state rather than inherit.
    (value * scale + 0.5).floor() / scale
}

/// How many passes make one sample: calibrated to `min_sample_time`, or one for a kernel that consumes its input.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Passes {
    Calibrated,
    OnePerSample,
}

/// Times `pass` over the pinned working set and reports one row.
///
/// A **sample** is `passes_per_sample` complete traversals of the working set,
/// bracketed by exactly two clock reads. `passes_per_sample` is chosen during
/// warm-up so a sample lasts at least `min_sample_time`, which is what bounds the
/// timing overhead — there is no per-call clock and therefore no adaptive stride.
/// The loop between the two clock reads contains no harness instructions at all.
pub fn measure(
    settings: &Settings,
    name: &str,
    spec: MeasureSpec,
    mut pass: impl FnMut(),
) -> Outcome {
    measure_core(
        settings,
        name,
        spec,
        &mut |passes| {
            let started = Instant::now();
            for _ in 0..passes {
                pass();
            }
            started.elapsed()
        },
        Passes::Calibrated,
    )
}

/// Times a kernel that consumes its input, rebuilding it outside the clock.
///
/// One pass per sample is forced: a rebuild between passes would land inside the
/// timed span. Sound only when a single pass clears `min_sample_time`, as a sort does.
pub fn measure_with_setup<State>(
    settings: &Settings,
    name: &str,
    spec: MeasureSpec,
    mut setup: impl FnMut() -> State,
    mut body: impl FnMut(&mut State),
) -> Outcome {
    measure_core(
        settings,
        name,
        spec,
        &mut |_passes| {
            let mut state = setup();
            let started = Instant::now();
            body(&mut state);
            started.elapsed()
        },
        Passes::OnePerSample,
    )
}

/// Shared statistics core behind `measure` and `measure_with_setup`, so convergence,
/// refusal and reporting cannot drift between them.
fn measure_core(
    settings: &Settings,
    name: &str,
    spec: MeasureSpec,
    run_sample: &mut dyn FnMut(u64) -> Duration,
    passes_rule: Passes,
) -> Outcome {
    // A refusal must be as visible as a result. Printing nothing is how a row goes
    // missing for weeks without anyone noticing.
    let refuse = |status: Status, samples: usize| {
        match &status {
            Status::Filtered => {}
            Status::TooFewSamples => print_row(
                name,
                &format!(
                    "REFUSED: {samples} samples in {} cap (need {DISPERSION_SAMPLES})",
                    spell_duration(settings.time_limit)
                ),
            ),
            Status::TooSlow {
                estimated_rate,
                projected_pass_seconds,
            } => print_row(
                name,
                &format!(
                    "TOO SLOW: ~{}, one pass ~{} against a {} cap",
                    spell_rate(*estimated_rate, 1.0, Unit::Bytes),
                    spell_duration(Duration::from_secs_f64(*projected_pass_seconds)),
                    spell_duration(settings.time_limit)
                ),
            ),
            other => print_row(name, &format!("REFUSED: {other:?}")),
        }
        // `selects` already noted a filtered row. A too-slow row is an expected
        // outcome, not a failure, so it gets its own bucket and does not force a
        // non-zero exit.
        match status {
            Status::Filtered => {}
            Status::TooSlow { .. } => note_row(name, RowStatus::TooSlow),
            _ => note_row(name, RowStatus::Refused),
        }
        Outcome {
            name: name.to_string(),
            median_rate: 0.0,
            spread: f64::NAN,
            samples,
            passes_per_sample: 0,
            status,
        }
    };
    if !settings.selects(name) {
        return refuse(Status::Filtered, 0);
    }

    let sample_floor = settings.min_sample_time;

    // Choose how many passes make one sample. Scaling by the observed shortfall
    // converges in a couple of steps even when a pass is nanoseconds long. A kernel
    // that consumes its input cannot be repeated inside a sample, so it stays at one
    // pass and skips calibration entirely.
    let mut passes: u64 = 1;
    let pass_cost = match passes_rule {
        Passes::Calibrated => loop {
            let sample_time = run_sample(passes);
            if sample_time >= sample_floor || passes >= MAX_PASSES_PER_SAMPLE {
                break sample_time.div_f64(passes as f64);
            }
            let shortfall = sample_floor.as_secs_f64() / sample_time.as_secs_f64().max(1e-9);
            passes = passes
                .saturating_mul((shortfall.ceil() as u64).max(2))
                .min(MAX_PASSES_PER_SAMPLE);
        },
        // A consuming kernel cannot batch passes, but one sample still prices one. Leaving
        // the cost at zero made the refusal below test `0 * 3 > time_limit`, so `sequence` -
        // the one suite whose work is superlinear - could never be refused.
        Passes::OnePerSample => run_sample(passes),
    };

    if pass_cost * DISPERSION_SAMPLES > settings.time_limit {
        let seconds = pass_cost.as_secs_f64();
        return refuse(
            Status::TooSlow {
                estimated_rate: spec.work.bytes.0 as f64 / seconds.max(1e-9),
                projected_pass_seconds: seconds,
            },
            1,
        );
    }

    // Warm-up: discard samples until three consecutive ones agree. Two cannot
    // distinguish agreement from coincidence. This is also what consumes the
    // cold-traversal climb that made `hash` look like it never converged.
    let window = DISPERSION_SAMPLES as usize;
    let warmup_deadline = Instant::now() + settings.warmup;
    let mut recent_sample_seconds: Vec<f64> = Vec::new();
    while recent_sample_seconds.len() < window || Instant::now() < warmup_deadline {
        let seconds = run_sample(passes).as_secs_f64();
        if recent_sample_seconds.len() == window {
            recent_sample_seconds.rotate_left(1);
            recent_sample_seconds[window - 1] = seconds;
        } else {
            recent_sample_seconds.push(seconds);
        }
        if recent_sample_seconds.len() == window {
            let (low, high) = recent_sample_seconds
                .iter()
                .fold((f64::MAX, 0.0f64), |(l, h), &v| (l.min(v), h.max(v)));
            if (high - low) / high.max(1e-12) <= settings.target_spread {
                break;
            }
        }
    }

    // Measure. Counters bracket the whole measured span rather than each sample, so
    // they cost two reads per row and cannot perturb the per-sample timing.
    let counters = HardwareCounters::start(settings.counters);
    let measure_start = Instant::now();
    let cap = settings.time_limit;
    let mut seconds_per_sample: Vec<f64> = Vec::new();
    let mut scratch: Vec<f64> = Vec::new();
    loop {
        seconds_per_sample.push(run_sample(passes).as_secs_f64());

        let elapsed = measure_start.elapsed();
        let enough = seconds_per_sample.len() >= settings.min_samples
            && elapsed >= settings.min_measure_time;
        if enough {
            scratch.clear();
            scratch.extend_from_slice(&seconds_per_sample);
            let sorted = &mut scratch;
            sorted.sort_unstable_by(f64::total_cmp);
            let median = quantile(sorted, 0.5);
            let halfwidth =
                (quantile(sorted, 0.9) - quantile(sorted, 0.1)) / (2.0 * median.max(1e-12));
            if halfwidth <= settings.target_spread {
                break;
            }
        }
        if elapsed >= cap {
            break;
        }
    }

    if seconds_per_sample.len() < window {
        let collected = seconds_per_sample.len();
        return refuse(Status::TooFewSamples, collected);
    }

    scratch.clear();
    scratch.extend_from_slice(&seconds_per_sample);
    let sorted = &mut scratch;
    sorted.sort_unstable_by(f64::total_cmp);
    let median_seconds = quantile(sorted, 0.5);
    let spread = (quantile(sorted, 0.9) - quantile(sorted, 0.1)) / median_seconds.max(1e-12);
    let mean_seconds = seconds_per_sample.iter().sum::<f64>() / seconds_per_sample.len() as f64;
    let gap = (mean_seconds - median_seconds).abs() / median_seconds.max(1e-12);

    let status = if gap > 2.0 * settings.target_spread {
        Status::NonStationary {
            mean_median_gap: gap,
        }
    } else if spread / 2.0 > settings.target_spread {
        Status::Unconverged
    } else {
        Status::Converged
    };

    let elements = spec.work.elements.saturating_mul(passes) as f64;
    let bytes = spec.work.bytes.0.saturating_mul(passes) as f64;
    let primary = match spec.unit {
        Unit::Bytes => bytes,
        _ => elements,
    } / median_seconds;

    let outcome = Outcome {
        name: name.to_string(),
        median_rate: primary,
        spread,
        samples: seconds_per_sample.len(),
        passes_per_sample: passes,
        status,
    };
    report_outcome(&outcome, &spec, bytes / median_seconds, counters.stop());
    record_outcome(settings, &outcome, &spec, bytes / median_seconds);
    note_row(name, RowStatus::Measured);
    outcome
}

/// Prints a measured row: the primary rate, the byte rate when it differs, the spread and sample count, then any caveat.
fn report_outcome(
    outcome: &Outcome,
    spec: &MeasureSpec,
    bytes_per_second: f64,
    counters: Option<(u64, u64)>,
) {
    let halfwidth = outcome.spread / 2.0;
    let mut columns = vec![spell_rate(outcome.median_rate, halfwidth, spec.unit)];
    if spec.unit != Unit::Bytes && spec.work.bytes.0 > 0 {
        columns.push(spell_rate(bytes_per_second, halfwidth, Unit::Bytes));
    }
    columns.push(format!("+-{:.1}% n={}", 100.0 * halfwidth, outcome.samples));
    if let Some((cycles, instructions)) = counters {
        let bytes = (spec.work.bytes.0 as f64)
            * (outcome.passes_per_sample as f64)
            * (outcome.samples as f64);
        if bytes > 0.0 {
            columns.push(format!("{:.2} cyc/B", cycles as f64 / bytes));
        }
        if cycles > 0 {
            columns.push(format!("{:.2} IPC", instructions as f64 / cycles as f64));
        }
    }
    match &outcome.status {
        Status::Converged => {}
        Status::Unconverged => columns.push(format!("UNCONVERGED {:.0}%", 100.0 * outcome.spread)),
        Status::NonStationary { mean_median_gap } => {
            columns.push(format!("NON-STATIONARY {:.0}%", 100.0 * mean_median_gap))
        }
        Status::TooFewSamples => columns.push(format!("REFUSED {} samples", outcome.samples)),
        // Reported by `refuse`, which never reaches this row-printing path.
        Status::TooSlow { .. } | Status::Filtered => return,
    }
    print_row(&outcome.name, &columns.join(" | "));
}

/// Batch size for a backend with `cores` parallel cores: `STRINGWARS_BATCH_PER_CORE`, or the
/// suite's `batch_per_core` in `stringwars.toml`, times `cores`. A CPU core and a GPU streaming
/// multiprocessor (an SM, not a warp or a CUDA core) each count as one core, so the batch scales
/// with the hardware instead of a fixed CPU/GPU multiplier.
pub fn batch_size(settings: &Settings, cores: usize) -> usize {
    let per_core = settings
        .batch_per_core
        .expect("this suite has no `batch_per_core` in stringwars.toml");
    per_core.saturating_mul(cores.max(1))
}

/// Number of streaming multiprocessors on the given CUDA device, queried from the CUDA runtime.
/// Each SM is counted as one core for batch sizing (an SM, not a warp or an individual CUDA core).
/// Returns None when CUDA is unavailable or the query fails, so callers fall back to a default
/// core count. The attribute id 16 is `cudaDevAttrMultiProcessorCount`.
#[cfg(feature = "cuda")]
pub fn gpu_multiprocessor_count(device_index: i32) -> Option<usize> {
    extern "C" {
        fn cudaDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> i32;
    }
    const MULTIPROCESSOR_COUNT_ATTRIBUTE: i32 = 16;
    let mut count: i32 = 0;
    let status =
        unsafe { cudaDeviceGetAttribute(&mut count, MULTIPROCESSOR_COUNT_ATTRIBUTE, device_index) };
    (status == 0 && count > 0).then_some(count as usize)
}

/// Without the CUDA feature there is no device to query, so callers use their fallback core count.
#[cfg(not(feature = "cuda"))]
pub fn gpu_multiprocessor_count(_device_index: i32) -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rank rule is written out rather than delegated to `f64::round`, which
    /// breaks ties away from zero while Python's `round` breaks them to even. At
    /// n = 10 — the modal terminating count — that put the two harnesses on
    /// different samples, and every median and spread is derived from this.
    #[test]
    fn quantile_rank_is_half_up() {
        let expected = |n: usize, fraction: f64| -> usize {
            ((fraction * (n as f64 - 1.0) + 0.5).floor() as usize).min(n - 1)
        };
        for n in [3usize, 6, 10, 12, 14, 18, 25, 100] {
            let samples: Vec<f64> = (0..n).map(|index| index as f64).collect();
            for fraction in [0.1, 0.5, 0.9] {
                assert_eq!(
                    quantile(&samples, fraction),
                    expected(n, fraction) as f64,
                    "n={n} fraction={fraction}"
                );
            }
        }
        // The case that actually diverged: 10 samples, median.
        let ten: Vec<f64> = (0..10).map(|index| index as f64).collect();
        assert_eq!(quantile(&ten, 0.5), 5.0);
    }

    #[test]
    fn quantile_handles_degenerate_input() {
        assert_eq!(quantile(&[], 0.5), 0.0);
        assert_eq!(quantile(&[7.0], 0.9), 7.0);
    }

    /// Digits are earned from dispersion: `239,350 MCUPS` for a number good to
    /// +-35% is lying with typography.
    #[test]
    fn significant_figures_track_dispersion() {
        // digits = the largest d with 10^-(d-1) >= halfwidth, so +-35% earns one
        // figure and +-2% earns two.
        assert_eq!(round_to_earned_digits(1234.0, 0.35), 1000.0);
        assert_eq!(round_to_earned_digits(1234.0, 0.02), 1200.0);
        assert_eq!(round_to_earned_digits(1234.0, 0.002), 1230.0);
        assert_eq!(round_to_earned_digits(0.0, 0.02), 0.0);
        assert!(round_to_earned_digits(f64::NAN, 0.02).is_nan());
        // Slow rows used to print `0.00` at a fixed two decimals.
        assert_eq!(
            spell_rate(0.004_71, 0.002, Unit::Comparisons),
            "0.00471 cmp/s"
        );
        assert_eq!(spell_rate(47.63, 0.02, Unit::Hashes), "48 hashes/s");
        assert_eq!(
            spell_rate(3.0 * 1024.0 * 1024.0, 0.002, Unit::Bytes),
            "3.00 MB/s"
        );
    }

    /// Keys and draws must match the C++ `stream_key` and `splitmix64_t` bit for bit.
    #[test]
    fn streams_match_the_cpp_harness() {
        let key = stream_key(Seed(42), "normalization/needles", 0);
        assert_eq!(key, 0x29a9_e279_7e65_1746);
        assert_eq!(stream_key(Seed(7), "", 3), 0xc778_81ed_89ce_9c24);
        assert_eq!(SplitMix64 { state: key }.next(), 0x5765_a4d9_6dc0_548e);
    }

    #[test]
    fn sizes_parse_as_iec() {
        assert_eq!(parse_size("256MB"), Some(Bytes(256 << 20)));
        assert_eq!(parse_size("1GB"), Some(Bytes(1 << 30)));
        assert_eq!(parse_size("16mb"), Some(Bytes(16 << 20)));
        assert_eq!(parse_size("512"), Some(Bytes(512)));
        for rejected in ["256MB", "1.5GB", " 16MB", "0", "0KB", "MB", "16b", ""] {
            assert_eq!(parse_size(rejected), None, "{rejected}");
        }
        assert_eq!(spell_size(256 << 20), "256MB");
        assert_eq!(spell_size(1000), "1000");
    }

    #[test]
    fn durations_need_a_unit_and_a_positive_integer() {
        assert_eq!(parse_duration("200ms"), Some(Duration::from_millis(200)));
        assert_eq!(parse_duration("10s"), Some(Duration::from_secs(10)));
        for rejected in ["10", "0s", "0ms", "1.5s", "1m", "+5s", " 5s", ""] {
            assert_eq!(parse_duration(rejected), None, "{rejected}");
        }
        assert_eq!(spell_duration(Duration::from_millis(1000)), "1s");
        assert_eq!(spell_duration(Duration::from_millis(1500)), "1500ms");
    }

    /// Both harnesses must resolve byte-identical working sets, so the tokenizer
    /// is the one rule that may never drift.
    #[test]
    fn fingerprint_distinguishes_token_boundaries() {
        let joined = fingerprint_tokens([&b"ab"[..], &b"c"[..]].into_iter());
        let single = fingerprint_tokens([&b"abc"[..]].into_iter());
        assert_ne!(joined, single, "NUL separator must survive the digest");
        assert_eq!(fingerprint_tokens([].into_iter()), 0);
    }

    /// A read cut at exactly `STRINGWARS_BYTES` can end mid-character; the cut backs off to
    /// the last whole one, and only at the end of the buffer.
    #[test]
    fn dataset_cut_never_tears_a_character() {
        let text = "мама мыла раму".as_bytes();
        for cut in 0..=text.len() {
            let end = char_boundary_floor(&text[..cut]);
            assert!(
                std::str::from_utf8(&text[..end]).is_ok(),
                "cut {cut} kept {end}"
            );
            assert!(cut - end < 4, "cut {cut} kept {end}");
        }
        assert_eq!(char_boundary_floor(b"a\xffb"), 3);
    }

    #[test]
    fn token_ranges_match_the_reference_state_machine() {
        // The pre-`split` implementation, kept only as a fuzzing oracle: this is the
        // normative tokenizer, so a silent divergence would corrupt every working set.
        fn reference(haystack: &[u8], tokenization: Tokenization) -> Vec<std::ops::Range<usize>> {
            if tokenization == Tokenization::File {
                return vec![0..haystack.len()];
            }
            let is_separator = |byte: u8| {
                if tokenization == Tokenization::Lines {
                    byte == b'\n'
                } else {
                    ASCII_WHITESPACE.contains(&byte)
                }
            };
            let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
            let mut start = None;
            for (index, &byte) in haystack.iter().enumerate() {
                if is_separator(byte) {
                    if let Some(begin) = start.take() {
                        ranges.push(begin..index);
                    }
                } else if start.is_none() {
                    start = Some(index);
                }
            }
            if let Some(begin) = start {
                ranges.push(begin..haystack.len());
            }
            ranges
        }

        // Deterministic xorshift, so a failure reproduces exactly.
        let alphabet = [
            b"a".as_slice(),
            b"bb",
            b" ",
            b"\n",
            b"\t",
            b"\r",
            b"\x0b",
            b"\x0c",
            "é".as_bytes(),
            "\u{4e2d}".as_bytes(),
        ];
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let mut haystack = Vec::new();
            for _ in 0..(next() % 24) {
                haystack.extend_from_slice(alphabet[(next() % alphabet.len() as u64) as usize]);
            }
            for tokenization in [Tokenization::Words, Tokenization::Lines, Tokenization::File] {
                assert_eq!(
                    token_ranges(&haystack, tokenization),
                    reference(&haystack, tokenization),
                    "tokenization={tokenization} haystack={haystack:?}",
                );
            }
        }
    }

    #[test]
    fn token_ranges_follow_the_normative_rule() {
        let text = b"alpha beta\n\ngamma";
        let words = token_ranges(text, Tokenization::Words);
        assert_eq!(words.len(), 3);
        assert_eq!(&text[words[0].clone()], b"alpha");
        assert_eq!(&text[words[2].clone()], b"gamma");

        // Empty tokens are dropped, so the blank line yields nothing.
        let lines = token_ranges(text, Tokenization::Lines);
        assert_eq!(lines.len(), 2);

        assert_eq!(token_ranges(text, Tokenization::File), vec![0..text.len()]);
    }

    #[test]
    fn ascii_whitespace_matches_the_documented_set() {
        assert_eq!(ASCII_WHITESPACE, b" \n\t\r\x0b\x0c");
        // Unicode spaces are deliberately excluded: Python's bare `str.split()`
        // also splits U+3000, which Rust never did, and the CJK corpora contain it.
        assert!(!ASCII_WHITESPACE.contains(&0xA0));
        // The lookup table the tokenizer actually consults must agree with the slice
        // it is derived from, for every byte.
        for byte in 0..=u8::MAX {
            assert_eq!(
                IS_ASCII_WHITESPACE[byte as usize],
                ASCII_WHITESPACE.contains(&byte),
                "byte {byte:#04x}",
            );
        }
    }
}
