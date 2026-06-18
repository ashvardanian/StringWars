#![doc = r#"
# StringWars: Multi-Way Word Hashing & Probabilistic Membership Benchmarks

Probabilistic membership structures — Bloom, cuckoo, and XOR/binary-fuse filters — need several independent hashes of
the same short key. This file benchmarks that operation in two layers.

**Layer 1 — multi-hash generation** produces `k` hashes per word for `k` in `{2, 4, 8, 16}`:

- StringZilla `hash_multiseed` normalizes the input into AES blocks once, then replays cheap per-seed rounds, emitting
  all `k` hashes in a single pass.
- StringZilla `hash` called once per seed, isolating what the multi-seed path amortizes.
- One 128-bit `xxh3_128` split into `(h1, h2)` with `g_i = h1 + i*h2` (Kirsch–Mitzenmacher) — the double-hashing scheme
  production Bloom and cuckoo filters actually use.

**Layer 2 — probabilistic membership** builds each filter from the unique words and queries a held-out set to measure
the false-positive rate, comparing a StringZilla-fed variant against the practical default while holding the filter
structure fixed.

## Usage

```sh
STRINGWARS_DATASET=xlsum.csv \
    STRINGWARS_TOKENS=words \
    cargo bench --features bench_containers --bench bench_containers
```
"#]

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hint::black_box;

use stringtape::{BytesCowsAuto, BytesTape};

use cuckoofilter::CuckooFilter;
use fastbloom::BloomFilter;
use stringzilla::sz;
use xorf::{BinaryFuse8, Filter};
use xxhash_rust::xxh3::{xxh3_128, xxh3_64};

#[path = "../utils.rs"]
mod utils;
use utils::{
    install_panic_hook, load_dataset_with_default_mode, log_stringzilla_metadata,
    measure_throughput, should_run, BenchBudget, ReportAs, ResultExt, WorkUnits,
};

/// Sixteen fixed odd seeds, enough for the largest `k`, shared across every multi-hash variant so
/// each produces the same family of hashes.
const SEEDS: [u64; 16] = [
    0x9E37_79B9_7F4A_7C15,
    0xC2B2_AE3D_27D4_EB4F,
    0x1656_67B1_9E37_79F9,
    0xD1B5_4A32_D192_ED03,
    0xA076_1D64_78BD_642F,
    0xE703_7ED1_A0B4_28DB,
    0x8EBC_6AF0_9C88_C6E3,
    0x5899_65CC_7537_4CC3,
    0x1D8E_4E27_C47D_124F,
    0xEB44_ACCA_B455_D165,
    0x2545_F491_4F6C_DD1D,
    0xFF51_AFD7_ED55_8CCD,
    0xC4CE_B9FE_1A85_EC53,
    0xBF58_476D_1CE4_E5B9,
    0x9417_5CC1_BAB3_5C97,
    0x4CF5_AD43_2745_937F,
];

/// The target false-positive rate configured for the Bloom and cuckoo filters.
const TARGET_FALSE_POSITIVE_RATE: f64 = 0.01;

/// Times one multi-hash variant, calling `fill` once per word to produce `k` hashes.
fn measure_multihash<Fill: FnMut(&[u8])>(
    name: &str,
    budget: &BenchBudget,
    k: usize,
    slices: &[&[u8]],
    mut fill: Fill,
) {
    if !should_run(name) {
        return;
    }
    let mut cursor = 0usize;
    measure_throughput(name, ReportAs::Hashes, budget, || {
        let token = slices[cursor % slices.len()];
        cursor += 1;
        fill(token);
        WorkUnits::new(k as u64, token.len() as u64)
    });
}

/// Times filter construction, rebuilding the whole filter from `count` keys on every pass.
fn measure_build<Build: FnMut()>(
    name: &str,
    budget: &BenchBudget,
    count: usize,
    bytes: u64,
    mut build: Build,
) {
    if !should_run(name) {
        return;
    }
    measure_throughput(name, ReportAs::Hashes, budget, || {
        build();
        WorkUnits::new(count as u64, bytes)
    });
}

/// Times membership queries, cycling one probe per call.
fn measure_query<Query: FnMut(&[u8]) -> bool>(
    name: &str,
    budget: &BenchBudget,
    probes: &[&[u8]],
    mut query: Query,
) {
    if !should_run(name) {
        return;
    }
    let mut cursor = 0usize;
    measure_throughput(name, ReportAs::Hashes, budget, || {
        let token = probes[cursor % probes.len()];
        cursor += 1;
        black_box(query(token));
        WorkUnits::new(1, token.len() as u64)
    });
}

/// Reports the measured false-positive rate over the held-out absent words plus bits-per-key.
fn report_quality<Contains: FnMut(&[u8]) -> bool>(
    label: &str,
    num_bits: usize,
    inserted_count: usize,
    absent: &[&[u8]],
    mut contains: Contains,
) {
    let false_positives = absent.iter().filter(|token| contains(token)).count();
    let rate = if absent.is_empty() {
        0.0
    } else {
        false_positives as f64 / absent.len() as f64 * 100.0
    };
    let bits_per_key = num_bits as f64 / inserted_count.max(1) as f64;
    println!(
        "    {:<38} {:5.2} bits/key, measured FPR {:.3}%",
        label, bits_per_key, rate
    );
}

/// Layer 1: produce `k` hashes per word, reporting hashes/s.
fn bench_multihash(budget: &BenchBudget, k: usize, slices: &[&[u8]]) {
    println!("# multihash (k={})", k);
    let mut hashes = [0u64; 16];

    measure_multihash(
        "multihash/stringzilla::hash_multiseed",
        budget,
        k,
        slices,
        |token| {
            sz::hash_multiseed_into(token, &SEEDS[..k], &mut hashes[..k]);
            black_box(&hashes[..k]);
        },
    );

    measure_multihash("multihash/stringzilla::hash", budget, k, slices, |token| {
        for (slot, seed) in hashes[..k].iter_mut().zip(&SEEDS[..k]) {
            *slot = sz::hash_with_seed(token, *seed);
        }
        black_box(&hashes[..k]);
    });

    measure_multihash("multihash/xxh3::xxh3_128", budget, k, slices, |token| {
        let wide = xxh3_128(token);
        let (first, second) = (wide as u64, (wide >> 64) as u64);
        for (index, slot) in hashes[..k].iter_mut().enumerate() {
            *slot = first.wrapping_add((index as u64).wrapping_mul(second));
        }
        black_box(&hashes[..k]);
    });
}

/// Bloom filter (fastbloom): SipHash default versus a single `sz::hash` fed through `insert_hash`.
fn bench_bloom(budget: &BenchBudget, inserted: &[&[u8]], absent: &[&[u8]], bytes: u64) {
    let count = inserted.len();

    let mut bloom = BloomFilter::with_false_pos(TARGET_FALSE_POSITIVE_RATE).expected_items(count);
    for token in inserted {
        bloom.insert(token);
    }
    report_quality(
        "bloom/fastbloom<siphash>",
        bloom.num_bits(),
        count,
        absent,
        |token| bloom.contains(token),
    );
    measure_build(
        "bloom/fastbloom::insert<siphash>",
        budget,
        count,
        bytes,
        || {
            let mut filter =
                BloomFilter::with_false_pos(TARGET_FALSE_POSITIVE_RATE).expected_items(count);
            for token in inserted {
                filter.insert(token);
            }
            black_box(&filter);
        },
    );
    measure_query(
        "bloom/fastbloom::contains<siphash>",
        budget,
        inserted,
        |token| bloom.contains(token),
    );

    let mut bloom_sz =
        BloomFilter::with_false_pos(TARGET_FALSE_POSITIVE_RATE).expected_items(count);
    for token in inserted {
        bloom_sz.insert_hash(sz::hash(token));
    }
    report_quality(
        "bloom/fastbloom<stringzilla>",
        bloom_sz.num_bits(),
        count,
        absent,
        |token| bloom_sz.contains_hash(sz::hash(token)),
    );
    measure_build(
        "bloom/fastbloom::insert<stringzilla>",
        budget,
        count,
        bytes,
        || {
            let mut filter =
                BloomFilter::with_false_pos(TARGET_FALSE_POSITIVE_RATE).expected_items(count);
            for token in inserted {
                filter.insert_hash(sz::hash(token));
            }
            black_box(&filter);
        },
    );
    measure_query(
        "bloom/fastbloom::contains<stringzilla>",
        budget,
        inserted,
        |token| bloom_sz.contains_hash(sz::hash(token)),
    );
}

/// Cuckoo filter (cuckoofilter): SipHash default. `cuckoofilter` hashes the key twice per op
/// (once for the fingerprint, once for the alternate bucket) through the `Hasher` trait and exposes
/// no precomputed-hash entry point, so there is no StringZilla-fed variant to compare here.
fn bench_cuckoo(budget: &BenchBudget, inserted: &[&[u8]], absent: &[&[u8]], bytes: u64) {
    let count = inserted.len();

    let mut cuckoo = CuckooFilter::<DefaultHasher>::with_capacity(count);
    for token in inserted {
        let _ = cuckoo.add(token);
    }
    report_quality(
        "cuckoo/cuckoofilter",
        cuckoo.memory_usage() * 8,
        count,
        absent,
        |token| cuckoo.contains(token),
    );
    measure_build("cuckoo/cuckoofilter::insert", budget, count, bytes, || {
        let mut filter = CuckooFilter::<DefaultHasher>::with_capacity(count);
        for token in inserted {
            let _ = filter.add(token);
        }
        black_box(&filter);
    });
    measure_query("cuckoo/cuckoofilter::contains", budget, inserted, |token| {
        cuckoo.contains(token)
    });
}

/// Collects the keys into a sorted, deduplicated `u64` array — the shape `xorf` consumes.
fn hashed_keys<Hash: Fn(&[u8]) -> u64>(inserted: &[&[u8]], hash: Hash) -> Vec<u64> {
    let mut keys: Vec<u64> = inserted.iter().map(|token| hash(token)).collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// Binary-fuse filter (xorf): a static filter built from pre-hashed keys, so the only variable is the
/// hash that produced them. Its ~0.4% false-positive rate is fixed by the 8-bit fingerprints.
fn bench_xorf(budget: &BenchBudget, inserted: &[&[u8]], absent: &[&[u8]], bytes: u64) {
    let count = inserted.len();

    let fuse_sz = BinaryFuse8::try_from(&hashed_keys(inserted, |token| sz::hash(token)))
        .expect("binary-fuse build from StringZilla keys");
    let fuse_xxh = BinaryFuse8::try_from(&hashed_keys(inserted, xxh3_64))
        .expect("binary-fuse build from xxh3 keys");
    report_quality(
        "xor/xorf::BinaryFuse8<stringzilla>",
        fuse_sz.len() * 8,
        count,
        absent,
        |token| fuse_sz.contains(&sz::hash(token)),
    );
    report_quality(
        "xor/xorf::BinaryFuse8<xxh3>",
        fuse_xxh.len() * 8,
        count,
        absent,
        |token| fuse_xxh.contains(&xxh3_64(token)),
    );

    measure_build(
        "xor/xorf::BinaryFuse8::build<stringzilla>",
        budget,
        count,
        bytes,
        || {
            black_box(
                BinaryFuse8::try_from(&hashed_keys(inserted, |token| sz::hash(token))).unwrap(),
            );
        },
    );
    measure_query(
        "xor/xorf::BinaryFuse8::contains<stringzilla>",
        budget,
        inserted,
        |token| fuse_sz.contains(&sz::hash(token)),
    );
    measure_build(
        "xor/xorf::BinaryFuse8::build<xxh3>",
        budget,
        count,
        bytes,
        || {
            black_box(
                BinaryFuse8::try_from(&hashed_keys(inserted, xxh3_64)).unwrap(),
            );
        },
    );
    measure_query(
        "xor/xorf::BinaryFuse8::contains<xxh3>",
        budget,
        inserted,
        |token| fuse_xxh.contains(&xxh3_64(token)),
    );
}

/// Layer 2: build and query each filter, holding out 20% of the unique words as absent probes.
fn bench_filters(budget: &BenchBudget, unique: &[&[u8]]) {
    let inserted_count = (unique.len() * 8 / 10).clamp(1, 1_000_000);
    let inserted = &unique[..inserted_count];
    let absent = &unique[inserted_count..];
    let inserted_bytes: u64 = inserted.iter().map(|token| token.len() as u64).sum();

    println!(
        "# filters ({} inserted, {} held-out absent)",
        inserted_count,
        absent.len()
    );
    bench_bloom(budget, inserted, absent, inserted_bytes);
    bench_cuckoo(budget, inserted, absent, inserted_bytes);
    bench_xorf(budget, inserted, absent, inserted_bytes);
}

/// Asserts the amortized multi-seed path emits exactly the hashes per-seed hashing would.
fn verify_multiseed_matches_naive() {
    let probe: &[u8] = b"multiseed-correctness-probe";
    let mut amortized = [0u64; 8];
    sz::hash_multiseed_into(probe, &SEEDS[..8], &mut amortized);
    for index in 0..8 {
        assert_eq!(
            amortized[index],
            sz::hash_with_seed(probe, SEEDS[index]),
            "hash_multiseed disagrees with hash_with_seed at seed index {}",
            index
        );
    }
    println!("- multiseed == naive per-seed hashing: OK");
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();
    verify_multiseed_matches_naive();

    let tokens: BytesCowsAuto = load_dataset_with_default_mode("words").unwrap_nice();

    let mut tape = BytesTape::<u64>::new();
    tape.extend(tokens.iter())
        .expect("Failed to build BytesTape");
    let view = tape.view();
    let slices: Vec<&[u8]> = (&view).into_iter().collect();

    let unique: Vec<&[u8]> = {
        let set: HashSet<&[u8]> = slices.iter().copied().collect();
        set.into_iter().collect()
    };
    println!("- {} tokens, {} unique\n", slices.len(), unique.len());

    let budget = BenchBudget::from_env(2.0, 10.0);

    for k in [2usize, 4, 8, 16] {
        bench_multihash(&budget, k, &slices);
    }
    bench_filters(&budget, &unique);
}
