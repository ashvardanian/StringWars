# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla",
#   "rbloom",
#   "cuckoofilter",
#   "xxhash",
# ]
# ///
"""
Multi-way word hashing and probabilistic membership benchmarks, mirroring containers/bench.rs.

Layer 1 produces `k` hashes per word for `k` in {2, 4, 8, 16}:
- StringZilla `hash_multiseed` emits all `k` hashes from one input preparation.
- StringZilla `hash` called once per seed, isolating what the multi-seed path amortizes.
- One 128-bit `xxh3_128` split into `(h1, h2)` with `g_i = h1 + i*h2` (the double-hashing baseline).

Layer 2 builds a Bloom filter (rbloom, default vs StringZilla-fed) and a cuckoo filter (cuckoofilter),
then queries a held-out set to measure the false-positive rate.

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  uv run containers/bench.py --dataset xlsum.csv --tokens words
  uv run containers/bench.py --dataset data.txt --tokens words -k "multiseed"
"""

import argparse
import re
import sys
from array import array
from collections.abc import Callable

import stringzilla as sz
import xxhash
from cuckoofilter import CuckooFilter
from rbloom import Bloom

from utils import (
    add_common_args,
    load_dataset,
    now_nanoseconds,
    paced_items,
    report_stats,
    resolve_tokens,
    should_run,
    tokenize_dataset,
)

# Sixteen fixed odd seeds shared by every multi-hash variant, matching containers/bench.rs.
SEEDS = [
    0x9E3779B97F4A7C15,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0xD1B54A32D192ED03,
    0xA0761D6478BD642F,
    0xE7037ED1A0B428DB,
    0x8EBC6AF09C88C6E3,
    0x589965CC75374CC3,
    0x1D8E4E27C47D124F,
    0xEB44ACCAB455D165,
    0x2545F4914F6CDD1D,
    0xFF51AFD7ED558CCD,
    0xC4CEB9FE1A85EC53,
    0xBF58476D1CE4E5B9,
    0x94175CC1BAB35C97,
    0x4CF5AD432745937F,
]

MASK_64 = (1 << 64) - 1
TARGET_FALSE_POSITIVE_RATE = 0.01


def stringzilla_hash_128(data: bytes) -> int:
    """Two seeded StringZilla hashes combined into the signed 128-bit integer rbloom expects."""
    value = (sz.hash(data, SEEDS[0]) << 64) | sz.hash(data, SEEDS[1])
    return value - (1 << 128) if value >= (1 << 127) else value


def log_system_info():
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- xxHash: {xxhash.VERSION}")
    print()


def bench_multihash(name: str, tokens: list[bytes], produce: Callable[[bytes], object], k: int, time_limit: float):
    """Time one multi-hash variant, calling `produce` once per word to emit `k` hashes."""
    start = now_nanoseconds()
    deadline = start + int(time_limit * 1e9)
    count = 0
    total_bytes = 0
    for token in paced_items(tokens, deadline):
        produce(token)
        count += 1
        total_bytes += len(token)
    seconds = (now_nanoseconds() - start) / 1e9
    report_stats(name, "hashes", seconds, count * k, total_bytes)


def bench_build(name: str, count: int, total_bytes: int, build: Callable[[], object], time_limit: float):
    """Time filter construction, rebuilding from every key on each pass."""
    start = now_nanoseconds()
    deadline = start + int(time_limit * 1e9)
    passes = 0
    while True:
        build()
        passes += 1
        if now_nanoseconds() >= deadline:
            break
    seconds = (now_nanoseconds() - start) / 1e9
    report_stats(name, "hashes", seconds, passes * count, passes * total_bytes)


def bench_query(name: str, probes: list[bytes], query: Callable[[bytes], bool], time_limit: float):
    """Time membership queries, cycling one probe per call."""
    start = now_nanoseconds()
    deadline = start + int(time_limit * 1e9)
    count = 0
    total_bytes = 0
    for token in paced_items(probes, deadline):
        query(token)
        count += 1
        total_bytes += len(token)
    seconds = (now_nanoseconds() - start) / 1e9
    report_stats(name, "hashes", seconds, count, total_bytes)


def report_quality(
    label: str, num_bits: int, inserted_count: int, absent: list[bytes], contains: Callable[[bytes], bool]
):
    """Report the measured false-positive rate over the held-out absent words plus bits-per-key."""
    false_positives = sum(1 for token in absent if contains(token))
    rate = false_positives / len(absent) * 100 if absent else 0.0
    bits = f"{num_bits / max(inserted_count, 1):5.2f} bits/key" if num_bits else "    n/a    "
    print(f"    {label:<38} {bits}, measured FPR {rate:.3f}%")


def run_multihash(tokens: list[bytes], filter_pattern: re.Pattern | None, time_limit: float):
    for k in (2, 4, 8, 16):
        print(f"# multihash (k={k})")
        seeds = array("Q", SEEDS[:k])
        out = array("Q", [0] * k)

        if should_run("multihash/stringzilla.hash_multiseed", filter_pattern):
            bench_multihash(
                "multihash/stringzilla.hash_multiseed",
                tokens,
                lambda token, seeds=seeds, out=out: sz.hash_multiseed(token, seeds, out),
                k,
                time_limit,
            )
        if should_run("multihash/stringzilla.hash", filter_pattern):
            bench_multihash(
                "multihash/stringzilla.hash",
                tokens,
                lambda token, seeds=seeds: [sz.hash(token, seed) for seed in seeds],
                k,
                time_limit,
            )
        if should_run("multihash/xxhash.xxh3_128", filter_pattern):

            def double_hash(token, k=k):
                wide = xxhash.xxh3_128(token).intdigest()
                first, second = wide & MASK_64, wide >> 64
                return [(first + index * second) & MASK_64 for index in range(k)]

            bench_multihash("multihash/xxhash.xxh3_128", tokens, double_hash, k, time_limit)


def run_filters(unique: list[bytes], filter_pattern: re.Pattern | None, time_limit: float):
    inserted_count = min(len(unique) * 8 // 10, 1_000_000) or 1
    inserted = unique[:inserted_count]
    absent = unique[inserted_count:]
    inserted_bytes = sum(len(token) for token in inserted)
    print(f"# filters ({inserted_count} inserted, {len(absent)} held-out absent)")

    # Bloom filter (rbloom): default Python hash versus a StringZilla-fed hash function.
    bloom = Bloom(inserted_count, TARGET_FALSE_POSITIVE_RATE)
    for token in inserted:
        bloom.add(token)
    report_quality("bloom/rbloom<siphash>", bloom.size_in_bits, inserted_count, absent, lambda t: t in bloom)
    if should_run("bloom/rbloom.insert<siphash>", filter_pattern):

        def build_bloom():
            filter_ = Bloom(inserted_count, TARGET_FALSE_POSITIVE_RATE)
            for token in inserted:
                filter_.add(token)

        bench_build("bloom/rbloom.insert<siphash>", inserted_count, inserted_bytes, build_bloom, time_limit)
    if should_run("bloom/rbloom.contains<siphash>", filter_pattern):
        bench_query("bloom/rbloom.contains<siphash>", inserted, lambda t: t in bloom, time_limit)

    bloom_sz = Bloom(inserted_count, TARGET_FALSE_POSITIVE_RATE, stringzilla_hash_128)
    for token in inserted:
        bloom_sz.add(token)
    report_quality("bloom/rbloom<stringzilla>", bloom_sz.size_in_bits, inserted_count, absent, lambda t: t in bloom_sz)
    if should_run("bloom/rbloom.insert<stringzilla>", filter_pattern):

        def build_bloom_sz():
            filter_ = Bloom(inserted_count, TARGET_FALSE_POSITIVE_RATE, stringzilla_hash_128)
            for token in inserted:
                filter_.add(token)

        bench_build("bloom/rbloom.insert<stringzilla>", inserted_count, inserted_bytes, build_bloom_sz, time_limit)
    if should_run("bloom/rbloom.contains<stringzilla>", filter_pattern):
        bench_query("bloom/rbloom.contains<stringzilla>", inserted, lambda t: t in bloom_sz, time_limit)

    # Cuckoo filter (cuckoofilter): pure-Python, hashes each key internally.
    cuckoo = CuckooFilter(capacity=inserted_count, fingerprint_size=2)
    for token in inserted:
        cuckoo.insert(token)
    report_quality("cuckoo/cuckoofilter", 0, inserted_count, absent, lambda t: cuckoo.contains(t))
    if should_run("cuckoo/cuckoofilter.insert", filter_pattern):

        def build_cuckoo():
            filter_ = CuckooFilter(capacity=inserted_count, fingerprint_size=2)
            for token in inserted:
                filter_.insert(token)

        bench_build("cuckoo/cuckoofilter.insert", inserted_count, inserted_bytes, build_cuckoo, time_limit)
    if should_run("cuckoo/cuckoofilter.contains", filter_pattern):
        bench_query("cuckoo/cuckoofilter.contains", inserted, lambda t: cuckoo.contains(t), time_limit)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multi-way word hashing and probabilistic membership structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()

    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as error:
            parser.error(f"Invalid regex for --filter: {error}")

    dataset = load_dataset(args.dataset, as_bytes=True, size_limit=args.dataset_limit)
    tokens = tokenize_dataset(dataset, resolve_tokens(args.tokens, "words"))
    if not tokens:
        print("No tokens found in dataset")
        return 1

    unique = list(dict.fromkeys(tokens))
    total_bytes = sum(len(token) for token in tokens)
    print(f"Dataset: {len(tokens):,} tokens, {total_bytes:,} bytes, {len(unique):,} unique")
    log_system_info()

    run_multihash(tokens, filter_pattern, args.time_limit)
    run_filters(unique, filter_pattern, args.time_limit)
    return 0


if __name__ == "__main__":
    exit(main())
