# /// script
# dependencies = [
#   "stringzilla",
#   "xxhash",
#   "blake3",
#   "google-crc32c",
#   "mmh3",
#   "cityhash",
# ]
# ///
"""
Python hash function benchmarks comparing various implementations.

Benchmarks hash function performance using consistent methodology with the Rust
bench_hash.rs implementation, focusing on both stateless and stateful hashing patterns.

Hash functions compared:
- Built-in Python: hash(), hashlib (MD5, SHA1, SHA256, Blake2b, Blake2s)
- StringZilla: sz.hash(), sz.bytesum() for performance comparison
- xxHash: xxh3_64, xxh64, xxh32 variants
- Blake3: Modern cryptographic hash for reference

Benchmark categories:
- Stateless: Hash each token independently
- Stateful: Incremental hashing across all tokens
- Crypto: Cryptographic hash functions (separate timing)

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  python bench_hash.py --dataset README.md --tokens lines
  python bench_hash.py --dataset xlsum.csv --tokens words -k "xxhash"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines python bench_hash.py
"""

import argparse
import re
import sys
from typing import List, Optional, Callable, Any
from importlib.metadata import version as pkg_version

import blake3
import stringzilla as sz
import xxhash
import google_crc32c
import mmh3
import cityhash

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns, name_matches


def log_system_info():
    """Log Python version and hash library versions."""
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- xxHash: {xxhash.VERSION}")
    print(f"- Blake3: {blake3.__version__}")
    print(f"- google-crc32c: {pkg_version('google-crc32c')}")
    print(f"- mmh3: {pkg_version('mmh3')}")
    print(f"- cityhash: {pkg_version('cityhash')}")
    print()  # Add blank line


def bench_hash_function(
    name: str,
    tokens: List[bytes],
    hash_func: Callable[[bytes], Any],
    time_limit_seconds: float = 10.0,
) -> None:
    """
    Benchmark a stateless hash function and report throughput.

    Processes tokens until time limit is reached, then reports results.
    """
    start_time = now_ns()
    time_limit_ns = int(time_limit_seconds * 1e9)

    processed_tokens = 0
    processed_bytes = 0

    # Stateless: hash each token independently
    next_check = 10_000
    for token in tokens:
        _ = hash_func(token)
        processed_tokens += 1
        processed_bytes += len(token)

        # Check time limit every 10,000 tokens
        if processed_tokens >= next_check:
            current_time = now_ns()
            if current_time >= start_time + time_limit_ns:
                break
            next_check += 10_000

    end_time = now_ns()

    duration_secs = (end_time - start_time) / 1e9
    throughput_gbs = processed_bytes / (1e9 * duration_secs)
    tokens_per_sec = processed_tokens / duration_secs

    print(f"{name:35s}: {duration_secs:8.3f}s ~ {throughput_gbs:8.3f} GB/s ~ {tokens_per_sec:10,.0f} tokens/s")


def run_stateless_benchmarks(
    tokens: List[bytes],
    filter_pattern: Optional[re.Pattern] = None,
    time_limit_seconds: float = 10.0,
):
    """Run stateless hash benchmarks (hash each token independently)."""
    print("\n=== Stateless Hash Benchmarks ===")

    # Python built-in hash
    if name_matches("hash", filter_pattern):
        bench_hash_function("hash", tokens, lambda x: hash(x), time_limit_seconds)

    # xxHash
    if name_matches("xxhash.xxh3_64", filter_pattern):
        bench_hash_function("xxhash.xxh3_64", tokens, lambda x: xxhash.xxh3_64(x).intdigest(), time_limit_seconds)

    # StringZilla hashes
    if name_matches("stringzilla.hash", filter_pattern):
        bench_hash_function("stringzilla.hash", tokens, lambda x: sz.hash(x), time_limit_seconds)

    # Google CRC32C (Castagnoli) one-shot
    if name_matches("google_crc32c.value", filter_pattern):
        bench_hash_function("google_crc32c.value", tokens, lambda x: google_crc32c.value(x), time_limit_seconds)

    # MurmurHash3 — stateless
    if name_matches("mmh3.hash32", filter_pattern):
        bench_hash_function("mmh3.hash32", tokens, lambda x: mmh3.hash(x, signed=False), time_limit_seconds)
    if name_matches("mmh3.hash64", filter_pattern):
        bench_hash_function("mmh3.hash64", tokens, lambda x: mmh3.hash64(x, signed=False)[0], time_limit_seconds)
    if name_matches("mmh3.hash128", filter_pattern):
        bench_hash_function("mmh3.hash128", tokens, lambda x: mmh3.hash128(x, signed=False), time_limit_seconds)

    # CityHash — stateless
    if name_matches("cityhash.CityHash64", filter_pattern):
        bench_hash_function("cityhash.CityHash64", tokens, lambda x: cityhash.CityHash64(x), time_limit_seconds)
    if name_matches("cityhash.CityHash128", filter_pattern):
        bench_hash_function("cityhash.CityHash128", tokens, lambda x: cityhash.CityHash128(x), time_limit_seconds)

    # Reference bounds
    if name_matches("blake3.digest", filter_pattern):
        bench_hash_function("blake3.digest", tokens, lambda x: blake3.blake3(x).digest(), time_limit_seconds)

    if name_matches("stringzilla.bytesum", filter_pattern):
        bench_hash_function("stringzilla.bytesum", tokens, lambda x: sz.bytesum(x), time_limit_seconds)


def bench_stateful_hash(
    name: str,
    tokens: List[bytes],
    hasher_factory: Callable,
    time_limit_seconds: float = 10.0,
) -> None:
    """Benchmark a stateful hash function and report throughput."""
    start_time = now_ns()
    time_limit_ns = int(time_limit_seconds * 1e9)

    processed_tokens = 0
    processed_bytes = 0

    hasher = hasher_factory()
    next_check = 10_000
    for token in tokens:
        hasher.update(token)
        processed_tokens += 1
        processed_bytes += len(token)

        # Check time limit every 10,000 tokens
        if processed_tokens >= next_check:
            current_time = now_ns()
            if current_time >= start_time + time_limit_ns:
                break
            next_check += 10_000

    result = hasher.digest() if hasattr(hasher, "digest") else hasher.intdigest()
    end_time = now_ns()

    duration_secs = (end_time - start_time) / 1e9
    throughput_gbs = processed_bytes / (1e9 * duration_secs)
    tokens_per_sec = processed_tokens / duration_secs
    print(f"{name:35s}: {duration_secs:8.3f}s ~ {throughput_gbs:8.3f} GB/s ~ {tokens_per_sec:10,.0f} tokens/s")


def run_stateful_benchmarks(
    tokens: List[bytes],
    filter_pattern: Optional[re.Pattern] = None,
    time_limit_seconds: float = 10.0,
):
    """Run stateful hash benchmarks (incremental/streaming hashing)."""
    print("\n=== Stateful Hash Benchmarks ===")

    # xxHash stateful
    if name_matches("xxhash.xxh3_64", filter_pattern):
        bench_stateful_hash("xxhash.xxh3_64", tokens, lambda: xxhash.xxh3_64(), time_limit_seconds)

    # StringZilla stateful hasher
    if name_matches("stringzilla.Hasher", filter_pattern):
        bench_stateful_hash("stringzilla.Hasher", tokens, lambda: sz.Hasher(), time_limit_seconds)

    # Google CRC32C (Castagnoli) stateful
    if name_matches("google_crc32c.Checksum", filter_pattern):
        bench_stateful_hash("google_crc32c.Checksum", tokens, lambda: google_crc32c.Checksum(), time_limit_seconds)


_main_epilog = """
Examples:

  # Benchmark all hash functions with default settings
  %(prog)s --dataset README.md --tokens lines

  # Test only specific hash functions
  %(prog)s --dataset data.txt --tokens lines -k "xxhash|stringzilla"

  # Compare stateless vs stateful hashing
  %(prog)s --dataset large.txt --tokens words -k "hash"

  # Test cryptographic hash performance
  %(prog)s --dataset text.txt --tokens lines -k "blake3"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark hash functions with StringZilla and other implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Compile filter pattern
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Load and tokenize dataset
    dataset = load_dataset(args.dataset, as_bytes=True, size_limit=args.dataset_limit)
    tokens = tokenize_dataset(dataset, args.tokens)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    # Report dataset info
    total_bytes = sum(len(token) for token in tokens)
    avg_token_length = total_bytes / len(tokens) if tokens else 0
    print(f"Dataset: {len(tokens):,} tokens, {total_bytes:,} bytes, {avg_token_length:.1f} avg token length")
    log_system_info()

    # Run benchmarks
    run_stateless_benchmarks(tokens, filter_pattern, args.time_limit)
    run_stateful_benchmarks(tokens, filter_pattern, args.time_limit)

    return 0


if __name__ == "__main__":
    exit(main())
