# /// script
# dependencies = [
#   "stringzilla",
#   "xxhash",
#   "blake3",
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
  python bench_hash.py --dataset test.txt --tokens words -k "xxhash"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines python bench_hash.py
"""

import argparse
import re
from typing import List, Optional, Callable, Any

import blake3
import stringzilla as sz
import xxhash

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns


def bench_hash_function(
    name: str,
    tokens: List[bytes],
    hash_func: Callable[[bytes], Any],
    stateful: bool = False,
) -> None:
    """Benchmark a hash function and report throughput."""
    total_bytes = sum(len(token) for token in tokens)

    start_time = now_ns()

    if stateful:
        # Simulate stateful/incremental hashing by accumulating results
        result = 0
        for token in tokens:
            result ^= hash(hash_func(token))  # XOR to prevent optimization
    else:
        # Stateless: hash each token independently
        for token in tokens:
            _ = hash_func(token)

    end_time = now_ns()

    duration_secs = (end_time - start_time) / 1e9
    throughput_gbs = total_bytes / (1e9 * duration_secs)
    tokens_per_sec = len(tokens) / duration_secs

    print(f"{name:25s}: {duration_secs:8.3f}s ~ {throughput_gbs:8.3f} GB/s ~ {tokens_per_sec:10,.0f} tokens/s")


def name_matches(name: str, pattern: Optional[re.Pattern]) -> bool:
    """Check if benchmark name matches filter pattern."""
    return True if pattern is None else bool(pattern.search(name))


def run_stateless_benchmarks(tokens: List[bytes], filter_pattern: Optional[re.Pattern] = None):
    """Run stateless hash benchmarks (hash each token independently)."""
    print("\n=== Stateless Hash Benchmarks ===")

    # Python built-in hash
    if name_matches("hash", filter_pattern):
        bench_hash_function("hash", tokens, lambda x: hash(x))

    # xxHash
    if name_matches("xxhash.xxh3_64", filter_pattern):
        bench_hash_function("xxhash.xxh3_64", tokens, lambda x: xxhash.xxh3_64(x).intdigest())

    # StringZilla hashes
    if name_matches("stringzilla.hash", filter_pattern):
        bench_hash_function("stringzilla.hash", tokens, lambda x: sz.hash(x))

    # Reference bounds
    if name_matches("blake3.digest", filter_pattern):
        bench_hash_function("blake3.digest", tokens, lambda x: blake3.blake3(x).digest())

    if name_matches("stringzilla.bytesum", filter_pattern):
        bench_hash_function("stringzilla.bytesum", tokens, lambda x: sz.bytesum(x))


def bench_stateful_hash(name: str, tokens: List[bytes], hasher_factory: Callable) -> None:
    """Benchmark a stateful hash function and report throughput."""
    total_bytes = sum(len(token) for token in tokens)

    start_time = now_ns()
    hasher = hasher_factory()
    for token in tokens:
        hasher.update(token)
    result = hasher.digest() if hasattr(hasher, "digest") else hasher.intdigest()
    end_time = now_ns()

    duration_secs = (end_time - start_time) / 1e9
    throughput_gbs = total_bytes / (1e9 * duration_secs)
    tokens_per_sec = len(tokens) / duration_secs
    print(f"{name:25s}: {duration_secs:8.3f}s ~ {throughput_gbs:8.3f} GB/s ~ {tokens_per_sec:10,.0f} tokens/s")


def run_stateful_benchmarks(tokens: List[bytes], filter_pattern: Optional[re.Pattern] = None):
    """Run stateful hash benchmarks (incremental/streaming hashing)."""
    print("\n=== Stateful Hash Benchmarks ===")

    # xxHash stateful
    if name_matches("xxhash.xxh3_64", filter_pattern):
        bench_stateful_hash("xxhash.xxh3_64", tokens, lambda: xxhash.xxh3_64())

    # StringZilla stateful hasher
    if name_matches("stringzilla.Hasher", filter_pattern):
        bench_stateful_hash("stringzilla.Hasher", tokens, lambda: sz.Hasher())


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark hash functions with StringZilla and other implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Load and tokenize dataset
    try:
        dataset = load_dataset(args.dataset, as_bytes=True)
        tokens = tokenize_dataset(dataset, args.tokens)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    if not tokens:
        print("No tokens found in dataset")
        return 1

    # Compile filter pattern
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            print(f"Invalid regex for --filter: {e}")
            return 1

    # Report dataset info
    total_bytes = sum(len(token) for token in tokens)
    avg_token_length = total_bytes / len(tokens) if tokens else 0
    print(f"Dataset: {len(tokens):,} tokens, {total_bytes:,} bytes, {avg_token_length:.1f} avg token length")

    # Run benchmarks
    run_stateless_benchmarks(tokens, filter_pattern)
    run_stateful_benchmarks(tokens, filter_pattern)

    return 0


if __name__ == "__main__":
    exit(main())
