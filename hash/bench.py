"""Hash benchmarks in Python: stateless, stateful and checksum digests. Mirrors `hash/bench.rs`.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group hash hash/bench.py
"""

import argparse
import hashlib
from collections.abc import Callable
from importlib.metadata import version as pkg_version
from typing import Any

import blake3
import cityhash
import google_crc32c
import mmh3
import stringzilla as sz
import xxhash

from stringwars import (
    MeasureSpec,
    Settings,
    finish,
    log_dataset,
    log_timing_overhead,
    measure,
    pass_over,
    print_machine,
    print_settings,
    read_settings,
    resolve_dataset,
)


def bench_hash_function(
    settings: Settings,
    name: str,
    tokens: list[bytes],
    hash_func: Callable[[bytes], Any],
    work: MeasureSpec,
) -> None:
    """
    Benchmark a stateless hash function over the whole working set.

    One pass hashes every token, driven from C so the interpreter never appears in
    the measured region — a Python-level loop body costs ~50-80 ns per item and
    would be attributed to the kernel. For the same reason `hash_func` is the
    library's own callable wherever one call suffices: an identity wrapper lambda
    measured 1.9x on `sz.hash` and on `xxh3`, and only some rows were paying it.
    """
    measure(settings, name, work, pass_over(hash_func, tokens))


def run_stateless_benchmarks(
    settings: Settings,
    tokens: list[bytes],
    work: MeasureSpec,
) -> None:
    print("\nStateless Hash Benchmarks")

    # No built-in `hash` row: CPython caches a `bytes` object's hash inside the object, so
    # every pass after the first reads the cache instead of hashing. Measured 5.19 GB/s
    # computing against 63 GB/s re-reading, and since warm-up discards the one real pass
    # the row reported 157 GB/s of cache lookups. Nothing here can be compared against
    # contenders that recompute, so the row is gone rather than misleading.

    # xxHash
    bench_hash_function(settings, "stateless/xxhash.xxh3_64", tokens, xxhash.xxh3_64_intdigest, work)

    # StringZilla hashes
    bench_hash_function(settings, "stateless/stringzilla.hash", tokens, sz.hash, work)

    # Google CRC32C (Castagnoli) one-shot
    bench_hash_function(settings, "stateless/google_crc32c.value", tokens, google_crc32c.value, work)

    # MurmurHash3 — stateless
    bench_hash_function(settings, "stateless/mmh3.hash32", tokens, lambda x: mmh3.hash(x, signed=False), work)
    bench_hash_function(settings, "stateless/mmh3.hash64", tokens, lambda x: mmh3.hash64(x, signed=False)[0], work)
    bench_hash_function(settings, "stateless/mmh3.hash128", tokens, lambda x: mmh3.hash128(x, signed=False), work)

    # CityHash — stateless
    bench_hash_function(settings, "stateless/cityhash.CityHash64", tokens, cityhash.CityHash64, work)
    bench_hash_function(settings, "stateless/cityhash.CityHash128", tokens, cityhash.CityHash128, work)


def bench_stateful_hash(
    settings: Settings,
    name: str,
    tokens: list[bytes],
    hasher_factory: Callable[[], Any],
    work: MeasureSpec,
) -> None:
    """
    Benchmark a stateful hash by streaming the whole working set through one hasher.

    The hasher is rebuilt per pass so every sample does identical work; the old
    version built it once for the entire run, which is not what the Rust side does.
    """

    def one_pass() -> None:
        hasher = hasher_factory()
        pass_over(hasher.update, tokens)()
        hasher.digest() if hasattr(hasher, "digest") else hasher.intdigest()

    measure(settings, name, work, one_pass)


def run_stateful_benchmarks(
    settings: Settings,
    tokens: list[bytes],
    work: MeasureSpec,
) -> None:
    print("\nStateful Hash Benchmarks")

    # xxHash stateful
    bench_stateful_hash(settings, "stateful/xxhash.xxh3_64", tokens, lambda: xxhash.xxh3_64(), work)

    # StringZilla stateful hasher
    bench_stateful_hash(settings, "stateful/stringzilla.Hasher", tokens, lambda: sz.Hasher(), work)

    # Google CRC32C (Castagnoli) stateful
    bench_stateful_hash(settings, "stateful/google_crc32c.Checksum", tokens, lambda: google_crc32c.Checksum(), work)


def run_checksum_benchmarks(
    settings: Settings,
    tokens: list[bytes],
    work: MeasureSpec,
) -> None:
    print("\nChecksum Hash Benchmarks")

    # StringZilla bytesum - reference lower bound
    bench_hash_function(settings, "checksum/stringzilla.bytesum", tokens, sz.bytesum, work)

    # Blake3 - cryptographic hash
    bench_hash_function(settings, "checksum/blake3.blake3", tokens, lambda x: blake3.blake3(x).digest(), work)

    # SHA256 via hashlib (Python standard library)
    bench_hash_function(settings, "checksum/hashlib.sha256", tokens, lambda x: hashlib.sha256(x).digest(), work)

    # SHA256 via StringZilla
    bench_hash_function(settings, "checksum/stringzilla.Sha256", tokens, lambda x: sz.Sha256().update(x).digest(), work)


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    print_machine(
        {
            "StringZilla": f"{sz.__version__} with {sz.__capabilities_str__}",
            "xxHash": xxhash.VERSION,
            "Blake3": blake3.__version__,
            "google-crc32c": pkg_version("google-crc32c"),
            "mmh3": pkg_version("mmh3"),
            "cityhash": pkg_version("cityhash"),
        }
    )
    settings = read_settings("hash")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    tokens = dataset.tokens
    log_dataset(dataset)

    work = MeasureSpec(unit="bytes", elements=dataset.token_count, total_bytes=dataset.token_bytes)
    log_timing_overhead(settings)
    run_stateless_benchmarks(settings, tokens, work)
    run_stateful_benchmarks(settings, tokens, work)
    run_checksum_benchmarks(settings, tokens, work)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
