"""Hash-container benchmarks in Python: multi-seed digests and filters. Mirrors `containers/bench.rs`.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group containers containers/bench.py
"""

import argparse
from array import array
from collections.abc import Callable

import numpy as np
import stringzilla as sz
import xxhash
from probables import BloomFilter

from stringwars import (
    Bytes,
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


def bench_multihash(
    settings: Settings,
    name: str,
    tokens: list[bytes],
    produce: Callable[[bytes], object],
    digest_bits: int,
    total_bytes: Bytes,
) -> None:
    """One pass emits `digest_bits` digest bits for every token."""
    work = MeasureSpec(
        unit="bits",
        elements=len(tokens) * digest_bits,
        total_bytes=total_bytes,
    )
    measure(settings, name, work, pass_over(produce, tokens))


def bench_build(settings: Settings, name: str, count: int, total_bytes: Bytes, build: Callable[[], object]) -> None:
    """One pass rebuilds the whole filter from every key."""
    work = MeasureSpec(unit="hashes", elements=count, total_bytes=total_bytes)
    measure(settings, name, work, build)


def bench_query(
    settings: Settings, name: str, probes: list[bytes], total_bytes: Bytes, query: Callable[[bytes], bool]
) -> None:
    """One pass queries every probe."""
    work = MeasureSpec(unit="hashes", elements=len(probes), total_bytes=total_bytes)
    measure(settings, name, work, pass_over(query, probes))


def report_quality(
    label: str,
    num_bits: int,
    inserted_count: int,
    absent: list[bytes],
    contains: Callable[[bytes], bool],
) -> None:
    """Report the measured false-positive rate over the held-out absent words plus bits-per-key."""
    false_positives = sum(1 for token in absent if contains(token))
    rate = false_positives / len(absent) * 100 if absent else 0.0
    bits = f"{num_bits / max(inserted_count, 1):5.2f} bits/key" if num_bits else "    n/a    "
    print(f"    {label:<38} {bits}, measured FPR {rate:.3f}%")


def run_multihash(settings: Settings, tokens: list[bytes], total_bytes: Bytes) -> None:
    for digest_bits in (128, 256, 512, 1024):
        print(f"# multihash ({digest_bits}-bit digest)")
        sz_hashes = digest_bits // 64
        xxh3_calls = digest_bits // 128
        seeds = array("Q", SEEDS[:sz_hashes])
        # One digest buffer reused every call, so no variant pays per-call allocation.
        # `array("Q")` and not numpy: the two per-dimension variants below store into it
        # from Python, and a numpy scalar `__setitem__` costs 43.5 ns against the 28.4 ns
        # hash it is storing — the comparison would be about numpy, not about hashing.
        out = array("Q", [0] * sz_hashes)

        def sz_hash_multiseed(token: bytes, seeds: array[int] = seeds, out: array[int] = out) -> None:
            sz.hash_multiseed(token, seeds, out)

        bench_multihash(
            settings,
            f"multihash-{digest_bits}/stringzilla.hash_multiseed",
            tokens,
            sz_hash_multiseed,
            digest_bits,
            total_bytes,
        )

        def sz_hash_fill(token: bytes, seeds: array[int] = seeds, out: array[int] = out) -> None:
            for index, seed in enumerate(seeds):
                out[index] = sz.hash(token, seed)

        bench_multihash(
            settings,
            f"multihash-{digest_bits}/stringzilla.hash",
            tokens,
            sz_hash_fill,
            digest_bits,
            total_bytes,
        )

        # One full 128-bit xxh3 hash per seed — every bit is independent, no double-hashing —
        # split across two 64-bit slots of the shared buffer.
        def xxh3_fill(token: bytes, n: int = xxh3_calls, out: array[int] = out) -> None:
            for index in range(n):
                wide = xxhash.xxh3_128_intdigest(token, seed=SEEDS[index])
                out[2 * index] = wide & MASK_64
                out[2 * index + 1] = wide >> 64

        bench_multihash(
            settings, f"multihash-{digest_bits}/xxhash.xxh3_128", tokens, xxh3_fill, digest_bits, total_bytes
        )


def run_filters(settings: Settings, unique: list[bytes]) -> None:
    inserted_count = min(len(unique) * 8 // 10, 1_000_000) or 1
    inserted = unique[:inserted_count]
    absent = unique[inserted_count:]
    inserted_bytes = Bytes(sum(len(token) for token in inserted))
    print(f"# filters ({inserted_count} inserted, {len(absent)} held-out absent)")

    # pyprobables Bloom filter, hashing each key with its default FNV-1a internally.
    bloom = BloomFilter(est_elements=inserted_count, false_positive_rate=TARGET_FALSE_POSITIVE_RATE)
    for token in inserted:
        bloom.add(token)
    report_quality("bloom/pyprobables<fnv>", bloom.number_bits, inserted_count, absent, bloom.check)

    def build_bloom() -> None:
        filter_ = BloomFilter(est_elements=inserted_count, false_positive_rate=TARGET_FALSE_POSITIVE_RATE)
        pass_over(filter_.add, inserted)()

    bench_build(settings, "bloom/pyprobables.add<fnv>", inserted_count, inserted_bytes, build_bloom)
    bench_query(settings, "bloom/pyprobables.check<fnv>", inserted, inserted_bytes, bloom.check)

    # Same filter, fed StringZilla's `hash_multiseed` digest through the precomputed-hash API: one
    # native call fills the reused buffer, then `add_alt`/`check_alt` consume it — no per-key callback.
    bloom_sz = BloomFilter(est_elements=inserted_count, false_positive_rate=TARGET_FALSE_POSITIVE_RATE)
    seeds = array("Q", SEEDS[: bloom_sz.number_hashes])
    out = np.empty(bloom_sz.number_hashes, dtype=np.uint64)

    def sz_digest(token: bytes) -> np.ndarray:
        sz.hash_multiseed(token, seeds, out)
        return out

    for token in inserted:
        bloom_sz.add_alt(sz_digest(token))
    report_quality(
        "bloom/pyprobables<stringzilla>",
        bloom_sz.number_bits,
        inserted_count,
        absent,
        lambda t: bloom_sz.check_alt(sz_digest(t)),
    )

    def build_bloom_sz() -> None:
        filter_ = BloomFilter(est_elements=inserted_count, false_positive_rate=TARGET_FALSE_POSITIVE_RATE)
        pass_over(lambda token: filter_.add_alt(sz_digest(token)), inserted)()

    bench_build(settings, "bloom/pyprobables.add<stringzilla>", inserted_count, inserted_bytes, build_bloom_sz)
    bench_query(
        settings,
        "bloom/pyprobables.check<stringzilla>",
        inserted,
        inserted_bytes,
        lambda t: bloom_sz.check_alt(sz_digest(t)),
    )


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    print_machine({"StringZilla": f"{sz.__version__} with {sz.__capabilities_str__}", "xxHash": xxhash.VERSION})
    settings = read_settings("containers")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    log_dataset(dataset)
    log_timing_overhead(settings)
    tokens = dataset.tokens

    run_multihash(settings, tokens, dataset.token_bytes)
    run_filters(settings, list(dict.fromkeys(tokens)))
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
