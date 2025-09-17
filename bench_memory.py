# /// script
# dependencies = [
#   "stringzilla",
#   "numpy",
#   "pycryptodome",
# ]
# ///
"""
Python memory-centric benchmarks analogous to bench_memory.rs.

Includes two groups:
- Lookup-table transforms (256-byte LUT): bytes.translate, stringzilla.Str.translate
- Random byte generation: NumPy PCG64, NumPy Philox, and PyCryptodome AES-CTR

Examples:
  python bench_memory.py --dataset README.md --tokens lines
  python bench_memory.py --dataset README.md --tokens words -k "translate|AES-CTR|PCG64|Philox"
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Callable, Iterable, List, Optional

import stringzilla as sz
import numpy as np
import Crypto as pycryptodome
from Crypto.Cipher import AES as PyCryptoDomeAES

from utils import add_common_args, load_dataset, name_matches, now_ns, tokenize_dataset


def log_system_info():
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- NumPy: {np.__version__}")
    print(f"- PyCryptoDome: {pycryptodome.__version__}")
    print()


def sz_translate(haystack: sz.Str, look_up_table: bytes) -> int:
    # StringZilla translation using 256-byte LUT
    result = haystack.translate(look_up_table)
    return len(result)


def bytes_translate(haystack_bytes: bytes, lut: bytes) -> int:
    result = haystack_bytes.translate(lut)
    return len(result)


def bench_translate(
    name: str,
    haystack,
    tables: List[bytes],
    op: Callable[[object, bytes], int],
    time_limit_seconds: float,
) -> None:
    start_time = now_ns()
    time_limit_ns = int(time_limit_seconds * 1e9)

    requested = 0
    produced_bytes = 0

    i = 0
    while True:
        table = tables[i % len(tables)]
        produced_bytes += op(haystack, table)
        requested += 1

        if requested % 10 == 0:
            if (now_ns() - start_time) >= time_limit_ns:
                break
        i += 1

    secs = (now_ns() - start_time) / 1e9
    gbps = produced_bytes / (1e9 * secs) if secs > 0 else 0.0
    qps = requested / secs if secs > 0 else 0.0
    print(f"{name:35s}: {secs:8.3f}s ~ {gbps:8.3f} GB/s ~ {qps:10,.2f} ops/s")


def sizes_from_tokens(tokens: Iterable[bytes]) -> List[int]:
    return [len(t) for t in tokens if len(t) > 0]


def bench_generator(name: str, sizes: List[int], gen_bytes: Callable[[int], bytes], time_limit_seconds: float) -> None:
    start = now_ns()
    limit = int(time_limit_seconds * 1e9)

    processed = 0
    total_bytes = 0
    next_check = 10_000

    for n in sizes:
        _ = gen_bytes(n)
        processed += 1
        total_bytes += n
        if processed >= next_check:
            now = now_ns()
            if now - start >= limit:
                break
            next_check += 10_000

    elapsed = (now_ns() - start) / 1e9
    if elapsed == 0:
        elapsed = 1e-9
    gbs = total_bytes / (1e9 * elapsed)
    rate = processed / elapsed
    print(f"{name:35s}: {elapsed:8.3f}s ~ {gbs:8.3f} GB/s ~ {rate:10,.0f} tokens/s")


def make_pycryptodome_aes_ctr():
    key = b"\x00" * 16
    cipher = PyCryptoDomeAES.new(key, PyCryptoDomeAES.MODE_CTR, nonce=b"")

    def gen_bytes(n: int) -> bytes:
        # Generate keystream by encrypting zero bytes
        return cipher.encrypt(b"\x00" * n)

    return gen_bytes


def make_stringzilla_fill_random():
    def gen_bytes(n: int):
        buf = bytearray(n)
        sz.fill_random(buf, 0)
        return buf

    return gen_bytes


def make_numpy_pcg64():
    gen = np.random.Generator(np.random.PCG64(0))
    rr = gen.bit_generator.random_raw

    def gen_bytes(n: int) -> bytes:
        words = (n + 7) // 8
        arr64 = rr(words)
        return arr64.view(np.uint8)[:n].tobytes()

    return gen_bytes


def make_numpy_philox():
    gen = np.random.Generator(np.random.Philox(0))
    rr = gen.bit_generator.random_raw

    def gen_bytes(n: int) -> bytes:
        words = (n + 7) // 8
        arr64 = rr(words)
        return arr64.view(np.uint8)[:n].tobytes()

    return gen_bytes


_main_epilog = """
Examples:

  # Benchmark lookup-table transforms and random generation
  %(prog)s --dataset README.md --tokens lines

  # Filter to only translations
  %(prog)s --dataset README.md --tokens words -k "translate"
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Memory-related benchmarks: LUT transforms and random generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )
    add_common_args(parser)
    args = parser.parse_args()

    # Dataset
    text = load_dataset(args.dataset, as_bytes=False, size_limit=args.dataset_limit)
    data = text.encode("utf-8", errors="ignore")
    tokens_b = tokenize_dataset(data, args.tokens)
    if not tokens_b:
        print("No tokens found in dataset")
        return 1

    total_bytes = sum(len(t) for t in tokens_b)
    avg_len = total_bytes / len(tokens_b)
    print(f"Dataset: {len(tokens_b):,} tokens, {total_bytes:,} bytes, {avg_len:.1f} avg token length")
    log_system_info()

    # Compile filter
    pattern: Optional[re.Pattern[str]] = None
    if args.filter:
        try:
            pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # ---------------- Lookup-table transforms ----------------
    print("\n=== Lookup-table Transforms ===")
    identity = bytes(range(256))
    reverse = bytes(reversed(identity))
    repeated = bytes(range(64)) * 4
    hex_table = b"0123456789abcdef" * 16

    # Operate on the full contiguous string via StringZilla's view
    sz_str = sz.Str(text)
    if name_matches("stringzilla.Str.translate(reverse)", pattern):
        bench_translate("stringzilla.Str.translate(reverse)", sz_str, [reverse], sz_translate, args.time_limit)
    if name_matches("stringzilla.Str.translate(repeated)", pattern):
        bench_translate("stringzilla.Str.translate(repeated)", sz_str, [repeated], sz_translate, args.time_limit)
    if name_matches("stringzilla.Str.translate(hex)", pattern):
        bench_translate("stringzilla.Str.translate(hex)", sz_str, [hex_table], sz_translate, args.time_limit)

    # Python bytes.translate on the contiguous bytes
    if name_matches("bytes.translate(reverse)", pattern):
        bench_translate("bytes.translate(reverse)", data, [reverse], bytes_translate, args.time_limit)
    if name_matches("bytes.translate(repeated)", pattern):
        bench_translate("bytes.translate(repeated)", data, [repeated], bytes_translate, args.time_limit)
    if name_matches("bytes.translate(hex)", pattern):
        bench_translate("bytes.translate(hex)", data, [hex_table], bytes_translate, args.time_limit)

    # ---------------- Random byte generation ----------------
    print("\n=== Random Byte Generation ===")
    sizes = sizes_from_tokens(tokens_b)

    if name_matches("pycryptodome.AES-CTR", pattern):
        bench_generator("pycryptodome.AES-CTR", sizes, make_pycryptodome_aes_ctr(), args.time_limit)
    if name_matches("stringzilla.fill_random", pattern):
        bench_generator("stringzilla.fill_random", sizes, make_stringzilla_fill_random(), args.time_limit)
    if name_matches("stringzilla.random", pattern):
        bench_generator("stringzilla.random", sizes, sz.random, args.time_limit)
    if name_matches("numpy.PCG64", pattern):
        bench_generator("numpy.PCG64", sizes, make_numpy_pcg64(), args.time_limit)
    if name_matches("numpy.Philox", pattern):
        bench_generator("numpy.Philox", sizes, make_numpy_philox(), args.time_limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
