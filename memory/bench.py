# /// script
# dependencies = [
#   "stringzilla",
#   "numpy",
#   "pycryptodome",
#   "opencv-python",
# ]
# ///
"""
Python memory-centric benchmarks analogous to bench_memory.rs.

Includes two groups:
- Lookup-table transforms (256-byte LUT): bytes.translate, stringzilla.Str.translate, OpenCV LUT, NumPy indexing
- Random byte generation: NumPy PCG64, NumPy Philox, and PyCryptodome AES-CTR

Examples:
  uv run memory/bench.py --dataset README.md --tokens lines
  uv run memory/bench.py --dataset README.md --tokens words -k "translate|LUT|AES-CTR|PCG64|Philox"
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
import cv2

from utils import add_common_args, load_dataset, should_run, now_ns, tokenize_dataset


def log_system_info():
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- NumPy: {np.__version__}")
    print(f"- PyCryptoDome: {pycryptodome.__version__}")
    print(f"- OpenCV: {cv2.__version__} (defaults to {cv2.getNumThreads()} threads)")
    print()


def sz_translate_allocating(haystack: bytes, look_up_table: bytes) -> int:
    """StringZilla translation with allocation (bytes input)."""
    result = sz.translate(haystack, look_up_table)
    return len(result)


def sz_translate_inplace(haystack: memoryview, look_up_table: bytes) -> int:
    """StringZilla translation in-place (memoryview input)."""
    sz.translate(haystack, look_up_table, inplace=True)
    return len(haystack)


def bytes_translate(haystack_bytes: bytes, lut: bytes) -> int:
    """Python bytes.translate (always allocating)."""
    result = haystack_bytes.translate(lut)
    return len(result)


def opencv_lut_allocating(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """OpenCV LUT with allocation."""
    result = cv2.LUT(haystack_array, lut)
    return len(result)


def opencv_lut_inplace(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """OpenCV LUT in-place."""
    cv2.LUT(haystack_array, lut, dst=haystack_array)
    return len(haystack_array)


def numpy_lut_indexing_allocating(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """NumPy array indexing (always allocating)."""
    result = lut[haystack_array]
    return len(result)


def numpy_lut_indexing_inplace(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """NumPy array indexing in-place."""
    haystack_array[:] = lut[haystack_array]
    return len(haystack_array)


def numpy_lut_take_allocating(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """NumPy take function (always allocating)."""
    result = np.take(lut, haystack_array)
    return len(result)


def numpy_lut_take_inplace(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """NumPy take function in-place."""
    np.take(lut, haystack_array, out=haystack_array)
    return len(haystack_array)


def bench_translate(
    name: str,
    tokens,
    table: bytes,
    op: Callable[[object, bytes], int],
    time_limit_seconds: float,
) -> None:
    start_time = now_ns()
    time_limit_ns = int(time_limit_seconds * 1e9)

    requested = 0
    produced_bytes = 0
    check_frequency = 100

    for token in tokens:
        produced_bytes += op(token, table)
        requested += 1
        check_frequency -= 1

        if check_frequency == 0:
            if (now_ns() - start_time) >= time_limit_ns:
                break
            check_frequency = 100

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
    tokens_b = tokenize_dataset(data, args.tokens, unique=False)
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

    # Disable OpenCV multithreading for more consistent results
    cv2.setNumThreads(1)

    # ---------------- Lookup-table transforms ----------------
    print()
    print("--- LUT Transforms ---")

    # Create reverse LUT
    reverse = bytes(reversed(range(256)))
    reverse_np = np.arange(255, -1, -1, dtype=np.uint8)

    # Convert tokens to numpy arrays for token-based benchmarks
    tokens_np = [np.array(np.frombuffer(token, dtype=np.uint8)) for token in tokens_b]
    tokens_mv = [memoryview(bytearray(token)) for token in tokens_b]

    # Python bytes.translate (always allocating)
    if should_run("lookup-table/std.bytes.translate(new)", pattern):
        bench_translate("std.bytes.translate(new)", tokens_b, reverse, bytes_translate, args.time_limit)

    # OpenCV allocating
    if should_run("lookup-table/opencv.LUT(new)", pattern):
        bench_translate("opencv.LUT(new)", tokens_np, reverse_np, opencv_lut_allocating, args.time_limit)

    # OpenCV in-place
    if should_run("lookup-table/opencv.LUT(inplace)", pattern):
        bench_translate("opencv.LUT(inplace)", tokens_np, reverse_np, opencv_lut_inplace, args.time_limit)

    # NumPy indexing allocating
    if should_run("lookup-table/numpy.indexing(new)", pattern):
        bench_translate("numpy.indexing(new)", tokens_np, reverse_np, numpy_lut_indexing_allocating, args.time_limit)

    # NumPy indexing in-place
    if should_run("lookup-table/numpy.indexing(inplace)", pattern):
        bench_translate("numpy.indexing(inplace)", tokens_np, reverse_np, numpy_lut_indexing_inplace, args.time_limit)

    # NumPy take allocating
    if should_run("lookup-table/numpy.take(new)", pattern):
        bench_translate("numpy.take(new)", tokens_np, reverse_np, numpy_lut_take_allocating, args.time_limit)

    # NumPy take in-place
    if should_run("lookup-table/numpy.take(inplace)", pattern):
        bench_translate("numpy.take(inplace)", tokens_np, reverse_np, numpy_lut_take_inplace, args.time_limit)

    # StringZilla allocating
    if should_run("lookup-table/stringzilla.translate(new)", pattern):
        bench_translate("stringzilla.translate(new)", tokens_b, reverse, sz_translate_allocating, args.time_limit)

    # StringZilla in-place (need memoryviews for each token)
    if should_run("lookup-table/stringzilla.translate(inplace)", pattern):
        bench_translate("stringzilla.translate(inplace)", tokens_mv, reverse, sz_translate_inplace, args.time_limit)

    # ---------------- Random byte generation ----------------
    print()
    print("--- Random Byte Generation ---")
    sizes = sizes_from_tokens(tokens_b)

    if should_run("generate-random/pycryptodome.AES-CTR()", pattern):
        bench_generator("pycryptodome.AES-CTR()", sizes, make_pycryptodome_aes_ctr(), args.time_limit)
    if should_run("generate-random/stringzilla.fill_random()", pattern):
        bench_generator("stringzilla.fill_random()", sizes, make_stringzilla_fill_random(), args.time_limit)
    if should_run("generate-random/stringzilla.random()", pattern):
        bench_generator("stringzilla.random()", sizes, sz.random, args.time_limit)
    if should_run("generate-random/numpy.PCG64()", pattern):
        bench_generator("numpy.PCG64()", sizes, make_numpy_pcg64(), args.time_limit)
    if should_run("generate-random/numpy.Philox()", pattern):
        bench_generator("numpy.Philox()", sizes, make_numpy_philox(), args.time_limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
