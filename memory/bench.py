"""Low-level memory benchmarks in Python: lookup tables, PRNG fills, copies. Mirrors `memory/bench.rs`.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group memory memory/bench.py
"""

import argparse
from collections.abc import Callable, Iterable, Sequence
from itertools import repeat
from typing import Any

import Crypto as pycryptodome
import cv2
import numpy as np
import stringzilla as sz
from Crypto.Cipher import AES as PyCryptoDomeAES

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


def sz_translate_allocating(haystack: bytes, look_up_table: bytes) -> int:
    """StringZilla translation with allocation (bytes input)."""
    result = sz.translate(haystack, look_up_table)
    return len(result)


def sz_translate_inplace(haystack: memoryview, look_up_table: bytes) -> int:
    """StringZilla translation in-place (memoryview input)."""
    sz.translate(haystack, look_up_table, inplace=True)
    return len(haystack)


def bytes_translate(haystack_bytes: bytes, lut: bytes) -> int:
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
    settings: Settings,
    name: str,
    tokens: Sequence[Any],
    table: bytes | np.ndarray,
    operation: Callable[[Any, Any], int],
) -> None:
    # The broadcast table column used to be built *inside* the timed region, so a
    # multi-million-element list allocation was charged to every contender.
    work = MeasureSpec(
        unit="bytes",
        elements=len(tokens),
        total_bytes=Bytes(sum(len(token) for token in tokens)),
    )
    measure(settings, name, work, pass_over(operation, tokens, repeat(table)))


def sizes_from_tokens(tokens: Iterable[bytes]) -> list[int]:
    return [len(token) for token in tokens if len(token) > 0]


def bench_generator(settings: Settings, name: str, sizes: list[int], generate_bytes: Callable[[int], object]) -> None:
    work = MeasureSpec(unit="bytes", elements=len(sizes), total_bytes=Bytes(sum(sizes)))
    measure(settings, name, work, pass_over(generate_bytes, sizes))


def make_pycryptodome_aes_ctr() -> Callable[[int], bytes]:
    key = b"\x00" * 16
    cipher = PyCryptoDomeAES.new(key, PyCryptoDomeAES.MODE_CTR, nonce=b"")

    # This row does pay an input allocation no other generator pays, but preallocating
    # buffers and passing `output=` measured *slower* (-17% at 9 B, -13% at 100 B): two
    # `memoryview` slices per call cost more than `b"\x00" * size`, which pymalloc serves
    # from its freelist.
    def generate_bytes(size: int) -> bytes:
        return cipher.encrypt(b"\x00" * size)

    return generate_bytes


def make_stringzilla_fill_random() -> Callable[[int], bytearray]:
    def generate_bytes(size: int) -> bytearray:
        buffer = bytearray(size)
        sz.fill_random(buffer, 0)
        return buffer

    return generate_bytes


def make_numpy_generator(bit_generator: np.random.BitGenerator) -> Callable[[int], np.ndarray]:
    """A byte generator over a NumPy PRNG's raw 64-bit words.

    The trailing slice is a view, not a `tobytes()` copy: the copy walked the whole buffer a
    second time and was charged to these two rows alone, and `pass_over` discards the value.
    """
    random_raw = np.random.Generator(bit_generator).bit_generator.random_raw

    def generate_bytes(size: int) -> np.ndarray:
        return random_raw((size + 7) // 8).view(np.uint8)[:size]

    return generate_bytes


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    print_machine(
        {
            "StringZilla": f"{sz.__version__} with {sz.__capabilities_str__}",
            "NumPy": np.__version__,
            "PyCryptoDome": pycryptodome.__version__,
            "OpenCV": f"{cv2.__version__}, pinned to 1 thread",
        }
    )
    settings = read_settings("memory")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    tokens_b = dataset.tokens
    log_dataset(dataset)
    log_timing_overhead(settings)

    # Disable OpenCV multithreading for more consistent results
    cv2.setNumThreads(1)

    # Lookup-table transforms
    print()
    print("LUT Transforms")

    reverse = bytes(reversed(range(256)))
    reverse_np = np.arange(255, -1, -1, dtype=np.uint8)

    tokens_np = [np.array(np.frombuffer(token, dtype=np.uint8)) for token in tokens_b]
    tokens_mv = [memoryview(bytearray(token)) for token in tokens_b]

    # Python bytes.translate (always allocating)
    bench_translate(settings, "lookup-table/bytes.translate<new>", tokens_b, reverse, bytes_translate)

    # OpenCV allocating
    bench_translate(settings, "lookup-table/opencv.LUT<new>", tokens_np, reverse_np, opencv_lut_allocating)

    # OpenCV in-place
    bench_translate(settings, "lookup-table/opencv.LUT<inplace>", tokens_np, reverse_np, opencv_lut_inplace)

    # NumPy indexing allocating
    bench_translate(settings, "lookup-table/numpy.indexing<new>", tokens_np, reverse_np, numpy_lut_indexing_allocating)

    # NumPy indexing in-place
    bench_translate(settings, "lookup-table/numpy.indexing<inplace>", tokens_np, reverse_np, numpy_lut_indexing_inplace)

    # NumPy take allocating
    bench_translate(settings, "lookup-table/numpy.take<new>", tokens_np, reverse_np, numpy_lut_take_allocating)

    # NumPy take in-place
    bench_translate(settings, "lookup-table/numpy.take<inplace>", tokens_np, reverse_np, numpy_lut_take_inplace)

    # StringZilla allocating
    bench_translate(settings, "lookup-table/stringzilla.translate<new>", tokens_b, reverse, sz_translate_allocating)

    # StringZilla in-place (need memoryviews for each token)
    bench_translate(settings, "lookup-table/stringzilla.translate<inplace>", tokens_mv, reverse, sz_translate_inplace)

    # Random byte generation
    print()
    print("Random Byte Generation")
    sizes = sizes_from_tokens(tokens_b)

    bench_generator(settings, "generate-random/pycryptodome.AES-CTR", sizes, make_pycryptodome_aes_ctr())
    bench_generator(settings, "generate-random/stringzilla.fill_random", sizes, make_stringzilla_fill_random())
    bench_generator(settings, "generate-random/stringzilla.random", sizes, sz.random)
    bench_generator(settings, "generate-random/numpy.PCG64", sizes, make_numpy_generator(np.random.PCG64(0)))
    bench_generator(settings, "generate-random/numpy.Philox", sizes, make_numpy_generator(np.random.Philox(0)))

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
