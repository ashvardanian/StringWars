"""Substring, reverse-substring and byte-set search benchmarks in Python. Mirrors `find/bench.rs`.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group find find/bench.py
"""

import argparse
import re
from collections.abc import Callable, Sequence
from functools import partial
from importlib.metadata import version as pkg_version
from typing import Any

import ahocorasick as ahoc
import stringzilla as sz

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

NEEDLES_PER_PASS = 16

# The three bytesets `find/bench.rs` scans, so a pass covers the haystack three times.
BYTESETS = ("\n\r\x0b\x0c", "</>&'\"=[]", "0123456789")
BYTESET_PATTERNS = tuple(re.compile("[" + re.escape(byteset) + "]") for byteset in BYTESETS)


def bench_op(
    settings: Settings,
    name: str,
    haystack: str | sz.Str,
    haystack_bytes: Bytes,
    patterns: Sequence[Any],
    operation: Callable[..., int],
) -> None:
    """
    One pass scans every pattern across the whole haystack.

    The pattern set is fixed, so every implementation scans identical work. The old
    loop ran against a deadline, so a faster engine got through more patterns than a
    slower one — and pattern cost varies by more than an order of magnitude.

    `haystack_bytes` is passed rather than measured: `len()` on a `str` counts
    codepoints, which published the four stdlib rows at 61% of their true rate on a
    Cyrillic corpus while the `sz.Str` rows, whose `len()` is bytes, were correct.
    """
    work = MeasureSpec(
        unit="bytes",
        elements=len(patterns),
        total_bytes=Bytes(haystack_bytes * len(patterns)),
    )
    measure(settings, name, work, pass_over(partial(operation, haystack), patterns))


def count_find(haystack: str | sz.Str, pattern: str) -> int:
    # Non-overlapping, as `find/bench.rs` counts. `sz.Str` indexes bytes and `str`
    # codepoints, so the stride is the needle measured in the haystack's own unit.
    stride = len(pattern) if isinstance(haystack, str) else len(pattern.encode())
    count, start = 0, 0
    while True:
        index = haystack.find(pattern, start)
        if index == -1:
            break
        count += 1
        start = index + stride
    return count


def count_rfind(haystack: str | sz.Str, pattern: str) -> int:
    count, start = 0, len(haystack) - 1
    while True:
        index = haystack.rfind(pattern, 0, start + 1)
        if index == -1:
            break
        count += 1
        start = index - 1
    return count


def count_regex(haystack: str, regex: re.Pattern[str]) -> int:
    return sum(1 for _ in regex.finditer(haystack))


def count_aho(haystack: str, automaton: ahoc.Automaton) -> int:
    return sum(1 for _ in automaton.iter(haystack))


def build_automaton(needle: str) -> ahoc.Automaton:
    """Built before the clock starts, as `find/bench.rs` builds its FSAs."""
    automaton = ahoc.Automaton()
    automaton.add_word(needle, 1)
    automaton.make_automaton()
    return automaton


def count_byteset(haystack: sz.Str, characters: str) -> int:
    count, start = 0, 0
    while True:
        index = haystack.find_first_of(characters, start)
        if index == -1:
            break
        count += 1
        start = index + 1
    return count


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    print_machine(
        {
            "StringZilla": f"{sz.__version__} with {sz.__capabilities_str__}",
            "PyAhoCorasick": pkg_version("pyahocorasick"),
        }
    )
    settings = read_settings("find")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    log_timing_overhead(settings)
    tokens = dataset.text_tokens()
    # The haystack is the concatenation of the resolved tokens, so the byte count the
    # rate is divided by is exactly `dataset.token_bytes`.
    pythonic_str = "".join(tokens)
    stringzilla_str = sz.Str(pythonic_str)
    # A fixed, evenly spaced needle sample, matching `find/bench.rs`, so every row
    # scans identical work rather than however far the deadline reached.
    stride = max(len(tokens) // NEEDLES_PER_PASS, 1)
    sample = tokens[::stride][:NEEDLES_PER_PASS]
    automatons = [build_automaton(needle) for needle in sample]
    haystack_bytes = dataset.token_bytes

    log_dataset(dataset)

    print("\nSubstring Search Benchmarks")
    bench_op(settings, "substring-forward/str.find", pythonic_str, haystack_bytes, sample, count_find)
    bench_op(settings, "substring-forward/stringzilla.Str.find", stringzilla_str, haystack_bytes, sample, count_find)
    bench_op(settings, "substring-backward/str.rfind", pythonic_str, haystack_bytes, sample, count_rfind)
    bench_op(settings, "substring-backward/stringzilla.Str.rfind", stringzilla_str, haystack_bytes, sample, count_rfind)
    bench_op(settings, "substring-forward/pyahocorasick.iter", pythonic_str, haystack_bytes, automatons, count_aho)

    print("\nCharacter Set Search")
    bench_op(settings, "byteset-forward/re.finditer", pythonic_str, haystack_bytes, BYTESET_PATTERNS, count_regex)
    bench_op(
        settings,
        "byteset-forward/stringzilla.Str.find_first_of",
        stringzilla_str,
        haystack_bytes,
        BYTESETS,
        count_byteset,
    )

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
