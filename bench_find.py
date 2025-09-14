# /// script
# dependencies = [
#   "stringzilla",
#   "pyahocorasick",
# ]
# ///
"""
Python substring, byteset, Aho–Corasick, and translate benches.

- Substring: str.find/rfind, Str.find/rfind (per token)
- Byteset: re.finditer(charclass), Str.find_first_of
- Aho–Corasick: per-token (build per pattern) and multi-token (one pass)
- Translate: bytes.translate and Str.translate (256-byte LUT)

Timing via time.monotonic_ns(); throughput in decimal GB/s. Filter with -k/--filter.
"""

import argparse
import re
import random
import time
from typing import List, Optional

from stringzilla import Str
import ahocorasick as ahoc


def _now_ns() -> int:
    return time.monotonic_ns()


def bench_op(name: str, haystack, patterns, op: callable):
    a = _now_ns()
    for pattern in patterns:
        op(haystack, pattern)
    b = _now_ns()
    bytes_length = len(haystack) * len(patterns)
    secs = (b - a) / 1e9
    gb_per_sec = bytes_length / (1e9 * secs)
    print(f"{name}: took {secs:.4f} seconds ~ {gb_per_sec:.3f} GB/s")


def count_find(haystack, pattern) -> int:
    count, start = 0, 0
    while True:
        index = haystack.find(pattern, start)
        if index == -1:
            break
        count += 1
        start = index + 1
    return count


def count_rfind(haystack, pattern) -> int:
    count, start = 0, len(haystack) - 1
    while True:
        index = haystack.rfind(pattern, 0, start + 1)
        if index == -1:
            break
        count += 1
        start = index - 1
    return count


def count_regex(haystack: str, regex: re.Pattern) -> int:
    return sum(1 for _ in regex.finditer(haystack))


def count_aho_multi(haystack: str, automaton) -> int:
    # Count all matches over all tokens in a single pass
    return sum(1 for _ in automaton.iter(haystack))


def count_aho(haystack: str, pattern: str) -> int:
    # Build automaton for a single pattern and count all inclusions
    automaton = ahoc.Automaton()
    automaton.add_word(pattern, 1)
    automaton.make_automaton()
    return sum(1 for _ in automaton.iter(haystack))


def count_byteset(haystack: Str, characters: str) -> int:
    count, start = 0, 0
    while True:
        index = haystack.find_first_of(characters, start)
        if index == -1:
            break
        count += 1
        start = index + 1
    return count


def sz_translate(haystack: Str, look_up_table: bytes) -> str:
    # StringZilla translation using 256-byte LUT
    return haystack.translate(look_up_table)


def bytes_translate(haystack_bytes: bytes, lut: bytes) -> bytes:
    # Python bytes.translate with 256-byte LUT
    return haystack_bytes.translate(lut)


def name_matches(name: str, pattern: Optional[re.Pattern]) -> bool:
    return True if pattern is None else bool(pattern.search(name))


def run_benches(
    tokens: List[str],
    pythonic_str: str,
    stringzilla_str: Str,
    filter_pattern: Optional[re.Pattern] = None,
):
    # Read-only Search (substring)
    if name_matches("str.find", filter_pattern):
        bench_op("str.find", pythonic_str, tokens, count_find)
    if name_matches("Str.find", filter_pattern):
        bench_op("Str.find", stringzilla_str, tokens, count_find)
    if name_matches("str.rfind", filter_pattern):
        bench_op("str.rfind", pythonic_str, tokens, count_rfind)
    if name_matches("Str.rfind", filter_pattern):
        bench_op("Str.rfind", stringzilla_str, tokens, count_rfind)

    # Aho–Corasick per-token variant (comparable to per-token substring search)
    if name_matches("pyahocorasick.single-token", filter_pattern):
        bench_op("pyahocorasick.single-token", pythonic_str, tokens, count_aho)

    # Aho-Corasick multi-pattern search (single pass) using shared log()
    automaton = ahoc.Automaton()
    for tok in tokens:
        automaton.add_word(tok, 1)
    automaton.make_automaton()
    if name_matches("pyahocorasick.iter(all tokens)", filter_pattern):
        bench_op("pyahocorasick.iter(all tokens)", pythonic_str, [automaton], count_aho_multi)


    # Character class byteset search: precompile regex and reuse
    cc_regex = re.compile(r"[\t\n\r ]")  # whitespace: space, tab, LF, CR
    if name_matches("re.finditer(charclass)", filter_pattern):
        bench_op("re.finditer(charclass)", pythonic_str, [cc_regex], count_regex)
    if name_matches("Str.find_first_of", filter_pattern):
        bench_op("Str.find_first_of", stringzilla_str, [" \t\n\r"], count_byteset)

    # Translate with byte-level LUT mappings
    identity = bytes(range(256))
    reverse = bytes(reversed(identity))
    repeated = bytes(range(64)) * 4
    hex_tbl = b"0123456789abcdef" * 16

    py_bytes = pythonic_str.encode("utf-8", errors="ignore")
    if name_matches("bytes.translate(reverse)", filter_pattern):
        bench_op("bytes.translate(reverse)", py_bytes, [reverse], bytes_translate)
    if name_matches("bytes.translate(repeated)", filter_pattern):
        bench_op("bytes.translate(repeated)", py_bytes, [repeated], bytes_translate)
    if name_matches("bytes.translate(hex)", filter_pattern):
        bench_op("bytes.translate(hex)", py_bytes, [hex_tbl], bytes_translate)
    if name_matches("Str.translate(reverse)", filter_pattern):
        bench_op("Str.translate(reverse)", stringzilla_str, [reverse], sz_translate)
    if name_matches("Str.translate(repeated)", filter_pattern):
        bench_op("Str.translate(repeated)", stringzilla_str, [repeated], sz_translate)
    if name_matches("Str.translate(hex)", filter_pattern):
        bench_op("Str.translate(hex)", stringzilla_str, [hex_tbl], sz_translate)


def bench(
    haystack_path: Optional[str] = None,
    haystack_pattern: Optional[str] = None,
    haystack_length: Optional[int] = None,
    tokens_mode: str = "words",
    sample: int = 100,
    seed: int = 42,
    filter_pattern: Optional[re.Pattern] = None,
):
    """Run string search benchmarks."""
    if haystack_path:
        pythonic_str: str = open(haystack_path, "r").read()
    else:
        haystack_length = int(haystack_length)
        repetitions = haystack_length // len(haystack_pattern)
        pythonic_str: str = haystack_pattern * repetitions

    stringzilla_str = Str(pythonic_str)
    if tokens_mode == "lines":
        tokens = pythonic_str.splitlines()
    elif tokens_mode == "words":
        tokens = pythonic_str.split()
    elif tokens_mode == "file":
        tokens = [pythonic_str]
    else:
        raise ValueError("tokens_mode must be one of: lines, words, file")
    total_tokens = len(tokens)
    mean_token_length = sum(len(t) for t in tokens) / total_tokens

    print(f"Prepared {total_tokens:,} tokens of {mean_token_length:.2f} mean length!")

    # Deterministic sampling for comparability
    random.seed(seed)
    if sample and sample < len(tokens):
        tokens = random.sample(tokens, sample)
    run_benches(
        tokens,
        pythonic_str,
        stringzilla_str,
        filter_pattern=filter_pattern,
    )


_main_epilog = """
Examples:

  # Benchmark with a file
  %(prog)s --haystack-path leipzig1M.txt

  # Benchmark with synthetic data
  %(prog)s --haystack-pattern "hello world " --haystack-length 1000000
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla find operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    parser.add_argument("--haystack-path", help="Path to input file")
    parser.add_argument(
        "--haystack-pattern", help="Pattern to repeat for synthetic data"
    )
    parser.add_argument(
        "--haystack-length", type=int, help="Length of synthetic haystack"
    )
    parser.add_argument(
        "--tokens-mode",
        choices=["lines", "words", "file"],
        default="words",
        help="Tokenization mode for substring benchmarks",
    )
    parser.add_argument(
        "--sample", type=int, default=100, help="Number of tokens to sample"
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "-k",
        "--filter",
        metavar="REGEX",
        help="Regex to select which benchmarks to run",
    )

    args = parser.parse_args()

    if args.haystack_path:
        if args.haystack_pattern or args.haystack_length:
            parser.error("Cannot specify both --haystack-path and synthetic options")
    else:
        if not (args.haystack_pattern and args.haystack_length):
            parser.error(
                "Must specify either --haystack-path or both --haystack-pattern and --haystack-length"
            )

    pattern = None
    if args.filter:
        try:
            pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter/-k: {e}")

    bench(
        args.haystack_path,
        args.haystack_pattern,
        args.haystack_length,
        tokens_mode=args.tokens_mode,
        sample=args.sample,
        seed=args.seed,
        filter_pattern=pattern,
    )


if __name__ == "__main__":
    main()
