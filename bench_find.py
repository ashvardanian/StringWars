# /// script
# dependencies = [
#   "stringzilla",
#   "pyahocorasick",
# ]
# ///
"""
Python substring, byteset, Aho–Corasick, and translate benches.

- Substring: str.find/rfind, sz.Str.find/rfind (per token)
- Byteset: re.finditer, sz.Str.find_first_of
- Aho–Corasick: per-token (build per pattern) and multi-token (one pass)
- Translate: bytes.translate and sz.Str.translate (256-byte LUT)

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  python bench_find.py --dataset README.md --tokens lines
  python bench_find.py --dataset xlsum.csv --tokens words -k "str.find"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines python bench_find.py

Timing via time.monotonic_ns(); throughput in decimal GB/s. Filter with -k/--filter.
"""

import argparse
import re
import sys
from typing import List, Optional, Generator
import importlib.metadata


import stringzilla as sz
import ahocorasick as ahoc

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns, name_matches


def log_system_info():
    """Log Python version and find library versions."""
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- PyAhoCorasick: {importlib.metadata.version('pyahocorasick')}")
    print()  # Add blank line


def bench_op(name: str, haystack, patterns, op: callable, time_limit_seconds: float = 10.0):
    start_time = now_ns()
    time_limit_ns = int(time_limit_seconds * 1e9)

    requested_queries = 0
    received_results = 0
    received_bytes = 0

    for pattern in patterns:
        received_results += op(haystack, pattern)
        requested_queries += 1
        received_bytes += len(haystack)

        # Check time limit every 10 iterations (since patterns might be fewer than tokens)
        current_time = now_ns()
        if (current_time - start_time) >= time_limit_ns:
            break

    end_time = now_ns()
    secs = (end_time - start_time) / 1e9

    queries_per_sec = requested_queries / secs if secs > 0 else 0.0
    results_per_sec = received_results / secs if secs > 0 else 0.0
    gb_per_sec = len(haystack) * queries_per_sec / 1e9

    print(
        f"{name:35s}: {secs:8.3f}s ~ {gb_per_sec:8.3f} GB/s ~ {queries_per_sec:10,.2f} queries/s ~ {results_per_sec:10,.0f} results/s"
    )


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


def count_byteset(haystack: sz.Str, characters: str) -> int:
    count, start = 0, 0
    while True:
        index = haystack.find_first_of(characters, start)
        if index == -1:
            break
        count += 1
        start = index + 1
    return count


def sz_translate(haystack: sz.Str, look_up_table: bytes) -> int:
    # StringZilla translation using 256-byte LUT
    result = haystack.translate(look_up_table)
    return len(result)


def bytes_translate(haystack_bytes: bytes, lut: bytes) -> int:
    # Python bytes.translate with 256-byte LUT
    result = haystack_bytes.translate(lut)
    return len(result)


_main_epilog = """
Examples:

  # Benchmark all find operations with default settings
  %(prog)s --dataset README.md --tokens lines

  # Test only substring search operations
  %(prog)s --dataset data.txt --tokens lines -k "str.find|sz.Str.find"

  # Benchmark character set searches
  %(prog)s --dataset large.txt --tokens words -k "find_first_of"

  # Test translation operations
  %(prog)s --dataset text.txt --tokens lines -k "translate"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla find operations",
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
    pythonic_str = load_dataset(args.dataset, as_bytes=False, size_limit=args.dataset_limit)
    tokens = tokenize_dataset(pythonic_str, args.tokens)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    stringzilla_str = sz.Str(pythonic_str)
    total_tokens = len(tokens)
    mean_token_length = sum(len(t) for t in tokens) / total_tokens

    print(f"Dataset: {total_tokens:,} tokens, {len(pythonic_str):,} bytes, {mean_token_length:.1f} avg token length")
    log_system_info()

    print("\n=== Substring Search Benchmarks ===")
    if name_matches("str.find", filter_pattern):
        bench_op("str.find", pythonic_str, tokens[::-1], count_find, args.time_limit)
    if name_matches("sz.Str.find", filter_pattern):
        bench_op("sz.Str.find", stringzilla_str, tokens[::-1], count_find, args.time_limit)
    if name_matches("str.rfind", filter_pattern):
        bench_op("str.rfind", pythonic_str, tokens, count_rfind, args.time_limit)
    if name_matches("sz.Str.rfind", filter_pattern):
        bench_op("sz.Str.rfind", stringzilla_str, tokens, count_rfind, args.time_limit)
    if name_matches("pyahocorasick.iter", filter_pattern):
        bench_op("pyahocorasick.iter", pythonic_str, tokens[::-1], count_aho, args.time_limit)

    print("\n=== Character Set Search ===")
    if args.tokens == "lines":
        re_chars = re.compile(r"[\n\r]")  # newlines: LF, CR
        sz_chars = "\n\r"
    else:
        re_chars = re.compile(r"[\t\n\r ]")  # whitespace: space, tab, LF, CR
        sz_chars = " \t\n\r"
    if name_matches("re.finditer", filter_pattern):
        bench_op("re.finditer", pythonic_str, [re_chars], count_regex, args.time_limit)
    if name_matches("sz.Str.find_first_of", filter_pattern):
        bench_op("sz.Str.find_first_of", stringzilla_str, [sz_chars], count_byteset, args.time_limit)

    print("\n=== Translation Benchmarks ===")
    # Translate with byte-level LUT mappings
    identity = bytes(range(256))
    reverse = bytes(reversed(identity))
    repeated = bytes(range(64)) * 4
    hex_table = b"0123456789abcdef" * 16

    py_bytes = pythonic_str.encode("utf-8", errors="ignore")
    if name_matches("bytes.translate(reverse)", filter_pattern):
        bench_op("bytes.translate(reverse)", py_bytes, [reverse], bytes_translate, args.time_limit)
    if name_matches("bytes.translate(repeated)", filter_pattern):
        bench_op("bytes.translate(repeated)", py_bytes, [repeated], bytes_translate, args.time_limit)
    if name_matches("bytes.translate(hex)", filter_pattern):
        bench_op("bytes.translate(hex)", py_bytes, [hex_table], bytes_translate, args.time_limit)
    if name_matches("sz.Str.translate(reverse)", filter_pattern):
        bench_op("sz.Str.translate(reverse)", stringzilla_str, [reverse], sz_translate, args.time_limit)
    if name_matches("sz.Str.translate(repeated)", filter_pattern):
        bench_op("sz.Str.translate(repeated)", stringzilla_str, [repeated], sz_translate, args.time_limit)
    if name_matches("sz.Str.translate(hex)", filter_pattern):
        bench_op("sz.Str.translate(hex)", stringzilla_str, [hex_table], sz_translate, args.time_limit)

    return 0


if __name__ == "__main__":
    exit(main())
