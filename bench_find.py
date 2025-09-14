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

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  python bench_find.py --dataset README.md --tokens lines
  python bench_find.py --dataset test.txt --tokens words -k "str\.find"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines python bench_find.py

Timing via time.monotonic_ns(); throughput in decimal GB/s. Filter with -k/--filter.
"""

import argparse
import re
from typing import List, Optional

from stringzilla import Str
import ahocorasick as ahoc

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns


def bench_op(name: str, haystack, patterns, op: callable):
    a = now_ns()
    for pattern in patterns:
        op(haystack, pattern)
    b = now_ns()
    bytes_length = len(haystack) * len(patterns)
    secs = (b - a) / 1e9
    gb_per_sec = bytes_length / (1e9 * secs)
    queries_per_sec = len(patterns) / secs
    print(f"{name:25s}: {secs:8.3f}s ~ {gb_per_sec:8.3f} GB/s ~ {queries_per_sec:10,.0f} queries/s")


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
    dataset_path: Optional[str] = None,
    tokens_mode: Optional[str] = None,
    filter_pattern: Optional[re.Pattern] = None,
):
    """Run string search benchmarks."""
    pythonic_str = load_dataset(dataset_path)
    tokens = tokenize_dataset(pythonic_str, tokens_mode)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    stringzilla_str = Str(pythonic_str)
    total_tokens = len(tokens)
    mean_token_length = sum(len(t) for t in tokens) / total_tokens

    print(f"Dataset: {total_tokens:,} tokens, {len(pythonic_str):,} bytes, {mean_token_length:.1f} avg token length")

    run_benches(
        tokens,
        pythonic_str,
        stringzilla_str,
        filter_pattern=filter_pattern,
    )


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla find operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Load and tokenize dataset
    try:
        filter_pattern = None
        if args.filter:
            try:
                filter_pattern = re.compile(args.filter)
            except re.error as e:
                parser.error(f"Invalid regex for --filter: {e}")

        bench(
            dataset_path=args.dataset,
            tokens_mode=args.tokens,
            filter_pattern=filter_pattern,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
