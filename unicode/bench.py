# /// script
# dependencies = [
#   "stringzilla>=4.5.0",
#   "regex",
#   "PyICU",
# ]
# ///
"""
Python Unicode case-insensitive benchmarks.

Benchmarks case-insensitive string operations:
- Case-insensitive comparison of adjacent token pairs
- Case-insensitive substring search
- Case folding transformation

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  uv run unicode/bench.py --dataset README.md --tokens lines
  uv run unicode/bench.py --dataset xlsum.csv --tokens words -k "casefold"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines uv run unicode/bench.py

Timing via time.monotonic_ns; throughput in decimal GB/s. Filter with -k/--filter.
"""

import argparse
import random
import re
import sys
from importlib.metadata import version as pkg_version
from typing import List, Tuple, Optional

import icu
import regex
import stringzilla as sz

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns, should_run



def log_system_info():
    """Log Python version and library versions."""
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- regex: {pkg_version('regex')}")
    print(f"- PyICU: {pkg_version('PyICU')} (ICU {icu.ICU_VERSION})")
    print()


def make_pairs(tokens: List[str]) -> List[Tuple[str, str]]:
    """Create adjacent token pairs for comparison benchmarks."""
    if len(tokens) < 2:
        return []
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def bench_case_compare(
    name: str,
    pairs: List[Tuple[str, str]],
    compare_fn: callable,
    time_limit_seconds: float = 10.0,
):
    """Benchmark case-insensitive comparison of string pairs."""
    if not pairs:
        print(f"{name:35s}: no pairs to compare")
        return

    start_time = now_ns()
    time_limit_ns = int(time_limit_seconds * 1e9)

    compared_pairs = 0
    compared_bytes = 0
    matches_found = 0

    while True:
        for s1, s2 in pairs:
            if compare_fn(s1, s2):
                matches_found += 1
            compared_pairs += 1
            compared_bytes += len(s1.encode('utf-8')) + len(s2.encode('utf-8'))

            if (now_ns() - start_time) >= time_limit_ns:
                break
        else:
            # Completed full loop, check time
            if (now_ns() - start_time) >= time_limit_ns:
                break
            continue
        break

    end_time = now_ns()
    secs = (end_time - start_time) / 1e9

    pairs_per_sec = compared_pairs / secs if secs > 0 else 0.0
    gb_per_sec = compared_bytes / secs / 1e9 if secs > 0 else 0.0

    print(
        f"{name:35s}: {secs:8.3f}s ~ {gb_per_sec:8.3f} GB/s ~ {pairs_per_sec:10,.0f} pairs/s ~ {matches_found:,} matches"
    )


def compare_casefold(s1: str, s2: str) -> bool:
    """Compare using Python's str.casefold()."""
    return s1.casefold() == s2.casefold()


def compare_regex_fullcase(s1: str, s2: str) -> bool:
    """Compare using regex with IGNORECASE | FULLCASE."""
    # Escape special regex characters and do a full match
    pattern = regex.compile(regex.escape(s1), regex.IGNORECASE | regex.FULLCASE)
    return pattern.fullmatch(s2) is not None


def compare_icu(s1: str, s2: str) -> bool:
    """Compare using ICU case folding."""
    folded1 = icu.UnicodeString(s1).foldCase()
    folded2 = icu.UnicodeString(s2).foldCase()
    return folded1 == folded2


def compare_stringzilla(s1: str, s2: str) -> bool:
    """Compare using StringZilla's utf8_case_insensitive_order."""
    return sz.utf8_case_insensitive_order(s1, s2) == 0


def bench_case_find(
    name: str,
    haystack: str,
    needles: List[str],
    find_fn: callable,
    time_limit_seconds: float = 10.0,
):
    """Benchmark case-insensitive substring search."""
    if not needles:
        print(f"{name:35s}: no needles to search")
        return

    haystack_bytes = len(haystack.encode('utf-8'))
    start_time = now_ns()
    time_limit_ns = int(time_limit_seconds * 1e9)

    queries_done = 0
    total_matches = 0

    while True:
        for needle in needles:
            total_matches += find_fn(haystack, needle)
            queries_done += 1

            if (now_ns() - start_time) >= time_limit_ns:
                break
        else:
            if (now_ns() - start_time) >= time_limit_ns:
                break
            continue
        break

    end_time = now_ns()
    secs = (end_time - start_time) / 1e9

    queries_per_sec = queries_done / secs if secs > 0 else 0.0
    # Throughput = haystack bytes searched per query
    gb_per_sec = (haystack_bytes * queries_done) / secs / 1e9 if secs > 0 else 0.0

    print(
        f"{name:35s}: {secs:8.3f}s ~ {gb_per_sec:8.3f} GB/s ~ {queries_per_sec:10,.2f} queries/s ~ {total_matches:,} matches"
    )


def find_casefold(haystack: str, needle: str) -> int:
    """Count occurrences using casefold on both strings."""
    haystack_folded = haystack.casefold()
    needle_folded = needle.casefold()
    if not needle_folded:
        return 0
    count = 0
    start = 0
    while True:
        pos = haystack_folded.find(needle_folded, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1
    return count


def find_regex_fullcase(haystack: str, needle: str) -> int:
    """Count occurrences using regex with IGNORECASE | FULLCASE."""
    if not needle:
        return 0
    pattern = regex.compile(regex.escape(needle), regex.IGNORECASE | regex.FULLCASE)
    # Use finditer for fair comparison (same Python loop overhead as StringZilla)
    return sum(1 for _ in pattern.finditer(haystack))


def find_icu(haystack: str, needle: str) -> int:
    """Count occurrences using ICU StringSearch."""
    if not needle:
        return 0
    collator = icu.Collator.createInstance(icu.Locale.getRoot())
    collator.setStrength(icu.Collator.SECONDARY)  # Case-insensitive
    searcher = icu.StringSearch(needle, haystack, collator)
    count = 0
    pos = searcher.nextMatch()
    while pos != -1:
        count += 1
        pos = searcher.nextMatch()
    return count


def find_stringzilla(haystack: str, needle: str) -> int:
    """Count occurrences using StringZilla's utf8_case_insensitive_find_iter."""
    if not needle:
        return 0
    return sum(1 for _ in sz.utf8_case_insensitive_find_iter(haystack, needle))


def bench_case_fold(
    name: str,
    strings: List[str],
    fold_fn: callable,
    time_limit_seconds: float = 10.0,
):
    """Benchmark case folding transformation."""
    if not strings:
        print(f"{name:35s}: no strings to fold")
        return

    # Pre-calculate total bytes for throughput
    total_bytes = sum(len(s.encode('utf-8')) for s in strings)

    start_time = now_ns()
    time_limit_ns = int(time_limit_seconds * 1e9)

    iterations = 0
    processed_bytes = 0

    while True:
        for s in strings:
            _ = fold_fn(s)
        iterations += 1
        processed_bytes += total_bytes

        if (now_ns() - start_time) >= time_limit_ns:
            break

    end_time = now_ns()
    secs = (end_time - start_time) / 1e9

    strings_per_sec = (iterations * len(strings)) / secs if secs > 0 else 0.0
    gb_per_sec = processed_bytes / secs / 1e9 if secs > 0 else 0.0

    print(
        f"{name:35s}: {secs:8.3f}s ~ {gb_per_sec:8.3f} GB/s ~ {strings_per_sec:10,.0f} strings/s"
    )


def fold_casefold(s: str) -> str:
    """Fold using Python's str.casefold() - full Unicode."""
    return s.casefold()


def fold_stringzilla(s: str) -> bytes:
    """Fold using StringZilla's utf8_case_fold() - full Unicode."""
    return sz.utf8_case_fold(s)


def fold_icu(s: str) -> str:
    """Fold using ICU case folding."""
    return str(icu.UnicodeString(s).foldCase())


_main_epilog = """
Examples:

  # Benchmark all Unicode case operations
  %(prog)s --dataset README.md --tokens lines

  # Test only case folding
  %(prog)s --dataset data.txt --tokens lines -k "casefold"

  # Test only comparison benchmarks
  %(prog)s --dataset large.txt --tokens words -k "compare"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark Unicode case-insensitive operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Compile filter pattern
    filter_pattern: Optional[re.Pattern] = None
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

    # Create adjacent pairs for comparison benchmarks
    pairs = make_pairs(tokens)

    # Select subset of tokens as needles for search benchmarks
    # Filter to needles with >= 3 UTF-8 bytes (matching Rust benchmark)
    candidates = [t for t in tokens if len(t.encode('utf-8')) >= 3]
    # Random sample with fixed seed for reproducibility
    random.seed(42)
    search_needles = random.sample(candidates, min(1000, len(candidates))) if candidates else []

    total_tokens = len(tokens)
    total_pairs = len(pairs)
    mean_token_length = sum(len(t) for t in tokens) / total_tokens
    total_bytes = len(pythonic_str)

    print(f"Dataset: {total_tokens:,} tokens, {total_bytes:,} bytes, {mean_token_length:.1f} avg token length")
    print(f"Pairs: {total_pairs:,}, Search needles: {len(search_needles)}")
    log_system_info()

    # === Case-Insensitive Comparison ===
    print("=== Case-Insensitive Comparison ===")
    if should_run("case-insensitive-compare/stringzilla.utf8_case_insensitive_order()", filter_pattern):
        bench_case_compare("stringzilla.utf8_case_insensitive_order()", pairs, compare_stringzilla, args.time_limit)
    if should_run("case-insensitive-compare/std.casefold().eq()", filter_pattern):
        bench_case_compare("std.casefold().eq()", pairs, compare_casefold, args.time_limit)
    if should_run("case-insensitive-compare/regex.fullmatch(FULLCASE)", filter_pattern):
        bench_case_compare("regex.fullmatch(FULLCASE)", pairs, compare_regex_fullcase, args.time_limit)
    if should_run("case-insensitive-compare/icu.CaseMap.foldCase().eq()", filter_pattern):
        bench_case_compare("icu.CaseMap.foldCase().eq()", pairs, compare_icu, args.time_limit)

    # === Case-Insensitive Substring Search ===
    print("\n=== Case-Insensitive Substring Search ===")
    if should_run("case-insensitive-find/stringzilla.utf8_case_insensitive_find()", filter_pattern):
        bench_case_find("stringzilla.utf8_case_insensitive_find()", pythonic_str, search_needles, find_stringzilla, args.time_limit)
    if should_run("case-insensitive-find/std.casefold().find()", filter_pattern):
        bench_case_find("std.casefold().find()", pythonic_str, search_needles, find_casefold, args.time_limit)
    if should_run("case-insensitive-find/regex.search(FULLCASE)", filter_pattern):
        bench_case_find("regex.search(FULLCASE)", pythonic_str, search_needles, find_regex_fullcase, args.time_limit)
    if should_run("case-insensitive-find/icu.StringSearch()", filter_pattern):
        bench_case_find("icu.StringSearch()", pythonic_str, search_needles, find_icu, args.time_limit)

    # === Case Folding Transformation ===
    print("\n=== Case Folding Transformation ===")
    if should_run("case-fold/stringzilla.utf8_case_fold()", filter_pattern):
        bench_case_fold("stringzilla.utf8_case_fold()", tokens, fold_stringzilla, args.time_limit)
    if should_run("case-fold/std.casefold()", filter_pattern):
        bench_case_fold("std.casefold()", tokens, fold_casefold, args.time_limit)
    if should_run("case-fold/icu.CaseMap.foldCase()", filter_pattern):
        bench_case_fold("icu.CaseMap.foldCase()", tokens, fold_icu, args.time_limit)

    return 0


if __name__ == "__main__":
    exit(main())
