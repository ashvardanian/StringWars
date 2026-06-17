# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=4.5.0",
#   "regex",
#   "PyICU",
# ]
# ///
"""
Python UTF-8 tokenization and iteration benchmarks.

Benchmarks segmentation and codepoint operations:
- TR29 word segmentation (lazy iterators), per document line
- Unicode whitespace and newline splitting, per document line
- UTF-8 codepoint counting, over the whole document

The splitter benches process one document line per call, cycling through the lines (default
tokenization mode is 'lines'); the codepoint counter scans the whole document buffer.

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file'); defaults to 'lines'

Examples:
  uv run tokenization/bench.py --dataset README.md
  STRINGWARS_DATASET=data.txt uv run tokenization/bench.py

Timing via time.monotonic_ns; throughput in decimal GB/s. Filter with -k/--filter.
"""

import argparse
import itertools
import re
import sys
from collections.abc import Callable
from importlib.metadata import version as pkg_version

import icu
import regex
import stringzilla as sz

from utils import (
    add_common_args,
    load_dataset,
    now_nanoseconds,
    paced_items,
    report_stats,
    resolve_tokens,
    should_run,
    tokenize_dataset,
)


def log_system_info():
    """Log Python version and library versions."""
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- regex: {pkg_version('regex')}")
    print(f"- PyICU: {pkg_version('PyICU')} (ICU {icu.ICU_VERSION})")
    print()


def bench_tokenize(
    name: str,
    text: str | bytes,
    count_function: Callable[[str | bytes], int],
    time_limit_seconds: float = 10.0,
):
    """Benchmark a whole-text tokenizer/scanner by counting what it yields.

    `count_function` consumes the entire `text` once per call and returns an integer
    count (token count, codepoint count, ...). The StringZilla and regex/ICU paths
    count lazily via `sum(1 for _ in ...)` so no token list is materialized; the
    stdlib `str.split()/splitlines()` paths do allocate a list, as that is the only
    idiom they offer. Throughput is reported as input bytes per second.
    """
    text_byte_length = len(text.encode("utf-8")) if isinstance(text, str) else len(text)

    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    passes = 0
    token_total = 0
    # Each yielded sentinel triggers one full-text pass; a pass is expensive, so we
    # check the deadline every iteration (step=1) rather than overshoot a large file.
    for _ in paced_items(itertools.count(), deadline_nanoseconds, step=1):
        token_total += count_function(text)
        passes += 1

    seconds = (now_nanoseconds() - start_time) / 1e9
    tokens_per_pass = token_total // passes if passes else 0

    print(f"{name}: {tokens_per_pass:,} tokens per pass over {passes:,} passes", file=sys.stderr)
    report_stats(name, "bytes", seconds, passes, text_byte_length * passes)


def bench_split_lines(
    name: str,
    lines: list[str],
    count_function: Callable[[str], int],
    time_limit_seconds: float = 10.0,
):
    """Benchmark a splitter by processing one document line per call, cycling the lines.

    Unlike the whole-text `bench_tokenize`, each call splits a single line, so the working set
    is one line rather than the entire file. Splitting is compute-bound, so the per-byte rate
    still mirrors a whole-file pass; only the working set changes. Throughput is reported as the
    sum of the byte lengths of every line processed.
    """
    line_byte_lengths = [len(line.encode("utf-8")) for line in lines]

    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    lines_processed = 0
    token_total = 0
    bytes_total = 0
    for line, line_byte_length in paced_items(
        itertools.cycle(zip(lines, line_byte_lengths, strict=True)), deadline_nanoseconds
    ):
        token_total += count_function(line)
        bytes_total += line_byte_length
        lines_processed += 1

    seconds = (now_nanoseconds() - start_time) / 1e9
    tokens_per_line = token_total / lines_processed if lines_processed else 0

    print(
        f"{name}: {tokens_per_line:,.1f} tokens per line over {lines_processed:,} lines",
        file=sys.stderr,
    )
    report_stats(name, "bytes", seconds, lines_processed, bytes_total)


def count_words_stringzilla(text: str) -> int:
    """Count TR29 words lazily via StringZilla's utf8_word_iter (no list built)."""
    return sum(1 for _ in sz.utf8_word_iter(text, skip_empty=True))


def count_words_regex(text: str) -> int:
    """Count word-like runs via the `regex` module's \\w+ finditer (lazy)."""
    return sum(1 for _ in regex.finditer(r"\w+", text))


def count_words_split(text: str) -> int:
    """Count whitespace-delimited words via stdlib str.split() (allocates a list)."""
    return len(text.split())


def make_count_words_icu() -> Callable[[str], int]:
    """Build an ICU word BreakIterator counter, reusing one iterator instance.

    Iterating the break iterator yields every boundary segment (words, punctuation,
    and whitespace), mirroring the Rust ICU WordSegmenter baseline which also counts
    boundary segments rather than only word-like ones.
    """
    break_iterator = icu.BreakIterator.createWordInstance(icu.Locale.getRoot())

    def count(text: str) -> int:
        break_iterator.setText(text)
        segments = 0
        for _ in break_iterator:
            segments += 1
        return segments

    return count


def count_whitespace_stringzilla(text: str) -> int:
    """Count whitespace-delimited tokens lazily via StringZilla's utf8_split_iter."""
    return sum(1 for _ in sz.utf8_split_iter(text, skip_empty=True))


def count_whitespace_split(text: str) -> int:
    """Count whitespace-delimited tokens via stdlib str.split() (allocates a list)."""
    return len(text.split())


def count_newlines_stringzilla(text: str) -> int:
    """Count lines lazily via StringZilla's utf8_splitlines_iter (no list built)."""
    return sum(1 for _ in sz.utf8_splitlines_iter(text))


def count_newlines_splitlines(text: str) -> int:
    """Count lines via stdlib str.splitlines() (allocates a list)."""
    return len(text.splitlines())


def count_codepoints_stringzilla(data: bytes) -> int:
    """Count UTF-8 codepoints in raw bytes via StringZilla's utf8_count (SIMD scan)."""
    return sz.utf8_count(data)


def count_codepoints_decode(data: bytes) -> int:
    """Count UTF-8 codepoints by decoding the bytes to a str, then len() (allocates)."""
    return len(data.decode("utf-8"))


_main_epilog = """
Examples:

  # Benchmark all tokenization and iteration operations
  %(prog)s --dataset README.md --tokens file

  # Test only word segmentation
  %(prog)s --dataset data.txt --tokens file -k "words"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark UTF-8 tokenization and iteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Compile filter pattern
    filter_pattern: re.Pattern | None = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Load the dataset and tokenize it into document lines (the per-line splitter work items).
    tokens_mode = resolve_tokens(args.tokens, "lines")
    pythonic_str = load_dataset(args.dataset, as_bytes=False, size_limit=args.dataset_limit)
    lines = tokenize_dataset(pythonic_str, tokens_mode)

    if not lines:
        print("No tokens found in dataset")
        return 1

    total_tokens = len(lines)
    mean_token_length = sum(len(t) for t in lines) / total_tokens
    total_bytes = len(pythonic_str)

    print(f"Dataset: {total_tokens:,} tokens, {total_bytes:,} bytes, {mean_token_length:.1f} avg token length")
    log_system_info()

    # UTF-8 word segmentation (TR29) per document line, cycling the lines.
    print("Word Segmentation (TR29)")
    if should_run("tokenize-words/stringzilla.utf8_word_iter()", filter_pattern):
        bench_split_lines("stringzilla.utf8_word_iter()", lines, count_words_stringzilla, args.time_limit)
    if should_run("tokenize-words/regex.finditer(\\w+)", filter_pattern):
        bench_split_lines("regex.finditer(\\w+)", lines, count_words_regex, args.time_limit)
    if should_run("tokenize-words/icu.BreakIterator()", filter_pattern):
        bench_split_lines("icu.BreakIterator()", lines, make_count_words_icu(), args.time_limit)
    if should_run("tokenize-words/std.str.split()", filter_pattern):
        bench_split_lines("std.str.split()", lines, count_words_split, args.time_limit)

    # UTF-8 whitespace splitting per document line, cycling the lines.
    print("\nWhitespace Splitting")
    if should_run("tokenize-whitespace/stringzilla.utf8_split_iter()", filter_pattern):
        bench_split_lines("stringzilla.utf8_split_iter()", lines, count_whitespace_stringzilla, args.time_limit)
    if should_run("tokenize-whitespace/std.str.split()", filter_pattern):
        bench_split_lines("std.str.split()", lines, count_whitespace_split, args.time_limit)

    # UTF-8 newline splitting per document line, cycling the lines.
    print("\nNewline Splitting")
    if should_run("tokenize-newlines/stringzilla.utf8_splitlines_iter()", filter_pattern):
        bench_split_lines("stringzilla.utf8_splitlines_iter()", lines, count_newlines_stringzilla, args.time_limit)
    if should_run("tokenize-newlines/std.str.splitlines()", filter_pattern):
        bench_split_lines("std.str.splitlines()", lines, count_newlines_splitlines, args.time_limit)

    # UTF-8 codepoint counting over the raw bytes (fair O(n)-from-bytes comparison;
    # `len(str)` is O(1) in CPython, so we decode-and-count as the stdlib baseline).
    print("\nCodepoint Counting")
    document_bytes = pythonic_str.encode("utf-8")
    if should_run("utf8-count/stringzilla.utf8_count()", filter_pattern):
        bench_tokenize("stringzilla.utf8_count()", document_bytes, count_codepoints_stringzilla, args.time_limit)
    if should_run("utf8-count/std.decode().len()", filter_pattern):
        bench_tokenize("std.decode().len()", document_bytes, count_codepoints_decode, args.time_limit)

    return 0


if __name__ == "__main__":
    exit(main())
