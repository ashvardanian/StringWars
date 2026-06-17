"""
Shared utilities for StringWars Python benchmarking scripts.

Common functions for dataset loading, tokenization, timing, and argument parsing
used across bench_find.py, bench_hash.py, and other benchmarking scripts.
"""

import os
import re
import time
from collections.abc import Callable

# region: Environment Variable Helpers
# Standardized functions for fetching environment variables consistently.
# Use these instead of raw os.environ.get() calls throughout the codebase.


def get_env(name: str) -> str | None:
    """Get an optional environment variable, returning None if not set."""
    return os.environ.get(name)


def get_env_or_default(name: str, default: str) -> str:
    """Get an environment variable with a default value."""
    return os.environ.get(name, default)


def get_env_parsed[T](name: str, default: T, parser: Callable[[str], T] = int) -> T:
    """
    Get an environment variable parsed to a type, with a default value.
    Returns the default if the variable is not set or cannot be parsed.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return parser(value)
    except (ValueError, TypeError):
        return default


def get_env_parsed_opt[T](name: str, parser: Callable[[str], T] = int) -> T | None:
    """
    Get an optional environment variable parsed to a type.
    Returns None if the variable is not set or cannot be parsed.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return parser(value)
    except (ValueError, TypeError):
        return None


def get_env_bool(name: str) -> bool:
    """
    Get a boolean environment variable.
    Accepts "1", "true", or "yes" (case-insensitive) as true values.
    Returns False if not set or set to any other value.
    """
    value = os.environ.get(name, "").lower()
    return value in ("1", "true", "yes")


# endregion: Environment Variable Helpers


def now_nanoseconds() -> int:
    """Get current time in nanoseconds for benchmarking."""
    return time.monotonic_ns()


# region: Benchmark loop profile

# Reporting stride: the smallest number of operations to run between progress logging
# and clock reads, so the cost of terminal rendering and timer syscalls is amortized
# away. This is independent of batch size. Benchmarks come in two shapes:
#
# - One item at a time: a single element per call, logged every LOGGING_STEP calls.
# - Batched: many items per call, where one batch already spans many items.
#
# Batched benchmarks use clamped_subranges to slice the input and hand each slice to a
# kernel that consumes a whole batch. One-at-a-time benchmarks use paced_items, which
# walks a single iterator and checks the clock from inside the loop, so it never builds
# a slice and works on any iterable.
LOGGING_STEP = 1024


def clamped_subranges(count: int, stride: int = LOGGING_STEP):
    """Yield (low, high) index windows covering [0, count) in stride-sized steps.

    Used by the batched benchmarks: slice the input with the returned indices, hand
    each slice to a kernel, then read the clock and repaint progress once per window.
    Pass stride explicitly to use the batch size as the window.
    """
    for low in range(0, count, stride):
        yield low, min(low + stride, count)


# Target wall-time between deadline clock reads in paced_items. The clock read costs tens of
# nanoseconds, so reading it once per ~1 ms of work keeps its overhead negligible (~0.01%).
PACING_TARGET_BETWEEN_CHECKS_NS = 1_000_000


def paced_items(items, deadline_nanoseconds: int, step: int = LOGGING_STEP, progress=None):
    """Yield items from a single iterator, checking the deadline from inside the loop.

    The companion to clamped_subranges for one-at-a-time benchmarks. It walks one
    iterator, repainting progress every step items, and stops once the deadline passes.
    Works on any iterable, such as a list, a zip of two lists, or an itertools.cycle,
    and never builds a slice.

    The cadence is *adaptive* rather than a fixed step. A fixed step would either read the
    clock too often on fine-grained items (e.g. hashing words) or overshoot the time limit
    on few-but-huge items (e.g. one whole file in `file` tokenization mode, where there is a
    single item). So `stride` — the number of items between checkpoints — starts at 1 and
    doubles toward `step` whenever a stride ran faster than `PACING_TARGET_BETWEEN_CHECKS_NS`:
    cheap items climb to the fully-amortized `step`, while a slow item leaves `stride` at 1
    and is checked every iteration, so the deadline cannot overshoot by more than one item.
    Progress repaint and the deadline check share the checkpoint, so the hot path is a single
    countdown — the same per-item cost as a non-adaptive loop.
    """
    stride = 1  # items between checkpoints; doubles toward `step` for cheap items
    countdown = 1
    last_check_nanoseconds = now_nanoseconds()
    for item in items:
        yield item
        countdown -= 1
        if countdown:
            continue
        current_nanoseconds = now_nanoseconds()
        if progress is not None:
            progress.update(stride)
        if current_nanoseconds >= deadline_nanoseconds:
            return
        if current_nanoseconds - last_check_nanoseconds < PACING_TARGET_BETWEEN_CHECKS_NS and stride < step:
            stride = min(stride * 2, step)
        last_check_nanoseconds = current_nanoseconds
        countdown = stride


def reduce_in_windows(
    function,
    *columns,
    deadline_nanoseconds: int,
    step: int = LOGGING_STEP,
    combine=sum,
    progress=None,
):
    """Apply function across the zipped columns one window at a time and reduce each window.

    The shared home for the trick of pushing a per-item loop into C: every window is
    handled by combine(map(...)), so the function calls and the reduction run in C with
    no Python bytecode per item, and the deadline and optional progress are checked once
    per window. This runs about 1.5 to 1.9 times faster than an explicit Python loop on
    real kernels, but only when the body is "call a function on each item and reduce the
    results" with no per-item side effect.

    Pass several equal-length sequences to vary more than one argument, for example two
    string columns. Pin a fixed argument with a leading functools.partial or a constant
    column such as [fixed] * n. Returns (reduced_total, processed_count).
    """
    count = min((len(column) for column in columns), default=0)
    total = 0
    processed = 0
    for low in range(0, count, step):
        if now_nanoseconds() >= deadline_nanoseconds:
            break
        high = min(low + step, count)
        total += combine(map(function, *(column[low:high] for column in columns)))
        if progress is not None:
            progress.update(high - low)
        processed = high
    return total, processed


# endregion: Benchmark loop profile


def parse_size(size_str: str) -> int:
    """
    Parse a size string like '128mb', '1gb', '500kb' into bytes.

    Supports: b, kb, mb, gb (case insensitive)
    Returns size in bytes.
    """
    if not size_str:
        raise ValueError("Size string cannot be empty")

    # Match number followed by optional unit
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(b|kb|mb|gb)?$", size_str.lower().strip())
    if not match:
        raise ValueError(f"Invalid size format: {size_str}. Use formats like '128mb', '1gb', '500kb'")

    number, unit = match.groups()
    number = float(number)

    # Convert to bytes
    multipliers = {
        None: 1,
        "b": 1,
        "kb": 1024,
        "mb": 1024 * 1024,
        "gb": 1024 * 1024 * 1024,
    }

    return int(number * multipliers[unit])


def load_dataset(
    dataset_path: str | None = None,
    as_bytes: bool = False,
    size_limit: str | None = None,
) -> str | bytes:
    """
    Load dataset from file path or environment variable.

    Args:
        dataset_path: Path to dataset file (uses STRINGWARS_DATASET env var if None)
        as_bytes: If True, return bytes; if False, return str
        size_limit: Maximum size to read (e.g., "128mb", "1gb"). If None, read entire file.

    Returns:
        Dataset contents as str or bytes based on as_bytes parameter
    """
    if dataset_path is None:
        dataset_path = get_env("STRINGWARS_DATASET")
        if dataset_path is None:
            raise ValueError("No dataset path provided and STRINGWARS_DATASET not set")

    # Parse size limit if provided
    max_bytes = None
    if size_limit:
        max_bytes = parse_size(size_limit)

    if as_bytes:
        with open(dataset_path, "rb") as f:
            if max_bytes is not None:
                return f.read(max_bytes)
            else:
                return f.read()
    else:
        with open(dataset_path, encoding="utf-8", errors="ignore") as f:
            if max_bytes is not None:
                return f.read(max_bytes)
            else:
                return f.read()


def tokenize_dataset(
    haystack: str | bytes,
    tokens_mode: str | None = None,
    unique: bool | None = None,
) -> list[str] | list[bytes]:
    """
    Tokenize haystack based on mode from argument or environment variable.

    Args:
        haystack: Input data to tokenize (str or bytes)
        tokens_mode: Tokenization mode ('lines', 'words', 'file') or None to use env var
        unique: If True, deduplicate tokens (preserving order). If None, uses STRINGWARS_UNIQUE.

    Returns:
        List of tokens in the same type as input (List[str] or List[bytes])
    """
    if tokens_mode is None:
        tokens_mode = get_env_or_default("STRINGWARS_TOKENS", "words")

    is_bytes = isinstance(haystack, bytes)

    if tokens_mode == "lines":
        # Split on LF only
        tokens = haystack.split(b"\n" if is_bytes else "\n")
    elif tokens_mode == "words":
        # Use default split() which handles all whitespace
        tokens = haystack.split()
    elif tokens_mode == "file":
        tokens = [haystack]
    else:
        raise ValueError(f"Unknown tokens mode: {tokens_mode}. Use 'lines', 'words', or 'file'.")

    if unique is None:
        unique = get_env_bool("STRINGWARS_UNIQUE")

    # Deduplicate, preserving the order of first appearance.
    if tokens_mode != "file" and unique:
        tokens = list(dict.fromkeys(tokens))

    return tokens


def add_common_args(parser):
    """Add common dataset and tokenization arguments to an ArgumentParser."""
    parser.add_argument(
        "--dataset",
        help="Path to input dataset file (overrides STRINGWARS_DATASET env var)",
    )
    parser.add_argument(
        "--tokens",
        choices=["lines", "words", "file"],
        help="Tokenization mode (overrides STRINGWARS_TOKENS env var)",
    )
    parser.add_argument(
        "-k",
        "--filter",
        metavar="REGEX",
        default=get_env("STRINGWARS_FILTER"),
        help="Regex to select which benchmarks to run (or set STRINGWARS_FILTER env var)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Time limit per benchmark function in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--dataset-limit",
        type=str,
        default="128mb",
        help="Maximum dataset size (default: 128mb). Supports formats like '1gb', '500mb', '10kb'",
    )


def should_run(name: str, pattern: re.Pattern | None) -> bool:
    """Check if benchmark should run based on filter pattern."""
    if pattern is None:
        return True
    assert hasattr(pattern, "search"), "Pattern must be a compiled regex"
    return bool(pattern.search(name))
