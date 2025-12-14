"""
Shared utilities for StringWars Python benchmarking scripts.

Common functions for dataset loading, tokenization, timing, and argument parsing
used across bench_find.py, bench_hash.py, and other benchmarking scripts.
"""

import os
import re
import time
from typing import Callable, List, Optional, TypeVar, Union

T = TypeVar("T")


# ============================================================================
# Environment Variable Helpers
# ============================================================================
# Standardized functions for fetching environment variables consistently.
# Use these instead of raw os.environ.get() calls throughout the codebase.


def get_env(name: str) -> Optional[str]:
    """Get an optional environment variable, returning None if not set."""
    return os.environ.get(name)


def get_env_or_default(name: str, default: str) -> str:
    """Get an environment variable with a default value."""
    return os.environ.get(name, default)


def get_env_parsed(name: str, default: T, parser: Callable[[str], T] = int) -> T:
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


def get_env_parsed_opt(name: str, parser: Callable[[str], T] = int) -> Optional[T]:
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


# ============================================================================


def now_ns() -> int:
    """Get current time in nanoseconds for benchmarking."""
    return time.monotonic_ns()


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
    dataset_path: Optional[str] = None,
    as_bytes: bool = False,
    size_limit: Optional[str] = None,
) -> Union[str, bytes]:
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
        with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
            if max_bytes is not None:
                return f.read(max_bytes)
            else:
                return f.read()


def tokenize_dataset(
    haystack: Union[str, bytes],
    tokens_mode: Optional[str] = None,
    unique: Optional[bool] = None,
) -> Union[List[str], List[bytes]]:
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
        default=10.0,
        help="Time limit per benchmark function in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--dataset-limit",
        type=str,
        default="128mb",
        help="Maximum dataset size (default: 128mb). Supports formats like '1gb', '500mb', '10kb'",
    )


def should_run(name: str, pattern: Optional[re.Pattern]) -> bool:
    """Check if benchmark should run based on filter pattern."""
    if pattern is None:
        return True
    assert hasattr(pattern, "search"), "Pattern must be a compiled regex"
    return bool(pattern.search(name))
