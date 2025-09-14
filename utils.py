"""
Shared utilities for StringWa.rs Python benchmarking scripts.

Common functions for dataset loading, tokenization, timing, and argument parsing
used across bench_find.py, bench_hash.py, and other benchmarking scripts.
"""

import os
import time
from typing import List, Optional, Union


def now_ns() -> int:
    """Get current time in nanoseconds for benchmarking."""
    return time.monotonic_ns()


def load_dataset(dataset_path: Optional[str] = None, as_bytes: bool = False) -> Union[str, bytes]:
    """
    Load dataset from file path or environment variable.

    Args:
        dataset_path: Path to dataset file (uses STRINGWARS_DATASET env var if None)
        as_bytes: If True, return bytes; if False, return str

    Returns:
        Dataset contents as str or bytes based on as_bytes parameter
    """
    if dataset_path is None:
        dataset_path = os.environ.get("STRINGWARS_DATASET")
        if dataset_path is None:
            raise ValueError("No dataset path provided and STRINGWARS_DATASET not set")

    if as_bytes:
        with open(dataset_path, "rb") as f:
            return f.read()
    else:
        with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def tokenize_dataset(haystack: Union[str, bytes], tokens_mode: Optional[str] = None) -> Union[List[str], List[bytes]]:
    """
    Tokenize haystack based on mode from argument or environment variable.

    Args:
        haystack: Input data to tokenize (str or bytes)
        tokens_mode: Tokenization mode ('lines', 'words', 'file') or None to use env var

    Returns:
        List of tokens in the same type as input (List[str] or List[bytes])
    """
    if tokens_mode is None:
        tokens_mode = os.environ.get("STRINGWARS_TOKENS", "words")

    is_bytes = isinstance(haystack, bytes)

    if tokens_mode == "lines":
        # Split on LF only
        return haystack.split(b"\n" if is_bytes else "\n")
    elif tokens_mode == "words":
        # Use default split() which handles all whitespace
        return haystack.split()
    elif tokens_mode == "file":
        return [haystack]
    else:
        raise ValueError(f"Unknown tokens mode: {tokens_mode}. Use 'lines', 'words', or 'file'.")


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
        help="Regex to select which benchmarks to run",
    )
