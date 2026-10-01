"""Shared harness for the StringWars Python suites. Mirrors `stringwars.rs`.

Every suite reads these once, at the top of `main`; defaults come from `stringwars.toml`.

Variable                     Default                      Meaning
`STRINGWARS_SEED`            `42`                         Seed of the drawn inputs, an integer or `random`
`STRINGWARS_FILTER`          none                         Regex over row names, falling back to a substring match
`STRINGWARS_WARMUP`          `1s`                         Warm-up cap per row, like `1s` or `200ms`
`STRINGWARS_TIME_LIMIT`      `10s`                        Measurement cap per row, like `10s` or `500ms`
`STRINGWARS_BYTES`           per suite, see the manifest  Bytes read from the dataset, like `256MB`
`STRINGWARS_BATCH_PER_CORE`  per suite, see the manifest  Items per core, or per GPU multiprocessor
`STRINGWARS_THREADS`         all cores                    Cores for multi-core rows, `0` for all
`STRINGWARS_DIMS`            per suite, see the manifest  MinHash widths, like `128` or `64,128,256`
`STRINGWARS_DATASET`         per suite, see the manifest  Path to the textual dataset
`STRINGWARS_TOKENS`          per suite, see the manifest  `lines`, `words` or `file`
`STRINGWARS_UNIQUE`          `false`                      Drops repeated tokens
`STRINGWARS_MIN_SAMPLES`     `10`                         Fewest samples a row needs before it may converge
`STRINGWARS_TARGET_SPREAD`   `0.025`                      Relative half-width at which a row converges
`STRINGWARS_RESULTS_DIR`     none                         Directory for per-row NDJSON records
"""

import functools
import json
import math
import os
import platform
import re
import secrets
import sys
import time
import tomllib
import zlib
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, NewType

Seed = NewType("Seed", int)  # 32-bit; derive streams from it, never add to it
Threads = NewType("Threads", int)  # resolved count, never 0: `parse_threads` maps 0 to all cores
Bytes = NewType("Bytes", int)  # a size in bytes, never an element count
Seconds = NewType("Seconds", float)  # a duration; clock readings stay `int` nanoseconds and say so in the name

# region: Environment variables


def env_text(name: str) -> str | None:
    """Reads `name`, or `None` when it is unset or empty."""
    return os.environ.get(name) or None


def env_parsed[T](name: str, fallback: T, parse: Callable[[str], T | None], expected: str) -> T:
    """Reads `name` through `parse`, or `fallback` when unset or empty; exits with status 1 if it does not parse."""
    text = env_text(name)
    if text is None:
        return fallback
    try:
        parsed = parse(text)
    except (ValueError, TypeError):
        parsed = None
    if parsed is None:
        raise SystemExit(f'{name}="{text}" does not parse, expected {expected}')
    return parsed


def env_count(name: str, fallback: int) -> int:
    """Reads a positive count like `128`, or `fallback` when unset or empty; exits if it does not parse."""
    return env_parsed(name, fallback, parse_count, "a positive count")


def env_duration(name: str, fallback: Seconds) -> Seconds:
    """Reads a duration like `200ms` or `10s`, or `fallback` when unset or empty; exits if it does not parse."""
    return env_parsed(name, fallback, parse_duration, "a duration like 200ms or 10s")


def env_size(name: str, fallback: Bytes) -> Bytes:
    """Reads a size like `256MB`, or `fallback` when unset or empty; exits if it does not parse."""
    return env_parsed(name, fallback, parse_size, "a size like 4096, 64KB or 1GB")


def env_flag(name: str, fallback: bool) -> bool:
    """Reads `0`, `1`, `true` or `false`, or `fallback` when unset or empty; exits if it does not parse."""
    return env_parsed(name, fallback, {"0": False, "false": False, "1": True, "true": True}.get, "0, 1, true or false")


def env_seed(name: str, fallback: Seed) -> Seed:
    """Reads a 32-bit seed or `random`, or `fallback` when unset or empty; exits if it does not parse."""
    return env_parsed(name, fallback, parse_seed, "an unsigned integer or random")


def parse_seed(text: str) -> Seed | None:
    """Parses a 32-bit unsigned integer, or `random` as 32 bits from the OS entropy source."""
    if text == "random":
        return Seed(secrets.randbits(32))
    return Seed(int(text)) if re.fullmatch(r"[0-9]+", text) and int(text) < 1 << 32 else None


def parse_threads(text: str) -> Threads | None:
    """Parses a thread count like `8`, or `0` as every core this process may run on."""
    count = (os.process_cpu_count() or 1) if text == "0" else parse_count(text)
    return None if count is None else Threads(count)


def parse_count(text: str) -> int | None:
    """Parses a positive whole number in ASCII digits, like `128`; zero is `None`."""
    digits = re.fullmatch(r"[0-9]+", text) is not None
    return (int(text) or None) if digits else None


def parse_dims(text: str) -> list[int] | None:
    """Parses one count like `128` or a comma list like `64,128,256,512`."""
    counts = [count for piece in text.split(",") if (count := parse_count(piece)) is not None]
    return counts if len(counts) == text.count(",") + 1 else None


def parse_duration(text: str) -> Seconds | None:
    """Parses a duration like `200ms` or `10s` into seconds; a bare number, a fraction or zero is `None`."""
    if text.endswith("ms"):
        count = parse_count(text.removesuffix("ms"))
        return None if count is None else Seconds(count / 1000)
    count = parse_count(text.removesuffix("s")) if text.endswith("s") else None
    return None if count is None else Seconds(count)


def parse_size(text: str) -> Bytes | None:
    """Parses a size like `256MB`: whole bytes, or `KB`, `MB`, `GB` or `TB` in any case; zero is `None`."""
    match = re.fullmatch(r"([0-9]+)(kb|mb|gb|tb)?", text.lower())
    if not match or not int(match[1]):
        return None
    return Bytes(int(match[1]) << {None: 0, "kb": 10, "mb": 20, "gb": 30, "tb": 40}[match[2]])


def spell_duration(seconds: Seconds) -> str:
    """Spells a duration the way `parse_duration` reads it: `1s`, `1500ms`."""
    milliseconds = round(seconds * 1000)
    return f"{milliseconds // 1000}s" if milliseconds % 1000 == 0 else f"{milliseconds}ms"


def spell_size(size: Bytes) -> str:
    """Spells a size the way `parse_size` reads it: `256MB`, `1000`."""
    count, unit = int(size), ""
    for larger in ("KB", "MB", "GB", "TB"):
        if count == 0 or count % 1024 != 0:
            break
        count, unit = count // 1024, larger
    return f"{count}{unit}"


_MASK64 = (1 << 64) - 1

# SplitMix64's increment, the golden ratio in 64 bits.
SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15


def mix(value: int) -> int:
    """SplitMix64's finalizer, a bijection that spreads every input bit over the whole output."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def stream_key(seed: Seed, name: str, index: int = 0) -> int:
    """The key of stream `index` of `name`, two mixes away from `seed`, so neighboring seeds and indices never meet."""
    hashed = 0xCBF29CE484222325
    for byte in name.encode():
        hashed = ((hashed ^ byte) * 0x100000001B3) & _MASK64
    return mix((mix(seed ^ hashed) + index) & _MASK64)


@dataclass
class SplitMix64:
    """A SplitMix64 stream seeded with a `stream_key`, bit-identical to the C++ `splitmix64_t`."""

    state: int

    def next(self) -> int:
        """The next 64 random bits."""
        self.state = (self.state + SPLITMIX64_GAMMA) & _MASK64
        return mix(self.state)

    def below(self, bound: int) -> int:
        """A draw below `bound`: the high half of a draw times `bound`, with a bias of at most `bound` over 2^64."""
        return (self.next() * bound) >> 64


# endregion: Environment variables


# region: Machine


def print_machine(libraries: Mapping[str, str]) -> None:
    """Prints the interpreter and the platform, then each library as "- Name: version", ahead of the settings."""
    print(f"Python {platform.python_version()}")
    print(f"- This machine: {platform.machine()}, {sys.platform}")
    for name, version in libraries.items():
        print(f"- {name}: {version}")


# Multiprocessors assumed when sizing a GPU batch without a GPU to ask.
FALLBACK_GPU_MULTIPROCESSORS = 64


def batch_size(settings: "Settings", cores: int) -> int:
    """Batch size for a backend with `cores` parallel cores: `STRINGWARS_BATCH_PER_CORE`, or the
    suite's `batch_per_core` in `stringwars.toml`, times `cores`. A CPU core and a GPU streaming
    multiprocessor (an SM, not a warp or a CUDA core) each count as one core, so the batch scales
    with the hardware instead of a fixed CPU/GPU multiplier. Mirrors the Rust `batch_size`.
    """
    if settings.batch_per_core is None:
        raise ValueError(f"suite {settings.suite!r} has no `batch_per_core` in stringwars.toml")
    return settings.batch_per_core * max(1, cores)


def gpu_multiprocessor_count(device_index: int = 0) -> int | None:
    """Number of streaming multiprocessors on the given CUDA device, queried straight from the
    CUDA runtime via ctypes (no cupy/torch needed). Each SM is counted as one core for batch
    sizing (an SM, not a warp or an individual CUDA core). Returns None when CUDA is unavailable
    or the query fails, so callers fall back to `FALLBACK_GPU_MULTIPROCESSORS`. Attribute id 16
    is `cudaDevAttrMultiProcessorCount`.
    """
    import ctypes
    import ctypes.util

    candidates = ["libcudart.so", "libcudart.so.12", ctypes.util.find_library("cudart")]
    multiprocessor_count_attribute = 16
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        count = ctypes.c_int(0)
        status = library.cudaDeviceGetAttribute(
            ctypes.byref(count),
            ctypes.c_int(multiprocessor_count_attribute),
            ctypes.c_int(device_index),
        )
        if status == 0 and count.value > 0:
            return count.value
    return None


# endregion: Machine


# region: Settings


class Tokenization(StrEnum):
    """How `STRINGWARS_TOKENS` splits the dataset; mirrors `stringwars.rs::Tokenization`."""

    LINES = "lines"
    WORDS = "words"
    FILE = "file"


# Whether repeated tokens stay in the working set; `STRINGWARS_UNIQUE=1` drops them.
Duplicates = Literal["keep", "drop"]


@functools.cache
def manifest() -> dict[str, Any]:
    """`stringwars.toml` — the single source of defaults shared with `stringwars.rs`."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stringwars.toml")
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def manifest_parsed[T](entry: Mapping[str, object], key: str, parse: Callable[[str], T | None], expected: str) -> T:
    """Parses `key` of a suite's manifest entry; a malformed `stringwars.toml` is a repository bug, so it raises."""
    parsed = parse(str(entry[key]))
    if parsed is None:
        raise ValueError(f"stringwars.toml: `{key}` is not {expected}")
    return parsed


@dataclass(frozen=True)
class Settings:
    """Every setting of one suite, read once at the top of `main` and passed down. Defaults come
    from `stringwars.toml`, the suite's `[suite]` entry first and `[limits]` second, so neither
    harness carries its own. Mirrors the Rust `Settings` field for field, less `counters` and
    `collisions`."""

    suite: str
    seed: Seed
    filter: str
    filter_pattern: re.Pattern[str] | None
    warmup: Seconds
    time_limit: Seconds
    dataset_bytes: Bytes
    batch_per_core: int | None
    threads: Threads
    dims: list[int]
    dataset_path: str
    tokenization: Tokenization
    duplicates: Duplicates
    min_measure_time: Seconds
    min_sample_time: Seconds
    min_samples: int
    target_spread: float
    results_dir: str | None

    def selects(self, name: str) -> bool:
        """Whether `STRINGWARS_FILTER` selects the row `name`, as a regex or else as a substring.

        Records a filtered row as it decides, because several suites consult the filter
        themselves and return early; those rows used to vanish from the tally entirely.
        """
        selected = bool(self.filter_pattern.search(name)) if self.filter_pattern else self.filter in name
        if not selected:
            _ROSTER.append((name, "filtered"))  # StringWars-only: the roster tallies filtered rows.
        return selected


def read_settings(suite: str) -> Settings:
    """Reads every `STRINGWARS_*` variable `suite` uses, exiting with status 1 on the first that does not parse."""
    entry = {**manifest()["limits"], **manifest()["suite"][suite]}
    filter = env_text("STRINGWARS_FILTER") or ""
    try:
        filter_pattern = re.compile(filter) if filter else None
    except re.error:
        filter_pattern = None
    return Settings(
        suite=suite,
        seed=env_seed("STRINGWARS_SEED", manifest_parsed(entry, "seed", parse_seed, "a 32-bit integer")),
        filter=filter,
        filter_pattern=filter_pattern,
        warmup=env_duration("STRINGWARS_WARMUP", manifest_parsed(entry, "warmup", parse_duration, "a duration")),
        time_limit=env_duration(
            "STRINGWARS_TIME_LIMIT", manifest_parsed(entry, "time_limit", parse_duration, "a duration")
        ),
        dataset_bytes=env_size("STRINGWARS_BYTES", manifest_parsed(entry, "bytes", parse_size, "a size")),
        batch_per_core=env_parsed(
            "STRINGWARS_BATCH_PER_CORE", entry.get("batch_per_core"), parse_count, "a positive count"
        ),
        threads=env_parsed(
            "STRINGWARS_THREADS", Threads(os.process_cpu_count() or 1), parse_threads, "a count, 0 for all cores"
        ),
        dims=env_parsed(
            "STRINGWARS_DIMS",
            manifest_parsed(entry, "dims", parse_dims, "a count list") if "dims" in entry else [],
            parse_dims,
            "positive counts like 64,128",
        ),
        dataset_path=env_text("STRINGWARS_DATASET") or entry["dataset"],
        tokenization=env_parsed(
            "STRINGWARS_TOKENS", Tokenization(entry.get("tokens", "lines")), Tokenization, "lines, words or file"
        ),
        duplicates="drop" if env_flag("STRINGWARS_UNIQUE", False) else "keep",
        min_measure_time=manifest_parsed(entry, "min_measure_time", parse_duration, "a duration"),
        min_sample_time=manifest_parsed(entry, "min_sample_time", parse_duration, "a duration"),
        min_samples=env_count("STRINGWARS_MIN_SAMPLES", entry["min_samples"]),
        target_spread=env_parsed(
            "STRINGWARS_TARGET_SPREAD",
            float(entry["target_spread"]),
            lambda text: spread if 0 < (spread := float(text)) < math.inf else None,
            "a fraction like 0.025",
        ),
        results_dir=env_text("STRINGWARS_RESULTS_DIR"),
    )


def print_settings(settings: Settings) -> None:
    """Prints every setting as "- Name: value", in the grammar it is read in."""
    print(f"- Seed: {settings.seed}")
    print(f"- Filter: {settings.filter or 'none'}")
    print(f"- Warm-up: {spell_duration(settings.warmup)}")
    print(f"- Time limit: {spell_duration(settings.time_limit)}")
    print(f"- Bytes: {spell_size(settings.dataset_bytes)}")
    if settings.batch_per_core is not None:
        print(f"- Batch per core: {settings.batch_per_core}")
    print(f"- Threads: {settings.threads}")
    if settings.dims:
        print(f"- Dims: {','.join(map(str, settings.dims))}")
    print(f"- Dataset: {settings.dataset_path}")
    print(f"- Tokens: {settings.tokenization}")
    print(f"- Unique: {str(settings.duplicates == 'drop').lower()}")
    print(f"- Min samples: {settings.min_samples}")
    print(f"- Target spread: {settings.target_spread}")
    print(f"- Results dir: {settings.results_dir or 'none'}")


# endregion: Settings


# region: Working set

_ASCII_WHITESPACE = b" \n\t\r\v\f"
_WORD_SEPARATORS = re.compile(b"[" + re.escape(_ASCII_WHITESPACE) + b"]+")


def tokenize_dataset(haystack: bytes, tokenization: Tokenization, duplicates: Duplicates) -> list[bytes]:
    """
    Split a buffer into tokens. Normative definition, shared with `stringwars.rs`:

      lines  split on \\n, drop empty
      words  split on ASCII whitespace {space \\n \\t \\r \\v \\f}, drop empty
      file   one token, the whole buffer

    Both harnesses must produce identical token counts and bytes; the identity CRC checks it.
    """
    match tokenization:
        case Tokenization.LINES:
            tokens = [token for token in haystack.split(b"\n") if token]
        case Tokenization.WORDS:
            tokens = [token for token in _WORD_SEPARATORS.split(haystack) if token]
        case Tokenization.FILE:
            return [haystack]
    return list(dict.fromkeys(tokens)) if duplicates == "drop" else tokens


@dataclass(frozen=True)
class Dataset:
    """A pinned working set. `token_bytes` is the denominator of every bytes/s figure."""

    tokens: list[bytes]
    token_bytes: Bytes
    token_count: int
    fingerprint: int
    tokenization: Tokenization
    path: str

    def text_tokens(self) -> list[str]:
        """The tokens decoded, for suites that benchmark `str` APIs."""
        return [token.decode("utf-8", errors="ignore") for token in self.tokens]


def _char_boundary_floor(raw: bytes) -> int:
    """Length of `raw` without an incomplete UTF-8 sequence at its end. Mirrors `stringwars.rs::char_boundary_floor`."""
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        if error.reason == "unexpected end of data":
            return error.start
    return len(raw)


def resolve_dataset(settings: Settings) -> Dataset:
    """
    Resolve the working set for one suite, identically to `stringwars.rs`: read the first
    `STRINGWARS_BYTES` of the dataset, round the read down to a power of two as StringZilla
    does, back off to a UTF-8 boundary, and keep every token in it, deduplicated under
    `STRINGWARS_UNIQUE`.
    """
    with open(settings.dataset_path, "rb") as handle:
        raw = handle.read(settings.dataset_bytes)
    if raw:
        raw = raw[: 1 << (len(raw).bit_length() - 1)]
    raw = raw[: _char_boundary_floor(raw)]

    tokens = tokenize_dataset(raw, settings.tokenization, settings.duplicates)
    if not tokens:
        raise ValueError(f"No tokens from {settings.dataset_path} in mode {settings.tokenization}")

    dataset = Dataset(
        tokens=tokens,
        token_bytes=Bytes(sum(map(len, tokens))),
        token_count=len(tokens),
        fingerprint=fingerprint_tokens(tokens),
        tokenization=settings.tokenization,
        path=settings.dataset_path,
    )
    note_working_set(settings, dataset)
    return dataset


def fingerprint_tokens(tokens: Iterable[bytes]) -> int:
    """
    Identity of a resolved working set: CRC32 over the tokens joined by a NUL.

    Deterministic across processes (unlike `hash()`, which is seed-randomized) and
    trivially mirrored in Rust with `crc32fast`, which the tree already depends on.
    Digesting the tokens rather than the file means capping the working set does
    not require re-hashing gigabytes.
    """
    return zlib.crc32(b"\x00".join(tokens)) & 0xFFFFFFFF


def log_dataset(dataset: Dataset) -> None:
    """
    Print the working set's identity, in the same format as `stringwars.rs`.

    This line is what makes the two harnesses checkable against each other: same
    mode, count, bytes and CRC means they resolved the same working set. They
    silently diverged for months (Rust split words on ASCII bytes, Python's
    `str.split()` also split U+3000 in the CJK corpora) and nothing in the output
    would have shown it.
    """
    gibibytes = dataset.token_bytes / 1024**3
    print(f"Dataset: {dataset.token_count:,} tokens, {dataset.token_bytes:,} bytes ({gibibytes:.2f} GB)")
    print(f"  Identity: mode {dataset.tokenization} crc 0x{dataset.fingerprint:08x}")


# endregion: Working set


# region: Reporting

# Width of the left-aligned row-name column, matching `stringwars.rs` and the C++ harnesses.
REPORT_NAME_WIDTH = 56

# Fewest samples a dispersion estimate needs; also the warm-up agreement window.
DISPERSION_SAMPLES = 3

# Cap on passes per sample, so calibrating a sub-nanosecond pass cannot run away.
MAX_PASSES_PER_SAMPLE = 1 << 40

# The primary unit a row reports; bytes/s is always shown as the secondary metric.
Unit = Literal["bytes", "cups", "hashes", "bits", "comparisons"]

# What became of one row the run reached, for the roster `finish` tallies.
RowStatus = Literal["measured", "refused", "too_slow", "skipped", "filtered"]

# How a row ended; mirrors `stringwars.rs::Status`.
Status = Literal["converged", "unconverged", "non_stationary", "too_few_samples", "too_slow", "filtered"]

# Each unit as NDJSON records name it, then its binary or decimal prefix step and its display suffix.
_UNITS: dict[Unit, tuple[str, int, str]] = {
    "bytes": ("bytes/s", 1024, "B/s"),
    "cups": ("CUPS", 1000, "CUPS"),
    "hashes": ("hashes/s", 1000, "hashes/s"),
    "bits": ("bits/s", 1000, "bits/s"),
    "comparisons": ("cmp/s", 1000, "cmp/s"),
}


@dataclass(frozen=True)
class RunIdentity:
    """What every NDJSON record is stamped with; mirrors `stringwars.rs::RunIdentity`."""

    suite: str
    dataset: str
    tokenization: Tokenization
    tokens: int
    token_bytes: Bytes
    crc: int


# Captured once when the working set resolves. Module-level because `measure` is handed only
# the row it is timing, and threading run context through every call site would be noise.
_RUN: RunIdentity | None = None

# The record fields that must agree across languages for a suite to be conformant.
_IDENTITY_FIELDS = ("dataset", "mode", "tokens", "token_bytes", "crc")

# Every row the run reached, in order, with what became of it. A row that vanishes
# silently is indistinguishable from a row that was never written: `similarities`
# lost four whole tables behind a `--bio` gate and the output looked complete.
_ROSTER: list[tuple[str, RowStatus]] = []


def note_working_set(settings: Settings, dataset: Dataset) -> None:
    """Registers the working set every NDJSON record is stamped with."""
    global _RUN
    _RUN = RunIdentity(
        suite=settings.suite,
        dataset=settings.dataset_path,
        tokenization=dataset.tokenization,
        tokens=dataset.token_count,
        token_bytes=dataset.token_bytes,
        crc=dataset.fingerprint,
    )


def print_row(name: str, text: str) -> None:
    """Prints one result line: the name in the shared 56-column field, then `text`."""
    print(f"{name:<{REPORT_NAME_WIDTH}} {text}")


def _earned_digits(relative_halfwidth: float) -> int:
    """Significant figures a relative half-width earns: the most, up to 4, with `10^-(digits-1)` still above it."""
    return max((digits for digits in range(1, 5) if 10.0 ** (-(digits - 1)) >= relative_halfwidth), default=1)


def _round_to_earned_digits(value: float, relative_halfwidth: float) -> float:
    """Round to the significant figures the measured dispersion justifies."""
    if value == 0.0 or not math.isfinite(value):
        return value
    magnitude = math.floor(math.log10(abs(value)))
    scale = 10.0 ** (_earned_digits(relative_halfwidth) - 1 - magnitude)
    # Half-up, spelled out, for the same reason as `_quantile`: rates are positive,
    # so this is the tie rule both harnesses can state rather than inherit.
    return math.floor(value * scale + 0.5) / scale


def spell_rate(per_second: float, relative_halfwidth: float, unit: Unit) -> str:
    """Spells `per_second` with the significant figures a `relative_halfwidth` earns, so a nonzero
    rate never prints as 0: "47.6 cmp/s", "1.2 GCUPS", "812 MB/s". Byte rates take binary prefixes."""
    _, step, suffix = _UNITS[unit]
    prefixes = ("", "K", "M", "G", "T") if step == 1024 else ("", "k", "M", "G", "T")
    value, prefix = per_second, prefixes[0]
    for larger in prefixes[1:]:
        if value < step:
            break
        value, prefix = value / step, larger
    shown = _round_to_earned_digits(value, relative_halfwidth)
    decimals = 0
    if shown != 0.0 and math.isfinite(shown):
        decimals = max(0, _earned_digits(relative_halfwidth) - 1 - math.floor(math.log10(abs(shown))))
    return f"{shown:.{decimals}f} {prefix}{suffix}"


def note_unavailable(name: str, reason: str) -> None:
    """
    Record a contender that could not run at all — a missing optional dependency, a
    gated backend. Prints a line so the absence is in the output rather than implied
    by a gap in the table.
    """
    print_row(name, f"SKIPPED: {reason}")
    _ROSTER.append((name, "skipped"))


def finish() -> int:
    """Prints the roster tally and returns the exit status: 1 if a row that was asked to run produced nothing."""
    tally = Counter(status for _, status in _ROSTER)
    print(
        f"\nRoster: {tally['measured']} measured, {tally['refused']} refused, "
        f"{tally['too_slow']} too slow, {tally['skipped']} skipped, {tally['filtered']} filtered",
    )
    refused = [name for name, status in _ROSTER if status == "refused"]
    for name in refused:
        print(f"  refused: {name}")
    return 1 if refused else 0


@dataclass(frozen=True)
class MeasureSpec:
    """
    What one pass over the pinned working set costs, declared up front.

    Declared rather than accumulated: counting work per call costs a Python-level
    increment per item, which at ~50-80 ns swamps a short kernel.
    """

    unit: Unit
    elements: int
    total_bytes: Bytes
    concurrency: Threads = Threads(1)


@dataclass(frozen=True)
class Outcome:
    """One finished row: its rate and dispersion, or how it ended without a number."""

    name: str
    median_rate: float
    spread: float
    samples: int
    passes_per_sample: int
    status: Status
    status_detail: float | None = None  # the gap behind `non_stationary`, the spread behind `unconverged`


def _record_outcome(settings: Settings, outcome: Outcome, spec: MeasureSpec, bytes_per_second: float) -> None:
    """
    Append one NDJSON record per row when `STRINGWARS_RESULTS_DIR` is set.

    Records exist so a published number can be traced back to the working set that
    produced it. They never touch a README, whose tables stay hand-written.
    """
    directory = settings.results_dir
    if not directory or _RUN is None or outcome.status == "filtered":
        return
    status = outcome.status if outcome.status_detail is None else f"{outcome.status}:{outcome.status_detail:.4f}"
    record = {
        "lang": "python",
        "suite": _RUN.suite,
        "dataset": _RUN.dataset,
        "mode": str(_RUN.tokenization),
        "tokens": _RUN.tokens,
        "token_bytes": _RUN.token_bytes,
        "crc": f"0x{_RUN.crc:08x}",
        "row": outcome.name,
        "unit": _UNITS[spec.unit][0],
        "rate": outcome.median_rate,
        "bytes_per_second": bytes_per_second,
        "spread": outcome.spread,
        "samples": outcome.samples,
        "passes_per_sample": outcome.passes_per_sample,
        "concurrency": spec.concurrency,
        "status": status,
    }
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{_RUN.suite}.ndjson"), "a") as handle:
        handle.write(json.dumps(record) + "\n")


def clock_overhead_nanoseconds() -> float:
    """
    Cost of one `time.monotonic_ns()`. The harness reads the clock exactly twice
    per sample, so this over `min_sample_time` is the whole timing overhead.
    """
    rounds = 10_000
    start = time.monotonic_ns()
    for _ in range(rounds):
        time.monotonic_ns()
    return (time.monotonic_ns() - start) / rounds


def log_timing_overhead(settings: Settings) -> None:
    """Proof obligation behind "the harness does not perturb the measurement"."""
    per_clock = clock_overhead_nanoseconds()
    floor_nanoseconds = settings.min_sample_time * 1e9
    print(
        f"Timing: {per_clock:.0f} ns/clock, {floor_nanoseconds / 1e6:.1f} ms sample floor "
        f"-> harness overhead <= {200.0 * per_clock / floor_nanoseconds:.4f}%",
    )


def _quantile(ordered: list[float], fraction: float) -> float:
    """
    Select a sample by rank, using a rule written out rather than delegated.

    Python's `round` breaks ties to even and Rust's `f64::round` breaks them away
    from zero, so the two harnesses picked different samples: at n = 10 — the modal
    terminating count, since `min_samples` is 10 and the convergence gate fires as
    soon as it is met — Python took the 5th-smallest and Rust the 6th. Every median
    rate and spread is derived from this, so the disagreement was systematic.
    """
    if not ordered:
        return 0.0
    rank = math.floor(fraction * (len(ordered) - 1) + 0.5)
    return ordered[min(rank, len(ordered) - 1)]


def measure(
    settings: Settings,
    name: str,
    spec: MeasureSpec,
    run_pass: Callable[[], object],
    setup: Callable[[], None] | None = None,
) -> Outcome:
    """
    Times `run_pass` over the pinned working set and reports one row.

    A sample is `passes_per_sample` complete traversals bracketed by exactly two
    clock reads; the count is chosen during warm-up so a sample lasts at least
    `min_sample_time`. There is no per-call clock and therefore no adaptive stride,
    and the loop between the two reads contains no harness bookkeeping.

    `setup` runs before each sample and outside the clock, for kernels that consume
    their input. It forces one pass per sample, since a rebuild between passes would
    land inside the timed span.
    """

    def run_sample(passes: int) -> int:
        """Nanoseconds for `passes` passes."""
        if setup is not None:
            setup()
        started = time.monotonic_ns()
        for _ in range(passes):
            run_pass()
        return time.monotonic_ns() - started

    def refuse(status: Status, samples: int, text: str) -> Outcome:
        # A refusal must be as visible as a result; printing nothing is how a row
        # goes missing for weeks without anyone noticing. A too-slow row is an expected
        # outcome, not a failure, so it gets its own bucket and does not force exit 1.
        print_row(name, text)
        _ROSTER.append((name, "too_slow" if status == "too_slow" else "refused"))
        return Outcome(name, 0.0, float("nan"), samples, 0, status)

    # `selects` already noted a filtered row.
    if not settings.selects(name):
        return Outcome(name, 0.0, float("nan"), 0, 0, "filtered")

    floor_nanoseconds = settings.min_sample_time * 1e9

    passes = 1
    if setup is None:
        while True:
            sample_nanoseconds = run_sample(passes)
            if sample_nanoseconds >= floor_nanoseconds or passes >= MAX_PASSES_PER_SAMPLE:
                break
            shortfall = math.ceil(floor_nanoseconds / max(sample_nanoseconds, 1))
            passes = min(max(2, shortfall) * passes, MAX_PASSES_PER_SAMPLE)
    else:
        # A consuming kernel cannot batch passes, but one sample still prices one. Skipping
        # it left the cost at zero, so the refusal below could never fire for `sequence` - the
        # one suite whose work is superlinear and therefore most able to outrun the cap.
        sample_nanoseconds = run_sample(passes)

    # Calibration is the first moment the cost of a pass is known. If one pass already
    # rules out the three samples a dispersion estimate needs, stop here rather than
    # after warm-up and a measured sample have each paid it again.
    pass_seconds = Seconds(sample_nanoseconds / 1e9 / passes)
    if pass_seconds * DISPERSION_SAMPLES > settings.time_limit:
        estimated_rate = spec.total_bytes / max(pass_seconds, 1e-9)
        return refuse(
            "too_slow",
            1,
            f"TOO SLOW: ~{spell_rate(estimated_rate, 1.0, 'bytes')}, one pass ~{spell_duration(pass_seconds)}"
            f" against a {spell_duration(settings.time_limit)} cap",
        )

    warmup_deadline = time.monotonic_ns() + int(settings.warmup * 1e9)
    recent_sample_seconds: deque[float] = deque(maxlen=DISPERSION_SAMPLES)
    # A single pass longer than the deadline used to yield zero warm-up samples, so the
    # row entered measurement cold - what warm-up exists to prevent.
    while len(recent_sample_seconds) < DISPERSION_SAMPLES or time.monotonic_ns() < warmup_deadline:
        recent_sample_seconds.append(run_sample(passes) / 1e9)
        if len(recent_sample_seconds) == DISPERSION_SAMPLES:
            low, high = min(recent_sample_seconds), max(recent_sample_seconds)
            if (high - low) / max(high, 1e-12) <= settings.target_spread:
                break

    measure_start = time.monotonic_ns()
    cap_nanoseconds = settings.time_limit * 1e9
    seconds_per_sample: list[float] = []
    while True:
        seconds_per_sample.append(run_sample(passes) / 1e9)

        elapsed_nanoseconds = time.monotonic_ns() - measure_start
        enough = len(seconds_per_sample) >= settings.min_samples
        if enough and elapsed_nanoseconds / 1e9 >= settings.min_measure_time:
            ordered = sorted(seconds_per_sample)
            median = _quantile(ordered, 0.5)
            halfwidth = (_quantile(ordered, 0.9) - _quantile(ordered, 0.1)) / (2.0 * max(median, 1e-12))
            if halfwidth <= settings.target_spread:
                break
        if elapsed_nanoseconds >= cap_nanoseconds:
            break

    if len(seconds_per_sample) < DISPERSION_SAMPLES:
        collected = len(seconds_per_sample)
        cap = spell_duration(settings.time_limit)
        return refuse(
            "too_few_samples", collected, f"REFUSED: {collected} samples in {cap} cap (need {DISPERSION_SAMPLES})"
        )

    ordered = sorted(seconds_per_sample)
    median_seconds = _quantile(ordered, 0.5)
    spread = (_quantile(ordered, 0.9) - _quantile(ordered, 0.1)) / max(median_seconds, 1e-12)
    mean_seconds = sum(seconds_per_sample) / len(seconds_per_sample)
    gap = abs(mean_seconds - median_seconds) / max(median_seconds, 1e-12)

    # `status` is what the NDJSON carries and must match the string `stringwars.rs` writes, or a
    # reader cannot compare the two languages; the console label only appears on the printed row.
    status: Status
    if gap > 2.0 * settings.target_spread:
        status, detail, label = "non_stationary", gap, f"NON-STATIONARY {100.0 * gap:.0f}%"
    elif spread / 2.0 > settings.target_spread:
        status, detail, label = "unconverged", spread, f"UNCONVERGED {100.0 * spread:.0f}%"
    else:
        status, detail, label = "converged", None, ""

    elements_per_second = spec.elements * passes / median_seconds
    bytes_per_second = spec.total_bytes * passes / median_seconds
    primary = bytes_per_second if spec.unit == "bytes" else elements_per_second

    halfwidth = spread / 2.0
    columns = [spell_rate(primary, halfwidth, spec.unit)]
    if spec.unit != "bytes" and spec.total_bytes > 0:
        columns.append(spell_rate(bytes_per_second, halfwidth, "bytes"))
    columns.append(f"+-{100.0 * halfwidth:.1f}% n={len(seconds_per_sample)}")
    if label:
        columns.append(label)
    print_row(name, " | ".join(columns))

    outcome = Outcome(name, primary, spread, len(seconds_per_sample), passes, status, detail)
    _record_outcome(settings, outcome, spec, bytes_per_second)
    _ROSTER.append((name, "measured"))
    return outcome


def measure_with_setup[State](
    settings: Settings,
    name: str,
    spec: MeasureSpec,
    setup: Callable[[], State],
    body: Callable[[State], object],
) -> Outcome:
    """
    Like `measure`, for kernels that consume their input — sorting, in-place edits.

    A thin wrapper over `measure` rather than a second implementation: the copy this
    replaced had no warm-up, no pass calibration, no `min_sample_time` floor and no
    non-stationary test, printed a different refusal wording, and recorded
    `status="converged"` into the NDJSON even when the line it had just printed said
    `UNCONVERGED` — so the record contradicted the console.
    """
    prepared: list[State] = []

    def prepare() -> None:
        prepared[:] = [setup()]

    return measure(settings, name, spec, lambda: body(prepared[0]), setup=prepare)


def pass_over(function: Callable[..., object], *columns: Iterable[Any]) -> Callable[[], None]:
    """
    Build a pass that drives `function` across whole columns inside C.

    The interpreter must not appear in the measured loop: a Python-level `for` body
    costs ~50-80 ns per item, which swamps a short kernel and gets attributed to it.
    `deque(..., maxlen=0)` drains the `map` at C speed and keeps no results.
    """

    def run() -> None:
        deque(map(function, *columns), maxlen=0)

    return run


# endregion: Reporting


def check_conformance(directory: str) -> int:
    """
    Verify the two harnesses resolved the same working set, from the records they wrote.

    Run both languages of a suite with `STRINGWARS_RESULTS_DIR` pointed here, then
    `uv run stringwars.py <dir>`. A mismatch means the numbers in that suite's table are
    not comparable, however plausible they look side by side — Rust and Python
    silently disagreed on tokenization for months and no output showed it.

    Records append, so a directory accumulates history. Only each language's most
    recent identity is compared: otherwise a since-fixed disagreement keeps failing.
    """
    import glob

    failures = 0
    for path in sorted(glob.glob(os.path.join(directory, "*.ndjson"))):
        suite = os.path.basename(path).removesuffix(".ndjson")
        identities: dict[str, tuple[object, ...]] = {}
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                identities[record["lang"]] = tuple(record[field] for field in _IDENTITY_FIELDS)
        distinct = set(identities.values())
        if len(identities) < 2:
            print(f"{suite:<16} SKIP  only {'/'.join(identities) or 'no'} records")
        elif len(distinct) == 1:
            _, mode, tokens, token_bytes, crc = distinct.pop()
            print(f"{suite:<16} OK    {mode} {tokens:,} tokens, {token_bytes:,} bytes, crc {crc}")
        else:
            failures += 1
            print(f"{suite:<16} FAIL  harnesses resolved different working sets")
            for lang, values in sorted(identities.items()):
                # Pair each value with its field name. Sorting the tuple instead raised
                # TypeError on int-against-str, crashing the one branch that reports a
                # mismatch.
                fields = " ".join(f"{f}={v}" for f, v in zip(_IDENTITY_FIELDS, values, strict=True))
                print(f"                  {lang:<8} {fields}")
    return failures


if __name__ == "__main__":
    raise SystemExit(check_conformance(sys.argv[1] if len(sys.argv) > 1 else "results"))
