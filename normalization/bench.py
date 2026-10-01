"""Unicode normalization and case-insensitive comparison benchmarks in Python. Mirrors `normalization/bench.rs`.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group normalization normalization/bench.py
"""

import argparse
import unicodedata
from collections.abc import Callable
from functools import partial
from importlib.metadata import version as pkg_version
from typing import Literal

import icu
import pyunormalize
import regex
import stringzilla as sz

from stringwars import (
    Bytes,
    MeasureSpec,
    Settings,
    SplitMix64,
    Tokenization,
    finish,
    log_dataset,
    log_timing_overhead,
    measure,
    note_unavailable,
    pass_over,
    print_machine,
    print_settings,
    read_settings,
    resolve_dataset,
    stream_key,
    tokenize_dataset,
)

NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


def bench_case_compare(
    settings: Settings,
    name: str,
    lefts: list[str],
    rights: list[str],
    compare_function: Callable[[str, str], bool],
    total_bytes: Bytes,
) -> None:
    """One pass compares every pair, so the pair mixture is identical in every sample."""
    if not lefts:
        note_unavailable(name, "fewer than two tokens to pair")
        return
    work = MeasureSpec(unit="bytes", elements=len(lefts), total_bytes=total_bytes)
    measure(settings, name, work, pass_over(compare_function, lefts, rights))


def compare_casefold(first_string: str, second_string: str) -> bool:
    return first_string.casefold() == second_string.casefold()


def compare_regex_fullcase(first_string: str, second_string: str) -> bool:
    """Escape, compile and full-match, all inside the timed pass — hence the row's name.

    Precompiling is not an option: the corpus holds millions of distinct words and `regex`
    caches only 500 patterns, so a hoisted variant would thrash that cache under its lock.
    `regex.escape` is a per-character Python loop and is paid on every call either way.
    """
    pattern = regex.compile(regex.escape(first_string), regex.IGNORECASE | regex.FULLCASE)
    return pattern.fullmatch(second_string) is not None


def compare_icu(first_string: str, second_string: str) -> bool:
    first_folded = icu.UnicodeString(first_string).foldCase()
    second_folded = icu.UnicodeString(second_string).foldCase()
    return first_folded == second_folded


def compare_stringzilla(first_string: str, second_string: str) -> bool:
    """Compare using StringZilla's utf8_uncased_order."""
    return sz.utf8_uncased_order(first_string, second_string) == 0


def bench_case_find(
    settings: Settings,
    name: str,
    haystack: str,
    needles: list[str],
    find_function: Callable[[str, str], int],
    haystack_bytes: Bytes,
) -> None:
    """One pass searches every needle across the whole haystack."""
    if not needles:
        note_unavailable(name, "no needles to search")
        return
    work = MeasureSpec(
        unit="bytes",
        elements=len(needles),
        total_bytes=Bytes(haystack_bytes * len(needles)),
    )
    measure(settings, name, work, pass_over(partial(find_function, haystack), needles))


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
    """Count occurrences with IGNORECASE | FULLCASE, escaping and compiling per needle.

    Only 16 needles run here, so the compile is a cache hit; the escape is still a
    per-character Python loop, and the row's name says so.
    """
    if not needle:
        return 0
    pattern = regex.compile(regex.escape(needle), regex.IGNORECASE | regex.FULLCASE)
    # Counted lazily, so the Python loop overhead matches the StringZilla row.
    return sum(1 for _ in pattern.finditer(haystack))


def make_find_icu() -> Callable[[str, str], int]:
    """Build an ICU StringSearch counter over one reused collator.

    The collator does not depend on the needle, so it is built here rather than inside the
    timed pass; only the `StringSearch` binding a needle to the haystack stays per-call.
    """
    collator = icu.Collator.createInstance(icu.Locale.getRoot())
    collator.setStrength(icu.Collator.SECONDARY)  # Case-insensitive

    def find(haystack: str, needle: str) -> int:
        if not needle:
            return 0
        searcher = icu.StringSearch(needle, haystack, collator)
        count = 0
        pos = searcher.nextMatch()
        while pos != -1:
            count += 1
            pos = searcher.nextMatch()
        return count

    return find


def find_stringzilla(haystack: str, needle: str) -> int:
    """Count occurrences using StringZilla's utf8_uncased_matches."""
    if not needle:
        return 0
    return sum(1 for _ in sz.utf8_uncased_matches(haystack, needle))


def bench_case_fold(
    settings: Settings,
    name: str,
    strings: list[str],
    fold_function: Callable[[str], str | bytes],
    total_bytes: Bytes,
) -> None:
    """One pass folds every string."""
    if not strings:
        note_unavailable(name, "nothing to process")
        return
    work = MeasureSpec(unit="bytes", elements=len(strings), total_bytes=total_bytes)
    measure(settings, name, work, pass_over(fold_function, strings))


def fold_casefold(s: str) -> str:
    """Fold using Python's str.casefold() - full Unicode."""
    return s.casefold()


def fold_stringzilla(s: str) -> bytes:
    """Fold using StringZilla's utf8_uncased_fold() - full Unicode."""
    return sz.utf8_uncased_fold(s)


def fold_icu(s: str) -> str:
    return str(icu.UnicodeString(s).foldCase())


NORMALIZATION_FORMS: tuple[NormalizationForm, ...] = ("NFC", "NFD", "NFKC", "NFKD")


def bench_normalize(
    settings: Settings,
    name: str,
    strings: list[str],
    normalize_function: Callable[[str], str | bytes],
    total_bytes: Bytes,
) -> None:
    """One pass normalizes every string."""
    if not strings:
        note_unavailable(name, "nothing to process")
        return
    work = MeasureSpec(unit="bytes", elements=len(strings), total_bytes=total_bytes)
    measure(settings, name, work, pass_over(normalize_function, strings))


def normalize_stringzilla(form: NormalizationForm, s: str) -> bytes:
    """Normalize using StringZilla's utf8_norm() - returns raw UTF-8 bytes."""
    return sz.utf8_norm(s, form)


def normalize_stdlib(form: NormalizationForm, s: str) -> str:
    """Normalize using Python's unicodedata.normalize()."""
    return unicodedata.normalize(form, s)


def normalize_pyunormalize(form: NormalizationForm, s: str) -> str:
    """Normalize using `pyunormalize`, a pure-Python implementation of UAX #15.

    Included as the reference point for what the algorithm costs without a native
    extension behind it - it ships its own Unicode tables rather than deferring to
    the interpreter's, so it also serves as an independent oracle for the forms.
    """
    return pyunormalize.normalize(form, s)


def make_normalize_icu(form: NormalizationForm) -> Callable[[str], str]:
    """Build an ICU Normalizer2-backed normalizer for one form.

    The `Normalizer2` instance is constructed once here, outside the hot loop, so
    the benchmark measures normalization rather than instance lookup. NFC/NFKC use
    COMPOSE, NFD/NFKD use DECOMPOSE; the underlying data set is `nfc` for the
    canonical forms and `nfkc` for the compatibility forms.
    """
    data_name = "nfkc" if form in ("NFKC", "NFKD") else "nfc"
    mode = icu.UNormalizationMode2.COMPOSE if form in ("NFC", "NFKC") else icu.UNormalizationMode2.DECOMPOSE
    normalizer = icu.Normalizer2.getInstance(None, data_name, mode)

    def normalize(s: str) -> str:
        return normalizer.normalize(s)

    return normalize


def strict_text(word: bytes) -> str | None:
    """The word decoded, or `None` if it is not valid UTF-8, as Rust's `str::from_utf8` decides."""
    try:
        return word.decode("utf-8")
    except UnicodeDecodeError:
        return None


# One pass scans the haystack once per needle, matching `find` and the Rust side.
NEEDLES_PER_PASS = 16


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    print_machine(
        {
            "StringZilla": f"{sz.__version__} with {sz.__capabilities_str__}",
            "regex": pkg_version("regex"),
            "PyICU": f"{pkg_version('PyICU')} with ICU {icu.ICU_VERSION}",
            "pyunormalize": f"{pkg_version('pyunormalize')} with Unicode {pyunormalize.UCD_VERSION}",
        }
    )
    settings = read_settings("normalization")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    tokens = dataset.text_tokens()
    pythonic_str = "".join(tokens)
    log_dataset(dataset)
    log_timing_overhead(settings)

    # The manifest gives this suite `tokens = "file"`, so `tokens` is one string.
    # Case-folding and normalization want that, but the find and compare groups need
    # real needles - searching for the whole haystack inside itself matches once, and
    # a single token yields no pairs at all. Rust splits the same buffer on the same
    # ASCII whitespace; `str.split()` would also split on Unicode spaces.
    word_bytes = tokenize_dataset(b"".join(dataset.tokens), Tokenization.WORDS, "keep")
    words = [word.decode("utf-8", errors="ignore") for word in word_bytes]
    lefts, rights = words[:-1], words[1:]

    # Valid UTF-8 needles of at least 3 bytes, drawn from the same stream as Rust, so both
    # languages search for the same words.
    candidates = [text for word in word_bytes if len(word) >= 3 and (text := strict_text(word)) is not None]
    generator = SplitMix64(stream_key(settings.seed, "normalization/needles", 0))
    search_needles = [
        candidates[generator.below(len(candidates))] for _ in range(min(NEEDLES_PER_PASS, len(candidates)))
    ]

    # `token_bytes` is the denominator every bytes/s figure is quoted against, and what
    # `log_dataset` has already printed; `len(pythonic_str)` is codepoints.
    total_bytes = dataset.token_bytes
    pair_bytes = Bytes(sum(map(len, word_bytes[:-1])) + sum(map(len, word_bytes[1:])))
    print(f"Pairs: {len(lefts):,}, Search needles: {len(search_needles)}")

    # Case-insensitive comparison
    print("Case-Insensitive Comparison")
    bench_case_compare(
        settings,
        "case-insensitive-compare/stringzilla.utf8_uncased_order",
        lefts,
        rights,
        compare_stringzilla,
        pair_bytes,
    )
    bench_case_compare(
        settings, "case-insensitive-compare/str.casefold.eq", lefts, rights, compare_casefold, pair_bytes
    )
    bench_case_compare(
        settings,
        "case-insensitive-compare/regex.fullmatch<compile+match>",
        lefts,
        rights,
        compare_regex_fullcase,
        pair_bytes,
    )
    bench_case_compare(
        settings, "case-insensitive-compare/icu.CaseMap.foldCase.eq", lefts, rights, compare_icu, pair_bytes
    )

    # Case-insensitive substring search
    print("\nCase-Insensitive Substring Search")
    # The row is named for the function actually called, `utf8_uncased_matches`.
    bench_case_find(
        settings,
        "case-insensitive-find/stringzilla.utf8_uncased_matches",
        pythonic_str,
        search_needles,
        find_stringzilla,
        total_bytes,
    )
    bench_case_find(
        settings, "case-insensitive-find/str.casefold.find", pythonic_str, search_needles, find_casefold, total_bytes
    )
    bench_case_find(
        settings,
        "case-insensitive-find/regex.finditer<compile+match>",
        pythonic_str,
        search_needles,
        find_regex_fullcase,
        total_bytes,
    )
    bench_case_find(
        settings, "case-insensitive-find/icu.StringSearch", pythonic_str, search_needles, make_find_icu(), total_bytes
    )

    # Case folding transformation
    print("\nCase Folding Transformation")
    bench_case_fold(settings, "case-fold/stringzilla.utf8_uncased_fold", tokens, fold_stringzilla, total_bytes)
    bench_case_fold(settings, "case-fold/str.casefold", tokens, fold_casefold, total_bytes)
    bench_case_fold(settings, "case-fold/icu.CaseMap.foldCase", tokens, fold_icu, total_bytes)

    # Unicode normalization (NFC / NFD / NFKC / NFKD) - all forms measured
    print("\nUnicode Normalization")
    for form in NORMALIZATION_FORMS:
        suffix = form.lower()
        bench_normalize(
            settings,
            f"normalize-{suffix}/stringzilla.utf8_norm",
            tokens,
            partial(normalize_stringzilla, form),
            total_bytes,
        )
        bench_normalize(
            settings, f"normalize-{suffix}/unicodedata.normalize", tokens, partial(normalize_stdlib, form), total_bytes
        )
        bench_normalize(settings, f"normalize-{suffix}/icu.Normalizer2", tokens, make_normalize_icu(form), total_bytes)
        bench_normalize(
            settings,
            f"normalize-{suffix}/pyunormalize.normalize",
            tokens,
            partial(normalize_pyunormalize, form),
            total_bytes,
        )

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
