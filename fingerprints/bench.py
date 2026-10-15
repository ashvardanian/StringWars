"""Exact window-overlap benchmarks in Python. Mirrors `fingerprints/bench.rs`.

StringZilla's `OverlapEngine` scores the exact n-gram overlap between texts. scikit-learn's sparse
window counts reproduce those scores, and cuDF's `jaccard_index` scores the same pairs under Jaccard's
normalization.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group fingerprints fingerprints/bench.py
"""

import argparse
import functools
from collections.abc import Callable
from dataclasses import replace
from importlib.metadata import version as pkg_version
from typing import Any

import numpy as np
import stringzilla as sz
from sklearn.feature_extraction.text import CountVectorizer

from stringwars import (
    Dataset,
    MeasureSpec,
    Settings,
    batch_size,
    finish,
    gpu_managed_array,
    gpu_ordinals,
    gpu_require,
    log_dataset,
    log_timing_overhead,
    measure,
    note_unavailable,
    print_machine,
    print_settings,
    read_settings,
    resolve_dataset,
    thread_pool,
)

# RAPIDS cuDF is outside the suite's dependency group, so its rows are optional.
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

NGRAM_WIDTHS = [5, 9, 17, 33]


def benchmark_overlap(settings: Settings, dataset: Dataset) -> None:
    """Scores the first `batch_per_core` documents as queries against every document as a candidate, at
    every n-gram width. One pass streams every candidate past every query, so the byte rate is
    candidate bytes per query batch. Mirrors `bench.rs`."""
    documents = dataset.text_tokens()
    queries = sz.Strings(documents[: min(batch_size(settings, 1), len(documents))])
    candidates = sz.Strings(documents)
    widths = len(NGRAM_WIDTHS)
    work = MeasureSpec(unit="bytes", elements=len(queries) * len(documents), total_bytes=dataset.token_bytes)
    call = "overlap/stringzilla.OverlapEngine.scores"
    print(f"\n# overlap - {len(queries)} queries, {len(documents)} candidates")

    name = f"{call}<1cpu>"
    if settings.selects(name):
        engine = sz.OverlapEngine(queries, NGRAM_WIDTHS)
        scores = np.zeros((len(queries), len(documents), widths), dtype=np.float32)
        measure(settings, name, work, lambda: engine.scores(candidates, scores))

    # One engine per pool thread, each holding every query against its share of the candidates.
    threads = settings.threads
    name = f"{call}<{threads}cpu>"
    if settings.selects(name):
        chunk = -(-len(documents) // threads)
        tasks = []
        for first in range(0, len(documents), chunk):
            share = sz.Strings(documents[first : first + chunk])
            engine = sz.OverlapEngine(queries, NGRAM_WIDTHS)
            scores = np.zeros((len(queries), len(share), widths), dtype=np.float32)
            tasks.append(functools.partial(engine.scores, share, scores))
        configure = functools.partial(sz.cpu_configure_thread, sz.cpu_capabilities_enabled())
        with thread_pool(threads, configure) as run_all:
            measure(settings, name, replace(work, concurrency=threads), functools.partial(run_all, tasks))

    # One engine per GPU, its first round checked against the host engine.
    for ordinal in gpu_ordinals(settings):
        name = f"{call}<cuda:{ordinal}>"
        if not settings.selects(name) or not gpu_require(settings, ordinal):
            continue
        stream = sz.cuda_stream_init(ordinal)
        try:
            benchmark_overlap_gpu(settings, name, ordinal, stream, queries, candidates, work)
        finally:
            sz.cuda_stream_free(stream)

    name = "overlap/sklearn.CountVectorizer<1cpu>"
    if settings.selects(name):
        benchmark_overlap_sklearn(settings, name, dataset.tokens[: len(queries)], dataset.tokens, work)

    for ordinal in gpu_ordinals(settings):
        name = f"overlap/cudf.Series.str.jaccard_index<aligned,cuda:{ordinal}>"
        if not CUDF_AVAILABLE:
            note_unavailable(name, "cudf not installed")
        elif settings.selects(name) and gpu_require(settings, ordinal):
            benchmark_overlap_cudf(settings, name, documents[: len(queries)], documents, work)


def benchmark_overlap_gpu(
    settings: Settings,
    name: str,
    ordinal: int,
    stream: int,
    queries: sz.Strings,
    candidates: sz.Strings,
    work: MeasureSpec,
) -> None:
    """One GPU overlap row, the candidates copied into the device's tape and the scores in managed memory."""
    capabilities = sz.cuda_capabilities_enabled(ordinal)
    try:
        engine = sz.OverlapEngine(
            queries, NGRAM_WIDTHS, candidates_budget=len(candidates), capabilities=capabilities, stream=stream
        )
    except Exception as error:  # a device that declines these widths skips its own row
        note_unavailable(name, str(error))
        return
    device_candidates = candidates.copy(capabilities, stream)
    shape = (len(queries), len(candidates), len(NGRAM_WIDTHS))
    scores = gpu_managed_array(shape, np.float32)

    def one_round() -> None:
        engine.scores(device_candidates, scores, stream=stream)
        sz.synchronize(capabilities, stream)

    one_round()
    expected = np.zeros(shape, dtype=np.float32)
    sz.OverlapEngine(queries, NGRAM_WIDTHS).scores(candidates, expected)
    if not np.allclose(scores, expected, atol=1e-4):
        raise SystemExit(f"{name} disagrees with the host engine on its first round")
    measure(settings, name, work, one_round)


def byte_windows(width: int, text: str) -> list[str]:
    """Every `width`-character window of `text`, which spells bytes when the text is decoded as Latin-1."""
    return [text[offset : offset + width] for offset in range(len(text) - width + 1)]


def sklearn_overlap(queries: list[bytes]) -> Callable[[list[str], np.ndarray], None]:
    """The shares `OverlapEngine` reports, from scikit-learn's sparse window counts.

    The product `Q × Cᵀ` of the queries' binary window indicators and the candidates' window counts is how
    many of a candidate's windows occur among a query's distinct ones, the engine's numerator, and the
    longer side's window count is its denominator. Candidates arrive Latin-1 decoded, one character per
    byte, so the character windows are the engine's byte windows.
    """
    query_lengths = np.fromiter(map(len, queries), dtype=np.int64, count=len(queries))
    query_texts = [query.decode("latin-1") for query in queries]
    stages = []
    for width in NGRAM_WIDTHS:
        # A callable analyzer, as `analyzer="char"` collapses the whitespace runs byte windows keep.
        windows = functools.partial(byte_windows, width)
        indicators = CountVectorizer(analyzer=windows, binary=True)
        query_matrix = indicators.fit_transform(query_texts)
        counter = CountVectorizer(analyzer=windows, vocabulary=indicators.vocabulary_)
        stages.append((width, query_matrix, counter))

    def scores_into(candidates: list[str], scores: np.ndarray) -> None:
        lengths = np.fromiter(map(len, candidates), dtype=np.int64, count=len(candidates))
        for index, (width, query_matrix, counter) in enumerate(stages):
            matches = (query_matrix @ counter.transform(candidates).T).toarray()
            longer = np.maximum.outer(np.maximum(query_lengths - width + 1, 0), np.maximum(lengths - width + 1, 0))
            np.divide(matches, longer, out=scores[:, :, index], where=longer > 0)

    return scores_into


def benchmark_overlap_sklearn(
    settings: Settings, name: str, queries: list[bytes], documents: list[bytes], work: MeasureSpec
) -> None:
    """scikit-learn's exact window overlap, checked against `OverlapEngine` over the first candidates."""
    try:
        scores_into = sklearn_overlap(queries)
    except ValueError as error:  # a width past every query leaves its vocabulary empty
        note_unavailable(name, str(error))
        return
    candidates = [document.decode("latin-1") for document in documents]
    probe = documents[:1024]
    expected = np.zeros((len(queries), len(probe), len(NGRAM_WIDTHS)), dtype=np.float32)
    sz.OverlapEngine(sz.Strings(queries), NGRAM_WIDTHS).scores(sz.Strings(probe), expected)
    actual = np.zeros_like(expected)
    scores_into(candidates[: len(probe)], actual)
    # StringZilla hashes windows into 32 bits, so a rare collision adds a match the exact count lacks.
    if np.mean(np.isclose(actual, expected, atol=1e-4)) < 0.999:
        raise SystemExit(f"{name} disagrees with StringZilla over the first {len(probe)} candidates")
    scores = np.zeros((len(queries), len(candidates), len(NGRAM_WIDTHS)), dtype=np.float32)
    measure(settings, name, work, functools.partial(scores_into, candidates, scores))


def benchmark_overlap_cudf(
    settings: Settings, name: str, queries: list[str], documents: list[str], work: MeasureSpec
) -> None:
    """cuDF's character n-gram Jaccard index over the same query and candidate pairs, laid out as aligned
    rows since it scores pairs rather than a cross product, its first round checked against `OverlapEngine`."""
    import cupy as cp

    left = cudf.Series(queries).repeat(len(documents))
    right = cudf.concat([cudf.Series(documents)] * len(queries), ignore_index=True)

    def one_pass() -> list[Any]:
        indices = [left.str.jaccard_index(right, width) for width in NGRAM_WIDTHS]
        cp.cuda.runtime.deviceSynchronize()
        return indices

    try:
        first = one_pass()
    except Exception as error:  # cuDF raises its own types for an unusable device or kernel
        note_unavailable(name, f"{type(error).__name__}: {error}")
        return
    shape = (len(queries), len(documents), len(NGRAM_WIDTHS))
    jaccard = np.stack([index.to_numpy() for index in first], axis=-1).reshape(shape)
    expected = np.zeros(shape, dtype=np.float32)
    sz.OverlapEngine(sz.Strings(queries), NGRAM_WIDTHS).scores(sz.Strings(documents), expected)
    # Jaccard normalizes differently, so only the pairs sharing no window must agree, up to hash collisions.
    if np.mean((jaccard > 0) == (expected > 0)) < 0.999:
        raise SystemExit(f"{name} disagrees with StringZilla on its first round")
    measure(settings, name, work, one_pass)


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    libraries = {
        "StringZilla": f"{sz.__version__} with {sz.cpu_capabilities_enabled().name}",
        "scikit-learn": pkg_version("scikit-learn"),
    }
    if CUDF_AVAILABLE:
        libraries["cuDF"] = cudf.__version__
    print_machine(libraries)
    settings = read_settings("fingerprints")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    log_dataset(dataset)
    log_timing_overhead(settings)

    benchmark_overlap(settings, dataset)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
