# /// script
# dependencies = [
#   "stringzilla",
#   "stringzillas-cpus",
#   "datasketch",
#   "numpy",
#   "tqdm",
# ]
# ///
"""
Fingerprinting benchmarks in Python: docs/s for MinHash operations.

- MinHash: datasketch.MinHash, stringzillas.Fingerprints, cudf.minhash_ngrams,
  and cudf.minhash64_ngrams (if RAPIDS cuDF is installed).

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  python bench_fingerprints.py --dataset README.md --tokens lines
  python bench_fingerprints.py --dataset xlsum.csv --tokens words -k "datasketch"
  STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=lines python bench_fingerprints.py
"""

import os
import argparse
import itertools
import re
import sys
from typing import Callable, Any, List, Optional, Iterable

from tqdm import tqdm
import numpy as np

from datasketch import MinHash
import stringzillas as szs
import stringzilla as sz

from utils import load_dataset, tokenize_dataset, add_common_args, now_ns, name_matches

# For RAPIDS cuDF GPU-accelerated MinHash
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

# Fixed n-gram widths for multi-scale fingerprinting (matching Rust benchmark)
NGRAM_WIDTHS = [5, 9, 17, 33]
NGRAM_WIDTHS_ARRAY = np.array(NGRAM_WIDTHS, dtype=np.uint64)


def log_system_info():
    """Log Python version and fingerprinting library versions."""

    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- DataSketch: {MinHash.__module__.split('.')[0]} (available)")
    if CUDF_AVAILABLE:
        print(f"- CuDF: {cudf.__version__}")
    print()  # Add blank line


def _checksum_from_results(result) -> int:
    """Normalize different return types to an integer checksum.

    - numpy arrays → sum of elements
    - tuple of arrays (hashes, counts) → sum both
    - iterable of numerics → sum
    - scalar → int(result)
    """
    try:
        if isinstance(result, tuple) and len(result) == 2:
            a, b = result
            sa = int(np.asarray(a).sum())
            sb = int(np.asarray(b).sum())
            return sa + sb
        if isinstance(result, np.ndarray):
            return int(result.sum())
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            return int(sum(int(x) for x in result))
        return int(result)
    except Exception:
        return 0


def bit_entropy(hash_matrix: np.ndarray) -> float:
    """Calculate bit entropy (how well distributed the bits are).

    Returns value between 0 and 1, where 1 means perfectly random bit distribution.
    """
    if hash_matrix.size == 0:
        return 0.0

    # Determine bit width based on dtype
    if hash_matrix.dtype == np.uint32:
        bits_per_hash = 32
    elif hash_matrix.dtype == np.uint64:
        bits_per_hash = 64
    else:
        bits_per_hash = 32  # default

    bit_ones_count = np.zeros(bits_per_hash, dtype=np.int64)
    total_hash_values = hash_matrix.size

    # Count 1s for each bit position across all hash values
    flat_matrix = hash_matrix.flatten()
    for bit_position in range(bits_per_hash):
        bit_ones_count[bit_position] = np.sum((flat_matrix >> bit_position) & 1)

    # Calculate entropy
    total_entropy = 0.0
    for ones_count in bit_ones_count:
        probability_of_one = ones_count / total_hash_values
        if 0 < probability_of_one < 1:
            total_entropy -= probability_of_one * np.log2(probability_of_one) + (1 - probability_of_one) * np.log2(
                1 - probability_of_one
            )

    return total_entropy / bits_per_hash  # Normalize to [0, 1]


def collision_rate(hash_matrix: np.ndarray) -> float:
    """Calculate collision rate (duplicate hash values).

    Returns fraction of hash values that are duplicates.
    """
    if hash_matrix.size == 0:
        return 0.0

    unique_hash_values = len(np.unique(hash_matrix.flatten()))
    total_hash_count = hash_matrix.size

    return 1.0 - (unique_hash_values / total_hash_count)


def print_quality_analysis(matrix_state: dict):
    """Print quality analysis for all fingerprinting engines."""
    print("\n=== Hash Quality Analysis ===")
    print(
        f"Documents processed: DataSketch={matrix_state['datasketch_doc_idx']}, "
        f"cuDF={matrix_state['cudf_doc_idx']}, StringZilla={matrix_state['szs_doc_idx']}"
    )

    if matrix_state["datasketch_doc_idx"] > 0:
        matrix_slice = matrix_state["datasketch_matrix"][: matrix_state["datasketch_doc_idx"]]
        print("\nDataSketch MinHash:")
        print(f"  Bit Entropy:  {bit_entropy(matrix_slice):.4f}")
        print(f"  Collision:    {collision_rate(matrix_slice)*100:.4f}%")

    if matrix_state["cudf_doc_idx"] > 0:
        matrix_slice = matrix_state["cudf_matrix"][: matrix_state["cudf_doc_idx"]]
        print("\ncuDF MinHash:")
        print(f"  Bit Entropy:  {bit_entropy(matrix_slice):.4f}")
        print(f"  Collision:    {collision_rate(matrix_slice)*100:.4f}%")

    if matrix_state["szs_doc_idx"] > 0:
        matrix_slice = matrix_state["szs_matrix"][: matrix_state["szs_doc_idx"]]
        print("\nStringZilla Fingerprints:")
        print(f"  Bit Entropy:  {bit_entropy(matrix_slice):.4f}")
        print(f"  Collision:    {collision_rate(matrix_slice)*100:.4f}%")


def log_with_matrix(
    name: str,
    documents: List[str],
    document_sizes: List[int],
    single_doc: Optional[Callable[[str], Any]] = None,
    batch_docs: Optional[Callable[[List[str]], Iterable[Any]]] = None,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    ops_counter: Optional[Callable[[List[str], Iterable[Any]], int]] = None,
    matrix_state: Optional[dict] = None,
    engine_type: str = "unknown",
):
    """Benchmark an operation with timeout, batching, progress, checksum, and matrix population."""
    processed_docs = 0
    processed_bytes = 0
    checksum = 0
    start_ns = now_ns()

    bar = tqdm(desc=name, unit="docs", leave=False, total=len(documents))

    try:
        for batch_indices in itertools.batched(range(len(documents)), max(1, batch_size)):
            if (now_ns() - start_ns) > int(timeout_seconds * 1e9):
                break

            batch_documents = [documents[i] for i in batch_indices]
            batch_bytes = [document_sizes[i] for i in batch_indices]

            # Choose batch vs single path explicitly
            results_iterable: Iterable[Any]
            if batch_docs is not None:
                results = batch_docs(batch_documents)
                if hasattr(results, "__iter__") and not isinstance(results, (str, bytes)):
                    results_iterable = results
                else:
                    results_iterable = [results]
            else:
                if single_doc is None:
                    raise ValueError("single_doc callable is required when batch_docs is not provided")
                results_iterable = (single_doc(doc) for doc in batch_documents)

            # Process results and populate matrix if provided
            for doc_idx, result in enumerate(results_iterable):
                checksum += _checksum_from_results(result)

                # Populate matrix if matrix_state is provided
                if matrix_state is not None and engine_type in ["datasketch", "cudf", "szs"]:
                    matrix_key = f"{engine_type}_matrix"
                    doc_idx_key = f"{engine_type}_doc_idx"

                    if (
                        matrix_key in matrix_state
                        and doc_idx_key in matrix_state
                        and matrix_state[doc_idx_key] < matrix_state[matrix_key].shape[0]
                    ):

                        # Extract hash values from result
                        if isinstance(result, np.ndarray) and result.size > 0:
                            hash_values = result.flatten()
                            if len(hash_values) == matrix_state[matrix_key].shape[1]:
                                matrix_state[matrix_key][matrix_state[doc_idx_key], :] = hash_values
                                matrix_state[doc_idx_key] += 1

            processed_docs += len(batch_documents)
            processed_bytes += sum(batch_bytes)

            # Count operations (hashes computed) if provided
            if ops_counter is not None:
                try:
                    # Recompute results for ops counting if generator was consumed
                    if batch_docs is not None and batch_size > 1:
                        results_for_ops = batch_docs(batch_documents)
                    else:
                        results_for_ops = [single_doc(doc) for doc in batch_documents]  # type: ignore[arg-type]
                    total_ops = total_ops + ops_counter(batch_documents, results_for_ops)
                except Exception:
                    pass

            elapsed_s = (now_ns() - start_ns) / 1e9
            if elapsed_s > 0:
                docs_per_sec = processed_docs / elapsed_s
                mb_per_sec = processed_bytes / (1e6 * elapsed_s)
                hashes_per_sec = (total_ops / elapsed_s) if "total_ops" in locals() else 0
                bar.set_postfix(
                    {
                        "docs/s": f"{docs_per_sec:.0f}",
                        "MB/s": f"{mb_per_sec:.1f}",
                        "hashes/s": f"{hashes_per_sec:.0f}",
                        "chk": checksum,
                    }
                )
            bar.update(len(batch_documents))
    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")
        return
    finally:
        bar.close()

    total_time_s = (now_ns() - start_ns) / 1e9
    if processed_docs:
        docs_per_sec = processed_docs / total_time_s
        mb_per_sec = processed_bytes / (1e6 * total_time_s)
        hashes_per_sec = (total_ops / total_time_s) if "total_ops" in locals() else 0
        extra = f", {hashes_per_sec:.0f} hashes/s" if hashes_per_sec else ""
        print(
            f"{name:35s}: {total_time_s:8.3f}s ~ {mb_per_sec:8.3f} MB/s ~ {docs_per_sec:10,.0f} docs/s{extra} ~ checksum={checksum}"
        )
    else:
        print(f"{name}: No documents processed")


def log(
    name: str,
    documents: List[str],
    document_sizes: List[int],
    single_doc: Optional[Callable[[str], Any]] = None,
    batch_docs: Optional[Callable[[List[str]], Iterable[Any]]] = None,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    ops_counter: Optional[Callable[[List[str], Iterable[Any]], int]] = None,
):
    """Benchmark an operation with timeout, batching, progress, and checksum.

    Provide one or both callables. If both are given and batch_size > 1, uses batch_docs.
    """
    processed_docs = 0
    processed_bytes = 0
    checksum = 0
    start_ns = now_ns()

    bar = tqdm(desc=name, unit="docs", leave=False, total=len(documents))

    try:
        for batch_indices in itertools.batched(range(len(documents)), max(1, batch_size)):
            if (now_ns() - start_ns) > int(timeout_seconds * 1e9):
                break

            batch_documents = [documents[i] for i in batch_indices]
            batch_bytes = [document_sizes[i] for i in batch_indices]

            # Choose batch vs single path explicitly
            results_iterable: Iterable[Any]
            if batch_docs is not None:
                results = batch_docs(batch_documents)
                if hasattr(results, "__iter__") and not isinstance(results, (str, bytes)):
                    results_iterable = results
                else:
                    results_iterable = [results]
            else:
                if single_doc is None:
                    raise ValueError("single_doc callable is required when batch_docs is not provided")
                results_iterable = (single_doc(doc) for doc in batch_documents)

            for result in results_iterable:
                checksum += _checksum_from_results(result)

            processed_docs += len(batch_documents)
            processed_bytes += sum(batch_bytes)

            # Count operations (hashes computed) if provided
            if ops_counter is not None:
                try:
                    # Recompute results for ops counting if generator was consumed
                    if batch_docs is not None and batch_size > 1:
                        results_for_ops = batch_docs(batch_documents)
                    else:
                        results_for_ops = [single_doc(doc) for doc in batch_documents]  # type: ignore[arg-type]
                    total_ops = total_ops + ops_counter(batch_documents, results_for_ops)
                except Exception:
                    pass

            elapsed_s = (now_ns() - start_ns) / 1e9
            if elapsed_s > 0:
                docs_per_sec = processed_docs / elapsed_s
                mb_per_sec = processed_bytes / (1e6 * elapsed_s)
                hashes_per_sec = (total_ops / elapsed_s) if "total_ops" in locals() else 0
                bar.set_postfix(
                    {
                        "docs/s": f"{docs_per_sec:.0f}",
                        "MB/s": f"{mb_per_sec:.1f}",
                        "hashes/s": f"{hashes_per_sec:.0f}",
                        "chk": checksum,
                    }
                )
            bar.update(len(batch_documents))
    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")
        return
    finally:
        bar.close()

    total_time_s = (now_ns() - start_ns) / 1e9
    if processed_docs:
        docs_per_sec = processed_docs / total_time_s
        mb_per_sec = processed_bytes / (1e6 * total_time_s)
        hashes_per_sec = (total_ops / total_time_s) if "total_ops" in locals() else 0
        extra = f", {hashes_per_sec:.0f} hashes/s" if hashes_per_sec else ""
        print(
            f"{name:35s}: {total_time_s:8.3f}s ~ {mb_per_sec:8.3f} MB/s ~ {docs_per_sec:10,.0f} docs/s{extra} ~ checksum={checksum}"
        )
    else:
        print(f"{name}: No documents processed")


def bench_3party_fingerprints(
    docs: List[str],
    docs_sizes: List[int],
    dimensions: int,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    filter_pattern: Optional[re.Pattern] = None,
    matrix_state: Optional[dict] = None,
):
    """Benchmark third-party fingerprinting/sketching implementations."""
    binary_docs = [doc.encode("utf-8") for doc in docs]

    # Create MinHash instances for each n-gram width (like Rust implementation)
    hashes_per_width = dimensions // len(NGRAM_WIDTHS)
    minhashers = [MinHash(num_perm=hashes_per_width) for _ in NGRAM_WIDTHS]

    def datasketch_multi_width_minhash(doc: bytes) -> np.ndarray:
        """Multi-width MinHash using standardized n-gram widths [5, 9, 17, 33]."""
        combined_signature = []

        for width_idx, width in enumerate(NGRAM_WIDTHS):
            # Generate n-grams of this width
            if len(doc) >= width:
                ngrams = [doc[i : i + width] for i in range(len(doc) - width + 1)]
                minhashers[width_idx].update_batch(ngrams)

            # Get signature and add to combined result
            signature = minhashers[width_idx].digest()
            combined_signature.extend(signature)
            minhashers[width_idx].clear()

        return np.array(combined_signature[:dimensions], dtype=np.uint32)  # Proper dtype for matrix

    if name_matches("datasketch.MinHash", filter_pattern):
        log_with_matrix(
            "datasketch.MinHash",
            binary_docs,
            docs_sizes,
            single_doc=datasketch_multi_width_minhash,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_docs, _res: sum(
                sum(max(len(d) - w + 1, 0) for w in NGRAM_WIDTHS) for d in batch_docs
            )
            * hashes_per_width,
            matrix_state=matrix_state,
            engine_type="datasketch",
        )

    # cuDF MinHash implementations (if available) - using batch_size for GPU efficiency
    if CUDF_AVAILABLE:
        # Precompute MinHash parameters once
        np.random.seed(42)  # For reproducibility
        a_vals_32 = np.random.randint(1, 2**32 - 1, dimensions, dtype=np.uint32)
        b_vals_32 = np.random.randint(0, 2**32 - 1, dimensions, dtype=np.uint32)
        a_vals_64 = np.random.randint(1, 2**32 - 1, dimensions, dtype=np.uint64)
        b_vals_64 = np.random.randint(0, 2**32 - 1, dimensions, dtype=np.uint64)

        def cudf_batch_minhash(batch_docs: List[str]) -> np.ndarray:
            """Process batch of documents with cuDF minhash_ngrams."""
            # Choose n-gram width based on minimum document length
            min_doc_len = min(len(doc) for doc in batch_docs) if batch_docs else 5
            ngram_width = min(5, max(1, min_doc_len))  # Use width between 1 and 5

            # Generate character n-grams for each document
            ngrams_list = []
            for doc in batch_docs:
                if len(doc) >= ngram_width:
                    doc_ngrams = [doc[i : i + ngram_width] for i in range(len(doc) - ngram_width + 1)]
                else:
                    doc_ngrams = [doc] if doc else [""]
                ngrams_list.append(doc_ngrams)

            # Create cuDF Series with list data (cuDF will auto-detect list dtype)
            ngrams_series = cudf.Series(ngrams_list)

            # Use cuDF's minhash_ngrams (ngrams parameter means how many ngrams to use per hash)
            # Ensure ngrams parameter is at least 2 (cuDF requirement)
            max_ngrams_available = len(ngrams_list[0]) if ngrams_list else 2
            ngrams_to_use = max(2, min(max_ngrams_available, 10))

            result = ngrams_series.str.minhash_ngrams(ngrams=ngrams_to_use, seed=42, a=a_vals_32, b=b_vals_32)

            # Convert cuDF list result to numpy array
            # minhash_ngrams returns a list Series, convert to host array
            result_host = result.to_pandas().tolist()
            return np.array(result_host, dtype=np.uint32)

        if name_matches("cudf.minhash_ngrams", filter_pattern):
            log_with_matrix(
                "cudf.minhash_ngrams",
                docs,
                docs_sizes,
                batch_docs=cudf_batch_minhash,
                timeout_seconds=timeout_seconds,
                batch_size=batch_size,  # Use the batch_size parameter properly
                ops_counter=lambda batch_docs, _res: sum(max(len(doc) - 4, 0) for doc in batch_docs) * dimensions,
                matrix_state=matrix_state,
                engine_type="cudf",
            )

        def cudf_batch_minhash64(batch_docs: List[str]) -> np.ndarray:
            """Process batch of documents with cuDF minhash64_ngrams."""
            # Choose n-gram width based on minimum document length
            min_doc_len = min(len(doc) for doc in batch_docs) if batch_docs else 5
            ngram_width = min(5, max(1, min_doc_len))  # Use width between 1 and 5

            # Generate character n-grams for each document
            ngrams_list = []
            for doc in batch_docs:
                if len(doc) >= ngram_width:
                    doc_ngrams = [doc[i : i + ngram_width] for i in range(len(doc) - ngram_width + 1)]
                else:
                    doc_ngrams = [doc] if doc else [""]
                ngrams_list.append(doc_ngrams)

            # Create cuDF Series with list data (cuDF will auto-detect list dtype)
            ngrams_series = cudf.Series(ngrams_list)

            # Use cuDF's 64-bit minhash_ngrams
            # Ensure ngrams parameter is at least 2 (cuDF requirement)
            max_ngrams_available = len(ngrams_list[0]) if ngrams_list else 2
            ngrams_to_use = max(2, min(max_ngrams_available, 10))

            result = ngrams_series.str.minhash64_ngrams(ngrams=ngrams_to_use, seed=42, a=a_vals_64, b=b_vals_64)

            # Convert cuDF list result to numpy array then to 32-bit for matrix compatibility
            result_host = result.to_pandas().tolist()
            result_array = np.array(result_host, dtype=np.uint64)
            return (result_array % (2**32)).astype(np.uint32)

        if name_matches("cudf.minhash64_ngrams", filter_pattern):
            log_with_matrix(
                "cudf.minhash64_ngrams",
                docs,
                docs_sizes,
                batch_docs=cudf_batch_minhash64,
                timeout_seconds=timeout_seconds,
                batch_size=batch_size,  # Use the batch_size parameter properly
                ops_counter=lambda batch_docs, _res: sum(max(len(doc) - 4, 0) for doc in batch_docs) * dimensions,
                matrix_state=matrix_state,
                engine_type="cudf",
            )


def bench_szs_fingerprints(
    docs: List[str],
    docs_sizes: List[int],
    dimensions: int,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    filter_pattern: Optional[re.Pattern] = None,
    matrix_state: Optional[dict] = None,
):
    """Benchmark StringZillas Fingerprints across device scopes and modes."""
    cpu_cores = os.cpu_count()
    default_scope = szs.DeviceScope()
    cpu_scope = szs.DeviceScope(cpu_cores=cpu_cores)
    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:
        gpu_scope = None

    # Per-doc kernels (single doc per call) for parity with scalar libs
    if name_matches("szs.Fingerprints(1xCPU)", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=default_scope, window_widths=NGRAM_WIDTHS_ARRAY)

        def kernel(doc: str) -> np.ndarray:
            hashes, counts = engine(sz.Strs([doc]), device=default_scope)
            # Return hash values as uint32 array for matrix population
            hash_array = np.asarray(hashes, dtype=np.uint32).flatten()
            return hash_array[:dimensions]  # Ensure exact dimensions

        log_with_matrix(
            name="szs.Fingerprints(1xCPU)",
            documents=docs,
            document_sizes=docs_sizes,
            single_doc=kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_documents, _results: len(batch_documents) * dimensions,
            matrix_state=matrix_state,
            engine_type="szs",
        )

    if name_matches(f"szs.Fingerprints({cpu_cores}xCPU)", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=cpu_scope, window_widths=NGRAM_WIDTHS_ARRAY)

        def kernel(doc: str) -> int:
            hashes, counts = engine(sz.Strs([doc]), device=cpu_scope)
            return _checksum_from_results((hashes, counts))

        log(
            name=f"szs.Fingerprints({cpu_cores}xCPU)",
            documents=docs,
            document_sizes=docs_sizes,
            single_doc=kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_documents, _results, d=dimensions: len(batch_documents) * d,
        )

    if gpu_scope is not None and name_matches("szs.Fingerprints(1xGPU)", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=gpu_scope, window_widths=NGRAM_WIDTHS_ARRAY)

        def kernel(doc: str) -> int:
            hashes, counts = engine(sz.Strs([doc]), device=gpu_scope)
            return _checksum_from_results((hashes, counts))

        log(
            name="szs.Fingerprints(1xGPU)",
            documents=docs,
            document_sizes=docs_sizes,
            single_doc=kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_documents, _results, d=dimensions: len(batch_documents) * d,
        )

    # Batched kernels (list[str] → list[int] checksums)
    if name_matches(f"szs.Fingerprints(1xCPU,batch={batch_size})", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=default_scope, window_widths=NGRAM_WIDTHS_ARRAY)

        def kernel_batch(batch_docs: List[str]) -> List[int]:
            hashes, counts = engine(sz.Strs(batch_docs), device=default_scope)
            # Reduce per document: sum of row in both arrays
            hashes = np.asarray(hashes)
            counts = np.asarray(counts)
            per_doc = hashes.sum(axis=1).astype(np.int64) + counts.sum(axis=1).astype(np.int64)
            return [int(x) for x in per_doc]

        log(
            name=f"szs.Fingerprints(1xCPU,batch={batch_size})",
            documents=docs,
            document_sizes=docs_sizes,
            batch_docs=kernel_batch,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            ops_counter=lambda batch_documents, _results, d=dimensions: len(batch_documents) * d,
        )

    if name_matches(f"szs.Fingerprints({cpu_cores}xCPU,batch={batch_size})", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=cpu_scope, window_widths=NGRAM_WIDTHS_ARRAY)

        def kernel_batch(batch_docs: List[str]) -> List[int]:
            hashes, counts = engine(sz.Strs(batch_docs), device=cpu_scope)
            hashes = np.asarray(hashes)
            counts = np.asarray(counts)
            per_doc = hashes.sum(axis=1).astype(np.int64) + counts.sum(axis=1).astype(np.int64)
            return [int(x) for x in per_doc]

        log(
            name=f"szs.Fingerprints({cpu_cores}xCPU,batch={batch_size})",
            documents=docs,
            document_sizes=docs_sizes,
            batch_docs=kernel_batch,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            ops_counter=lambda batch_documents, _results, d=dimensions: len(batch_documents) * d,
        )

    if gpu_scope is not None and name_matches(f"szs.Fingerprints(1xGPU,batch={batch_size})", filter_pattern):
        engine = szs.Fingerprints(ndim=dimensions, capabilities=gpu_scope, window_widths=NGRAM_WIDTHS_ARRAY)

        def kernel_batch(batch_docs: List[str]) -> List[int]:
            hashes, counts = engine(sz.Strs(batch_docs), device=gpu_scope)
            hashes = np.asarray(hashes)
            counts = np.asarray(counts)
            per_doc = hashes.sum(axis=1).astype(np.int64) + counts.sum(axis=1).astype(np.int64)
            return [int(x) for x in per_doc]

        log(
            name=f"szs.Fingerprints(1xGPU,batch={batch_size})",
            documents=docs,
            document_sizes=docs_sizes,
            batch_docs=kernel_batch,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
            ops_counter=lambda batch_documents, _results, d=dimensions: len(batch_documents) * d,
        )


_main_epilog = """
Examples:

  # Benchmark all algorithms with default settings
  %(prog)s --dataset leipzig1M.txt

  # Benchmark with limited docs and specific dimensions
  %(prog)s --dataset leipzig1M.txt --max-docs 1000 --dimensions 128

  # Test only specific algorithms
  %(prog)s --dataset leipzig1M.txt -k "(datasketch|szs.Fingerprints)"

  # GPU-only benchmarks
  %(prog)s --dataset leipzig1M.txt -k "(cudf|GPU)"

  # High-throughput batch processing
  %(prog)s --dataset leipzig1M.txt --batch-size 1024
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla fingerprinting operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)
    parser.add_argument("-n", "--max-docs", type=int, help="Maximum number of docs to process")
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        default=256,
        help="Number of hash functions for MinHash (default: 256)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for batch-capable APIs (default: 1)",
    )

    args = parser.parse_args()

    # Compile filter pattern
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Load and tokenize dataset
    dataset = load_dataset(args.dataset, size_limit=args.dataset_limit)
    tokens = tokenize_dataset(dataset, args.tokens)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    # Limit number of documents if specified
    if args.max_docs is not None:
        tokens = tokens[: args.max_docs]

    docs_sizes = [len(doc.encode("utf-8")) for doc in tokens]
    total_bytes = sum(docs_sizes)
    avg_doc_length = total_bytes / len(tokens) if tokens else 0

    print(f"Dataset: {len(tokens):,} docs, {total_bytes:,} bytes, {avg_doc_length:.1f} avg doc length")
    log_system_info()

    # Pre-allocated matrices for quality analysis: N_docs × N_dims (engine-specific types)
    total_documents = len(tokens)
    datasketch_matrix = np.zeros((total_documents, args.dimensions), dtype=np.uint32)
    cudf_matrix = np.zeros((total_documents, args.dimensions), dtype=np.uint32)
    szs_matrix = np.zeros((total_documents, args.dimensions), dtype=np.uint32)

    # Track which documents have been processed for each implementation
    matrix_state = {
        "datasketch_matrix": datasketch_matrix,
        "cudf_matrix": cudf_matrix,
        "szs_matrix": szs_matrix,
        "datasketch_doc_idx": 0,
        "cudf_doc_idx": 0,
        "szs_doc_idx": 0,
    }

    print("\n=== Fingerprinting Benchmarks ===")

    # Run benchmarks with matrix population
    bench_3party_fingerprints(
        tokens, docs_sizes, args.dimensions, args.time_limit, args.batch_size, filter_pattern, matrix_state
    )
    bench_szs_fingerprints(
        tokens, docs_sizes, args.dimensions, args.time_limit, args.batch_size, filter_pattern, matrix_state
    )

    # Print quality analysis
    print_quality_analysis(matrix_state)

    return 0


if __name__ == "__main__":
    exit(main())
