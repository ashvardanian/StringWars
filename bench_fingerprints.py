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

- MinHash: datasketch.MinHash, stringzillas.Fingerprints, cuDF (optional)

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
from pathlib import Path
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

# Global state for MinHash to avoid repeated initialization
_datasketch_min_hash_state = None


def log_system_info():
    """Log Python version and fingerprinting library versions."""
    import sys
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


def bit_entropy(hash_matrix: List[List[int]]) -> float:
    """Calculate bit entropy (how well distributed the bits are).

    Returns value between 0 and 1, where 1 means perfectly random bit distribution.
    """
    if not hash_matrix or not hash_matrix[0]:
        return 0.0

    # Determine bit width (assuming 32-bit integers)
    bits_per_hash = 32
    bit_ones_count = [0] * bits_per_hash  # Count of 1s for each bit position
    total_hash_values = len(hash_matrix) * len(hash_matrix[0])

    for document_hashes in hash_matrix:
        for hash_value in document_hashes:
            for bit_position in range(bits_per_hash):
                if (hash_value >> bit_position) & 1 == 1:
                    bit_ones_count[bit_position] += 1

    # Calculate entropy
    total_entropy = 0.0
    for ones_count in bit_ones_count:
        probability_of_one = ones_count / total_hash_values
        if 0 < probability_of_one < 1:
            total_entropy -= probability_of_one * np.log2(probability_of_one) + (1 - probability_of_one) * np.log2(
                1 - probability_of_one
            )

    return total_entropy / bits_per_hash  # Normalize to [0, 1]


def collision_rate(hash_matrix: List[List[int]]) -> float:
    """Calculate collision rate (duplicate hash values).

    Returns fraction of hash values that are duplicates.
    """
    if not hash_matrix or not hash_matrix[0]:
        return 0.0

    unique_hash_values = set()
    total_hash_count = 0

    for document_hashes in hash_matrix:
        for hash_value in document_hashes:
            unique_hash_values.add(hash_value)
            total_hash_count += 1

    return 1.0 - (len(unique_hash_values) / total_hash_count)


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

    try:
        bar = tqdm(desc=name, unit="docs", leave=False, total=len(documents))
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
        bar.close()
    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")
        return

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


def benchmark_third_party_fingerprints(
    docs: List[str],
    docs_sizes: List[int],
    dimensions: int,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    filter_pattern: Optional[re.Pattern] = None,
):
    """Benchmark third-party fingerprinting/sketching implementations."""
    global _datasketch_min_hash_state

    binary_docs = [doc.encode("utf-8") for doc in docs]

    # Create MinHash instances for each n-gram width
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

        return np.array(combined_signature[:dimensions])  # Trim to exact dimensions

    if name_matches("datasketch.MinHash.multi_width", filter_pattern):
        log(
            "datasketch.MinHash.multi_width",
            binary_docs,
            docs_sizes,
            single_doc=datasketch_multi_width_minhash,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_docs, _res: sum(
                sum(max(len(d) - w + 1, 0) for w in NGRAM_WIDTHS) for d in batch_docs
            )
            * hashes_per_width,
        )

    # Legacy single-width n-gram implementation for comparison
    if _datasketch_min_hash_state is None:
        _datasketch_min_hash_state = MinHash(num_perm=dimensions)

    def datasketch_minhash_update_batch_legacy(doc: bytes) -> np.ndarray:
        """Legacy single-width n-gram MinHash for comparison."""
        width = 5  # Use first width from NGRAM_WIDTHS
        if len(doc) >= width:
            ngrams = [doc[i : i + width] for i in range(len(doc) - width + 1)]
            _datasketch_min_hash_state.update_batch(ngrams)
        digest = _datasketch_min_hash_state.digest()
        _datasketch_min_hash_state.clear()
        return digest

    if name_matches("datasketch.MinHash.single_width", filter_pattern):
        log(
            "datasketch.MinHash.single_width",
            binary_docs,
            docs_sizes,
            single_doc=datasketch_minhash_update_batch_legacy,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_docs, _res: sum(max(len(d) - 5 + 1, 0) for d in batch_docs) * dimensions,
        )

    # cuDF MinHash implementations (if available)
    if CUDF_AVAILABLE:
        # Precompute MinHash parameters for cuDF - distribute across widths
        np.random.seed(42)  # For reproducibility
        hashes_per_width = dimensions // len(NGRAM_WIDTHS)

        # Test multiple cuDF MinHash variants with different widths
        for width in NGRAM_WIDTHS:
            a_vals = np.random.randint(1, 2**32 - 1, hashes_per_width, dtype=np.uint32)
            b_vals = np.random.randint(0, 2**32 - 1, hashes_per_width, dtype=np.uint32)

            # cuDF minhash - single document per width
            if name_matches(f"cudf.minhash(width={width})", filter_pattern):

                def kernel(doc: str, w=width, a=a_vals, b=b_vals) -> int:
                    # Create cuDF Series from document
                    s = cudf.Series([doc])
                    # Use cuDF's minhash function with specific width
                    result = s.str.minhash(seed=42, a=a, b=b, width=w)
                    # Extract minhash values and compute checksum
                    minhash_vals = result.iloc[0]  # List of minhash values
                    return sum(minhash_vals) % (2**32)

                log(
                    name=f"cudf.minhash(width={width})",
                    documents=docs,
                    document_sizes=docs_sizes,
                    single_doc=kernel,
                    timeout_seconds=timeout_seconds,
                    batch_size=1,
                    ops_counter=lambda batch_documents, _results, w=width: sum(
                        max(len(d.encode("utf-8")) - w + 1, 0) for d in batch_documents
                    )
                    * hashes_per_width,
                )

        # cuDF multi-width combined approach
        if name_matches("cudf.minhash.multi_width", filter_pattern):
            # Create parameter sets for all widths
            all_params = []
            for width in NGRAM_WIDTHS:
                a_vals = np.random.randint(1, 2**32 - 1, hashes_per_width, dtype=np.uint32)
                b_vals = np.random.randint(0, 2**32 - 1, hashes_per_width, dtype=np.uint32)
                all_params.append((width, a_vals, b_vals))

            def multi_width_kernel(doc: str) -> int:
                """Multi-width cuDF MinHash combining all widths."""
                s = cudf.Series([doc])
                combined_checksum = 0

                for width, a_vals, b_vals in all_params:
                    result = s.str.minhash(seed=42, a=a_vals, b=b_vals, width=width)
                    minhash_vals = result.iloc[0]
                    combined_checksum += sum(minhash_vals) % (2**32)

                return combined_checksum % (2**32)

            log(
                name="cudf.minhash.multi_width",
                documents=docs,
                document_sizes=docs_sizes,
                single_doc=multi_width_kernel,
                timeout_seconds=timeout_seconds,
                batch_size=1,
                ops_counter=lambda batch_documents, _results: sum(
                    sum(max(len(d.encode("utf-8")) - w + 1, 0) for w in NGRAM_WIDTHS) for d in batch_documents
                )
                * hashes_per_width,
            )

        # cuDF minhash64 variant (if different performance characteristics)
        if name_matches("cudf.minhash64", filter_pattern):
            a_vals_64 = np.random.randint(1, 2**32 - 1, dimensions, dtype=np.uint32)
            b_vals_64 = np.random.randint(0, 2**32 - 1, dimensions, dtype=np.uint32)

            def kernel_64(doc: str) -> int:
                s = cudf.Series([doc])
                result = s.str.minhash64(seed=42, a=a_vals_64, b=b_vals_64, width=NGRAM_WIDTHS[0])
                minhash_vals = result.iloc[0]
                return sum(minhash_vals) % (2**64)

            log(
                name="cudf.minhash64",
                documents=docs,
                document_sizes=docs_sizes,
                single_doc=kernel_64,
                timeout_seconds=timeout_seconds,
                batch_size=1,
                ops_counter=lambda batch_documents, _results: sum(
                    max(len(d.encode("utf-8")) - NGRAM_WIDTHS[0] + 1, 0) for d in batch_documents
                )
                * dimensions,
            )


def benchmark_szs_fingerprints(
    docs: List[str],
    docs_sizes: List[int],
    dimensions: int,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    filter_pattern: Optional[re.Pattern] = None,
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

        def kernel(doc: str) -> int:
            hashes, counts = engine(sz.Strs([doc]), device=default_scope)
            return _checksum_from_results((hashes, counts))

        log(
            name="szs.Fingerprints(1xCPU)",
            documents=docs,
            document_sizes=docs_sizes,
            single_doc=kernel,
            timeout_seconds=timeout_seconds,
            batch_size=1,
            ops_counter=lambda batch_documents, _results: len(batch_documents) * dimensions,
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


def bench(
    dataset_path: Optional[str] = None,
    tokens_mode: Optional[str] = None,
    max_docs: Optional[int] = None,
    dimensions: int = 256,
    timeout_seconds: int = 10,
    batch_size: int = 1,
    filter_pattern: Optional[re.Pattern] = None,
    dataset_limit: Optional[str] = None,
):
    """Run fingerprinting benchmarks."""

    # Load dataset using unified utilities
    dataset = load_dataset(dataset_path, size_limit=dataset_limit)
    tokens = tokenize_dataset(dataset, tokens_mode)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    # Use tokens as documents (each token is a document)
    docs = tokens[:max_docs] if max_docs else tokens

    docs_sizes = [len(doc.encode("utf-8")) for doc in docs]

    # Dataset statistics (matching Rust benchmark style)
    total_docs = len(docs)
    total_bytes = sum(docs_sizes)
    total_chars = sum(len(doc) for doc in docs)
    avg_bytes_per_doc = total_bytes / total_docs
    avg_chars_per_doc = total_chars / total_docs
    hashes_per_width = dimensions // len(NGRAM_WIDTHS)

    print("Dataset statistics:")
    print(f"- Source: {dataset_path}")
    print(f"- Processing mode: documents")
    print(f"- Total documents: {total_docs:,}")
    print(f"- Average doc length: {avg_bytes_per_doc:.1f} bytes, {avg_chars_per_doc:.1f} chars")
    print(f"- Total dataset size: {total_bytes:,} bytes, {total_chars:,} chars")
    print(f"- Batch size (for batch APIs): {batch_size}")
    print(f"- Fingerprint dimensions: {dimensions}")
    print(f"- N-gram widths: {NGRAM_WIDTHS} bytes")
    print(f"- Hashes per width: {hashes_per_width} (total NDIM: {dimensions})")
    print()
    log_system_info()

    print("=== Fingerprinting & Sketching Benchmarks ===")
    benchmark_third_party_fingerprints(docs, docs_sizes, dimensions, timeout_seconds, batch_size, filter_pattern)
    benchmark_szs_fingerprints(docs, docs_sizes, dimensions, timeout_seconds, batch_size, filter_pattern)

    # TODO: Add hash quality analysis here
    # This would require collecting hash matrices during benchmarking
    # and then computing bit_entropy() and collision_rate() for each algorithm
    print("\n=== Hash Quality Analysis ===")
    print("(Hash quality metrics not yet implemented - requires matrix collection during benchmarking)")


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

    # Compile regex pattern
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Run benchmark
    return bench(args.dataset, args.tokens, args.max_docs, args.dimensions, args.time_limit, args.batch_size, filter_pattern, args.dataset_limit)


if __name__ == "__main__":
    exit(main())
