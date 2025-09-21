# StringWa.rs: Text Processing on CPUs & GPUs, in Python & Rust 🦀

![StringWa.rs Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringWa.rs.jpg?raw=true)

There are many __great__ libraries for string processing!
Mostly, of course, written in Assembly, C, and C++, but some in Rust as well. 😅

Where Rust decimates C and C++, is the __simplicity__ of dependency management, making it great for benchmarking "Systems Software" and lining up apples-to-apples across native crates and their Python bindings.
So, to accelerate the development of the [`StringZilla`](https://github.com/ashvardanian/StringZilla) C, C++, and CUDA libraries (with Rust and Python bindings), I've created this repository to compare it against some of my & communities most beloved Rust projects, like:

- [`memchr`](https://github.com/BurntSushi/memchr) for substring search.
- [`rapidfuzz`](https://github.com/rapidfuzz/rapidfuzz-rs) for edit distances.
- [`aHash`](https://github.com/tkaitchuck/aHash) and [`crc32fast`](https://github.com/srijs/rust-crc32fast) for hashing.
- [`aho_corasick`](https://github.com/BurntSushi/aho-corasick) and [`regex`](https://github.com/rust-lang/regex) for multi-search.
- [`arrow`](https://github.com/apache/arrow-rs) and [`polars`](https://github.com/pola-rs/polars) for collections.

Of course, the functionality of the projects is different, as are the APIs and the usage patterns.
So, I focus on the workloads for which StringZilla was designed and compare the throughput of the core operations.
Notably, I also favor modern hardware with support for a wider range SIMD instructions, like mask-equipped AVX-512 on x86 starting from the 2015 Intel Skylake-X CPUs or more recent predicated variable-length SVE and SVE2 on Arm, that aren't often supported by existing libraries and tooling.

> [!IMPORTANT]  
> The numbers in the tables below are provided for reference only and may vary depending on the CPU, compiler, dataset, and tokenization method.
> Most of them were obtained on Intel Sapphire Rapids CPUs and Nvidia H100 GPUs, using Rust with `-C target-cpu=native` optimization flag.
> To replicate the results, please refer to the [Replicating the Results](#replicating-the-results) section below.

## Hash

Many great hashing libraries exist in Rust, C, and C++.
Typical top choices are `aHash`, `xxHash`, `blake3`, `gxhash`, `CityHash`, `MurmurHash`, `crc32fast`, or the native `std::hash`.
Many of them have similar pitfalls:

- They are not always documented to have a certain reproducible output and are recommended for use only for local in-memory construction of hash tables, not for serialization or network communication.
- They don't always support streaming and require the whole input to be available in memory at once.
- They don't always pass the SMHasher test suite, especially with `--extra` checks enabled.
- They generally don't have a dynamic dispatch mechanism to simplify shipping of precompiled software.
- They are rarely available for multiple programming languages.

StringZilla addresses those issues and seems to provide competitive performance.
On Intel Sapphire Rapids CPU, on `xlsum.csv` dataset, the following numbers can be expected for hashing individual whitespace-delimited words and newline-delimited lines:

| Library               | Bits  | Ports ¹ |    Short Words |      Long Lines |
| --------------------- | :---: | :-----: | -------------: | --------------: |
| Rust 🦀                |       |         |                |                 |
| `std::hash`           |  64   |    ❌    |     0.43 GiB/s |      3.74 GiB/s |
| `crc32fast::hash`     |  32   |    ✅    |     0.49 GiB/s |      8.45 GiB/s |
| `xxh3::xxh3_64`       |  64   |    ✅    |     1.08 GiB/s |      9.48 GiB/s |
| `aHash::hash_one`     |  64   |    ❌    |     1.23 GiB/s |      8.61 GiB/s |
| `gxhash::gxhash64`    |  64   |    ❌    | __2.68 GiB/s__ |      9.19 GiB/s |
| `stringzilla::hash`   |  64   |    ✅    |     1.84 GiB/s | __11.23 GiB/s__ |
|                       |       |         |                |                 |
| Python 🐍              |       |         |                |                 |
| `hash`                | 32/64 |    ❌    |     0.13 GiB/s |      4.27 GiB/s |
| `xxhash.xxh3_64`      |  64   |    ✅    |     0.04 GiB/s |      6.38 GiB/s |
| `google_crc32c.value` |  32   |    ✅    |     0.04 GiB/s |      5.96 GiB/s |
| `mmh3.hash32`         |  32   |    ✅    |     0.05 GiB/s |      2.65 GiB/s |
| `mmh3.hash64`         |  64   |    ✅    |     0.03 GiB/s |      4.45 GiB/s |
| `cityhash.CityHash64` |  64   |    ✅    |     0.06 GiB/s |      4.87 GiB/s |
| `stringzilla.hash`    |  64   |    ✅    | __0.14 GiB/s__ |  __9.19 GiB/s__ |


> ¹ Portability means availability in multiple other programming languages, like C, C++, Python, Java, Go, JavaScript, etc.

In larger systems, however, we often need the ability to incrementally hash the data.
This is especially important in distributed systems, where the data is too large to fit into memory at once.

| Library                    | Bits  | Ports ¹ |    Short Words |     Long Lines |
| -------------------------- | :---: | :-----: | -------------: | -------------: |
| Rust 🦀                     |       |         |                |                |
| `std::hash::DefaultHasher` |  64   |    ❌    |     0.51 GiB/s |     3.92 GiB/s |
| `aHash::AHasher`           |  64   |    ❌    | __1.30 GiB/s__ | __8.56 GiB/s__ |
| `crc32fast::Hasher`        |  32   |    ✅    |     0.37 GiB/s |     8.39 GiB/s |
| `stringzilla::Hasher`      |  64   |    ✅    |     0.89 GiB/s |     6.39 GiB/s |
|                            |       |         |                |                |
| Python 🐍                   |       |         |                |                |
| `xxhash.xxh3_64`           |  64   |    ✅    |     0.09 GiB/s |      7.09 GB/s |
| `google_crc32c.Checksum`   |  32   |    ✅    |     0.04 GiB/s |     5.96 GiB/s |
| `stringzilla.Hasher`       |  64   |    ✅    | __0.35 GiB/s__ |  __6.04 GB/s__ |

For reference, one may want to put those numbers next to check-sum calculation speeds on one end of complexity and cryptographic hashing speeds on the other end.

| Library                | Bits  | Ports ¹ | Short Words |  Long Lines |
| ---------------------- | :---: | :-----: | ----------: | ----------: |
| Rust 🦀                 |       |         |             |             |
| `stringzilla::bytesum` |  64   |    ✅    |  2.16 GiB/s | 11.65 GiB/s |
| `blake3::hash`         |  256  |    ✅    |  0.10 GiB/s |  1.97 GiB/s |
|                        |       |         |             |             |
| Python 🐍               |       |         |             |             |
| `stringzilla.bytesum`  |  64   |    ✅    |  0.16 GiB/s |  8.62 GiB/s |
| `blake3.digest`        |  256  |    ✅    |  0.02 GiB/s |  1.82 GiB/s |


## Substring Search

Substring search is one of the most common operations in text processing, and one of the slowest.
Most of the time, programmers don't think about replacing the `str::find` method, as it's already expected to be optimized.
In many languages it's offloaded to the C standard library [`memmem`](https://man7.org/linux/man-pages/man3/memmem.3.html) or [`strstr`](https://en.cppreference.com/w/c/string/byte/strstr) for `NULL`-terminated strings.
The C standard library is, however, also implemented by humans, and a better solution can be created.

| Library             | Short Word Queries | Long Line Queries |
| ------------------- | -----------------: | ----------------: |
| Rust 🦀              |                    |                   |
| `std::str::find`    |         9.45 GiB/s |       10.88 GiB/s |
| `memmem::find`      |         9.48 GiB/s |       10.83 GiB/s |
| `memmem::Finder`    |         9.51 GiB/s |   __10.99 GiB/s__ |
| `stringzilla::find` |    __10.51 GiB/s__ |       10.82 GiB/s |
|                     |                    |                   |
| Python 🐍            |                    |                   |
| `str.find`          |         1.05 GiB/s |        1.23 GiB/s |
| `stringzilla.find`  |    __10.82 GiB/s__ |   __11.79 GiB/s__ |

Interestingly, the reverse order search is almost never implemented in SIMD, assuming fewer people ever need it.
Still, those are provided by StringZilla mostly for parsing tasks and feature parity.

| Library              | Short Word Queries | Long Line Queries |
| -------------------- | -----------------: | ----------------: |
| Rust 🦀               |                    |                   |
| `std::str::rfind`    |         2.72 GiB/s |        5.94 GiB/s |
| `memmem::rfind`      |         2.70 GiB/s |        5.90 GiB/s |
| `memmem::FinderRev`  |         2.79 GiB/s |        5.81 GiB/s |
| `stringzilla::rfind` |    __10.34 GiB/s__ |   __10.66 GiB/s__ |
|                      |                    |                   |
| Python 🐍             |                    |                   |
| `str.rfind`          |         1.54 GiB/s |        3.84 GiB/s |
| `stringzilla.rfind`  |     __7.15 GiB/s__ |   __11.56 GiB/s__ |


## Byte-Set Search

StringWa.rs takes a few representative examples of various character sets that appear in real parsing or string validation tasks:

- tabulation characters, like `\n\r\v\f`;
- HTML and XML markup characters, like `</>&'\"=[]`;
- numeric characters, like `0123456789`.

It's common in such cases, to pre-construct some library-specific filter-object or Finite State Machine (FSM) to search for a set of characters.
Once that object is constructed, all of it's inclusions in each token (word or line) are counted.
Current numbers should look like this:

| Library                         |    Short Words |     Long Lines |
| ------------------------------- | -------------: | -------------: |
| Rust 🦀                          |                |                |
| `bstr::iter`                    |     0.26 GiB/s |     0.25 GiB/s |
| `regex::find_iter`              |     0.23 GiB/s |     5.22 GiB/s |
| `aho_corasick::find_iter`       |     0.41 GiB/s |     0.50 GiB/s |
| `stringzilla::find_byteset`     | __1.61 GiB/s__ | __8.17 GiB/s__ |
|                                 |                |                |
| Python 🐍                        |                |                |
| `re.finditer`                   |     0.04 GiB/s |     0.19 GiB/s |
| `stringzilla.Str.find_first_of` | __0.11 GiB/s__ | __8.79 GiB/s__ |

## Sequence Operations

Rust has several Dataframe libraries, DBMS and Search engines that heavily rely on string sorting and intersections.
Those operations mostly are implemented using conventional algorithms:

- Comparison-based Quicksort or Mergesort for sorting.
- Hash-based or Tree-based algorithms for intersections.

Assuming the compares can be accelerated with SIMD and so can be the hash functions, StringZilla could already provide a performance boost in such applications, but starting with v4 it also provides specialized algorithms for sorting and intersections.
Those are directly compatible with arbitrary string-comparable collection types with a support of an indexed access to the elements.

| Library                                     |               Short Words |              Long Lines |
| ------------------------------------------- | ------------------------: | ----------------------: |
| Rust 🦀                                      |                           |                         |
| `std::sort_unstable_by_key`                 |        54.35 M compares/s |      57.70 M compares/s |
| `rayon::par_sort_unstable_by_key` on 1x CPU |        47.08 M compares/s |      50.35 M compares/s |
| `polars::Series::sort`                      |       200.34 M compares/s |      65.44 M compares/s |
| `polars::Series::arg_sort`                  |        25.01 M compares/s |      14.05 M compares/s |
| `arrow::lexsort_to_indices`                 |       122.20 M compares/s |  __84.73 M compares/s__ |
| `stringzilla::argsort_permutation`          |   __213.73 M compares/s__ |      74.64 M compares/s |
|                                             |                           |                         |
| Python 🐍                                    |                           |                         |
| `list.sort` on 1x CPU                       |        47.06 M compares/s |      22.36 M compares/s |
| `pandas.Series.sort_values` on 1x CPU       |         9.39 M compares/s |      11.93 M compares/s |
| `pyarrow.compute.sort_indices` on 1x CPU    |        62.17 M compares/s |       5.53 M compares/s |
| `polars.Series.sort` on 1x CPU              |       223.38 M compares/s | __181.60 M compares/s__ |
| `cudf.Series.sort_values` on 1x GPU         | __9'463.59 M compares/s__ |      66.44 M compares/s |
| `stringzilla.Strs.sorted` on 1x CPU         |       171.13 M compares/s |      77.88 M compares/s |

## Random Generation & Lookup Tables

Some of the most common operations in data processing are random generation and lookup tables.
That's true not only for strings but for any data type, and StringZilla has been extensively used in Image Processing and Bioinformatics for those purposes.
Generating random byte-streams:

| Library                        |    Short Words |      Long Lines |
| ------------------------------ | -------------: | --------------: |
| Rust 🦀                         |                |                 |
| `getrandom::fill`              |     0.18 GiB/s |      0.45 GiB/s |
| `rand_chacha::ChaCha20Rng`     |     0.62 GiB/s |      1.85 GiB/s |
| `rand_xoshiro::Xoshiro128Plus` |     0.83 GiB/s |      3.85 GiB/s |
| `zeroize::zeroize`             |     0.66 GiB/s |      4.73 GiB/s |
| `stringzilla::fill_random`     | __2.47 GiB/s__ | __10.57 GiB/s__ |
|                                |                |                 |
| Python 🐍                       |                |                 |
| `numpy.PCG64`                  |     0.01 GiB/s |      1.28 GiB/s |
| `numpy.Philox`                 |     0.01 GiB/s |      1.59 GiB/s |
| `pycryptodome.AES-CTR`         |     0.01 GiB/s |     13.16 GiB/s |
| `stringzilla.random`           | __0.11 GiB/s__ | __20.37 GiB/s__ |

Performing in-place lookups in a precomputed table of 256 bytes:

| Library                         |    Short Words |     Long Lines |
| ------------------------------- | -------------: | -------------: |
| Rust 🦀                          |                |                |
| serial code                     | __0.61 GiB/s__ |     1.49 GiB/s |
| `stringzilla::lookup_inplace`   |     0.54 GiB/s | __9.90 GiB/s__ |
|                                 |                |                |
| Python 🐍                        |                |                |
| `bytes.translate`               |     0.05 GiB/s |     1.92 GiB/s |
| `numpy.take`                    |     0.01 GiB/s |     0.85 GiB/s |
| `opencv.LUT`                    |     0.01 GiB/s |     1.95 GiB/s |
| `opencv.LUT` inplace            |     0.01 GiB/s |     2.16 GiB/s |
| `stringzilla.translate`         |     0.07 GiB/s |     7.92 GiB/s |
| `stringzilla.translate` inplace | __0.06 GiB/s__ | __8.14 GiB/s__ |


## Similarities Scoring

Edit Distance calculation is a common component of Search Engines, Data Cleaning, and Natural Language Processing, as well as in Bioinformatics.
It's a computationally expensive operation, generally implemented using dynamic programming, with a quadratic time complexity upper bound.

| Library                                               | ≅ 100 bytes lines | ≅ 1'000 bytes lines |
| ----------------------------------------------------- | ----------------: | ------------------: |
| Rust 🦀                                                |                   |
| `rapidfuzz::levenshtein<Bytes>`                       |       4'633 MCUPS |        14'316 MCUPS |
| `rapidfuzz::levenshtein<Chars>`                       |       3'877 MCUPS |        13'179 MCUPS |
| `stringzillas::LevenshteinDistances` on 1x CPU        |       3'315 MCUPS |        13'084 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8` on 1x CPU    |       3'283 MCUPS |        11'690 MCUPS |
| `stringzillas::LevenshteinDistances` on 16x CPUs      |      29'430 MCUPS |       105'400 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8` on 16x CPUs  |      38'954 MCUPS |       103'500 MCUPS |
| `stringzillas::LevenshteinDistances` on 1x GPU        |  __31'913 MCUPS__ |   __624'730 MCUPS__ |
|                                                       |                   |                     |
| Python 🐍                                              |                   |                     |
| `nltk.edit_distance`                                  |           2 MCUPS |             2 MCUPS |
| `jellyfish.levenshtein_distance`                      |          81 MCUPS |           228 MCUPS |
| `rapidfuzz.Levenshtein.distance`                      |         108 MCUPS |         9'272 MCUPS |
| `editdistance.eval`                                   |          89 MCUPS |           660 MCUPS |
| `edlib.align`                                         |          82 MCUPS |         7'262 MCUPS |
| `polyleven.levenshtein`                               |          89 MCUPS |         3'887 MCUPS |
| `stringzillas.LevenshteinDistances` on 1x CPU         |          53 MCUPS |         3'407 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8` on 1x CPU     |          57 MCUPS |         3'693 MCUPS |
| `cudf.edit_distance` batch on 1x GPU                  |      24'754 MCUPS |         6'976 MCUPS |
| `stringzillas.LevenshteinDistances` batch on 1x CPU   |       2'343 MCUPS |        12'141 MCUPS |
| `stringzillas.LevenshteinDistances` batch on 16x CPUs |       3'762 MCUPS |       119'261 MCUPS |
| `stringzillas.LevenshteinDistances` batch on 1x GPU   |  __18'081 MCUPS__ |   __320'109 MCUPS__ |


For biological sequences, the Needleman-Wunsch and Smith-Waterman algorithms are more appropriate, as they allow overriding the default substitution costs.
Another common adaptation is to used Gotoh's affine gap penalties, which better model the evolutionary events in DNA and Protein sequences.

| Library                                                | ≅ 100 bytes lines | ≅ 1'000 bytes lines |
| ------------------------------------------------------ | ----------------: | ------------------: |
| Rust 🦀 with linear gaps                                |                   |
| `stringzillas::NeedlemanWunschScores` on 1x CPU        |         278 MCUPS |           612 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 16x CPUs      |       4'057 MCUPS |         8'492 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 1x GPU        |         131 MCUPS |    __12'113 MCUPS__ |
| `stringzillas::SmithWatermanScores` on 1x CPU          |         263 MCUPS |           552 MCUPS |
| `stringzillas::SmithWatermanScores` on 16x CPUs        |       3'883 MCUPS |         8'011 MCUPS |
| `stringzillas::SmithWatermanScores` on 1x GPU          |         143 MCUPS |    __12'921 MCUPS__ |
|                                                        |                   |                     |
| Python 🐍 with linear gaps                              |                   |                     |
| `biopython.PairwiseAligner.score` on 1x CPU            |          95 MCUPS |           557 MCUPS |
| `stringzillas.NeedlemanWunschScores` on 1x CPU         |          30 MCUPS |           481 MCUPS |
| `stringzillas.NeedlemanWunschScores` batch on 1x CPU   |         246 MCUPS |           570 MCUPS |
| `stringzillas.NeedlemanWunschScores` batch on 16x CPUs |       3'103 MCUPS |         9'208 MCUPS |
| `stringzillas.NeedlemanWunschScores` batch on 1x GPU   |         127 MCUPS |        12'246 MCUPS |
| `stringzillas.SmithWatermanScores` on 1x CPU           |          28 MCUPS |           440 MCUPS |
| `stringzillas.SmithWatermanScores` batch on 1x CPU     |         255 MCUPS |           582 MCUPS |
| `stringzillas.SmithWatermanScores` batch on 16x CPUs   |   __3'535 MCUPS__ |         8'235 MCUPS |
| `stringzillas.SmithWatermanScores` batch on 1x GPU     |         130 MCUPS |    __12'702 MCUPS__ |
|                                                        |                   |                     |
| Rust 🦀 with affine gaps                                |                   |                     |
| `stringzillas::NeedlemanWunschScores` on 1x CPU        |          83 MCUPS |           354 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 16x CPUs      |       1'267 MCUPS |         4'694 MCUPS |
| `stringzillas::NeedlemanWunschScores` on 1x GPU        |         128 MCUPS |    __13'799 MCUPS__ |
| `stringzillas::SmithWatermanScores` on 1x CPU          |          79 MCUPS |           284 MCUPS |
| `stringzillas::SmithWatermanScores` on 16x CPUs        |   __1'026 MCUPS__ |         3'776 MCUPS |
| `stringzillas::SmithWatermanScores` on 1x GPU          |         127 MCUPS |    __13'205 MCUPS__ |

## Byte-level Fingerprinting & Sketching Benchmarks

In large-scale Retrieval workloads a common technique is to convert variable-length messy strings into some fixed-length representations.
Those are often called "fingerprints" or "sketches", like "Min-Hashing" or "Count-Min-Sketching".
There are a million variations of those algorithms, all resulting in different speed-vs-accuracy tradeoffs.
Two of the approximations worth considering is the number of collisions of produced individual hashes withing fingerprints, and the bit-distribution entropy of the produced fingerprints.
Adjusting all implementation to the same tokenization scheme, one my experience following numbers:

| Library                                    | ≅ 100 bytes lines | ≅ 1'000 bytes lines |
| ------------------------------------------ | ----------------: | ------------------: |
| serial `<ByteGrams>` 🦀                     |        0.44 MiB/s |          0.47 MiB/s |
|                                            | 92.81% collisions |   94.58% collisions |
|                                            |    0.8528 entropy |      0.7979 entropy |
|                                            |                   |                     |
| `pc::MinHash<ByteGrams>`🦀                  |        2.41 MiB/s |          3.16 MiB/s |
|                                            | 91.80% collisions |   93.17% collisions |
|                                            |    0.9343 entropy |      0.8779 entropy |
|                                            |                   |                     |
| `stringzillas::Fingerprints` on 1x CPU 🦀   |        0.56 MiB/s |          0.51 MiB/s |
| `stringzillas::Fingerprints` on 16x CPUs 🦀 |        6.62 MiB/s |          8.03 MiB/s |
| `stringzillas::Fingerprints` on 1x GPU 🦀   |  __102.07 MiB/s__ |    __392.37 MiB/s__ |
|                                            | 86.80% collisions |   93.21% collisions |
|                                            |    0.9992 entropy |      0.9967 entropy |

The trickiest part, however, is analyzing the retrieval quality of those fingerprints and comparing them to other approaches.
So, how many bits per fingerprint are needed to achieve a specific recall rate for a given dataset?
Or, how does the average Levenshtein distance among the top-k nearest neighbors change with the fingerprint size?
It must clearly decrease, but how fast, and how does that compare to ground truth?

## Replicating the Results

### Replicating the Results in Rust 🦀

Before running benchmarks, you can test your Rust environment running:

```bash
cargo install cargo-criterion --locked
```

To pull and compile all the dependencies, you can call:

```bash
cargo fetch --all-features
cargo build --all-features
```

Wars always take long, and so do these benchmarks.
Every one of them includes a few seconds of a warm-up phase to ensure that the CPU caches are filled and the results are not affected by cold start or SIMD-related frequency scaling.
Each of them accepts a few environment variables to control the dataset, the tokenization, and the error bounds.
You can log those by printing file-level documentation using `awk` on Linux:

```bash
awk '/^\/\/!/ { print } !/^\/\/!/ { exit }' bench_find.rs
```

Commonly used environment variables are:

- `STRINGWARS_DATASET` - the path to the textual dataset file.
- `STRINGWARS_TOKENS` - the tokenization mode: `file`, `lines`, or `words`.
- `STRINGWARS_ERROR_BOUND` - the maximum allowed error in the Levenshtein distance.

Here is an example of a common benchmark run on a Unix-like system:

```bash
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_hash bench_hash --jobs $(nproc)
```

On Windows using PowerShell you'd need to set the environment variable differently:

```powershell
$env:STRINGWARS_DATASET="README.md"
cargo criterion --jobs $(nproc)
```

### Replicating the Results in Python 🐍

It's recommended to use `uv` for Python dependency management and running the benchmarks.
To install all dependencies for all benchmarks:

```sh
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-cuda.txt
uv pip install --only-binary=:all: -r requirements.txt -r requirements-cuda.txt
```

To install dependencies for individual benchmarks:

```sh
PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com \
uv pip install '.[find,hash,sequence,fingerprints,similarities]'
```

To run individual benchmarks, you can call:

```sh
uv run --no-project python bench_hash.py --help
uv run --no-project python bench_find.py --help
uv run --no-project python bench_memory.py --help
uv run --no-project python bench_sequence.py --help
uv run --no-project python bench_similarities.py --help
uv run --no-project python bench_fingerprints.py --help 🔜
```

## Datasets

### ASCII Corpus

For benchmarks on ASCII data I've used the English Leipzig Corpora Collection.
It's 124 MB in size, 1'000'000 lines long, and contains 8'388'608 tokens of mean length 5.

```bash
wget --no-clobber -O leipzig1M.txt https://introcs.cs.princeton.edu/python/42sort/leipzig1m.txt 
STRINGWARS_DATASET=leipzig1M.txt cargo criterion --jobs $(nproc)
```

### UTF8 Corpus

For richer mixed UTF data, I've used the XL Sum dataset for multilingual extractive summarization.
It's 4.7 GB in size (1.7 GB compressed), 1'004'598 lines long, and contains 268'435'456 tokens of mean length 8.
To download, unpack, and run the benchmarks, execute the following bash script in your terminal:

```bash
wget --no-clobber -O xlsum.csv.gz https://github.com/ashvardanian/xl-sum/releases/download/v1.0.0/xlsum.csv.gz
gzip -d xlsum.csv.gz
STRINGWARS_DATASET=xlsum.csv cargo criterion --jobs $(nproc)
```
