# StringWa.rs

![StringWa.rs Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringWa.rs.jpg?raw=true)

_Not to pick a fight, but let there be String Wars!_ 😅
Jokes aside, many __great__ libraries for string processing exist.
_Mostly, of course, written in Assembly, C, and C++, but some in Rust as well._ 😅

Where Rust decimates C and C++, however, is the __simplicity__ of dependency management, making it great for benchmarking "Systems Software"!
So, to accelerate the development of the [`StringZilla`](https://github.com/ashvardanian/StringZilla) C library, I've created this repository to compare it against some of my & communities most beloved Rust projects, like:

- [`memchr`](https://github.com/BurntSushi/memchr) for substring search.
- [`rapidfuzz`](https://github.com/rapidfuzz/rapidfuzz-rs) for edit distances.
- [`aHash`](https://github.com/tkaitchuck/aHash) for hashing.
- [`aho_corasick`](https://github.com/BurntSushi/aho-corasick) for multi-pattern search.
- [`tantivy`](https://github.com/quickwit-oss/tantivy) for document retrieval.

Of course, the functionality of the projects is different, as are the APIs and the usage patterns.
So, I focus on the workloads for which StringZilla was designed and compare the throughput of the core operations.
Notably, I also favor modern hardware with support for a wider range SIMD instructions, like mask-equipped AVX-512 on x86 starting from the 2015 Intel Skylake-X CPUs or more recent predicated variable-length SVE and SVE2 on Arm, that aren't supported by most of the existing libraries and Rust tooling.

> [!IMPORTANT]  
> The numbers in the tables below are provided for reference only and may vary depending on the CPU, compiler, dataset, and tokenization method.
> Most of them were obtained on Intel Sapphire Rapids CPUs and Nvidia H100 GPUs, using Rust with `-C target-cpu=native` optimization flag.
> To replicate the results, please refer to the [Replicating the Results](#replicating-the-results) section below.

## String Hashing Benchmarks

Many great hashing libraries exist in Rust, C, and C++.
Typical top choices are `aHash`, `xxHash`, `blake3`, `gxhash`, `CityHash`, `MurmurHash`, or the native `std::hash`.
Many of them have similar pitfalls:

- They are not always documented to have a certain reproducible output and are recommended for use only for local in-memory construction of hash tables, not for serialization or network communication.
- They don't always support streaming and require the whole input to be available in memory at once.
- They don't always pass the SMHasher test suite, especially with `--extra` checks enabled.
- They generally don't have a dynamic dispatch mechanism to simplify shipping of precompiled software to a wide range of users.
- They are rarely available for multiple programming languages.

StringZilla addresses those issues and seems to provide competitive performance.
On Intel Sapphire Rapids CPU, on `xlsum.csv` dataset, the following numbers can be expected for hashing individual whitespace-delimited words and newline-delimited lines:

| Library                | Languages         |  Shorter Words |    Longer Lines |
| ---------------------- | ----------------- | -------------: | --------------: |
| `std::hash`            | Rs                |     0.43 GiB/s |      3.74 GiB/s |
| `xxh3::xxh3_64`        | C, C++, Rs, Py... |     1.08 GiB/s |      9.48 GiB/s |
| `aHash::hash_one`      | Rs                |     1.23 GiB/s |      8.61 GiB/s |
| `gxhash::gxhash64`     | Rs                | __2.68 GiB/s__ |     10.81 GiB/s |
| `stringzilla::hash`    | C, C++, Rs, Py... |     1.84 GiB/s | __11.23 GiB/s__ |
|                        |                   |                |                 |
| `blake3::hash`         | C, Rs, Py, Go     |     0.10 GiB/s |      1.97 GiB/s |
| `stringzilla::bytesum` | C, C++, Rs, Py... |     2.16 GiB/s |     11.65 GiB/s |

> Blake3 and byte-level summation are provided as a reference for expected lower and upper bounds.
> Blake3 is a cryptographic hash function and is obliged to provide a certain level of security, which comes at a cost.
> Byte-level summation is a simple operation, that is still sometimes used in practice, and is expected to be the fastest.

In larger systems, however, we often need the ability to incrementally hash the data.
This is especially important in distributed systems, where the data is too large to fit into memory at once.

| Library                    | Languages         |  Shorter Words |   Longer Lines |
| -------------------------- | ----------------- | -------------: | -------------: |
| `std::hash::DefaultHasher` | Rs                |     0.51 GiB/s |     3.92 GiB/s |
| `aHash::AHasher`           | Rs                | __1.30 GiB/s__ | __8.56 GiB/s__ |
| `stringzilla::HashState`   | C, C++, Rs, Py... |     0.89 GiB/s |     6.39 GiB/s |

## Substring & Character-Set Search Benchmarks

Substring search is one of the most common operations in text processing, and one of the slowest.
Most of the time, programmers don't think about replacing the `str::find` method, as it's already expected to be optimized.
In many languages it's offloaded to the C standard library [`memmem`](https://man7.org/linux/man-pages/man3/memmem.3.html) or [`strstr`](https://en.cppreference.com/w/c/string/byte/strstr) for NULL-terminated strings.
The C standard library is, however, also implemented by humans, and a better solution can be created.

| Library              |   Shorter Words |    Longer Lines |
| -------------------- | --------------: | --------------: |
| `std::str::find`     |      9.48 GiB/s |     10.88 GiB/s |
| `memmem::find`       |      9.51 GiB/s |     10.83 GiB/s |
| `stringzilla::find`  | __10.45 GiB/s__ | __10.89 GiB/s__ |
|                      |                 |                 |
| `std::str::rfind`    |      2.96 GiB/s |      3.65 GiB/s |
| `memmem::rfind`      |      2.95 GiB/s |      3.71 GiB/s |
| `stringzilla::rfind` |  __9.78 GiB/s__ | __10.43 GiB/s__ |

> Higher-throughput evaluation with `memmem` is possible, if the "matcher" object is reused to iterate through the string instead of constructing a new one for each search.

Similarly, one can search a string for a set of characters.
StringWa.rs takes a few representative examples of various character sets that appear in real parsing or string validation tasks:

- tabulation characters, like `\n\r\v\f`;
- HTML and XML markup characters, like `</>&'\"=[]`;
- numeric characters, like `0123456789`.

It's common in such cases, to pre-construct some library-specific filter-object or Finite State Machine (FSM) to search for a set of characters.
Once that object is constructed, all of it's inclusions in each token (word or line) are counted.
Current numbers should look like this:

| Library                     |  Shorter Words |   Longer Lines |
| --------------------------- | -------------: | -------------: |
| `bstr::iter`                |     0.26 GiB/s |     0.25 GiB/s |
| `regex::find_iter`          |     0.23 GiB/s |     5.22 GiB/s |
| `aho_corasick::find_iter`   |     0.41 GiB/s |     0.50 GiB/s |
| `stringzilla::find_byteset` | __1.61 GiB/s__ | __8.17 GiB/s__ |

## Strings Sorting & Intersections Benchmarks

Rust has several Dataframe libraries, DBMS and Search engines that heavily rely on string sorting and intersections.
Those operations mostly are implemented using conventional algorithms:

- Comparison-based Quicksort or Mergesort for sorting.
- Hash-based or Tree-based algorithms for intersections.

Assuming the comparisons can be accelerated with SIMD and so can be the hash functions, StringZilla could already provide a performance boost in such applications, but starting with v4 it also provides specialized algorithms for sorting and intersections.
Those are directly compatible with arbitrary string-comparable collection types with a support of an indexed access to the elements.

| Library                                     |      Shorter Words |      Longer Lines |
| ------------------------------------------- | -----------------: | ----------------: |
| `std::sort_unstable_by_key`                 |      54.35 Melem/s |     57.70 Melem/s |
| `rayon::par_sort_unstable_by_key` on 1x CPU |      47.08 Melem/s |     50.35 Melem/s |
| `arrow::lexsort_to_indices`                 |     122.20 Melem/s | __84.73 Melem/s__ |
| `stringzilla::argsort_permutation`          | __182.88 Melem/s__ |     74.64 Melem/s |

## Random Generation & Lookup Tables

Some of the most common operations in data processing are random generation and lookup tables.
That's true not only for strings but for any data type, and StringZilla has been extensively used in Image Processing and Bioinformatics for those purposes.
Generating random byte-streams:

| Library                        | ≅ 100 bytes lines | ≅ 1000 bytes lines |
| ------------------------------ | ----------------: | -----------------: |
| `getrandom::fill`              |        0.18 GiB/s |         0.40 GiB/s |
| `rand_chacha::ChaCha20Rng`     |        0.62 GiB/s |         1.72 GiB/s |
| `rand_xoshiro::Xoshiro128Plus` |        2.66 GiB/s |         3.72 GiB/s |
| `zeroize::zeroize`             |        4.62 GiB/s |         4.35 GiB/s |
| `sz::fill_random`              |   __17.30 GiB/s__ |    __10.57 GiB/s__ |

Performing in-place lookups in a precomputed table of 256 bytes:

| Library              | ≅ 100 bytes lines | ≅ 1000 bytes lines |
| -------------------- | ----------------: | -----------------: |
| Serial Rust          |        1.64 GiB/s |         1.61 GiB/s |
| `sz::lookup_inplace` |    __2.28 GiB/s__ |    __13.39 GiB/s__ |


## String Edit Distance Benchmarks

Edit Distance calculation is a common component of Search Engines, Data Cleaning, and Natural Language Processing, as well as in Bioinformatics.
It's a computationally expensive operation, generally implemented using dynamic programming, with a quadratic time complexity upper bound.

| Library                                     | ≅ 100 bytes lines | ≅ 1000 bytes lines |
| ------------------------------------------- | ----------------: | -----------------: |
| Binary inputs                               |                   |                    |
| `rapidfuzz::levenshtein<Bytes>`             |       4'633 MCUPS |       14'316 MCUPS |
| `szs::LevenshteinDistances` on 1x CPU       |       3'315 MCUPS |       13'084 MCUPS |
| `szs::LevenshteinDistances` on 16x CPUs     |      29'430 MCUPS |      105'400 MCUPS |
| `szs::LevenshteinDistances` on 1x GPU       |      31'913 MCUPS |  __624'730 MCUPS__ |
|                                             |                   |                    |
| UTF-8 inputs                                |                   |                    |
| `rapidfuzz::levenshtein<Chars>`             |       3'877 MCUPS |       13'179 MCUPS |
| `szs::LevenshteinDistancesUtf8` on 1x CPU   |       3'283 MCUPS |       11'690 MCUPS |
| `szs::LevenshteinDistancesUtf8` on 16x CPUs |      38'954 MCUPS |      103'500 MCUPS |

For biological sequences, the Needleman-Wunsch and Smith-Waterman algorithms are more appropriate, as they allow overriding the default substitution costs.
Another common adaptation is to used Gotoh's affine gap penalties, which better model the evolutionary events in DNA and Protein sequences.

| Library                                  | ≅ 100 bytes lines | ≅ 1000 bytes lines |
| ---------------------------------------- | ----------------: | -----------------: |
| Linear gaps                              |                   |                    |
| `szs::NeedlemanWunschScores` on 1x CPU   |         278 MCUPS |          612 MCUPS |
| `szs::NeedlemanWunschScores` on 16x CPUs |       4'057 MCUPS |        8'492 MCUPS |
| `szs::NeedlemanWunschScores` on 1x GPU   |         131 MCUPS |   __12'113 MCUPS__ |
| `szs::SmithWatermanScores` on 1x CPU     |         263 MCUPS |          552 MCUPS |
| `szs::SmithWatermanScores` on 16x CPUs   |       3'883 MCUPS |        8'011 MCUPS |
| `szs::SmithWatermanScores` on 1x GPU     |         143 MCUPS |   __12'921 MCUPS__ |
|                                          |                   |                    |
| Affine gaps                              |                   |                    |
| `szs::NeedlemanWunschScores` on 1x CPU   |          83 MCUPS |          354 MCUPS |
| `szs::NeedlemanWunschScores` on 16x CPUs |       1'267 MCUPS |        4'694 MCUPS |
| `szs::NeedlemanWunschScores` on 1x GPU   |         128 MCUPS |   __13'799 MCUPS__ |
| `szs::SmithWatermanScores` on 1x CPU     |          79 MCUPS |          284 MCUPS |
| `szs::SmithWatermanScores` on 16x CPUs   |       1'026 MCUPS |        3'776 MCUPS |
| `szs::SmithWatermanScores` on 1x GPU     |         127 MCUPS |   __13'205 MCUPS__ |

## Byte-level Fingerprinting & Sketching Benchmarks

In large-scale Retrieval workloads a common technique is to convert variable-length messy strings into some fixed-length representations.
Those are often called "fingerprints" or "sketches", like "Min-Hashing" or "Count-Min-Sketching".
There are a million variations of those algorithms, all resulting in different speed-vs-accuracy tradeoffs.
Two of the approximations worth considering is the number of collisions of produced individual hashes withing fingerprints, and the bit-distribution entropy of the produced fingerprints.
Adjusting all implementation to the same tokenization scheme, one my experience following numbers:

| Library                         | ≅ 100 bytes lines | ≅ 1000 bytes lines |
| ------------------------------- | ----------------: | -----------------: |
| Base MinHash for `<ByteGrams>`  |        0.44 MiB/s |         0.47 MiB/s |
|                                 | 92.81% collisions |  94.58% collisions |
|                                 |    0.8528 entropy |     0.7979 entropy |
|                                 |                   |                    |
| `pc::MinHash<ByteGrams>`        |        2.41 MiB/s |         3.16 MiB/s |
|                                 | 91.80% collisions |  93.17% collisions |
|                                 |    0.9343 entropy |     0.8779 entropy |
|                                 |                   |                    |
| `szs::Fingerprints` on 1x CPU   |        0.56 MiB/s |         0.51 MiB/s |
| `szs::Fingerprints` on 16x CPUs |        6.62 MiB/s |         8.03 MiB/s |
| `szs::Fingerprints` on 1x GPU   |  __102.07 MiB/s__ |   __392.37 MiB/s__ |
|                                 | 86.80% collisions |  93.21% collisions |
|                                 |    0.9992 entropy |     0.9967 entropy |

## Replicating the Results

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
uv run --no-project python bench_find.py --help
uv run --no-project python bench_sequence.py --help
uv run --no-project python bench_similarities.py --help
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
