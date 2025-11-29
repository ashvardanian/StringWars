# StringWars

## Text Processing on CPUs & GPUs, in Python & Rust

![StringWars Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringWa.rs.jpg?raw=true)

There are many **great** libraries for string processing!
Mostly, of course, written in Assembly, C, and C++, but some in Rust as well.

Where Rust decimates C and C++, is the **simplicity** of dependency management, making it great for benchmarking "Systems Software" and lining up apples-to-apples across native crates and their Python bindings.
So, to accelerate the development of the [`StringZilla`](https://github.com/ashvardanian/StringZilla) C, C++, and CUDA libraries (with Rust and Python bindings), I've created this repository to compare it against some of my & communities most beloved Rust projects, like:

- [`memchr`](https://github.com/BurntSushi/memchr) for substring search.
- [`rapidfuzz`](https://github.com/rapidfuzz/rapidfuzz-rs) and [`bio`](https://github.com/rust-bio/rust-bio) for edit distances and alignments.
- [`aHash`](https://github.com/tkaitchuck/aHash), [`xxhash-rust`](https://github.com/DoumanAsh/xxhash-rust), [`foldhash`](https://github.com/orlp/foldhash), and [`blake3`](https://github.com/BLAKE3-team/BLAKE3) for hashing.
- [`aho_corasick`](https://github.com/BurntSushi/aho-corasick) and [`regex`](https://github.com/rust-lang/regex) for multi-pattern search.
- [`arrow`](https://github.com/apache/arrow-rs) and [`polars`](https://github.com/pola-rs/polars) for collections and sorting.
- [`icu`](https://github.com/unicode-org/icu4x) for Unicode processing.
- [`ring`](https://github.com/briansmith/ring) and [`sodiumoxide`](https://github.com/sodiumoxide/sodiumoxide) for encryption.

Of course, the functionality of the projects is different, as are the APIs and the usage patterns.
So, I focus on the workloads for which StringZilla was designed and compare the throughput of the core operations.
Notably, I also favor modern hardware with support for a wider range SIMD instructions, like mask-equipped AVX-512 on x86 starting from the 2015 Intel Skylake-X CPUs or more recent predicated variable-length SVE and SVE2 on Arm, that aren't often supported by existing libraries and tooling.

> [!IMPORTANT]  
> The numbers in the tables below are provided for reference only and may vary depending on the CPU, compiler, dataset, and tokenization method.
> Most of them were obtained on Intel Sapphire Rapids **(SPR)** and Granite Rapids **(GNR)** CPUs and Nvidia Hopper-based **H100** and Blackwell-based **RTX 6000** Pro GPUs, using Rust with `-C target-cpu=native` optimization flag.
> To replicate the results, please refer to the [Replicating the Results](#replicating-the-results) section below.

## Benchmarks at a Glance

### Hash

Many hashing libraries exist, but they often lack reproducible outputs, streaming support, or cross-language availability.
Throughput on short words and long lines:

```
                    Short Words                  Long Lines
Rust:
stringzilla::hash   ████████████████████ 1.84    ████████████████████ 11.38 GB/s
aHash::hash_one     █████████████▍       1.23    ███████████████▏      8.61 GB/s
xxh3::xxh3_64       ███████████▊         1.08    ████████████████▋     9.48 GB/s
std::hash           ████▋                0.43    ██████▌               3.74 GB/s

Python:
stringzilla.hash    ████████████████████ 0.14    ████████████████████  9.19 GB/s
hash                ██████████████████▌  0.13    █████████▎            4.27 GB/s
xxhash.xxh3_64      █████▋               0.04    █████████████▉        6.38 GB/s
```

See [bench_hash.md](bench_hash.md) for details

### Substring Search

Substring search is offloaded to C's `memmem` or `strstr` in most languages, but SIMD-optimized implementations can do better.
Throughput on long lines:

```
                    Left to right                Reverse order
Rust:
memmem::Finder      ████████████████████ 10.99
stringzilla         ███████████████████▋ 10.82   ████████████████████ 10.66 GB/s
std::str            ███████████████████▊ 10.88   ███████████▏          5.94 GB/s

Python:
stringzilla         ████████████████████ 11.79   ████████████████████ 11.56 GB/s
str                 ██                    1.23   ██████▋               3.84 GB/s
```

See [bench_find.md](bench_find.md) for details

### Byte-Set Search

Searching for character sets (tabs, HTML markup, digits) commonly uses regex or Aho-Corasick automata.
Throughput counting all matches on long lines:

```
Rust:
stringzilla         ████████████████████   8.17 GB/s
regex::find_iter    ████████████▊          5.22 GB/s
aho_corasick        █▏                     0.50 GB/s

Python:
stringzilla         ████████████████████   8.79 GB/s
re.finditer         ▍                      0.19 GB/s
```

See [bench_find.md](bench_find.md) for details

### UTF-8 Processing

Different scripts stress UTF-8 differently: Korean has 3-byte Hangul with single-byte whitespace (representative for tokenization), Arabic uses 2-byte characters, English is mostly 1-byte ASCII.
Throughput on AMD Zen5 Turin:

```
                      English                     Arabic
Newline splitting:
stringzilla           ████████████████ 15.45      ████████████████████ 18.34 GB/s
stdlib                ██                1.90      ██                    1.82 GB/s

                      English                     Korean
Whitespace splitting:
stringzilla           ████████████████████ 0.82   ████████████████████ 1.88 GB/s
stdlib                ██████████████████▊  0.77   ██████████▍          0.98 GB/s
icu::WhiteSpace       ██▋                  0.11   █▌                   0.15 GB/s
```

Case folding on bicameral scripts (Latin, Cyrillic, Greek, Armenian) plus Chinese for reference:

```
                      English 16x                 German 6x
Case folding:
stringzilla           ████████████████████ 7.53   ████████████████████ 2.59 GB/s
stdlib                ██▌                  0.48   ███▎                 0.43 GB/s

                      Russian 10x                 French 5x
stringzilla           ████████████████████ 2.20   ████████████████████ 1.84 GB/s
stdlib                ██                   0.22   ███▊                 0.35 GB/s

                      Greek 5x                    Armenian 4x
stringzilla           ████████████████████ 1.00   ████████████████████  908 MB/s
stdlib                ████▍                0.22   ████▉                 223 MB/s

                      Vietnamese 1.3x             Chinese 4x
stringzilla           ████████████████████  352   ████████████████████ 1.21 GB/s
stdlib                █████████████▏        265   █████▍                325 MB/s
```

See [bench_unicode.md](bench_unicode.md) for details

### Sequence Operations

Dataframe libraries and search engines rely heavily on string sorting.
SIMD-accelerated comparisons and specialized radix sorts can outperform generic algorithms.
Throughput on short words:

```
Rust:
stringzilla         ████████████████████  213.73 M cmp/s
polars::sort        ██████████████████▊   200.34 M cmp/s
arrow::lexsort      ███████████▍          122.20 M cmp/s
std::sort           █████                  54.35 M cmp/s

Python:
polars.sort         ████████████████████  223.38 M cmp/s
stringzilla.sorted  ███████████████▎      171.13 M cmp/s
pyarrow.sort        █████▌                 62.17 M cmp/s
list.sort           ████▏                  47.06 M cmp/s
```

GPU: `cudf` on H100 reaches **9,463 M cmp/s** on short words.

See [bench_sequence.md](bench_sequence.md) for details

### Random Generation

Random byte generation and lookup tables are common in image processing and bioinformatics.
Throughput on long lines:

```
Rust:
stringzilla         ████████████████████  10.57 GB/s
zeroize             ████████▉              4.73 GB/s
rand_xoshiro        ███████▎               3.85 GB/s

Python:
stringzilla         ████████████████████  20.37 GB/s
pycryptodome        ████████████▉         13.16 GB/s
numpy.Philox        █▌                     1.59 GB/s
```

See [bench_memory.md](bench_memory.md) for details

### Similarity Scoring

Edit distance is essential for search engines, data cleaning, NLP, and bioinformatics.
It's computationally expensive with O(n\*m) complexity, but GPUs and multi-core parallelism help.
Levenshtein distance on ~1,000 byte lines (MCUPS = Million Cell Updates Per Second):

```
                        1 Core                       1 Socket
Rust:
bio::levenshtein        █▏                      823
rapidfuzz               ████████████████████ 14,316
stringzilla (384x GNR)  ██████████████████▎  13,084  ████████████████████ 3,084,270 MCUPS
stringzilla (B200)                                   ██████▍                998,620 MCUPS
stringzilla (H100)                                   ██████                 925,890 MCUPS
```

See [bench_similarities.md](bench_similarities.md) for details

### Fingerprinting

Converting variable-length strings into fixed-length sketches (like Min-Hashing) enables fast approximate matching in large-scale retrieval.
Throughput on ~1,000 byte lines:

```
                        1 Core                       1 Socket
Rust:
pc::MinHash             ████████████████████   3.16
stringzilla (384x GNR)  ███▏                   0.51  ███████████████▍      302.30 MB/s
stringzilla (H100)                                   ████████████████████  392.37 MB/s
```

See [bench_fingerprints.md](bench_fingerprints.md) for details

### Encryption

ChaCha20 and AES256 encryption throughput comparison on long lines:

```
Rust:
ring::aes256        ████████████████████   2.89 GB/s
ring::chacha20      ████████▏              1.19 GB/s
libsodium::chacha20 █████                  0.71 GB/s
```

See [bench_encryption.md](bench_encryption.md) for details

## Replicating the Results

### Replicating the Results in Rust

Before running benchmarks, you can test your Rust environment running:

```bash
cargo install cargo-criterion --locked
```

To pull and compile all the dependencies, you can call:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --benches --all-features                  # to compile everything
RUSTFLAGS="-C target-cpu=native" cargo check --benches --all-features --all-targets    # to fail on warnings
```

By default StringWars links `stringzilla` in CPU mode.
If the machine has an NVIDIA GPU with CUDA installed, enable the CUDA kernels explicitly when running benches, for example:

```bash
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    STRINGWARS_FILTER=GPU \
    cargo criterion --features "cuda bench_similarities" bench_similarities --jobs 1
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

### Replicating the Results in Python

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
uv run --no-project python bench_fingerprints.py --help
```

## Datasets

### UTF8 Corpus

For mixed UTF data, I've used the XL Sum dataset for multilingual extractive summarization.
It's 4.7 GB in size (1.7 GB compressed), 1'004'598 lines long, and contains 268'435'456 tokens of mean length 8.
To download, unpack, and run the benchmarks, execute the following bash script in your terminal:

```bash
curl -fL -o xlsum.csv.gz https://github.com/ashvardanian/xl-sum/releases/download/v1.0.0/xlsum.csv.gz
gzip -d xlsum.csv.gz
STRINGWARS_DATASET=xlsum.csv cargo criterion --jobs $(nproc)
```

### Multilingual Wikipedia Corpus

The Cohere Wikipedia dataset provides pre-processed JSONL files for different languages.
This may be the optimal dataset for relative comparison of UTF-8 decoding and matching enginges in each individual environment.
Not all Wikipedia languages are available, but the following have been selected specifically:

- **Chinese (zh)**: 3-byte CJK characters, rare 1-byte punctuation
- **Korean (ko)**: 3-byte Hangul syllables, frequent 1-byte punctuation
- **Arabic (ar)**: 2-byte Arabic script, with regular 1-byte punctuation
- **French (fr)**: Mixed 1-2 byte Latin with high diacritic density
- **English (en)**: Mostly 1-byte ASCII baseline

To download and decompress one file from each language:

```bash
curl -fL -o wiki_en.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/en/000.jsonl.gz && gunzip wiki_en.jsonl.gz
curl -fL -o wiki_zh.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/zh/000.jsonl.gz && gunzip wiki_zh.jsonl.gz
curl -fL -o wiki_ko.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/ko/000.jsonl.gz && gunzip wiki_ko.jsonl.gz
curl -fL -o wiki_ar.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/ar/000.jsonl.gz && gunzip wiki_ar.jsonl.gz
curl -fL -o wiki_fr.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/fr/000.jsonl.gz && gunzip wiki_fr.jsonl.gz
curl -fL -o wiki_de.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/de/000.jsonl.gz && gunzip wiki_de.jsonl.gz
curl -fL -o wiki_es.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/es/000.jsonl.gz && gunzip wiki_es.jsonl.gz
curl -fL -o wiki_it.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/it/000.jsonl.gz && gunzip wiki_it.jsonl.gz
```

Each JSONL file contains one JSON object per line with fields: `id`, `title`, `text` (paragraph content), `url`, `wiki_id`, and `paragraph_id`.

### CC-100 Corpus

The [CC-100](https://data.statmt.org/cc-100/) corpus provides large monolingual text files (1-80 GB) for 100+ languages, extracted from Common Crawl.
Files are XZ-compressed plain text with documents separated by double-newlines.

| Workload                    | Relevant Scripts                  | Best Test Languages                                  |
| --------------------------- | --------------------------------- | ---------------------------------------------------- |
| **Case Folding**            | Latin, Cyrillic, Greek, Armenian  | Turkish (I/i), German (ss->SS), Greek, Russian       |
| **Normalization**           | Indic, Arabic, Vietnamese, Korean | Vietnamese, Hindi, Korean, Arabic                    |
| **Whitespace Tokenization** | Most scripts except CJK/Thai      | English, Russian, Arabic vs. Chinese, Japanese, Thai |
| **Grapheme Clusters**       | Indic, Thai, Khmer, Myanmar       | Thai, Tamil, Myanmar, Khmer                          |
| **RTL Handling**            | Arabic, Hebrew                    | Arabic, Hebrew, Persian                              |

**Bicameral scripts** with various case folding rules:

```bash
curl -fL https://data.statmt.org/cc-100/en.txt.xz | xz -d > cc100_en.txt      # 82 GB - English
curl -fL https://data.statmt.org/cc-100/de.txt.xz | xz -d > cc100_de.txt      # 18 GB - German
curl -fL https://data.statmt.org/cc-100/tr.txt.xz | xz -d > cc100_tr.txt      # 5.4 GB - Turkish
curl -fL https://data.statmt.org/cc-100/ru.txt.xz | xz -d > cc100_ru.txt      # 46 GB - Russian
curl -fL https://data.statmt.org/cc-100/uk.txt.xz | xz -d > cc100_uk.txt      # 14 GB - Ukrainian
curl -fL https://data.statmt.org/cc-100/el.txt.xz | xz -d > cc100_el.txt      # 7.4 GB - Greek
curl -fL https://data.statmt.org/cc-100/hy.txt.xz | xz -d > cc100_hy.txt      # 776 MB - Armenian
curl -fL https://data.statmt.org/cc-100/ka.txt.xz | xz -d > cc100_ka.txt      # 1.1 GB - Georgian
curl -fL https://data.statmt.org/cc-100/pl.txt.xz | xz -d > cc100_pl.txt      # 12 GB - Polish
curl -fL https://data.statmt.org/cc-100/cs.txt.xz | xz -d > cc100_cs.txt      # 4.4 GB - Czech
curl -fL https://data.statmt.org/cc-100/nl.txt.xz | xz -d > cc100_nl.txt      # 7.9 GB - Dutch
curl -fL https://data.statmt.org/cc-100/fr.txt.xz | xz -d > cc100_fr.txt      # 14 GB - French
curl -fL https://data.statmt.org/cc-100/es.txt.xz | xz -d > cc100_es.txt      # 14 GB - Spanish
curl -fL https://data.statmt.org/cc-100/pt.txt.xz | xz -d > cc100_pt.txt      # 13 GB - Portuguese
curl -fL https://data.statmt.org/cc-100/it.txt.xz | xz -d > cc100_it.txt      # 7.8 GB - Italian
```

**Unicameral scripts** without case folding, but with other normalization/segmentation challenges:

```bash
curl -fL https://data.statmt.org/cc-100/ar.txt.xz | xz -d > cc100_ar.txt      # 5.4 GB - Arabic (RTL)
curl -fL https://data.statmt.org/cc-100/he.txt.xz | xz -d > cc100_he.txt      # 6.1 GB - Hebrew (RTL)
curl -fL https://data.statmt.org/cc-100/fa.txt.xz | xz -d > cc100_fa.txt      # 20 GB - Persian (RTL)
curl -fL https://data.statmt.org/cc-100/hi.txt.xz | xz -d > cc100_hi.txt      # 2.5 GB - Hindi (Devanagari)
curl -fL https://data.statmt.org/cc-100/bn.txt.xz | xz -d > cc100_bn.txt      # 860 MB - Bengali
curl -fL https://data.statmt.org/cc-100/ta.txt.xz | xz -d > cc100_ta.txt      # 1.3 GB - Tamil
curl -fL https://data.statmt.org/cc-100/te.txt.xz | xz -d > cc100_te.txt      # 536 MB - Telugu
curl -fL https://data.statmt.org/cc-100/th.txt.xz | xz -d > cc100_th.txt      # 8.7 GB - Thai (no spaces)
curl -fL https://data.statmt.org/cc-100/vi.txt.xz | xz -d > cc100_vi.txt      # 28 GB - Vietnamese
curl -fL https://data.statmt.org/cc-100/zh-Hans.txt.xz | xz -d > cc100_zh.txt # 14 GB - Chinese
curl -fL https://data.statmt.org/cc-100/ja.txt.xz | xz -d > cc100_ja.txt      # 15 GB - Japanese
curl -fL https://data.statmt.org/cc-100/ko.txt.xz | xz -d > cc100_ko.txt      # 14 GB - Korean (Jamo)
curl -fL https://data.statmt.org/cc-100/my.txt.xz | xz -d > cc100_my.txt      # 46 MB - Myanmar
curl -fL https://data.statmt.org/cc-100/km.txt.xz | xz -d > cc100_km.txt      # 153 MB - Khmer
curl -fL https://data.statmt.org/cc-100/am.txt.xz | xz -d > cc100_am.txt      # 133 MB - Amharic (Ethiopic)
curl -fL https://data.statmt.org/cc-100/si.txt.xz | xz -d > cc100_si.txt      # 452 MB - Sinhala
```

### Leipzig Corpora Collection

The [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/) provides pre-segmented sentences in 200+ languages.
Each tar.gz contains `*-sentences.txt` (tab-separated `id\tsentence`), `*-words.txt` (frequencies), and co-occurrence files.
Standard sizes: 10K, 30K, 100K, 300K, 1M sentences. Check for newer years at the download page.

**Bicameral scripts** with various case folding rules:

```bash
curl -fL https://downloads.wortschatz-leipzig.de/corpora/eng_wikipedia_2016_1M.tar.gz | tar -xzf - -O 'eng_wikipedia_2016_1M/eng_wikipedia_2016_1M-sentences.txt' | cut -f2 > leipzig1M_en.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/deu_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'deu_wikipedia_2021_1M/deu_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_de.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/tur_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'tur_wikipedia_2021_1M/tur_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_tr.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/rus_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'rus_wikipedia_2021_1M/rus_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_ru.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ukr_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ukr_wikipedia_2021_1M/ukr_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_uk.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ell_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ell_wikipedia_2021_1M/ell_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_el.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/hye_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'hye_wikipedia_2021_1M/hye_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_hy.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/kat_wikipedia_2021_300K.tar.gz | tar -xzf - -O 'kat_wikipedia_2021_300K/kat_wikipedia_2021_300K-sentences.txt' | cut -f2 > leipzig300K_ka.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/pol_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'pol_wikipedia_2021_1M/pol_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_pl.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ces_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ces_wikipedia_2021_1M/ces_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_cs.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/nld_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'nld_wikipedia_2021_1M/nld_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_nl.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/fra_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'fra_wikipedia_2021_1M/fra_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_fr.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/spa_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'spa_wikipedia_2021_1M/spa_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_es.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/por_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'por_wikipedia_2021_1M/por_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_pt.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ita_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ita_wikipedia_2021_1M/ita_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_it.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/lit_wikipedia_2021_300K.tar.gz | tar -xzf - -O 'lit_wikipedia_2021_300K/lit_wikipedia_2021_300K-sentences.txt' | cut -f2 > leipzig300K_lt.txt
```

**Unicameral scripts** without case folding, but with other normalization/segmentation challenges:

```bash
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ara_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ara_wikipedia_2021_1M/ara_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_ar.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/heb_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'heb_wikipedia_2021_1M/heb_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_he.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/fas_wikipedia_2014_1M.tar.gz | tar -xzf - -O 'fas_wikipedia_2014_1M/fas_wikipedia_2014_1M-sentences.txt' | cut -f2 > leipzig1M_fa.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/hin_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'hin_wikipedia_2021_1M/hin_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_hi.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ben_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ben_wikipedia_2021_1M/ben_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_bn.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/tam_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'tam_wikipedia_2021_1M/tam_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_ta.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/tel_wikipedia_2021_300K.tar.gz | tar -xzf - -O 'tel_wikipedia_2021_300K/tel_wikipedia_2021_300K-sentences.txt' | cut -f2 > leipzig300K_te.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/tha_wikipedia_2021_10K.tar.gz | tar -xzf - -O 'tha_wikipedia_2021_10K/tha_wikipedia_2021_10K-sentences.txt' | cut -f2 > leipzig10K_th.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/vie_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'vie_wikipedia_2021_1M/vie_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_vi.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/zho_wikipedia_2018_1M.tar.gz | tar -xzf - -O 'zho_wikipedia_2018_1M/zho_wikipedia_2018_1M-sentences.txt' | cut -f2 > leipzig1M_zh.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/jpn_wikipedia_2018_1M.tar.gz | tar -xzf - -O 'jpn_wikipedia_2018_1M/jpn_wikipedia_2018_1M-sentences.txt' | cut -f2 > leipzig1M_ja.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/kor_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'kor_wikipedia_2021_1M/kor_wikipedia_2021_1M-sentences.txt' | cut -f2 > leipzig1M_ko.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/amh_wikipedia_2021_30K.tar.gz | tar -xzf - -O 'amh_wikipedia_2021_30K/amh_wikipedia_2021_30K-sentences.txt' | cut -f2 > leipzig30K_am.txt
```

To produce a mixed dataset with rows in all languages:

```bash
cat leipzig*.txt | shuf | head -c 1G > leipzig1GB.txt
```

### DNA Corpus

For bioinformatics workloads, I use the following datasets with increasing string lengths:

```bash
curl -fL -o acgt_100.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_100.txt?download=true'
curl -fL -o acgt_1k.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_1k.txt?download=true'
curl -fL -o acgt_10k.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_10k.txt?download=true'
curl -fL -o acgt_100k.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_100k.txt?download=true'
curl -fL -o acgt_1m.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_1m.txt?download=true'
curl -fL -o acgt_10m.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_10m.txt?download=true'
```

## Deep Profiling

In case you are profiling the some of the internal kernels of mentioned libraries, here are a few example commands to get around.
Such as using `ncu` for NVIDIA GPUs to evaluate the register usage and occupancy of the CUDA kernels used in StringZilla's Levenshtein distance calculation:

```bash
/usr/local/cuda/bin/ncu \
  --metrics launch__registers_per_thread,launch__occupancy_per_block_size,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum \
  --target-processes all \
  --kernel-name "levenshtein_on_each_cuda_thread" \
  --launch-skip 5 \
  --launch-count 1 \
  bash -c 'STRINGWARS_DATASET=acgt_100.txt STRINGWARS_BATCH=65536 STRINGWARS_TOKENS=lines STRINGWARS_FILTER="uniform/stringzillas::LevenshteinDistances\(1xGPU\)" cargo criterion --features "cuda bench_similarities" bench_similarities --jobs 1'
```

Using `perf` on Linux to analyze the CPU-side performance of SIMD-accelerated substring search:

```bash
perf record -e cpu-clock -g graph,0x400000 -o perf.data -- cargo criterion --features "bench_similarities" bench_similarities --jobs 1
perf report -i perf.data
```
