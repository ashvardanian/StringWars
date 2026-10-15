# Case Folding & Normalization Benchmarks

Benchmarks for Unicode case-insensitive operations and normalization — case folding, case-insensitive comparison and substring search, and NFC/NFD/NFKC/NFKD normalization — across different languages and hardware platforms.

Case folding, case-insensitive search and NFC normalization are each measured single-threaded, one row per Leipzig corpus language.
Case folding runs on a 512 KB prefix of each corpus, as one `file`-mode token.
The 512 KB size is set by `pyunormalize`, which sustains ~53 MChar/s to 800 K characters and then collapses by more than four orders of magnitude.
`stringzilla::utf8_uncased_fold` is StringZilla's full Unicode case fold.
Its Rust alternative is ICU4X's `icu::CaseMapper::fold_string` and its Python alternative is PyICU's `icu.CaseMap.foldCase`.
`std::to_lowercase` lowercases rather than folds, so it is not an alternative.
A `–` alternative is a chip where that ICU rival was not measured.

The `utf8_uncased_search` and `utf8_uncased_matches` rows search a case-insensitive needle, against `memchr::Finder` over ICU-folded text in Rust and PyICU's `icu.StringSearch` in Python.

The `utf8_norm` rows normalize each corpus into NFC; NFD, NFKC and NFKD were measured too and follow the same ordering, so only NFC is tabulated.
Rust races `icu::ComposingNormalizer` and the `unicode-normalization` crate, and Python races PyICU's `icu.Normalizer2`, with `unicodedata.normalize` and [`pyunormalize`](https://github.com/mlodel/pyunormalize) — a pure-Python UAX #15 implementation that ships its own Unicode tables, included as the reference for what the algorithm costs with no native code behind it — listed as other benchmarked rivals.

Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.
A language has rows on a chip only where it was measured there.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `std::to_lowercase`, `pcre2::pre-jit`, `pcre2::jit-on-fly`, `pcre2::no-jit`, `unicode-normalization::nfc`.

| StringZilla                                    |     Speed | Speedup | Alternative                    |      Speed |
| :--------------------------------------------- | --------: | ------: | :----------------------------- | ---------: |
| `stringzilla::utf8_uncased_fold`, Arabic       |  4.7 GB/s |   54.7× | `icu::CaseMapper::fold_string` |    86 MB/s |
| `stringzilla::utf8_uncased_fold`, Armenian     |  710 MB/s |   4.97× | `icu::CaseMapper::fold_string` |   143 MB/s |
| `stringzilla::utf8_uncased_fold`, Bengali      |  4.3 GB/s |   18.7× | `icu::CaseMapper::fold_string` |   230 MB/s |
| `stringzilla::utf8_uncased_fold`, Chinese      |  430 MB/s |   2.26× | `icu::CaseMapper::fold_string` |   190 MB/s |
| `stringzilla::utf8_uncased_fold`, Czech        |  2.4 GB/s |   32.0× | `icu::CaseMapper::fold_string` |    75 MB/s |
| `stringzilla::utf8_uncased_fold`, Dutch        |   21 GB/s |    285× | `icu::CaseMapper::fold_string` |  73.6 MB/s |
| `stringzilla::utf8_uncased_fold`, English      |   20 GB/s |    292× | `icu::CaseMapper::fold_string` |  68.6 MB/s |
| `stringzilla::utf8_uncased_fold`, Farsi        |  2.4 GB/s |   15.0× | `icu::CaseMapper::fold_string` |   160 MB/s |
| `stringzilla::utf8_uncased_fold`, French       |  2.7 GB/s |   37.5× | `icu::CaseMapper::fold_string` |    72 MB/s |
| `stringzilla::utf8_uncased_fold`, Georgian     |  1.6 GB/s |   11.4× | `icu::CaseMapper::fold_string` |   140 MB/s |
| `stringzilla::utf8_uncased_fold`, German       |  4.1 GB/s |   59.5× | `icu::CaseMapper::fold_string` |  68.9 MB/s |
| `stringzilla::utf8_uncased_fold`, Greek        |  2.4 GB/s |   24.5× | `icu::CaseMapper::fold_string` |  97.8 MB/s |
| `stringzilla::utf8_uncased_fold`, Hebrew       |  4.9 GB/s |   37.4× | `icu::CaseMapper::fold_string` |   131 MB/s |
| `stringzilla::utf8_uncased_fold`, Hindi        |  4.3 GB/s |   19.5× | `icu::CaseMapper::fold_string` |   220 MB/s |
| `stringzilla::utf8_uncased_fold`, Italian      |  7.5 GB/s |    107× | `icu::CaseMapper::fold_string` |  70.3 MB/s |
| `stringzilla::utf8_uncased_fold`, Japanese     |  1.4 GB/s |   7.37× | `icu::CaseMapper::fold_string` |   190 MB/s |
| `stringzilla::utf8_uncased_fold`, Korean       |  2.2 GB/s |   12.5× | `icu::CaseMapper::fold_string` |   176 MB/s |
| `stringzilla::utf8_uncased_fold`, Lithuanian   |  1.6 GB/s |   24.2× | `icu::CaseMapper::fold_string` |  66.2 MB/s |
| `stringzilla::utf8_uncased_fold`, Polish       |  2.1 GB/s |   29.8× | `icu::CaseMapper::fold_string` |  70.4 MB/s |
| `stringzilla::utf8_uncased_fold`, Portuguese   |  3.4 GB/s |   47.8× | `icu::CaseMapper::fold_string` |  71.2 MB/s |
| `stringzilla::utf8_uncased_fold`, Russian      |  1.6 GB/s |   14.5× | `icu::CaseMapper::fold_string` |   110 MB/s |
| `stringzilla::utf8_uncased_fold`, Spanish      |  3.8 GB/s |   52.6× | `icu::CaseMapper::fold_string` |  72.3 MB/s |
| `stringzilla::utf8_uncased_fold`, Tamil        |  4.4 GB/s |   21.0× | `icu::CaseMapper::fold_string` |   210 MB/s |
| `stringzilla::utf8_uncased_fold`, Turkish      |  2.0 GB/s |   28.4× | `icu::CaseMapper::fold_string` |  70.3 MB/s |
| `stringzilla::utf8_uncased_fold`, Ukrainian    |  1.5 GB/s |   13.2× | `icu::CaseMapper::fold_string` |   114 MB/s |
| `stringzilla::utf8_uncased_fold`, Vietnamese   |  2.5 GB/s |   29.1× | `icu::CaseMapper::fold_string` |  85.8 MB/s |
| `stringzilla::utf8_uncased_search`, Arabic     |   13 GB/s |   95.6× | `memchr::Finder<icu-fold>`     |   136 MB/s |
| `stringzilla::utf8_uncased_search`, Armenian   |  805 MB/s |   5.88× | `memchr::Finder<icu-fold>`     |   137 MB/s |
| `stringzilla::utf8_uncased_search`, Bengali    |   28 GB/s |    122× | `memchr::Finder<icu-fold>`     |   230 MB/s |
| `stringzilla::utf8_uncased_search`, Chinese    | 22.7 GB/s |    119× | `memchr::Finder<icu-fold>`     |   190 MB/s |
| `stringzilla::utf8_uncased_search`, Czech      |  8.8 GB/s |    119× | `memchr::Finder<icu-fold>`     |    74 MB/s |
| `stringzilla::utf8_uncased_search`, Dutch      |  8.5 GB/s |    117× | `memchr::Finder<icu-fold>`     |  72.7 MB/s |
| `stringzilla::utf8_uncased_search`, English    | 10.4 GB/s |    149× | `memchr::Finder<icu-fold>`     |    70 MB/s |
| `stringzilla::utf8_uncased_search`, Farsi      |   27 GB/s |    176× | `memchr::Finder<icu-fold>`     |   153 MB/s |
| `stringzilla::utf8_uncased_search`, French     |  7.1 GB/s |   97.9× | `memchr::Finder<icu-fold>`     |  72.5 MB/s |
| `stringzilla::utf8_uncased_search`, Georgian   | 8.96 GB/s |   64.9× | `memchr::Finder<icu-fold>`     |   138 MB/s |
| `stringzilla::utf8_uncased_search`, German     | 6.01 GB/s |   87.7× | `memchr::Finder<icu-fold>`     |  68.5 MB/s |
| `stringzilla::utf8_uncased_search`, Greek      | 4.25 GB/s |   43.3× | `memchr::Finder<icu-fold>`     |  98.1 MB/s |
| `stringzilla::utf8_uncased_search`, Hebrew     |   15 GB/s |    115× | `memchr::Finder<icu-fold>`     |   131 MB/s |
| `stringzilla::utf8_uncased_search`, Hindi      |   11 GB/s |   52.6× | `memchr::Finder<icu-fold>`     |   209 MB/s |
| `stringzilla::utf8_uncased_search`, Italian    | 10.6 GB/s |    152× | `memchr::Finder<icu-fold>`     |  69.6 MB/s |
| `stringzilla::utf8_uncased_search`, Japanese   | 20.2 GB/s |    105× | `memchr::Finder<icu-fold>`     |   193 MB/s |
| `stringzilla::utf8_uncased_search`, Korean     |   32 GB/s |    183× | `memchr::Finder<icu-fold>`     |   175 MB/s |
| `stringzilla::utf8_uncased_search`, Lithuanian |   13 GB/s |    197× | `memchr::Finder<icu-fold>`     |  66.0 MB/s |
| `stringzilla::utf8_uncased_search`, Polish     | 10.3 GB/s |    143× | `memchr::Finder<icu-fold>`     |  71.9 MB/s |
| `stringzilla::utf8_uncased_search`, Portuguese |   11 GB/s |    154× | `memchr::Finder<icu-fold>`     |  71.2 MB/s |
| `stringzilla::utf8_uncased_search`, Russian    | 7.28 GB/s |   66.9× | `memchr::Finder<icu-fold>`     | 108.8 MB/s |
| `stringzilla::utf8_uncased_search`, Spanish    | 7.41 GB/s |    103× | `memchr::Finder<icu-fold>`     | 71.93 MB/s |
| `stringzilla::utf8_uncased_search`, Tamil      |   21 GB/s |    107× | `memchr::Finder<icu-fold>`     |   196 MB/s |
| `stringzilla::utf8_uncased_search`, Turkish    | 12.0 GB/s |    172× | `memchr::Finder<icu-fold>`     |  69.8 MB/s |
| `stringzilla::utf8_uncased_search`, Ukrainian  | 9.94 GB/s |   88.0× | `memchr::Finder<icu-fold>`     |   113 MB/s |
| `stringzilla::utf8_uncased_search`, Vietnamese |  4.6 GB/s |   53.9× | `memchr::Finder<icu-fold>`     | 85.32 MB/s |
| `stringzilla::utf8_norm`, Arabic               |  600 MB/s |   1.30× | `icu::ComposingNormalizer`     |   460 MB/s |
| `stringzilla::utf8_norm`, Armenian             |  590 MB/s |   1.23× | `icu::ComposingNormalizer`     |   480 MB/s |
| `stringzilla::utf8_norm`, Bengali              |  320 MB/s |   1.23× | `icu::ComposingNormalizer`     |   260 MB/s |
| `stringzilla::utf8_norm`, Chinese              |  690 MB/s |   0.92× | `icu::ComposingNormalizer`     |   750 MB/s |
| `stringzilla::utf8_norm`, Czech                |  9.5 GB/s |   4.75× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, Dutch                |   27 GB/s |   13.5× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, English              |   13 GB/s |   6.50× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, Farsi                |  550 MB/s |   1.20× | `icu::ComposingNormalizer`     |   460 MB/s |
| `stringzilla::utf8_norm`, French               |  6.2 GB/s |   3.10× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, Georgian             |  640 MB/s |   0.93× | `icu::ComposingNormalizer`     |   690 MB/s |
| `stringzilla::utf8_norm`, German               |  6.9 GB/s |   3.45× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, Greek                |  600 MB/s |   1.22× | `icu::ComposingNormalizer`     |   490 MB/s |
| `stringzilla::utf8_norm`, Hebrew               |  570 MB/s |   1.21× | `icu::ComposingNormalizer`     |   470 MB/s |
| `stringzilla::utf8_norm`, Hindi                |  570 MB/s |   1.36× | `icu::ComposingNormalizer`     |   420 MB/s |
| `stringzilla::utf8_norm`, Italian              |   14 GB/s |   7.00× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, Japanese             |  660 MB/s |   0.90× | `icu::ComposingNormalizer`     |   730 MB/s |
| `stringzilla::utf8_norm`, Korean               |  530 MB/s |   0.90× | `icu::ComposingNormalizer`     |   590 MB/s |
| `stringzilla::utf8_norm`, Lithuanian           |  3.6 GB/s |   1.80× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, Polish               |  5.8 GB/s |   2.90× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, Portuguese           |   14 GB/s |   7.00× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, Russian              |  4.8 GB/s |   10.0× | `icu::ComposingNormalizer`     |   480 MB/s |
| `stringzilla::utf8_norm`, Spanish              |   14 GB/s |   7.00× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, Tamil                |  490 MB/s |   1.96× | `icu::ComposingNormalizer`     |   250 MB/s |
| `stringzilla::utf8_norm`, Turkish              |  6.6 GB/s |   3.30× | `icu::ComposingNormalizer`     |     2 GB/s |
| `stringzilla::utf8_norm`, Ukrainian            |  2.3 GB/s |   4.51× | `icu::ComposingNormalizer`     |   510 MB/s |
| `stringzilla::utf8_norm`, Vietnamese           |  520 MB/s |   0.57× | `icu::ComposingNormalizer`     |   910 MB/s |

> Measured October 15, 2026.

### Python

Also benchmarked: `str.casefold`, `str.casefold.find`, `regex.finditer`, `unicodedata.normalize`, `pyunormalize.normalize`.

| StringZilla                                    |     Speed | Speedup | Alternative            |      Speed |
| :--------------------------------------------- | --------: | ------: | :--------------------- | ---------: |
| `stringzilla.utf8_uncased_fold`, Arabic        |  4.7 GB/s |   16.8× | `icu.CaseMap.foldCase` |   280 MB/s |
| `stringzilla.utf8_uncased_fold`, Armenian      |  710 MB/s |   2.73× | `icu.CaseMap.foldCase` |   260 MB/s |
| `stringzilla.utf8_uncased_fold`, Bengali       |  4.3 GB/s |   4.78× | `icu.CaseMap.foldCase` |   900 MB/s |
| `stringzilla.utf8_uncased_fold`, Chinese       |  440 MB/s |   1.10× | `icu.CaseMap.foldCase` |   400 MB/s |
| `stringzilla.utf8_uncased_fold`, Czech         |  2.3 GB/s |   6.22× | `icu.CaseMap.foldCase` |   370 MB/s |
| `stringzilla.utf8_uncased_fold`, Dutch         |   19 GB/s |   57.6× | `icu.CaseMap.foldCase` |   330 MB/s |
| `stringzilla.utf8_uncased_fold`, English       |   18 GB/s |   62.1× | `icu.CaseMap.foldCase` |   290 MB/s |
| `stringzilla.utf8_uncased_fold`, Farsi         |  2.4 GB/s |   6.86× | `icu.CaseMap.foldCase` |   350 MB/s |
| `stringzilla.utf8_uncased_fold`, French        | 2.66 GB/s |   7.82× | `icu.CaseMap.foldCase` |   340 MB/s |
| `stringzilla.utf8_uncased_fold`, Georgian      |  1.6 GB/s |   8.00× | `icu.CaseMap.foldCase` |   200 MB/s |
| `stringzilla.utf8_uncased_fold`, German        |  4.4 GB/s |   17.6× | `icu.CaseMap.foldCase` |   250 MB/s |
| `stringzilla.utf8_uncased_fold`, Greek         |  2.4 GB/s |   21.8× | `icu.CaseMap.foldCase` |   110 MB/s |
| `stringzilla.utf8_uncased_fold`, Hebrew        |  4.8 GB/s |   18.5× | `icu.CaseMap.foldCase` |   260 MB/s |
| `stringzilla.utf8_uncased_fold`, Hindi         |  4.3 GB/s |   5.51× | `icu.CaseMap.foldCase` |   780 MB/s |
| `stringzilla.utf8_uncased_fold`, Italian       |  7.1 GB/s |   20.9× | `icu.CaseMap.foldCase` |   340 MB/s |
| `stringzilla.utf8_uncased_fold`, Japanese      |  1.4 GB/s |   2.98× | `icu.CaseMap.foldCase` |   470 MB/s |
| `stringzilla.utf8_uncased_fold`, Korean        |  2.2 GB/s |   3.06× | `icu.CaseMap.foldCase` |   720 MB/s |
| `stringzilla.utf8_uncased_fold`, Lithuanian    |  1.6 GB/s |   4.85× | `icu.CaseMap.foldCase` |   330 MB/s |
| `stringzilla.utf8_uncased_fold`, Polish        |  2.1 GB/s |   6.36× | `icu.CaseMap.foldCase` |   330 MB/s |
| `stringzilla.utf8_uncased_fold`, Portuguese    |  3.3 GB/s |   10.3× | `icu.CaseMap.foldCase` |   320 MB/s |
| `stringzilla.utf8_uncased_fold`, Russian       |  1.6 GB/s |   12.3× | `icu.CaseMap.foldCase` |   130 MB/s |
| `stringzilla.utf8_uncased_fold`, Spanish       |  3.7 GB/s |   11.2× | `icu.CaseMap.foldCase` |   330 MB/s |
| `stringzilla.utf8_uncased_fold`, Tamil         | 4.39 GB/s |   9.98× | `icu.CaseMap.foldCase` |   440 MB/s |
| `stringzilla.utf8_uncased_fold`, Turkish       | 2.00 GB/s |   6.01× | `icu.CaseMap.foldCase` |   333 MB/s |
| `stringzilla.utf8_uncased_fold`, Ukrainian     |  1.5 GB/s |   10.0× | `icu.CaseMap.foldCase` |   150 MB/s |
| `stringzilla.utf8_uncased_fold`, Vietnamese    |  2.5 GB/s |   10.0× | `icu.CaseMap.foldCase` |   250 MB/s |
| `stringzilla.utf8_uncased_matches`, Arabic     |  9.7 GB/s |    157× | `icu.StringSearch`     |  61.9 MB/s |
| `stringzilla.utf8_uncased_matches`, Armenian   |  800 MB/s |   9.55× | `icu.StringSearch`     |  83.8 MB/s |
| `stringzilla.utf8_uncased_matches`, Bengali    |   24 GB/s |    212× | `icu.StringSearch`     |   113 MB/s |
| `stringzilla.utf8_uncased_matches`, Chinese    |   15 GB/s |    225× | `icu.StringSearch`     |  66.8 MB/s |
| `stringzilla.utf8_uncased_matches`, Czech      | 8.48 GB/s |    183× | `icu.StringSearch`     |  46.4 MB/s |
| `stringzilla.utf8_uncased_matches`, Dutch      | 4.94 GB/s |    101× | `icu.StringSearch`     |  48.8 MB/s |
| `stringzilla.utf8_uncased_matches`, English    |  7.3 GB/s |    151× | `icu.StringSearch`     |  48.5 MB/s |
| `stringzilla.utf8_uncased_matches`, Farsi      |   20 GB/s |    282× | `icu.StringSearch`     | 70.91 MB/s |
| `stringzilla.utf8_uncased_matches`, French     | 6.40 GB/s |    133× | `icu.StringSearch`     |  48.2 MB/s |
| `stringzilla.utf8_uncased_matches`, Georgian   | 8.27 GB/s |   67.2× | `icu.StringSearch`     |   123 MB/s |
| `stringzilla.utf8_uncased_matches`, German     |  4.8 GB/s |   99.8× | `icu.StringSearch`     |  48.1 MB/s |
| `stringzilla.utf8_uncased_matches`, Greek      | 3.95 GB/s |   56.8× | `icu.StringSearch`     |  69.5 MB/s |
| `stringzilla.utf8_uncased_matches`, Hebrew     |   12 GB/s |    155× | `icu.StringSearch`     |  77.5 MB/s |
| `stringzilla.utf8_uncased_matches`, Hindi      |  9.1 GB/s |   78.4× | `icu.StringSearch`     |   116 MB/s |
| `stringzilla.utf8_uncased_matches`, Italian    | 9.31 GB/s |    185× | `icu.StringSearch`     |  50.2 MB/s |
| `stringzilla.utf8_uncased_matches`, Japanese   |   15 GB/s |    195× | `icu.StringSearch`     |  76.9 MB/s |
| `stringzilla.utf8_uncased_matches`, Korean     |   29 GB/s |    609× | `icu.StringSearch`     |  47.6 MB/s |
| `stringzilla.utf8_uncased_matches`, Lithuanian |   13 GB/s |    274× | `icu.StringSearch`     |  47.4 MB/s |
| `stringzilla.utf8_uncased_matches`, Polish     | 9.28 GB/s |    194× | `icu.StringSearch`     |  47.9 MB/s |
| `stringzilla.utf8_uncased_matches`, Portuguese |   10 GB/s |    207× | `icu.StringSearch`     |  48.2 MB/s |
| `stringzilla.utf8_uncased_matches`, Russian    | 7.21 GB/s |   91.9× | `icu.StringSearch`     | 78.49 MB/s |
| `stringzilla.utf8_uncased_matches`, Spanish    |  6.3 GB/s |    130× | `icu.StringSearch`     |  48.6 MB/s |
| `stringzilla.utf8_uncased_matches`, Tamil      |   20 GB/s |    164× | `icu.StringSearch`     |   122 MB/s |
| `stringzilla.utf8_uncased_matches`, Turkish    |   11 GB/s |    219× | `icu.StringSearch`     |  50.2 MB/s |
| `stringzilla.utf8_uncased_matches`, Ukrainian  | 9.29 GB/s |    118× | `icu.StringSearch`     |    79 MB/s |
| `stringzilla.utf8_uncased_matches`, Vietnamese |  4.2 GB/s |   96.1× | `icu.StringSearch`     |  43.7 MB/s |
| `stringzilla.utf8_norm`, Arabic                |  510 MB/s |   0.94× | `icu.Normalizer2`      |   540 MB/s |
| `stringzilla.utf8_norm`, Armenian              |  500 MB/s |   0.93× | `icu.Normalizer2`      |   540 MB/s |
| `stringzilla.utf8_norm`, Bengali               |  300 MB/s |   0.77× | `icu.Normalizer2`      |   390 MB/s |
| `stringzilla.utf8_norm`, Chinese               |  580 MB/s |   0.64× | `icu.Normalizer2`      |   900 MB/s |
| `stringzilla.utf8_norm`, Czech                 |  2.5 GB/s |   2.60× | `icu.Normalizer2`      |   960 MB/s |
| `stringzilla.utf8_norm`, Dutch                 |  2.9 GB/s |   3.33× | `icu.Normalizer2`      |   870 MB/s |
| `stringzilla.utf8_norm`, English               |  2.6 GB/s |   2.89× | `icu.Normalizer2`      |   900 MB/s |
| `stringzilla.utf8_norm`, Farsi                 |  470 MB/s |   0.84× | `icu.Normalizer2`      |   560 MB/s |
| `stringzilla.utf8_norm`, French                |  2.1 GB/s |   2.36× | `icu.Normalizer2`      |   890 MB/s |
| `stringzilla.utf8_norm`, Georgian              |  540 MB/s |   0.68× | `icu.Normalizer2`      |   800 MB/s |
| `stringzilla.utf8_norm`, German                |  2.2 GB/s |   2.53× | `icu.Normalizer2`      |   870 MB/s |
| `stringzilla.utf8_norm`, Greek                 |  500 MB/s |   0.93× | `icu.Normalizer2`      |   540 MB/s |
| `stringzilla.utf8_norm`, Hebrew                |  480 MB/s |   0.92× | `icu.Normalizer2`      |   520 MB/s |
| `stringzilla.utf8_norm`, Hindi                 |  490 MB/s |   0.78× | `icu.Normalizer2`      |   630 MB/s |
| `stringzilla.utf8_norm`, Italian               |  2.6 GB/s |   2.99× | `icu.Normalizer2`      |   870 MB/s |
| `stringzilla.utf8_norm`, Japanese              |  550 MB/s |   0.60× | `icu.Normalizer2`      |   910 MB/s |
| `stringzilla.utf8_norm`, Korean                |  450 MB/s |   0.65× | `icu.Normalizer2`      |   690 MB/s |
| `stringzilla.utf8_norm`, Lithuanian            |  1.6 GB/s |   1.80× | `icu.Normalizer2`      |   890 MB/s |
| `stringzilla.utf8_norm`, Polish                |  2.1 GB/s |   2.33× | `icu.Normalizer2`      |   900 MB/s |
| `stringzilla.utf8_norm`, Portuguese            |  2.5 GB/s |   2.81× | `icu.Normalizer2`      |   890 MB/s |
| `stringzilla.utf8_norm`, Russian               |  1.9 GB/s |   3.58× | `icu.Normalizer2`      |   530 MB/s |
| `stringzilla.utf8_norm`, Spanish               |  2.5 GB/s |   2.81× | `icu.Normalizer2`      |   890 MB/s |
| `stringzilla.utf8_norm`, Tamil                 |  430 MB/s |   0.98× | `icu.Normalizer2`      |   440 MB/s |
| `stringzilla.utf8_norm`, Turkish               |  2.2 GB/s |   2.34× | `icu.Normalizer2`      |   940 MB/s |
| `stringzilla.utf8_norm`, Ukrainian             |  1.3 GB/s |   2.28× | `icu.Normalizer2`      |   570 MB/s |
| `stringzilla.utf8_norm`, Vietnamese            |  450 MB/s |   0.76× | `icu.Normalizer2`      |   590 MB/s |

> Measured October 15, 2026.

## AMD Zen5 Turin

### Rust

| StringZilla                                  |      Speed | Speedup | Alternative |  Speed |
| :------------------------------------------- | ---------: | ------: | :---------- | -----: |
| `stringzilla::utf8_uncased_fold`, Arabic     | 957.5 MB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Armenian   |   866 MB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Bengali    |  5.75 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Chinese    |  1.13 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Czech      |   789 MB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Dutch      |  4.41 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, English    |  7.01 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Farsi      |   818 MB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, French     |  1.71 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Georgian   |   183 MB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, German     |  2.41 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Greek      | 0.931 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Hebrew     | 0.941 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Italian    |  2.13 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Japanese   |  3.27 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Korean     |   821 MB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Lithuanian |   824 MB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Polish     |   896 MB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Portuguese |  2.22 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Russian    |  2.05 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Spanish    |  2.22 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Tamil      |  5.63 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Turkish    |   813 MB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Ukrainian  |  1.95 GB/s |       – | –           |      – |
| `stringzilla::utf8_uncased_fold`, Vietnamese |   336 MB/s |       – | –           |      – |

### Python

| StringZilla                                 |      Speed | Speedup | Alternative |  Speed |
| :------------------------------------------ | ---------: | ------: | :---------- | -----: |
| `stringzilla.utf8_uncased_fold`, Arabic     |  1.68 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Armenian   |   711 MB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Bengali    |  2.71 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Chinese    |   845 MB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Czech      |   656 MB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Dutch      |  2.77 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, English    |  2.92 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Farsi      |  1.32 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, French     |  1.28 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Georgian   |   465 MB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, German     |  1.69 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Greek      |   743 MB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Hebrew     |  1.73 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Italian    |  1.80 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Japanese   |  1.86 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Korean     |  2.61 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Lithuanian |   694 MB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Polish     |   750 MB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Portuguese |  1.67 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Russian    |  1.45 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Spanish    |  1.68 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Tamil      |  2.82 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Turkish    |   673 MB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Ukrainian  |  1.47 GB/s |       – | –           |      – |
| `stringzilla.utf8_uncased_fold`, Vietnamese |   278 MB/s |       – | –           |      – |

## Apple M5 Pro

### Rust

Also benchmarked: `std::to_lowercase`, `pcre2::no-jit`, `pcre2::jit-on-fly`, `pcre2::pre-jit`, `unicode-normalization::nfc`.

| StringZilla                                          |      Speed | Speedup | Alternative                |      Speed |
| :--------------------------------------------------- | ---------: | ------: | :------------------------- | ---------: |
| `stringzilla::utf8_uncased_fold`, Arabic             |  2.20 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Armenian           |   902 MB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Bengali            |  6.29 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Chinese            |   515 MB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Czech              |  1.30 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Dutch              | 14.64 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, English            | 14.88 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Farsi              |  2.47 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, French             |  1.33 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Georgian           |   774 MB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, German             |  2.19 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Greek              |  1.62 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Hebrew             |  2.65 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Hindi              |  6.83 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Italian            |  3.89 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Japanese           |  1.68 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Korean             |  1.73 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Polish             |  1.29 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Portuguese         |  1.69 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Russian            |  1.04 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Spanish            |  1.91 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Tamil              |  5.84 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Turkish            |  1.18 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Ukrainian          |  1.05 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_fold`, Vietnamese         |  1.34 GB/s |       – | –                          |          – |
| `stringzilla::utf8_uncased_search`, Arabic           | 10.44 GB/s |   37.0× | `memchr::Finder<icu-fold>` |   282 MB/s |
| `stringzilla::utf8_uncased_search`, Armenian         |  1.07 GB/s |   3.69× | `memchr::Finder<icu-fold>` |   290 MB/s |
| `stringzilla::utf8_uncased_search`, Bengali          | 22.68 GB/s |   51.3× | `memchr::Finder<icu-fold>` |   442 MB/s |
| `stringzilla::utf8_uncased_search`, Chinese          | 11.64 GB/s |   29.2× | `memchr::Finder<icu-fold>` |   398 MB/s |
| `stringzilla::utf8_uncased_search`, Czech            |  4.07 GB/s |   25.6× | `memchr::Finder<icu-fold>` |   159 MB/s |
| `stringzilla::utf8_uncased_search`, Dutch            |  5.31 GB/s |   32.6× | `memchr::Finder<icu-fold>` |   163 MB/s |
| `stringzilla::utf8_uncased_search`, English          |  6.43 GB/s |   42.9× | `memchr::Finder<icu-fold>` |   150 MB/s |
| `stringzilla::utf8_uncased_search`, Farsi            | 20.96 GB/s |   65.9× | `memchr::Finder<icu-fold>` |   318 MB/s |
| `stringzilla::utf8_uncased_search`, French           |  6.58 GB/s |   42.4× | `memchr::Finder<icu-fold>` |   155 MB/s |
| `stringzilla::utf8_uncased_search`, Georgian         |  3.32 GB/s |   11.6× | `memchr::Finder<icu-fold>` |   287 MB/s |
| `stringzilla::utf8_uncased_search`, German           |  7.00 GB/s |   45.2× | `memchr::Finder<icu-fold>` |   155 MB/s |
| `stringzilla::utf8_uncased_search`, Greek            |  1.61 GB/s |   8.26× | `memchr::Finder<icu-fold>` |   195 MB/s |
| `stringzilla::utf8_uncased_search`, Hebrew           | 15.77 GB/s |   59.1× | `memchr::Finder<icu-fold>` |   267 MB/s |
| `stringzilla::utf8_uncased_search`, Hindi            | 14.94 GB/s |   35.2× | `memchr::Finder<icu-fold>` |   425 MB/s |
| `stringzilla::utf8_uncased_search`, Italian          |  5.23 GB/s |   34.0× | `memchr::Finder<icu-fold>` |   154 MB/s |
| `stringzilla::utf8_uncased_search`, Japanese         | 10.27 GB/s |   25.5× | `memchr::Finder<icu-fold>` |   402 MB/s |
| `stringzilla::utf8_uncased_search`, Korean           | 28.14 GB/s |   91.4× | `memchr::Finder<icu-fold>` |   308 MB/s |
| `stringzilla::utf8_uncased_search`, Polish           |  4.76 GB/s |   31.3× | `memchr::Finder<icu-fold>` |   152 MB/s |
| `stringzilla::utf8_uncased_search`, Portuguese       |  4.33 GB/s |   27.4× | `memchr::Finder<icu-fold>` |   158 MB/s |
| `stringzilla::utf8_uncased_search`, Russian          |  3.53 GB/s |   16.2× | `memchr::Finder<icu-fold>` |   218 MB/s |
| `stringzilla::utf8_uncased_search`, Spanish          |  4.94 GB/s |   32.1× | `memchr::Finder<icu-fold>` |   154 MB/s |
| `stringzilla::utf8_uncased_search`, Tamil            | 18.47 GB/s |   48.5× | `memchr::Finder<icu-fold>` |   381 MB/s |
| `stringzilla::utf8_uncased_search`, Turkish          |  2.55 GB/s |   17.2× | `memchr::Finder<icu-fold>` |   148 MB/s |
| `stringzilla::utf8_uncased_search`, Ukrainian        |  2.47 GB/s |   10.6× | `memchr::Finder<icu-fold>` |   234 MB/s |
| `stringzilla::utf8_uncased_search`, Vietnamese       |  1.85 GB/s |   10.5× | `memchr::Finder<icu-fold>` |   176 MB/s |
| `stringzilla::utf8_norm`, Arabic                     |  1.09 GB/s |   2.26× | `icu::ComposingNormalizer` |   483 MB/s |
| `stringzilla::utf8_norm`, Armenian                   |  1.04 GB/s |   2.15× | `icu::ComposingNormalizer` |   484 MB/s |
| `stringzilla::utf8_norm`, Bengali                    |   404 MB/s |   1.13× | `icu::ComposingNormalizer` |   357 MB/s |
| `stringzilla::utf8_norm`, Chinese                    |  1.08 GB/s |   1.50× | `icu::ComposingNormalizer` |   722 MB/s |
| `stringzilla::utf8_norm`, Czech                      | 14.33 GB/s |   3.98× | `icu::ComposingNormalizer` |  3.60 GB/s |
| `stringzilla::utf8_norm`, Dutch                      | 35.89 GB/s |   9.62× | `icu::ComposingNormalizer` |  3.73 GB/s |
| `stringzilla::utf8_norm`, English                    | 20.48 GB/s |   5.75× | `icu::ComposingNormalizer` |  3.56 GB/s |
| `stringzilla::utf8_norm`, Farsi                      | 0.987 GB/s |   2.04× | `icu::ComposingNormalizer` |   484 MB/s |
| `stringzilla::utf8_norm`, French                     | 12.66 GB/s |   3.67× | `icu::ComposingNormalizer` |  3.45 GB/s |
| `stringzilla::utf8_norm`, Georgian                   | 0.978 GB/s |   1.50× | `icu::ComposingNormalizer` |   653 MB/s |
| `stringzilla::utf8_norm`, German                     | 15.45 GB/s |   4.40× | `icu::ComposingNormalizer` |  3.51 GB/s |
| `stringzilla::utf8_norm`, Greek                      |  1.19 GB/s |   2.34× | `icu::ComposingNormalizer` |   509 MB/s |
| `stringzilla::utf8_norm`, Hebrew                     |  1.02 GB/s |   2.10× | `icu::ComposingNormalizer` |   485 MB/s |
| `stringzilla::utf8_norm`, Hindi                      |   731 MB/s |   1.53× | `icu::ComposingNormalizer` |   479 MB/s |
| `stringzilla::utf8_norm`, Italian                    | 27.25 GB/s |   7.35× | `icu::ComposingNormalizer` |  3.71 GB/s |
| `stringzilla::utf8_norm`, Japanese                   | 0.950 GB/s |   1.36× | `icu::ComposingNormalizer` |   700 MB/s |
| `stringzilla::utf8_norm`, Korean                     |   824 MB/s |   1.45× | `icu::ComposingNormalizer` |   570 MB/s |
| `stringzilla::utf8_norm`, Polish                     | 11.00 GB/s |   3.21× | `icu::ComposingNormalizer` |  3.43 GB/s |
| `stringzilla::utf8_norm`, Portuguese                 | 18.71 GB/s |   5.04× | `icu::ComposingNormalizer` |  3.71 GB/s |
| `stringzilla::utf8_norm`, Russian                    | 10.78 GB/s |   22.3× | `icu::ComposingNormalizer` |   484 MB/s |
| `stringzilla::utf8_norm`, Spanish                    | 20.19 GB/s |   5.43× | `icu::ComposingNormalizer` |  3.72 GB/s |
| `stringzilla::utf8_norm`, Tamil                      |   526 MB/s |   1.50× | `icu::ComposingNormalizer` |   350 MB/s |
| `stringzilla::utf8_norm`, Turkish                    | 12.65 GB/s |   3.61× | `icu::ComposingNormalizer` |  3.50 GB/s |
| `stringzilla::utf8_norm`, Ukrainian                  |  5.82 GB/s |   11.3× | `icu::ComposingNormalizer` |   515 MB/s |
| `stringzilla::utf8_norm`, Vietnamese                 |   821 MB/s |   0.59× | `icu::ComposingNormalizer` |  1.40 GB/s |

> Measured July 29, 2026.

### Python

Also benchmarked: `str.casefold`, `regex`, `casefold.find`, `unicodedata.normalize`, `pyunormalize`.

| StringZilla                                         |      Speed | Speedup | Alternative               |      Speed |
| :-------------------------------------------------- | ---------: | ------: | :------------------------ | ---------: |
| `stringzilla.utf8_uncased_fold`, Arabic             |  2.17 GB/s |   5.00× | `icu.CaseMap.foldCase`    |   434 MB/s |
| `stringzilla.utf8_uncased_fold`, Armenian           |   871 MB/s |   2.04× | `icu.CaseMap.foldCase`    |   426 MB/s |
| `stringzilla.utf8_uncased_fold`, Bengali            |  6.21 GB/s |   4.70× | `icu.CaseMap.foldCase`    |  1.32 GB/s |
| `stringzilla.utf8_uncased_fold`, Chinese            |   573 MB/s |   0.86× | `icu.CaseMap.foldCase`    |   668 MB/s |
| `stringzilla.utf8_uncased_fold`, Czech              |  1.27 GB/s |   2.37× | `icu.CaseMap.foldCase`    |   535 MB/s |
| `stringzilla.utf8_uncased_fold`, Dutch              | 14.21 GB/s |   29.7× | `icu.CaseMap.foldCase`    |   478 MB/s |
| `stringzilla.utf8_uncased_fold`, English            | 14.85 GB/s |   34.9× | `icu.CaseMap.foldCase`    |   426 MB/s |
| `stringzilla.utf8_uncased_fold`, Farsi              |  2.52 GB/s |   4.96× | `icu.CaseMap.foldCase`    |   508 MB/s |
| `stringzilla.utf8_uncased_fold`, French             |  1.29 GB/s |   2.66× | `icu.CaseMap.foldCase`    |   485 MB/s |
| `stringzilla.utf8_uncased_fold`, Georgian           |   714 MB/s |   1.82× | `icu.CaseMap.foldCase`    |   392 MB/s |
| `stringzilla.utf8_uncased_fold`, German             |  2.10 GB/s |   5.80× | `icu.CaseMap.foldCase`    |   362 MB/s |
| `stringzilla.utf8_uncased_fold`, Greek              |  1.46 GB/s |   7.45× | `icu.CaseMap.foldCase`    |   196 MB/s |
| `stringzilla.utf8_uncased_fold`, Hebrew             |  2.65 GB/s |   6.78× | `icu.CaseMap.foldCase`    |   391 MB/s |
| `stringzilla.utf8_uncased_fold`, Hindi              |  6.74 GB/s |   5.76× | `icu.CaseMap.foldCase`    |  1.17 GB/s |
| `stringzilla.utf8_uncased_fold`, Italian            |  3.79 GB/s |   7.90× | `icu.CaseMap.foldCase`    |   480 MB/s |
| `stringzilla.utf8_uncased_fold`, Japanese           |  1.79 GB/s |   2.39× | `icu.CaseMap.foldCase`    |   749 MB/s |
| `stringzilla.utf8_uncased_fold`, Korean             |  1.63 GB/s |   1.55× | `icu.CaseMap.foldCase`    |  1.05 GB/s |
| `stringzilla.utf8_uncased_fold`, Polish             |  1.25 GB/s |   2.65× | `icu.CaseMap.foldCase`    |   472 MB/s |
| `stringzilla.utf8_uncased_fold`, Portuguese         |  1.62 GB/s |   3.49× | `icu.CaseMap.foldCase`    |   464 MB/s |
| `stringzilla.utf8_uncased_fold`, Russian            | 0.950 GB/s |   3.74× | `icu.CaseMap.foldCase`    |   254 MB/s |
| `stringzilla.utf8_uncased_fold`, Spanish            |  1.81 GB/s |   3.83× | `icu.CaseMap.foldCase`    |   472 MB/s |
| `stringzilla.utf8_uncased_fold`, Tamil              |  5.81 GB/s |   8.18× | `icu.CaseMap.foldCase`    |   710 MB/s |
| `stringzilla.utf8_uncased_fold`, Turkish            |  1.15 GB/s |   2.42× | `icu.CaseMap.foldCase`    |   476 MB/s |
| `stringzilla.utf8_uncased_fold`, Ukrainian          | 0.969 GB/s |   3.35× | `icu.CaseMap.foldCase`    |   289 MB/s |
| `stringzilla.utf8_uncased_fold`, Vietnamese         |  1.29 GB/s |   3.40× | `icu.CaseMap.foldCase`    |   379 MB/s |
| `stringzilla.utf8_uncased_matches`, Arabic          | 10.41 GB/s |   82.6× | `icu.StringSearch`        |   126 MB/s |
| `stringzilla.utf8_uncased_matches`, Armenian        |  1.01 GB/s |   5.94× | `icu.StringSearch`        |   170 MB/s |
| `stringzilla.utf8_uncased_matches`, Bengali         | 27.87 GB/s |    124× | `icu.StringSearch`        |   224 MB/s |
| `stringzilla.utf8_uncased_matches`, Chinese         | 13.88 GB/s |    104× | `icu.StringSearch`        |   134 MB/s |
| `stringzilla.utf8_uncased_matches`, Czech           |  2.70 GB/s |   27.7× | `icu.StringSearch`        |  97.3 MB/s |
| `stringzilla.utf8_uncased_matches`, Dutch           |  4.86 GB/s |   45.0× | `icu.StringSearch`        |   108 MB/s |
| `stringzilla.utf8_uncased_matches`, English         |  5.68 GB/s |   61.1× | `icu.StringSearch`        |    93 MB/s |
| `stringzilla.utf8_uncased_matches`, Farsi           | 13.76 GB/s |   98.3× | `icu.StringSearch`        |   140 MB/s |
| `stringzilla.utf8_uncased_matches`, French          |  4.00 GB/s |   40.0× | `icu.StringSearch`        |   100 MB/s |
| `stringzilla.utf8_uncased_matches`, Georgian        |  2.82 GB/s |   11.0× | `icu.StringSearch`        |   257 MB/s |
| `stringzilla.utf8_uncased_matches`, German          |  5.32 GB/s |   54.2× | `icu.StringSearch`        |  98.2 MB/s |
| `stringzilla.utf8_uncased_matches`, Greek           |  1.89 GB/s |   13.1× | `icu.StringSearch`        |   144 MB/s |
| `stringzilla.utf8_uncased_matches`, Hebrew          | 10.10 GB/s |   62.0× | `icu.StringSearch`        |   163 MB/s |
| `stringzilla.utf8_uncased_matches`, Hindi           | 12.59 GB/s |   48.8× | `icu.StringSearch`        |   258 MB/s |
| `stringzilla.utf8_uncased_matches`, Italian         |  6.04 GB/s |   58.1× | `icu.StringSearch`        |   104 MB/s |
| `stringzilla.utf8_uncased_matches`, Japanese        |  8.97 GB/s |   57.5× | `icu.StringSearch`        |   156 MB/s |
| `stringzilla.utf8_uncased_matches`, Korean          | 27.99 GB/s |    304× | `icu.StringSearch`        |    92 MB/s |
| `stringzilla.utf8_uncased_matches`, Polish          |  4.27 GB/s |   42.7× | `icu.StringSearch`        |   100 MB/s |
| `stringzilla.utf8_uncased_matches`, Portuguese      |  5.08 GB/s |   49.3× | `icu.StringSearch`        |   103 MB/s |
| `stringzilla.utf8_uncased_matches`, Russian         |  2.62 GB/s |   15.9× | `icu.StringSearch`        |   165 MB/s |
| `stringzilla.utf8_uncased_matches`, Spanish         |  4.57 GB/s |   45.7× | `icu.StringSearch`        |   100 MB/s |
| `stringzilla.utf8_uncased_matches`, Tamil           | 19.79 GB/s |   74.7× | `icu.StringSearch`        |   265 MB/s |
| `stringzilla.utf8_uncased_matches`, Turkish         |  2.78 GB/s |   26.2× | `icu.StringSearch`        |   106 MB/s |
| `stringzilla.utf8_uncased_matches`, Ukrainian       |  3.00 GB/s |   17.9× | `icu.StringSearch`        |   168 MB/s |
| `stringzilla.utf8_uncased_matches`, Vietnamese      |  1.81 GB/s |   20.1× | `icu.StringSearch`        |    90 MB/s |
| `stringzilla.utf8_norm`, Arabic                     |  1.13 GB/s |   1.21× | `icu.Normalizer2`         | 0.931 GB/s |
| `stringzilla.utf8_norm`, Armenian                   |  1.07 GB/s |   1.12× | `icu.Normalizer2`         | 0.959 GB/s |
| `stringzilla.utf8_norm`, Bengali                    |   387 MB/s |   0.49× | `icu.Normalizer2`         |   791 MB/s |
| `stringzilla.utf8_norm`, Chinese                    |  1.09 GB/s |   0.69× | `icu.Normalizer2`         |  1.57 GB/s |
| `stringzilla.utf8_norm`, Czech                      | 11.57 GB/s |   11.9× | `icu.Normalizer2`         | 0.969 GB/s |
| `stringzilla.utf8_norm`, Dutch                      | 21.23 GB/s |   23.6× | `icu.Normalizer2`         |   899 MB/s |
| `stringzilla.utf8_norm`, English                    | 14.92 GB/s |   16.7× | `icu.Normalizer2`         |   891 MB/s |
| `stringzilla.utf8_norm`, Farsi                      | 0.997 GB/s |   1.00× | `icu.Normalizer2`         | 0.997 GB/s |
| `stringzilla.utf8_norm`, French                     | 9.993 GB/s |   10.8× | `icu.Normalizer2`         |   924 MB/s |
| `stringzilla.utf8_norm`, Georgian                   | 0.987 GB/s |   0.68× | `icu.Normalizer2`         |  1.46 GB/s |
| `stringzilla.utf8_norm`, German                     | 11.90 GB/s |   13.2× | `icu.Normalizer2`         |   901 MB/s |
| `stringzilla.utf8_norm`, Greek                      |  1.18 GB/s |   1.25× | `icu.Normalizer2`         | 0.941 GB/s |
| `stringzilla.utf8_norm`, Hebrew                     |  1.03 GB/s |   1.11× | `icu.Normalizer2`         |   925 MB/s |
| `stringzilla.utf8_norm`, Hindi                      |   726 MB/s |   0.52× | `icu.Normalizer2`         |  1.40 GB/s |
| `stringzilla.utf8_norm`, Italian                    | 18.81 GB/s |   20.7× | `icu.Normalizer2`         |   909 MB/s |
| `stringzilla.utf8_norm`, Japanese                   | 0.969 GB/s |   0.60× | `icu.Normalizer2`         |  1.61 GB/s |
| `stringzilla.utf8_norm`, Korean                     |   758 MB/s |   0.58× | `icu.Normalizer2`         |  1.31 GB/s |
| `stringzilla.utf8_norm`, Polish                     | 9.360 GB/s |   10.0× | `icu.Normalizer2`         |   934 MB/s |
| `stringzilla.utf8_norm`, Portuguese                 | 13.26 GB/s |   14.4× | `icu.Normalizer2`         |   920 MB/s |
| `stringzilla.utf8_norm`, Russian                    |  9.15 GB/s |   9.54× | `icu.Normalizer2`         | 0.959 GB/s |
| `stringzilla.utf8_norm`, Spanish                    | 13.90 GB/s |   15.0× | `icu.Normalizer2`         |   926 MB/s |
| `stringzilla.utf8_norm`, Tamil                      |   508 MB/s |   0.55× | `icu.Normalizer2`         |   923 MB/s |
| `stringzilla.utf8_norm`, Turkish                    | 9.937 GB/s |   10.5× | `icu.Normalizer2`         | 0.950 GB/s |
| `stringzilla.utf8_norm`, Ukrainian                  |  5.27 GB/s |   5.29× | `icu.Normalizer2`         | 0.997 GB/s |
| `stringzilla.utf8_norm`, Vietnamese                 |   779 MB/s |   0.89× | `icu.Normalizer2`         |   874 MB/s |

> Measured July 29, 2026.

## Reproducing

To rerun the benchmarks for all languages:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bench bench_normalization --features bench_normalization
bin=$(find target/release/deps -name 'bench_normalization-*' -executable -type f | head -1)

for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold" "$bin"
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold/" uv run --group normalization normalization/bench.py
done
```

To rerun the benchmarks for all languages:

```bash
for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-insensitive-find" "$bin"
done
```

---

See [README.md](../README.md) for dataset information and replication instructions.
