# UTF-8 Tokenization & Iteration Benchmarks

Benchmarks for UTF-8 segmentation and codepoint iteration — whitespace, newline, and TR29 word splitting, UTF-8 character counting and decoding, and locating the Nth codepoint — across different languages and hardware platforms.
Different scripts stress UTF-8 processing in different ways.
Sections run from the coarsest splits to the finest units.
Tables below cover:

- __Korean__: 3-byte Hangul syllables with single-byte whitespace between words - representative for tokenization workloads
- __Chinese__: 3-byte CJK characters with rare whitespace - tests raw byte throughput
- __Arabic__: 2-byte Arabic script with regular punctuation - good for newline splitting benchmarks
- __French__: Mixed 1-2 byte Latin with high diacritic density
- __English__: Mostly 1-byte ASCII baseline

Each row is one segmentation or indexing function over one Leipzig corpus language, single-threaded.
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.
Languages without a measurement on a chip have no rows there.

The `utf8_split_newlines` rows split on the forced newline delimiters.
There are 8 characters classified as forced newline delimiters in Unicode.
Five overlap with the ASCII whitespace set: line feed _U+000A_, vertical tab _U+000B_, form feed _U+000C_, carriage return _U+000D_, and the carriage return plus line feed combination _U+000D U+000A_ treated as a single break.
The remaining three are the Next Line control _U+0085_, the line separator _U+2028_, and the paragraph separator _U+2029_.
Unlike the optional break opportunities of UAX#14, these always end a line.

The `utf8_linebreaks` rows mark every UAX#14 position a renderer may wrap.
Mandatory breaks (classes BK, CR, LF, NL — the same forced breaks as the newline rows) are separated from the allowed soft-wrap points.
Around 40 rules (LB1–LB31) govern soft wraps after spaces and hyphens, around CJK ideographs, and East-Asian-Width cases such as LB19.
It emits far more opportunities than the hard newline set.

The `utf8_sentences` rows split TR29 sentences.
UAX#29 sentence boundaries use 12 Sentence_Break classes (STerm, ATerm, Close, SContinue, Sep, CR, LF, Sp, Lower, Upper, OLetter, Numeric, Extend, Format).
Sentences end at `.!?` and the Unicode terminators.
Abbreviations (U.S.A.), decimals (3.14), and a following continuation do not break.

The `utf8_split_whitespaces` rows split on Unicode whitespace.
There are 25 characters classified as whitespace in Unicode.
Six are ASCII separators: horizontal tab _U+0009_, line feed _U+000A_, vertical tab _U+000B_, form feed _U+000C_, carriage return _U+000D_, and space _U+0020_.
Two more are the Next Line control _U+0085_ and the no-break space _U+00A0_.
Then the Ogham space mark _U+1680_, followed by the eleven General Punctuation separators: en quad _U+2000_, em quad _U+2001_, en space _U+2002_, em space _U+2003_, three-per-em space _U+2004_, four-per-em space _U+2005_, six-per-em space _U+2006_, figure space _U+2007_, punctuation space _U+2008_, thin space _U+2009_, and hair space _U+200A_.
The same block adds the line and paragraph separators _U+2028_ and _U+2029_, the narrow no-break space _U+202F_, the medium mathematical space _U+205F_, and the CJK ideographic space _U+3000_.
The zero-width space _U+200B_, despite its name, is not among them — its `White_Space` property is `No`.

The `utf8_wordbreaks` rows find TR29 word boundaries.
UAX#29 word boundaries use 18 Word_Break classes (ALetter, Hebrew_Letter, Numeric, Katakana, ExtendNumLet, MidLetter, MidNum, MidNumLet, Single_Quote, Double_Quote, WSegSpace, Format, Extend, ZWJ, Regional_Indicator, CR, LF, Newline).
Letter and number runs stay together while punctuation and spaces split.
Space-less scripts (Chinese, Japanese, Thai) need a dictionary, hence ICU's `WordSegmenter::new_dictionary`.
`stringzilla::utf8_wordbreaks` emits every boundary segment (tiling the input contiguously), so it is compared only against the baselines that also tile — `split_word_bounds` and ICU — and `std::split_whitespace` and the word-like-only `unicode_words` filter are not alternatives.

The `utf8_graphemes` rows find grapheme clusters.
A grapheme cluster is one user-perceived character: a base plus combining marks, a Hangul L/V/T syllable, a Regional_Indicator pair (flag), or an Extended_Pictographic base joined by ZWJ _U+200D_.
UAX#29 assigns 13 Grapheme_Cluster_Break classes (CR, LF, Control, Extend, ZWJ, Regional_Indicator, Prepend, SpacingMark, L, V, T, LV, LVT).
It breaks between every pair except where rules GB3–GB11 forbid it.

The `find_nth_utf8` rows return the byte offset of the Nth codepoint by scanning lead bytes, targeting the last codepoint so it forces a full-buffer scan.
UTF-8 encodes each codepoint in 1–4 bytes (lead bytes `0xxxxxxx`, `110xxxxx`, `1110xxxx`, `11110xxx`).
CPython stores `str` in a fixed-width representation (PEP 393), so `s[n]` and `len(s)` are O(1) — there is no comparable Python byte scan, so these rows are Rust-only.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `icu::LineSegmenter`, `unicode-segmentation::split_sentence_bounds`, `icu::WhiteSpace`, `std::split_whitespace`, `unicode-segmentation::graphemes`.

| StringZilla                                    |      Speed | Speedup | Alternative                               |      Speed |
| :--------------------------------------------- | ---------: | ------: | :---------------------------------------- | ---------: |
| `stringzilla::utf8_split_newlines`, English    |  5.44 GB/s |   5.44× | `std::split<is_unicode_newline>`          |  1000 MB/s |
| `stringzilla::utf8_split_newlines`, Chinese    |   3.5 GB/s |   4.73× | `std::split<is_unicode_newline>`          |   740 MB/s |
| `stringzilla::utf8_split_newlines`, Arabic     |   4.5 GB/s |   9.78× | `std::split<is_unicode_newline>`          |   460 MB/s |
| `stringzilla::utf8_split_newlines`, French     |   3.9 GB/s |   7.96× | `std::split<is_unicode_newline>`          |   490 MB/s |
| `stringzilla::utf8_split_newlines`, Korean     |   5.3 GB/s |   5.96× | `std::split<is_unicode_newline>`          |   890 MB/s |
| `stringzilla::utf8_linebreaks`, English        |   316 MB/s |   0.88× | `unicode-linebreak::linebreaks`           |   358 MB/s |
| `stringzilla::utf8_linebreaks`, Chinese        |   150 MB/s |   0.31× | `unicode-linebreak::linebreaks`           |   480 MB/s |
| `stringzilla::utf8_linebreaks`, Arabic         |   200 MB/s |   0.67× | `unicode-linebreak::linebreaks`           |   300 MB/s |
| `stringzilla::utf8_linebreaks`, French         |   180 MB/s |   0.69× | `unicode-linebreak::linebreaks`           |   262 MB/s |
| `stringzilla::utf8_linebreaks`, Korean         |   260 MB/s |   0.41× | `unicode-linebreak::linebreaks`           |   640 MB/s |
| `stringzilla::utf8_sentences`, English         |   685 MB/s |   2.62× | `icu::SentenceSegmenter`                  |   261 MB/s |
| `stringzilla::utf8_sentences`, Chinese         |   440 MB/s |   2.59× | `icu::SentenceSegmenter`                  |   170 MB/s |
| `stringzilla::utf8_sentences`, Arabic          |   484 MB/s |   2.69× | `icu::SentenceSegmenter`                  |   180 MB/s |
| `stringzilla::utf8_sentences`, French          |   460 MB/s |   3.54× | `icu::SentenceSegmenter`                  |   130 MB/s |
| `stringzilla::utf8_sentences`, Korean          |   660 MB/s |   2.54× | `icu::SentenceSegmenter`                  |   260 MB/s |
| `stringzilla::utf8_split_whitespaces`, English |   1.3 GB/s |   3.27× | `std::split<is_whitespace>`               |   398 MB/s |
| `stringzilla::utf8_split_whitespaces`, Chinese |   2.5 GB/s |   4.24× | `std::split<is_whitespace>`               |   590 MB/s |
| `stringzilla::utf8_split_whitespaces`, Arabic  |   1.2 GB/s |   2.84× | `std::split<is_whitespace>`               |   422 MB/s |
| `stringzilla::utf8_split_whitespaces`, French  |   850 MB/s |   2.66× | `std::split<is_whitespace>`               |   320 MB/s |
| `stringzilla::utf8_split_whitespaces`, Korean  |   1.9 GB/s |   2.41× | `std::split<is_whitespace>`               |   790 MB/s |
| `stringzilla::utf8_wordbreaks`, English        |   520 MB/s |   3.91× | `icu::WordSegmenter`                      |   133 MB/s |
| `stringzilla::utf8_wordbreaks`, Chinese        |   241 MB/s |   2.68× | `unicode-segmentation::split_word_bounds` |    90 MB/s |
| `stringzilla::utf8_wordbreaks`, Arabic         |   300 MB/s |   1.92× | `icu::WordSegmenter`                      |   156 MB/s |
| `stringzilla::utf8_wordbreaks`, French         |   245 MB/s |   2.95× | `icu::WordSegmenter`                      |    83 MB/s |
| `stringzilla::utf8_wordbreaks`, Korean         |   450 MB/s |   1.96× | `icu::WordSegmenter`                      |   230 MB/s |
| `stringzilla::utf8_graphemes`, English         |   376 MB/s |   2.39× | `icu::GraphemeClusterSegmenter`           |   157 MB/s |
| `stringzilla::utf8_graphemes`, Chinese         |   270 MB/s |   1.93× | `icu::GraphemeClusterSegmenter`           |   140 MB/s |
| `stringzilla::utf8_graphemes`, Arabic          |   273 MB/s |   2.10× | `icu::GraphemeClusterSegmenter`           |   130 MB/s |
| `stringzilla::utf8_graphemes`, French          |   190 MB/s |   2.09× | `icu::GraphemeClusterSegmenter`           |    91 MB/s |
| `stringzilla::utf8_graphemes`, Korean          |   343 MB/s |   1.72× | `icu::GraphemeClusterSegmenter`           |   200 MB/s |
| `stringzilla::find_nth_utf8`, English          |  25.8 GB/s |   14.3× | `std::char_indices.nth`                   |   1.8 GB/s |
| `stringzilla::find_nth_utf8`, Chinese          |    22 GB/s |   21.6× | `std::char_indices.nth`                   |  1.02 GB/s |
| `stringzilla::find_nth_utf8`, Arabic           |    25 GB/s |   40.3× | `std::char_indices.nth`                   |   620 MB/s |
| `stringzilla::find_nth_utf8`, French           |    24 GB/s |   24.0× | `std::char_indices.nth`                   |     1 GB/s |
| `stringzilla::find_nth_utf8`, Korean           |    26 GB/s |   28.0× | `std::char_indices.nth`                   |   930 MB/s |

> Measured October 15, 2026.

### Python

Also benchmarked: `grapheme.graphemes`.

| StringZilla                                    |      Speed | Speedup | Alternative         |      Speed |
| :--------------------------------------------- | ---------: | ------: | :------------------ | ---------: |
| `stringzilla.utf8_split_newlines`, English     |   1.7 GB/s |       – | –                   |          – |
| `stringzilla.utf8_split_newlines`, Chinese     |   1.4 GB/s |       – | –                   |          – |
| `stringzilla.utf8_split_newlines`, Arabic      |   2.0 GB/s |       – | –                   |          – |
| `stringzilla.utf8_split_newlines`, French      |   2.1 GB/s |       – | –                   |          – |
| `stringzilla.utf8_split_newlines`, Korean      |   2.3 GB/s |       – | –                   |          – |
| `stringzilla.utf8_linebreaks`, English         |  73.4 MB/s |   1.51× | `icu.BreakIterator` |  48.7 MB/s |
| `stringzilla.utf8_linebreaks`, Chinese         |  48.1 MB/s |   1.05× | `icu.BreakIterator` |  45.6 MB/s |
| `stringzilla.utf8_linebreaks`, Arabic          |   100 MB/s |   1.16× | `icu.BreakIterator` |    86 MB/s |
| `stringzilla.utf8_linebreaks`, French          |   120 MB/s |   1.54× | `icu.BreakIterator` |  77.9 MB/s |
| `stringzilla.utf8_linebreaks`, Korean          |  78.0 MB/s |   1.09× | `icu.BreakIterator` |  71.5 MB/s |
| `stringzilla.utf8_sentences`, English          |   400 MB/s |   2.86× | `icu.BreakIterator` |   140 MB/s |
| `stringzilla.utf8_sentences`, Chinese          |   349 MB/s |   1.20× | `icu.BreakIterator` |   290 MB/s |
| `stringzilla.utf8_sentences`, Arabic           |   400 MB/s |   1.60× | `icu.BreakIterator` |   250 MB/s |
| `stringzilla.utf8_sentences`, French           |   560 MB/s |   2.33× | `icu.BreakIterator` |   240 MB/s |
| `stringzilla.utf8_sentences`, Korean           |   570 MB/s |   0.93× | `icu.BreakIterator` |   610 MB/s |
| `stringzilla.utf8_split_whitespaces`, English  |   115 MB/s |   5.75× | `regex.finditer`    |   ~20 MB/s |
| `stringzilla.utf8_split_whitespaces`, Chinese  |   870 MB/s |   6.69× | `regex.finditer`    |   130 MB/s |
| `stringzilla.utf8_split_whitespaces`, Arabic   |   190 MB/s |   6.55× | `regex.finditer`    |    29 MB/s |
| `stringzilla.utf8_split_whitespaces`, French   |   180 MB/s |   5.77× | `regex.finditer`    |  31.2 MB/s |
| `stringzilla.utf8_split_whitespaces`, Korean   |   280 MB/s |   5.50× | `regex.finditer`    |  50.9 MB/s |
| `stringzilla.utf8_wordbreaks`, English         |    77 MB/s |   2.57× | `icu.BreakIterator` |    30 MB/s |
| `stringzilla.utf8_wordbreaks`, Chinese         |    49 MB/s |   5.44× | `icu.BreakIterator` |    ~9 MB/s |
| `stringzilla.utf8_wordbreaks`, Arabic          |  73.7 MB/s |   1.42× | `icu.BreakIterator` |    52 MB/s |
| `stringzilla.utf8_wordbreaks`, French          |    78 MB/s |   1.63× | `icu.BreakIterator` |  47.8 MB/s |
| `stringzilla.utf8_wordbreaks`, Korean          |   113 MB/s |   2.63× | `icu.BreakIterator` |    43 MB/s |
| `stringzilla.utf8_graphemes`, English          |  20.0 MB/s |   2.00× | `icu.BreakIterator` |   ~10 MB/s |
| `stringzilla.utf8_graphemes`, Chinese          |    44 MB/s |   1.21× | `icu.BreakIterator` |  36.5 MB/s |
| `stringzilla.utf8_graphemes`, Arabic           |  33.3 MB/s |   1.33× | `icu.BreakIterator` |  25.0 MB/s |
| `stringzilla.utf8_graphemes`, French           |    30 MB/s |   1.15× | `icu.BreakIterator` |    26 MB/s |
| `stringzilla.utf8_graphemes`, Korean           |    60 MB/s |   0.97× | `icu.BreakIterator` |  61.7 MB/s |

> Measured October 15, 2026.

## Apple M5 Pro

### Rust

Also benchmarked: `icu::LineSegmenter`, `unicode-segmentation::split_sentence_bounds`, `icu::WhiteSpace`, `std::split_whitespace`, `unicode-segmentation::graphemes`.

| StringZilla                                    |      Speed | Speedup | Alternative                               |      Speed |
| :--------------------------------------------- | ---------: | ------: | :---------------------------------------- | ---------: |
| `stringzilla::utf8_split_newlines`, English    |  2.68 GB/s |   1.56× | `std::split<is_unicode_newline>`          |  1.72 GB/s |
| `stringzilla::utf8_split_newlines`, Chinese    |  2.30 GB/s |   1.12× | `std::split<is_unicode_newline>`          |  2.06 GB/s |
| `stringzilla::utf8_split_newlines`, Arabic     |  3.08 GB/s |   2.88× | `std::split<is_unicode_newline>`          |  1.07 GB/s |
| `stringzilla::utf8_split_newlines`, French     |  2.51 GB/s |   1.95× | `std::split<is_unicode_newline>`          |  1.29 GB/s |
| `stringzilla::utf8_split_newlines`, Korean     |  2.64 GB/s |   1.78× | `std::split<is_unicode_newline>`          |  1.48 GB/s |
| `stringzilla::utf8_linebreaks`, English        |  0.23 GB/s |   1.15× | `unicode-linebreak::linebreaks`           |  0.20 GB/s |
| `stringzilla::utf8_linebreaks`, Chinese        |  0.20 GB/s |   0.38× | `unicode-linebreak::linebreaks`           |  0.53 GB/s |
| `stringzilla::utf8_linebreaks`, Arabic         |  0.23 GB/s |   0.42× | `unicode-linebreak::linebreaks`           |  0.55 GB/s |
| `stringzilla::utf8_linebreaks`, French         |  0.22 GB/s |   1.10× | `unicode-linebreak::linebreaks`           |  0.20 GB/s |
| `stringzilla::utf8_linebreaks`, Korean         |  0.20 GB/s |   0.34× | `unicode-linebreak::linebreaks`           |  0.59 GB/s |
| `stringzilla::utf8_sentences`, English         |  0.41 GB/s |   1.08× | `icu::SentenceSegmenter`                  |  0.38 GB/s |
| `stringzilla::utf8_sentences`, Chinese         |  0.51 GB/s |   1.00× | `icu::SentenceSegmenter`                  |  0.51 GB/s |
| `stringzilla::utf8_sentences`, Arabic          |  0.47 GB/s |   1.07× | `icu::SentenceSegmenter`                  |  0.44 GB/s |
| `stringzilla::utf8_sentences`, French          |  0.38 GB/s |   1.06× | `icu::SentenceSegmenter`                  |  0.36 GB/s |
| `stringzilla::utf8_sentences`, Korean          |  0.52 GB/s |   1.27× | `icu::SentenceSegmenter`                  |  0.41 GB/s |
| `stringzilla::utf8_split_whitespaces`, English |  0.55 GB/s |   0.83× | `std::split<is_whitespace>`               |  0.66 GB/s |
| `stringzilla::utf8_split_whitespaces`, Chinese |  1.70 GB/s |   1.09× | `std::split<is_whitespace>`               |  1.56 GB/s |
| `stringzilla::utf8_split_whitespaces`, Arabic  |  0.74 GB/s |   0.76× | `std::split<is_whitespace>`               | 0.969 GB/s |
| `stringzilla::utf8_split_whitespaces`, French  |  0.54 GB/s |   0.86× | `std::split<is_whitespace>`               |  0.63 GB/s |
| `stringzilla::utf8_split_whitespaces`, Korean  |  0.75 GB/s |   0.56× | `std::split<is_whitespace>`               |  1.34 GB/s |
| `stringzilla::utf8_wordbreaks`, English        |  0.43 GB/s |   2.15× | `icu::WordSegmenter`                      |  0.20 GB/s |
| `stringzilla::utf8_wordbreaks`, Chinese        |  0.32 GB/s |   1.39× | `unicode-segmentation::split_word_bounds` |  0.23 GB/s |
| `stringzilla::utf8_wordbreaks`, Arabic         |  0.33 GB/s |   1.06× | `icu::WordSegmenter`                      |  0.31 GB/s |
| `stringzilla::utf8_wordbreaks`, French         |  0.33 GB/s |   1.65× | `icu::WordSegmenter`                      |  0.20 GB/s |
| `stringzilla::utf8_wordbreaks`, Korean         |  0.33 GB/s |   0.85× | `icu::WordSegmenter`                      |  0.39 GB/s |
| `stringzilla::utf8_graphemes`, English         |  0.20 GB/s |   0.74× | `icu::GraphemeClusterSegmenter`           |  0.27 GB/s |
| `stringzilla::utf8_graphemes`, Chinese         |  0.38 GB/s |   0.88× | `icu::GraphemeClusterSegmenter`           |  0.43 GB/s |
| `stringzilla::utf8_graphemes`, Arabic          |  0.27 GB/s |   0.79× | `icu::GraphemeClusterSegmenter`           |  0.34 GB/s |
| `stringzilla::utf8_graphemes`, French          |  0.20 GB/s |   0.74× | `icu::GraphemeClusterSegmenter`           |  0.27 GB/s |
| `stringzilla::utf8_graphemes`, Korean          |  0.27 GB/s |   0.79× | `icu::GraphemeClusterSegmenter`           |  0.34 GB/s |
| `stringzilla::find_nth_utf8`, English          | 31.80 GB/s |   12.8× | `std::char_indices.nth`                   |  2.49 GB/s |
| `stringzilla::find_nth_utf8`, Chinese          | 31.92 GB/s |   30.7× | `std::char_indices.nth`                   |  1.04 GB/s |
| `stringzilla::find_nth_utf8`, Arabic           | 31.93 GB/s |   52.3× | `std::char_indices.nth`                   |  0.61 GB/s |
| `stringzilla::find_nth_utf8`, French           | 31.91 GB/s |   19.1× | `std::char_indices.nth`                   |  1.67 GB/s |
| `stringzilla::find_nth_utf8`, Korean           | 31.92 GB/s |   35.9× | `std::char_indices.nth`                   |  0.89 GB/s |

> Measured July 29, 2026.

### Python

Also benchmarked: `grapheme.graphemes`.

| StringZilla                                     |      Speed | Speedup | Alternative         |      Speed |
| :---------------------------------------------- | ---------: | ------: | :------------------ | ---------: |
| `stringzilla.utf8_split_newlines`, English      |  2.25 GB/s |       – | –                   |          – |
| `stringzilla.utf8_split_newlines`, Chinese      |  1.88 GB/s |       – | –                   |          – |
| `stringzilla.utf8_split_newlines`, Arabic       |  2.67 GB/s |       – | –                   |          – |
| `stringzilla.utf8_split_newlines`, French       |  2.09 GB/s |       – | –                   |          – |
| `stringzilla.utf8_split_newlines`, Korean       |  2.20 GB/s |       – | –                   |          – |
| `stringzilla.utf8_linebreaks`, English          |  0.15 GB/s |   1.25× | `icu.BreakIterator` |  0.12 GB/s |
| `stringzilla.utf8_linebreaks`, Chinese          |  0.10 GB/s |   0.83× | `icu.BreakIterator` |  0.12 GB/s |
| `stringzilla.utf8_linebreaks`, Arabic           |  0.18 GB/s |   0.90× | `icu.BreakIterator` |  0.20 GB/s |
| `stringzilla.utf8_linebreaks`, French           |  0.15 GB/s |   1.25× | `icu.BreakIterator` |  0.12 GB/s |
| `stringzilla.utf8_linebreaks`, Korean           |  0.10 GB/s |   0.91× | `icu.BreakIterator` |  0.11 GB/s |
| `stringzilla.utf8_sentences`, English           |  0.39 GB/s |   1.03× | `icu.BreakIterator` |  0.38 GB/s |
| `stringzilla.utf8_sentences`, Chinese           |  0.47 GB/s |   0.69× | `icu.BreakIterator` |  0.68 GB/s |
| `stringzilla.utf8_sentences`, Arabic            |  0.45 GB/s |   0.70× | `icu.BreakIterator` |  0.64 GB/s |
| `stringzilla.utf8_sentences`, French            |  0.36 GB/s |   0.95× | `icu.BreakIterator` |  0.38 GB/s |
| `stringzilla.utf8_sentences`, Korean            |  0.48 GB/s |   0.61× | `icu.BreakIterator` |  0.79 GB/s |
| `stringzilla.utf8_split_whitespaces`, English   |  0.25 GB/s |   4.17× | `regex.finditer`    |  0.06 GB/s |
| `stringzilla.utf8_split_whitespaces`, Chinese   |  1.24 GB/s |   3.10× | `regex.finditer`    |  0.40 GB/s |
| `stringzilla.utf8_split_whitespaces`, Arabic    |  0.38 GB/s |   4.09× | `regex.finditer`    | 0.093 GB/s |
| `stringzilla.utf8_split_whitespaces`, French    |  0.25 GB/s |   4.17× | `regex.finditer`    |  0.06 GB/s |
| `stringzilla.utf8_split_whitespaces`, Korean    |  0.36 GB/s |   3.87× | `regex.finditer`    | 0.093 GB/s |
| `stringzilla.utf8_wordbreaks`, English          |  0.12 GB/s |   1.71× | `icu.BreakIterator` |  0.07 GB/s |
| `stringzilla.utf8_wordbreaks`, Chinese          |  0.11 GB/s |   3.67× | `icu.BreakIterator` |  0.03 GB/s |
| `stringzilla.utf8_wordbreaks`, Arabic           |  0.16 GB/s |   1.23× | `icu.BreakIterator` |  0.13 GB/s |
| `stringzilla.utf8_wordbreaks`, French           |  0.12 GB/s |   1.50× | `icu.BreakIterator` |  0.08 GB/s |
| `stringzilla.utf8_wordbreaks`, Korean           |  0.15 GB/s |   2.14× | `icu.BreakIterator` |  0.07 GB/s |
| `stringzilla.utf8_graphemes`, English           |  0.05 GB/s |   1.25× | `icu.BreakIterator` |  0.04 GB/s |
| `stringzilla.utf8_graphemes`, Chinese           |  0.11 GB/s |   1.10× | `icu.BreakIterator` |  0.10 GB/s |
| `stringzilla.utf8_graphemes`, Arabic            |  0.07 GB/s |   1.00× | `icu.BreakIterator` |  0.07 GB/s |
| `stringzilla.utf8_graphemes`, French            |  0.05 GB/s |   1.00× | `icu.BreakIterator` |  0.05 GB/s |
| `stringzilla.utf8_graphemes`, Korean            | 0.093 GB/s |   0.93× | `icu.BreakIterator` |  0.10 GB/s |

> Measured July 29, 2026.

## Reproducing

To rerun the benchmarks for all languages:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bench bench_tokenization --features bench_tokenization
bin=$(find target/release/deps -name 'bench_tokenization-*' -executable -type f | head -1)

for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="tokenize" "$bin"
done
```

---

See [README.md](../README.md) for dataset information and replication instructions.
