# UTF-8 Processing Benchmarks

Benchmarks for UTF-8 text processing, including whitespace and newline splitting across different languages and hardware platforms.

## Overview

Different scripts stress UTF-8 processing in different ways:

- __Korean__: 3-byte Hangul syllables with single-byte whitespace between words - representative for tokenization workloads
- __Chinese__: 3-byte CJK characters with rare whitespace - tests raw byte throughput
- __Arabic__: 2-byte Arabic script with regular punctuation - good for newline splitting benchmarks
- __French__: Mixed 1-2 byte Latin with high diacritic density
- __English__: Mostly 1-byte ASCII baseline

## AMD Zen5 Turin

On AMD Zen5 Turin CPUs on different datasets, StringZilla provides the following throughput for splitting around whitespace and newline characters on 5 vastly different languages.

| Library                                   |     English |     Chinese |      Arabic |      French |      Korean |
| ----------------------------------------- | ----------: | ----------: | ----------: | ----------: | ----------: |
| Split around 25 whitespace characters:    |             |             |             |             |             |
| `stringzilla::utf8_whitespace_splits`     |  0.82 GiB/s |  2.40 GiB/s |  2.40 GiB/s |  0.92 GiB/s |  1.88 GiB/s |
| `stdlib::split(char::is_whitespace)`      |  0.77 GiB/s |  1.87 GiB/s |  1.04 GiB/s |  0.72 GiB/s |  0.98 GiB/s |
| `icu::WhiteSpace`                         |  0.11 GiB/s |  0.16 GiB/s |  0.15 GiB/s |  0.12 GiB/s |  0.15 GiB/s |
|                                           |             |             |             |             |             |
| Split around 8 newline combinations:      |             |             |             |             |             |
| `stringzilla::utf8_newline_splits`        | 15.45 GiB/s | 16.65 GiB/s | 18.34 GiB/s | 14.52 GiB/s | 16.71 GiB/s |
| `stdlib::split(char::is_unicode_newline)` |  1.90 GiB/s |  1.93 GiB/s |  1.82 GiB/s |  1.78 GiB/s |  1.81 GiB/s |

## Apple M2 Pro

| Library                                   |    English |    Chinese |     Arabic |     French |     Korean |
| ----------------------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Split around 25 whitespace characters:    |            |            |            |            |            |
| `stringzilla::utf8_whitespace_splits`     | 0.57 GiB/s | 2.45 GiB/s | 1.18 GiB/s | 0.61 GiB/s | 0.92 GiB/s |
| `stdlib::split(char::is_whitespace)`      | 0.59 GiB/s | 1.16 GiB/s | 0.99 GiB/s | 0.63 GiB/s | 0.89 GiB/s |
| `icu::WhiteSpace`                         | 0.10 GiB/s | 0.16 GiB/s | 0.14 GiB/s | 0.11 GiB/s | 0.14 GiB/s |
|                                           |            |            |            |            |            |
| Split around 8 newline combinations:      |            |            |            |            |            |
| `stringzilla::utf8_newline_splits`        | 5.69 GiB/s | 6.24 GiB/s | 6.58 GiB/s | 6.70 GiB/s | 6.29 GiB/s |
| `stdlib::split(char::is_unicode_newline)` | 1.12 GiB/s | 1.11 GiB/s | 1.11 GiB/s | 1.11 GiB/s | 1.13 GiB/s |

---

See [README.md](README.md) for dataset information and replication instructions.
