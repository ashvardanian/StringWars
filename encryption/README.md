# Encryption and Decryption Benchmarks

Benchmarks for encryption and decryption operations across Rust and Python implementations.

## Overview

These benchmarks compare ChaCha20-Poly1305 and AES-256-GCM AEAD throughput across different libraries.
Rust covers `ring`, `openssl`, `libsodium`, and `stringzilla`; Python covers `cryptography` (OpenSSL backend), `pynacl` (libsodium), `pycryptodome`, and `stringzilla`.
The `cryptography` and `ring`/`openssl` paths reach the same OpenSSL/AES-NI kernels, while `pycryptodome` pays a large per-message Python-object overhead that dominates on short lines.

Each Rust row seals and opens into a buffer sized before the timed loop, so the rows price the cipher rather than the allocator.
OpenSSL is the exception: it builds one cipher context per message, which the crate gives no way to re-key, and at 100-byte lines that setup is most of the call — roughly 750 ns of 1.62 µs.

StringZilla also contributes an AES-256-CTR row.
Counter mode is unauthenticated, so it is not comparable to the AEAD rows on security — it is the floor, the same AES round function without a tag to accumulate, and the gap to `stringzilla::aes256` is what GHASH costs.

## Encryption

### Intel Xeon4 Sapphire Rapids

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |        0.20 GB/s |          0.75 GB/s |
| `libsodium::xchacha20`          |        0.17 GB/s |          0.69 GB/s |
| `ring::chacha20`                |        0.43 GB/s |          1.29 GB/s |
| `ring::aes256`                  |    __0.75 GB/s__ |          3.69 GB/s |
| `openssl::chacha20`             |        0.06 GB/s |          0.49 GB/s |
| `openssl::aes256`               |        0.10 GB/s |          0.86 GB/s |
| `stringzilla::aes256`           |        0.61 GB/s |      __4.14 GB/s__ |
|                                 |                  |                    |
| _`stringzilla::aes256ctr`_      |      _2.95 GB/s_ |        _9.46 GB/s_ |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |      152.41 MB/s |          1.09 GB/s |
| `cryptography.ChaCha20Poly1305` |       59.48 MB/s |        483.42 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       30.74 MB/s |        236.38 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        5.84 MB/s |         48.03 MB/s |
| `pycryptodome.AES-GCM`          |        2.94 MB/s |         26.80 MB/s |
| `stringzilla.Aes256GcmKey`      |  __330.00 MB/s__ |      __2.03 GB/s__ |
|                                 |                  |                    |
| _`stringzilla.Aes256CtrKey`_    |    _606.80 MB/s_ |        _3.26 GB/s_ |

> Measured August 5, 2026.

### Apple M5 Pro

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |      367.91 MB/s |        692.71 MB/s |
| `libsodium::xchacha20`          |      303.01 MB/s |        663.46 MB/s |
| `ring::chacha20`                |      571.15 MB/s |          1.56 GB/s |
| `ring::aes256`                  |    __1.89 GB/s__ |      __5.18 GB/s__ |
| `openssl::chacha20`             |      206.12 MB/s |          1.02 GB/s |
| `openssl::aes256`               |      333.64 MB/s |          2.37 GB/s |
| `stringzilla::aes256`           |                — |                  — |
|                                 |                  |                    |
| _`stringzilla::aes256ctr`_      |                — |                  — |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |  __251.95 MB/s__ |      __1.88 GB/s__ |
| `cryptography.ChaCha20Poly1305` |      153.80 MB/s |        894.47 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       26.04 MB/s |         58.59 MB/s |
| `pycryptodome.ChaCha20Poly1305` |       15.97 MB/s |        122.80 MB/s |
| `pycryptodome.AES-GCM`          |        6.99 MB/s |         52.11 MB/s |
| `stringzilla.Aes256GcmKey`      |                — |                  — |
|                                 |                  |                    |
| _`stringzilla.Aes256CtrKey`_    |                — |                  — |

> Measured July 29, 2026.

## Decryption

### Intel Xeon4 Sapphire Rapids

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |        0.19 GB/s |          0.75 GB/s |
| `libsodium::xchacha20`          |        0.17 GB/s |          0.70 GB/s |
| `ring::chacha20`                |        0.41 GB/s |          1.34 GB/s |
| `ring::aes256`                  |    __0.82 GB/s__ |          3.85 GB/s |
| `openssl::chacha20`             |        0.06 GB/s |          0.48 GB/s |
| `openssl::aes256`               |        0.10 GB/s |          0.86 GB/s |
| `stringzilla::aes256`           |        0.63 GB/s |      __4.39 GB/s__ |
|                                 |                  |                    |
| _`stringzilla::aes256ctr`_      |      _2.94 GB/s_ |        _9.67 GB/s_ |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |      122.99 MB/s |          0.94 GB/s |
| `cryptography.ChaCha20Poly1305` |       53.33 MB/s |        434.76 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       28.10 MB/s |        175.52 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        3.18 MB/s |         22.26 MB/s |
| `pycryptodome.AES-GCM`          |        1.93 MB/s |         17.73 MB/s |
| `stringzilla.Aes256GcmKey`      |  __230.10 MB/s__ |      __1.70 GB/s__ |
|                                 |                  |                    |
| _`stringzilla.Aes256CtrKey`_    |    _338.42 MB/s_ |        _2.32 GB/s_ |

> Measured August 5, 2026.

### Apple M5 Pro

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |      356.95 MB/s |        686.24 MB/s |
| `libsodium::xchacha20`          |      293.16 MB/s |        658.93 MB/s |
| `ring::chacha20`                |      571.49 MB/s |          1.25 GB/s |
| `ring::aes256`                  |    __1.93 GB/s__ |      __5.22 GB/s__ |
| `openssl::chacha20`             |      203.11 MB/s |          1.02 GB/s |
| `openssl::aes256`               |      335.52 MB/s |          2.39 GB/s |
| `stringzilla::aes256`           |                — |                  — |
|                                 |                  |                    |
| _`stringzilla::aes256ctr`_      |                — |                  — |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |  __257.58 MB/s__ |      __1.88 GB/s__ |
| `cryptography.ChaCha20Poly1305` |      157.05 MB/s |        840.71 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       25.42 MB/s |         58.26 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        8.76 MB/s |         75.12 MB/s |
| `pycryptodome.AES-GCM`          |        5.11 MB/s |         40.08 MB/s |
| `stringzilla.Aes256GcmKey`      |                — |                  — |
|                                 |                  |                    |
| _`stringzilla.Aes256CtrKey`_    |                — |                  — |

> Measured July 29, 2026.

## Key Setup

Expanding a key is per-connection work rather than per-message, but at 100-byte lines it is a large share of the call.
StringZilla's GCM key derives eight Galois hash powers beyond the round-key schedule its CTR key stops at, which is the whole distance between those two rows.
`libsodium` only copies 32 bytes here, deferring expansion to the first message, which is why it sits an order of magnitude below the rest.

| Library                  | Median latency |
| ------------------------ | -------------: |
| `libsodium::chacha20`    |        7.89 ns |
| `ring::chacha20`         |       39.55 ns |
| `stringzilla::aes256ctr` |       61.62 ns |
| `stringzilla::aes256`    |       96.09 ns |
| `ring::aes256`           |      142.22 ns |
| `openssl::chacha20`      |      718.46 ns |
| `openssl::aes256`        |      780.69 ns |

> Measured August 5, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
