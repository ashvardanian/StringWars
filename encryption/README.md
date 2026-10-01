# Encryption and Decryption Benchmarks

Benchmarks for encryption and decryption operations across Rust and Python implementations.

## Overview

These benchmarks compare ChaCha20-Poly1305 and AES-256-GCM AEAD throughput across different libraries.
Rust covers `ring`, `openssl`, and `libsodium`; Python covers `cryptography` (OpenSSL backend), `pynacl` (libsodium), and `pycryptodome`.
The `cryptography` and `ring`/`openssl` paths reach the same OpenSSL/AES-NI kernels, while `pycryptodome` pays a large per-message Python-object overhead that dominates on short lines.

## Encryption

### Intel Xeon4 Sapphire Rapids

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |        0.15 GB/s |          0.49 GB/s |
| `libsodium::xchacha20`          |                — |                  — |
| `ring::chacha20`                |        0.25 GB/s |          0.75 GB/s |
| `ring::aes256`                  |    __0.36 GB/s__ |      __2.05 GB/s__ |
| `openssl::chacha20`             |                — |                  — |
| `openssl::aes256`               |                — |                  — |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |   __70.51 MB/s__ |    __662.52 MB/s__ |
| `cryptography.ChaCha20Poly1305` |       31.77 MB/s |        344.90 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       16.75 MB/s |        152.19 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        3.69 MB/s |         34.77 MB/s |
| `pycryptodome.AES-GCM`          |        1.33 MB/s |         18.69 MB/s |

> Measured June 17, 2026.

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
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |  __251.95 MB/s__ |      __1.88 GB/s__ |
| `cryptography.ChaCha20Poly1305` |      153.80 MB/s |        894.47 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       26.04 MB/s |         58.59 MB/s |
| `pycryptodome.ChaCha20Poly1305` |       15.97 MB/s |        122.80 MB/s |
| `pycryptodome.AES-GCM`          |        6.99 MB/s |         52.11 MB/s |

> Measured July 29, 2026.

## Decryption

### Intel Xeon4 Sapphire Rapids

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |        0.27 GB/s |          1.05 GB/s |
| `libsodium::xchacha20`          |                — |                  — |
| `ring::chacha20`                |        0.32 GB/s |          0.69 GB/s |
| `ring::aes256`                  |    __0.61 GB/s__ |      __1.97 GB/s__ |
| `openssl::chacha20`             |                — |                  — |
| `openssl::aes256`               |                — |                  — |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |   __67.69 MB/s__ |    __546.57 MB/s__ |
| `cryptography.ChaCha20Poly1305` |       34.02 MB/s |        270.83 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       18.16 MB/s |        133.35 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        2.13 MB/s |         16.81 MB/s |
| `pycryptodome.AES-GCM`          |        1.28 MB/s |         12.85 MB/s |

> Measured June 17, 2026.

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
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |  __257.58 MB/s__ |      __1.88 GB/s__ |
| `cryptography.ChaCha20Poly1305` |      157.05 MB/s |        840.71 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       25.42 MB/s |         58.26 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        8.76 MB/s |         75.12 MB/s |
| `pycryptodome.AES-GCM`          |        5.11 MB/s |         40.08 MB/s |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
