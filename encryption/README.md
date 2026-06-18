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
| `libsodium::chacha20`           |        0.16 GB/s |          0.53 GB/s |
| `ring::chacha20`                |        0.27 GB/s |          0.80 GB/s |
| `ring::aes256`                  |    __0.39 GB/s__ |      __2.20 GB/s__ |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |   __73.93 MB/s__ |    __694.70 MB/s__ |
| `cryptography.ChaCha20Poly1305` |       33.31 MB/s |        361.65 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       17.56 MB/s |        159.58 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        3.87 MB/s |         36.46 MB/s |
| `pycryptodome.AES-GCM`          |        1.39 MB/s |         19.60 MB/s |

> Measured June 17, 2026 on an Intel Xeon4 Sapphire Rapids.

## Decryption

### Intel Xeon4 Sapphire Rapids

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |        0.29 GB/s |          1.13 GB/s |
| `ring::chacha20`                |        0.34 GB/s |          0.74 GB/s |
| `ring::aes256`                  |    __0.65 GB/s__ |      __2.11 GB/s__ |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |   __70.98 MB/s__ |    __573.12 MB/s__ |
| `cryptography.ChaCha20Poly1305` |       35.67 MB/s |        283.99 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       19.04 MB/s |        139.83 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        2.23 MB/s |         17.63 MB/s |
| `pycryptodome.AES-GCM`          |        1.34 MB/s |         13.47 MB/s |

> Measured June 17, 2026 on an Intel Xeon4 Sapphire Rapids.

---

See [README.md](README.md) for dataset information and replication instructions.
