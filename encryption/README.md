# Encryption and Decryption Benchmarks

Benchmarks for encryption and decryption operations in Rust implementations.

## Overview

These benchmarks compare ChaCha20 and AES256 encryption/decryption throughput across different libraries.

## Encryption

| Library               | ~100 bytes lines | ~1,000 bytes lines |
| --------------------- | ---------------: | -----------------: |
| Rust                  |                  |                    |
| `libsodium::chacha20` |        0.16 GB/s |          0.53 GB/s |
| `ring::chacha20`      |        0.27 GB/s |          0.80 GB/s |
| `ring::aes256`        |        0.39 GB/s |          2.20 GB/s |

> Measured 2026-06-17 on an Intel Xeon Platinum 8468 (Sapphire Rapids).

## Decryption

| Library               | ~100 bytes lines | ~1,000 bytes lines |
| --------------------- | ---------------: | -----------------: |
| Rust                  |                  |                    |
| `libsodium::chacha20` |        0.29 GB/s |          1.13 GB/s |
| `ring::chacha20`      |        0.34 GB/s |          0.74 GB/s |
| `ring::aes256`        |        0.65 GB/s |          2.11 GB/s |

> Measured 2026-06-17 on an Intel Xeon Platinum 8468 (Sapphire Rapids).

---

See [README.md](README.md) for dataset information and replication instructions.
