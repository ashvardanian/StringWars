# Encryption and Decryption Benchmarks

Benchmarks for encryption and decryption operations in Rust implementations.

## Overview

These benchmarks compare ChaCha20 and AES256 encryption/decryption throughput across different libraries.

## Encryption

| Library               | ~100 bytes lines | ~1,000 bytes lines |
| --------------------- | ---------------: | -----------------: |
| Rust                  |                  |                    |
| `libsodium::chacha20` |        0.20 GB/s |          0.71 GB/s |
| `ring::chacha20`      |        0.39 GB/s |          1.19 GB/s |
| `ring::aes256`        |        0.61 GB/s |          2.89 GB/s |

## Decryption

| Library               | ~100 bytes lines | ~1,000 bytes lines |
| --------------------- | ---------------: | -----------------: |
| Rust                  |                  |                    |
| `libsodium::chacha20` |        0.20 GB/s |          0.69 GB/s |
| `ring::chacha20`      |        0.42 GB/s |          1.08 GB/s |
| `ring::aes256`        |        0.85 GB/s |          2.48 GB/s |

---

See [README.md](README.md) for dataset information and replication instructions.
