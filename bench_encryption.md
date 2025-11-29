# Encryption and Decryption Benchmarks

Benchmarks for encryption and decryption operations in Rust implementations.

## Overview

These benchmarks compare ChaCha20 and AES256 encryption/decryption throughput across different libraries.

## Encryption

| Library               | ~100 bytes lines | ~1,000 bytes lines |
| --------------------- | ---------------: | -----------------: |
| Rust                  |                  |                    |
| `libsodium::chacha20` |       0.20 GiB/s |         0.71 GiB/s |
| `ring::chacha20`      |       0.39 GiB/s |         1.19 GiB/s |
| `ring::aes256`        |       0.61 GiB/s |         2.89 GiB/s |

## Decryption

| Library               | ~100 bytes lines | ~1,000 bytes lines |
| --------------------- | ---------------: | -----------------: |
| Rust                  |                  |                    |
| `libsodium::chacha20` |       0.20 GiB/s |         0.69 GiB/s |
| `ring::chacha20`      |       0.42 GiB/s |         1.08 GiB/s |
| `ring::aes256`        |       0.85 GiB/s |         2.48 GiB/s |

---

See [README.md](README.md) for dataset information and replication instructions.
