# Encryption and Decryption Benchmarks

Benchmarks for encryption and decryption operations across Rust and Python implementations.

## Overview

These benchmarks compare ChaCha20-Poly1305 and AES-256-GCM AEAD throughput across different libraries.
Rust covers `ring`, `openssl`, `libsodium`, and `stringzilla`; Python covers `cryptography` (OpenSSL backend), `pynacl` (libsodium), `pycryptodome`, and `stringzilla`.
The `cryptography` and `ring`/`openssl` paths reach the same OpenSSL/AES-NI kernels, while `pycryptodome` pays a large per-message Python-object overhead that dominates on short lines.

Each Rust row seals and opens into a buffer sized before the timed loop, so the rows price the cipher rather than the allocator.
OpenSSL is the exception: it builds one cipher context per message, which the crate gives no way to re-key, so at 100-byte lines it pays that setup on every call.

StringZilla also contributes an AES-256-CTR row.
Counter mode is unauthenticated, so it is not comparable to the AEAD rows on security — it is the floor, the same AES round function without a tag to accumulate, and the gap to `stringzilla::aes256` is what GHASH costs.

The `encrypt_into` and `decrypt_into` rows seal and open one line at a time, and the `xor_into` rows are the symmetric counter-mode transform, so one row covers both directions.
Alternatives are restricted to the same cipher: AES-256-GCM rows race `ring::aes256gcm`, `openssl::aes256gcm` and `cryptography.AESGCM`, and the unauthenticated AES-256-CTR rows have no rival in this suite.

The `new` rows expand a 32-byte key, which is per-connection work rather than per-message, but at 100-byte lines it is a large share of the call.
StringZilla's GCM key derives eight Galois hash powers beyond the round-key schedule its CTR key stops at, which is the whole distance between those two rows.
`libsodium` only copies 32 bytes there, deferring expansion to the first message, so it is not an alternative.

Every table reports single-core throughput in bytes per second, on lines of about 100 and about 1,000 bytes.
Speedup is StringZilla's speed over the alternative's, `–` marks a row without an alternative, `✗` a result that failed its check, and `⋯` an alternative not benchmarked on that chip.
Apple M5 Pro has no table because no StringZilla encryption row was measured there.

## Intel Xeon5 Emerald Rapids

### Rust

Also benchmarked: `openssl::aes256gcm`, `ring::chacha20poly1305`, `openssl::chacha20poly1305`, `libsodium::chacha20poly1305_ietf`, `libsodium::xchacha20poly1305_ietf`.

| StringZilla                                                  |     Speed | Speedup | Alternative       |     Speed |
| :----------------------------------------------------------- | --------: | ------: | :---------------- | --------: |
| `stringzilla::Aes256GcmKey::encrypt_into`, ~100-byte lines   |  972 MB/s |   0.99× | `ring::aes256gcm` |  980 MB/s |
| `stringzilla::Aes256GcmKey::encrypt_into`, ~1,000-byte lines |  4.9 GB/s |   1.04× | `ring::aes256gcm` |  4.7 GB/s |
| `stringzilla::Aes256GcmKey::decrypt_into`, ~100-byte lines   | 1.03 GB/s |   0.98× | `ring::aes256gcm` | 1.05 GB/s |
| `stringzilla::Aes256GcmKey::decrypt_into`, ~1,000-byte lines | 5.30 GB/s |   1.18× | `ring::aes256gcm` |  4.5 GB/s |
| `stringzilla::Aes256CtrKey::xor_into`, ~100-byte lines       | 4.27 GB/s |       – | –                 |         – |
| `stringzilla::Aes256CtrKey::xor_into`, ~1,000-byte lines     |   12 GB/s |       – | –                 |         – |
| `stringzilla::Aes256GcmKey::new`, 32-byte key                |  220 MB/s |   1.05× | `ring::aes256gcm` |  210 MB/s |
| `stringzilla::Aes256CtrKey::new`, 32-byte key                |  580 MB/s |   2.76× | `ring::aes256gcm` |  210 MB/s |

> Measured October 15, 2026.

### Python

Also benchmarked: `cryptography.ChaCha20Poly1305`, `pynacl.chacha20poly1305_ietf`, `pycryptodome.AES-GCM`, `pycryptodome.ChaCha20Poly1305`.

| StringZilla                                                |     Speed | Speedup | Alternative           |      Speed |
| :--------------------------------------------------------- | --------: | ------: | :-------------------- | ---------: |
| `stringzilla.Aes256GcmKey.encrypt`, ~100-byte lines        |  626 MB/s |   2.64× | `cryptography.AESGCM` |   237 MB/s |
| `stringzilla.Aes256GcmKey.encrypt`, ~1,000-byte lines      |  2.1 GB/s |   1.17× | `cryptography.AESGCM` |   1.8 GB/s |
| `stringzilla.Aes256GcmKey.decrypt`, ~100-byte lines        |  660 MB/s |   2.77× | `cryptography.AESGCM` |   238 MB/s |
| `stringzilla.Aes256GcmKey.decrypt`, ~1,000-byte lines      |  3.7 GB/s |   3.70× | `cryptography.AESGCM` | 1,000 MB/s |
| `stringzilla.Aes256CtrKey.xor`, ~100-byte lines            | 1.64 GB/s |       – | –                     |          – |
| `stringzilla.Aes256CtrKey.xor`, ~1,000-byte lines          |  3.9 GB/s |       – | –                     |          – |

> Measured October 15, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
