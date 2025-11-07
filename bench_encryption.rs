#![doc = r#"
# StringWars: Encryption Benchmarks

This file contains benchmarks for various Rust encryption libraries using Criterion,
comparing AEAD (Authenticated Encryption with Associated Data) ciphers commonly used
in TLS 1.2/1.3 and the Noise Protocol Framework.

The benchmarks focus on:
- **AES-256-GCM**: Hardware-accelerated AEAD cipher
- **ChaCha20-Poly1305**: Software-optimized AEAD cipher

The benchmarks are organized into three categories:

**Key Generation/Setup**:
- Ring key generation
- OpenSSL cipher initialization

**Encryption** (encrypting dataset tokens):
- ChaCha20-Poly1305 via Ring
- AES-256-GCM via Ring
- ChaCha20-Poly1305 IETF via OpenSSL
- AES-256-GCM via OpenSSL
- ChaCha20-Poly1305 IETF via libsodium
- XChaCha20-Poly1305 IETF via libsodium (extended nonce)

**Decryption** (decrypting previously encrypted data):
- ChaCha20-Poly1305 via Ring
- AES-256-GCM via Ring
- ChaCha20-Poly1305 IETF via OpenSSL
- AES-256-GCM via OpenSSL
- ChaCha20-Poly1305 IETF via libsodium
- XChaCha20-Poly1305 IETF via libsodium (extended nonce)

## System Dependencies

Before running these benchmarks, ensure the following system packages are installed:

```sh
sudo apt install -y build-essential pkg-config libssl-dev libsodium-dev # for Ubuntu/Debian
sudo dnf install -y gcc pkg-config openssl-devel libsodium-devel # for RHEL/Fedora
brew install pkg-config openssl libsodium # for macOS
```

## Usage Examples

The benchmarks use environment variables to control the input dataset:

- `STRINGWARS_DATASET`: Path to the input dataset file (required)
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line
  - `words`: Process the dataset word by word
  - `file`: Process the entire file as a single token
- `STRINGWARS_FILTER`: Regex pattern to filter which benchmarks to run (e.g., `aes` for AES benchmarks, `encryption/.*chacha` for ChaCha)

To run the benchmarks with the appropriate CPU features enabled:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_encryption bench_encryption --jobs $(nproc)
```

To benchmark Ring vs OpenSSL encryption and decryption:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=acgt_100.txt \
    STRINGWARS_TOKENS=lines \
    STRINGWARS_FILTER="(cryption/openssl|cryption/ring)" \
    cargo criterion --features bench_encryption bench_encryption
```
"#]
use std::hint::black_box;

use criterion::{Criterion, Throughput};
use stringtape::BytesCowsAuto;
use stringzilla::sz;

mod utils;
use utils::{load_dataset, should_run};

fn log_stringzilla_metadata() {
    let v = sz::version();
    println!("StringZilla v{}.{}.{}", v.major, v.minor, v.patch);
    println!("- uses dynamic dispatch: {}", sz::dynamic_dispatch());
    println!("- capabilities: {}", sz::capabilities().as_str());
}

fn configure_bench() -> Criterion {
    Criterion::default()
        .configure_from_args()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_secs(5))
        .measurement_time(std::time::Duration::from_secs(10))
}

/// Benchmarks key generation and cipher setup overhead
fn bench_key_generation(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    use ring::aead;

    // Benchmark: ring ChaCha20-Poly1305 key generation
    if should_run("keygen/ring::chacha20poly1305") {
        group.bench_function("ring::chacha20poly1305", |b| {
            b.iter(|| {
                let key_bytes = [0u8; 32]; // 256-bit key
                let key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes);
                black_box(key)
            })
        });
    }

    // Benchmark: ring AES-256-GCM key generation
    if should_run("keygen/ring::aes256gcm") {
        group.bench_function("ring::aes256gcm", |b| {
            b.iter(|| {
                let key_bytes = [0u8; 32]; // 256-bit key
                let key = aead::UnboundKey::new(&aead::AES_256_GCM, &key_bytes);
                black_box(key)
            })
        });
    }

    // Benchmark: OpenSSL ChaCha20-Poly1305 initialization
    if should_run("keygen/openssl::chacha20poly1305") {
        group.bench_function("openssl::chacha20poly1305", |b| {
            use openssl::symm::{Cipher, Crypter, Mode};
            b.iter(|| {
                let key = [0u8; 32];
                let cipher = Cipher::chacha20_poly1305();
                let crypter = Crypter::new(cipher, Mode::Encrypt, &key, None);
                black_box(crypter)
            })
        });
    }

    // Benchmark: OpenSSL AES-256-GCM initialization
    if should_run("keygen/openssl::aes256gcm") {
        group.bench_function("openssl::aes256gcm", |b| {
            use openssl::symm::{Cipher, Crypter, Mode};
            b.iter(|| {
                let key = [0u8; 32];
                let cipher = Cipher::aes_256_gcm();
                let crypter = Crypter::new(cipher, Mode::Encrypt, &key, None);
                black_box(crypter)
            })
        });
    }

    // Benchmark: libsodium ChaCha20-Poly1305 IETF key generation
    if should_run("keygen/libsodium::chacha20poly1305_ietf") {
        use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key};
        group.bench_function("libsodium::chacha20poly1305_ietf", |b| {
            b.iter(|| {
                let key = Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
                black_box(key)
            })
        });
    }

    // Benchmark: libsodium XChaCha20-Poly1305 IETF key generation
    if should_run("keygen/libsodium::xchacha20poly1305_ietf") {
        use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key};
        group.bench_function("libsodium::xchacha20poly1305_ietf", |b| {
            b.iter(|| {
                let key = Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
                black_box(key)
            })
        });
    }
}

/// Benchmarks AEAD encryption (encrypt + authenticate)
fn bench_encryption(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &BytesCowsAuto,
) {
    use ring::aead::{self, Aad, LessSafeKey, Nonce, UnboundKey, NONCE_LEN};

    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = tokens.iter().map(|t| t.len()).sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Benchmark: ring ChaCha20-Poly1305 encryption
    if should_run("encryption/ring::chacha20poly1305") {
        let key_bytes = [0u8; 32];
        let unbound_key = UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes).unwrap();
        let key = LessSafeKey::new(unbound_key);

        group.bench_function("ring::chacha20poly1305", |b| {
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for token in tokens.iter() {
                    let mut in_out = token.to_vec();
                    in_out.reserve(aead::CHACHA20_POLY1305.tag_len());

                    let mut nonce_bytes = [0u8; NONCE_LEN];
                    nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);
                    let nonce = Nonce::try_assume_unique_for_key(&nonce_bytes).unwrap();

                    let _ =
                        black_box(key.seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out));
                }
            })
        });
    }

    // Benchmark: ring AES-256-GCM encryption
    if should_run("encryption/ring::aes256gcm") {
        let key_bytes = [0u8; 32];
        let unbound_key = UnboundKey::new(&aead::AES_256_GCM, &key_bytes).unwrap();
        let key = LessSafeKey::new(unbound_key);

        group.bench_function("ring::aes256gcm", |b| {
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for token in tokens.iter() {
                    let mut in_out = token.to_vec();
                    // Reserve space for the authentication tag
                    in_out.reserve(aead::AES_256_GCM.tag_len());

                    let mut nonce_bytes = [0u8; NONCE_LEN];
                    nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);
                    let nonce = Nonce::try_assume_unique_for_key(&nonce_bytes).unwrap();

                    let _ =
                        black_box(key.seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out));
                }
            })
        });
    }

    // Benchmark: OpenSSL ChaCha20-Poly1305 encryption
    if should_run("encryption/openssl::chacha20poly1305") {
        use openssl::symm::{encrypt_aead, Cipher};

        group.bench_function("openssl::chacha20poly1305", |b| {
            let key = [0u8; 32];
            let cipher = Cipher::chacha20_poly1305();
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for token in tokens.iter() {
                    let mut iv = [0u8; 12];
                    iv[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);

                    let mut tag = vec![0u8; 16];
                    let _ = black_box(encrypt_aead(cipher, &key, Some(&iv), &[], token, &mut tag));
                }
            })
        });
    }

    // Benchmark: OpenSSL AES-256-GCM encryption
    if should_run("encryption/openssl::aes256gcm") {
        use openssl::symm::{encrypt_aead, Cipher};

        group.bench_function("openssl::aes256gcm", |b| {
            let key = [0u8; 32];
            let cipher = Cipher::aes_256_gcm();
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for token in tokens.iter() {
                    let mut iv = [0u8; 12];
                    iv[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);

                    let mut tag = vec![0u8; 16];
                    let _ = black_box(encrypt_aead(cipher, &key, Some(&iv), &[], token, &mut tag));
                }
            })
        });
    }

    // Benchmark: libsodium ChaCha20-Poly1305 IETF encryption
    if should_run("encryption/libsodium::chacha20poly1305_ietf") {
        use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key, Nonce};

        group.bench_function("libsodium::chacha20poly1305_ietf", |b| {
            let key = Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for token in tokens.iter() {
                    let mut nonce_bytes = [0u8; chacha20poly1305_ietf::NONCEBYTES];
                    nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);
                    let nonce = Nonce(nonce_bytes);

                    let _ = black_box(chacha20poly1305_ietf::seal(token, None, &nonce, &key));
                }
            })
        });
    }

    // Benchmark: libsodium XChaCha20-Poly1305 IETF encryption
    if should_run("encryption/libsodium::xchacha20poly1305_ietf") {
        use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key, Nonce};

        group.bench_function("libsodium::xchacha20poly1305_ietf", |b| {
            let key = Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for token in tokens.iter() {
                    let mut nonce_bytes = [0u8; xchacha20poly1305_ietf::NONCEBYTES];
                    nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);
                    let nonce = Nonce(nonce_bytes);

                    let _ = black_box(xchacha20poly1305_ietf::seal(token, None, &nonce, &key));
                }
            })
        });
    }
}

/// Benchmarks AEAD decryption (verify + decrypt)
fn bench_decryption(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    tokens: &BytesCowsAuto,
) {
    use ring::aead::{self, Aad, LessSafeKey, Nonce, UnboundKey, NONCE_LEN};

    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = tokens.iter().map(|t| t.len()).sum();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Prepare encrypted data for ring ChaCha20-Poly1305
    let key_bytes = [0u8; 32];
    let unbound_key_chacha = UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes).unwrap();
    let key_chacha = LessSafeKey::new(unbound_key_chacha);
    let mut encrypted_tokens_chacha: Vec<Vec<u8>> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let mut in_out = token.to_vec();
            in_out.reserve(aead::CHACHA20_POLY1305.tag_len());

            let mut nonce_bytes = [0u8; NONCE_LEN];
            nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
            nonce_counter = nonce_counter.wrapping_add(1);
            let nonce = Nonce::try_assume_unique_for_key(&nonce_bytes).unwrap();

            key_chacha
                .seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out)
                .unwrap();
            encrypted_tokens_chacha.push(in_out);
        }
    }

    // Prepare encrypted data for ring AES-256-GCM
    let unbound_key_aes = UnboundKey::new(&aead::AES_256_GCM, &key_bytes).unwrap();
    let key_aes = LessSafeKey::new(unbound_key_aes);
    let mut encrypted_tokens_aes: Vec<Vec<u8>> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let mut in_out = token.to_vec();
            in_out.reserve(aead::AES_256_GCM.tag_len());

            let mut nonce_bytes = [0u8; NONCE_LEN];
            nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
            nonce_counter = nonce_counter.wrapping_add(1);
            let nonce = Nonce::try_assume_unique_for_key(&nonce_bytes).unwrap();

            key_aes
                .seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out)
                .unwrap();
            encrypted_tokens_aes.push(in_out);
        }
    }

    // Benchmark: ring ChaCha20-Poly1305 decryption
    if should_run("decryption/ring::chacha20poly1305") {
        group.bench_function("ring::chacha20poly1305", |b| {
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for encrypted in &encrypted_tokens_chacha {
                    let mut in_out = encrypted.clone();

                    let mut nonce_bytes = [0u8; NONCE_LEN];
                    nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);
                    let nonce = Nonce::try_assume_unique_for_key(&nonce_bytes).unwrap();

                    let _ = black_box(key_chacha.open_in_place(nonce, Aad::empty(), &mut in_out));
                }
            })
        });
    }

    // Benchmark: ring AES-256-GCM decryption
    if should_run("decryption/ring::aes256gcm") {
        group.bench_function("ring::aes256gcm", |b| {
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for encrypted in &encrypted_tokens_aes {
                    let mut in_out = encrypted.clone();

                    let mut nonce_bytes = [0u8; NONCE_LEN];
                    nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);
                    let nonce = Nonce::try_assume_unique_for_key(&nonce_bytes).unwrap();

                    let _ = black_box(key_aes.open_in_place(nonce, Aad::empty(), &mut in_out));
                }
            })
        });
    }

    // Prepare encrypted data for OpenSSL ChaCha20-Poly1305
    use openssl::symm::{decrypt_aead, encrypt_aead, Cipher};
    let cipher_chacha = Cipher::chacha20_poly1305();
    let mut encrypted_tokens_openssl_chacha: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let mut iv = [0u8; 12];
            iv[..8].copy_from_slice(&nonce_counter.to_le_bytes());
            nonce_counter = nonce_counter.wrapping_add(1);

            let mut tag = vec![0u8; 16];
            let ciphertext =
                encrypt_aead(cipher_chacha, &key_bytes, Some(&iv), &[], token, &mut tag).unwrap();
            encrypted_tokens_openssl_chacha.push((ciphertext, tag));
        }
    }

    // Prepare encrypted data for OpenSSL AES-256-GCM
    let cipher_aes = Cipher::aes_256_gcm();
    let mut encrypted_tokens_openssl_aes: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let mut iv = [0u8; 12];
            iv[..8].copy_from_slice(&nonce_counter.to_le_bytes());
            nonce_counter = nonce_counter.wrapping_add(1);

            let mut tag = vec![0u8; 16];
            let ciphertext =
                encrypt_aead(cipher_aes, &key_bytes, Some(&iv), &[], token, &mut tag).unwrap();
            encrypted_tokens_openssl_aes.push((ciphertext, tag));
        }
    }

    // Benchmark: OpenSSL ChaCha20-Poly1305 decryption
    if should_run("decryption/openssl::chacha20poly1305") {
        group.bench_function("openssl::chacha20poly1305", |b| {
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for (ciphertext, tag) in &encrypted_tokens_openssl_chacha {
                    let mut iv = [0u8; 12];
                    iv[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);

                    let _ = black_box(decrypt_aead(
                        cipher_chacha,
                        &key_bytes,
                        Some(&iv),
                        &[],
                        ciphertext,
                        tag,
                    ));
                }
            })
        });
    }

    // Benchmark: OpenSSL AES-256-GCM decryption
    if should_run("decryption/openssl::aes256gcm") {
        group.bench_function("openssl::aes256gcm", |b| {
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for (ciphertext, tag) in &encrypted_tokens_openssl_aes {
                    let mut iv = [0u8; 12];
                    iv[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);

                    let _ = black_box(decrypt_aead(
                        cipher_aes,
                        &key_bytes,
                        Some(&iv),
                        &[],
                        ciphertext,
                        tag,
                    ));
                }
            })
        });
    }

    // Prepare encrypted data for libsodium ChaCha20-Poly1305 IETF
    use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{
        self, Key as Key_ChaCha, Nonce as Nonce_ChaCha,
    };
    let key_sodium_chacha = Key_ChaCha([0u8; chacha20poly1305_ietf::KEYBYTES]);
    let mut encrypted_tokens_sodium_chacha: Vec<Vec<u8>> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let mut nonce_bytes = [0u8; chacha20poly1305_ietf::NONCEBYTES];
            nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
            nonce_counter = nonce_counter.wrapping_add(1);
            let nonce = Nonce_ChaCha(nonce_bytes);

            let ciphertext = chacha20poly1305_ietf::seal(token, None, &nonce, &key_sodium_chacha);
            encrypted_tokens_sodium_chacha.push(ciphertext);
        }
    }

    // Benchmark: libsodium ChaCha20-Poly1305 IETF decryption
    if should_run("decryption/libsodium::chacha20poly1305_ietf") {
        group.bench_function("libsodium::chacha20poly1305_ietf", |b| {
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for ciphertext in &encrypted_tokens_sodium_chacha {
                    let mut nonce_bytes = [0u8; chacha20poly1305_ietf::NONCEBYTES];
                    nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);
                    let nonce = Nonce_ChaCha(nonce_bytes);

                    let _ = black_box(chacha20poly1305_ietf::open(
                        ciphertext,
                        None,
                        &nonce,
                        &key_sodium_chacha,
                    ));
                }
            })
        });
    }

    // Prepare encrypted data for libsodium XChaCha20-Poly1305 IETF
    use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{
        self, Key as Key_XChaCha, Nonce as Nonce_XChaCha,
    };
    let key_sodium_xchacha = Key_XChaCha([0u8; xchacha20poly1305_ietf::KEYBYTES]);
    let mut encrypted_tokens_sodium_xchacha: Vec<Vec<u8>> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let mut nonce_bytes = [0u8; xchacha20poly1305_ietf::NONCEBYTES];
            nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
            nonce_counter = nonce_counter.wrapping_add(1);
            let nonce = Nonce_XChaCha(nonce_bytes);

            let ciphertext = xchacha20poly1305_ietf::seal(token, None, &nonce, &key_sodium_xchacha);
            encrypted_tokens_sodium_xchacha.push(ciphertext);
        }
    }

    // Benchmark: libsodium XChaCha20-Poly1305 IETF decryption
    if should_run("decryption/libsodium::xchacha20poly1305_ietf") {
        group.bench_function("libsodium::xchacha20poly1305_ietf", |b| {
            b.iter(|| {
                let mut nonce_counter: u64 = 0;
                for ciphertext in &encrypted_tokens_sodium_xchacha {
                    let mut nonce_bytes = [0u8; xchacha20poly1305_ietf::NONCEBYTES];
                    nonce_bytes[..8].copy_from_slice(&nonce_counter.to_le_bytes());
                    nonce_counter = nonce_counter.wrapping_add(1);
                    let nonce = Nonce_XChaCha(nonce_bytes);

                    let _ = black_box(xchacha20poly1305_ietf::open(
                        ciphertext,
                        None,
                        &nonce,
                        &key_sodium_xchacha,
                    ));
                }
            })
        });
    }
}

fn main() {
    log_stringzilla_metadata();

    // Initialize libsodium
    sodiumoxide::init().expect("Failed to initialize libsodium");

    // Load the dataset defined by the environment variables, and panic if the content is missing
    let tape = load_dataset();
    if tape.is_empty() {
        panic!("No tokens found in the dataset.");
    }

    let mut criterion = configure_bench();

    // Profile key generation and cipher initialization overhead
    let mut group = criterion.benchmark_group("keygen");
    bench_key_generation(&mut group);
    group.finish();

    // Profile encryption operations
    let mut group = criterion.benchmark_group("encryption");
    bench_encryption(&mut group, &tape);
    group.finish();

    // Profile decryption operations
    let mut group = criterion.benchmark_group("decryption");
    bench_decryption(&mut group, &tape);
    group.finish();

    criterion.final_summary();
}
