#![doc = r#"
# StringWars: Encryption Benchmarks

This file contains benchmarks for various Rust encryption libraries, comparing AEAD (Authenticated Encryption with
Associated Data) ciphers commonly used in TLS 1.2/1.3 and the Noise Protocol Framework.

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
- `STRINGWARS_FILTER`: Regex pattern to filter which benchmarks to run (e.g., `aes` for AES benchmarks,
  `encryption/.*chacha` for ChaCha)

To run the benchmarks with the appropriate CPU features enabled:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo bench --features bench_encryption --bench bench_encryption
```

To benchmark Ring vs OpenSSL encryption and decryption:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=acgt_100.txt \
    STRINGWARS_TOKENS=lines \
    STRINGWARS_FILTER="(cryption/openssl|cryption/ring)" \
    cargo bench --features bench_encryption --bench bench_encryption
```
"#]
use std::hint::black_box;

use stringtape::BytesCowsAuto;

#[path = "../utils.rs"]
mod utils;
use utils::{
    install_panic_hook, load_dataset, log_stringzilla_metadata, measure_throughput, BenchBudget,
    ReportAs, ResultExt, WorkUnits,
};

/// Constructs the next Ring AEAD nonce from a monotonically incrementing counter.
/// The counter occupies the first 8 bytes of the 12-byte nonce in little-endian order.
fn next_ring_nonce(counter: &mut u64) -> ring::aead::Nonce {
    let mut nonce_bytes = [0u8; ring::aead::NONCE_LEN];
    nonce_bytes[..8].copy_from_slice(&counter.to_le_bytes());
    *counter = counter.wrapping_add(1);
    ring::aead::Nonce::try_assume_unique_for_key(&nonce_bytes).unwrap()
}

/// Constructs the next OpenSSL 12-byte IV from a monotonically incrementing counter.
/// The counter occupies the first 8 bytes in little-endian order.
fn next_openssl_iv(counter: &mut u64) -> [u8; 12] {
    let mut iv = [0u8; 12];
    iv[..8].copy_from_slice(&counter.to_le_bytes());
    *counter = counter.wrapping_add(1);
    iv
}

/// Constructs the next libsodium ChaCha20-Poly1305 IETF nonce from a monotonically incrementing counter.
/// The counter occupies the first 8 bytes of the nonce in little-endian order.
fn next_sodium_chacha20_nonce(
    counter: &mut u64,
) -> sodiumoxide::crypto::aead::chacha20poly1305_ietf::Nonce {
    use sodiumoxide::crypto::aead::chacha20poly1305_ietf;
    let mut nonce_bytes = [0u8; chacha20poly1305_ietf::NONCEBYTES];
    nonce_bytes[..8].copy_from_slice(&counter.to_le_bytes());
    *counter = counter.wrapping_add(1);
    chacha20poly1305_ietf::Nonce(nonce_bytes)
}

/// Constructs the next libsodium XChaCha20-Poly1305 IETF nonce from a monotonically incrementing counter.
/// The counter occupies the first 8 bytes of the nonce in little-endian order.
fn next_sodium_xchacha20_nonce(
    counter: &mut u64,
) -> sodiumoxide::crypto::aead::xchacha20poly1305_ietf::Nonce {
    use sodiumoxide::crypto::aead::xchacha20poly1305_ietf;
    let mut nonce_bytes = [0u8; xchacha20poly1305_ietf::NONCEBYTES];
    nonce_bytes[..8].copy_from_slice(&counter.to_le_bytes());
    *counter = counter.wrapping_add(1);
    xchacha20poly1305_ietf::Nonce(nonce_bytes)
}

/// Benchmarks key generation and cipher setup overhead. Each variant builds one key/cipher per
/// call and cycles for the budget; throughput is reported as bytes/s over the 32-byte key.
fn bench_key_generation(budget: &BenchBudget) {
    use ring::aead;

    // Benchmark: ring ChaCha20-Poly1305 key generation
    measure_throughput(
        "keygen/ring::chacha20poly1305",
        ReportAs::Bytes,
        budget,
        || {
            let key_bytes = [0u8; 32]; // 256-bit key
            let key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes);
            let _ = black_box(key);
            WorkUnits::bytes(32)
        },
    );

    // Benchmark: ring AES-256-GCM key generation
    measure_throughput("keygen/ring::aes256gcm", ReportAs::Bytes, budget, || {
        let key_bytes = [0u8; 32]; // 256-bit key
        let key = aead::UnboundKey::new(&aead::AES_256_GCM, &key_bytes);
        let _ = black_box(key);
        WorkUnits::bytes(32)
    });

    // Benchmark: OpenSSL ChaCha20-Poly1305 initialization
    {
        use openssl::symm::{Cipher, Crypter, Mode};
        measure_throughput(
            "keygen/openssl::chacha20poly1305",
            ReportAs::Bytes,
            budget,
            || {
                let key = [0u8; 32];
                let cipher = Cipher::chacha20_poly1305();
                let crypter = Crypter::new(cipher, Mode::Encrypt, &key, None);
                let _ = black_box(crypter);
                WorkUnits::bytes(32)
            },
        );
    }

    // Benchmark: OpenSSL AES-256-GCM initialization
    {
        use openssl::symm::{Cipher, Crypter, Mode};
        measure_throughput("keygen/openssl::aes256gcm", ReportAs::Bytes, budget, || {
            let key = [0u8; 32];
            let cipher = Cipher::aes_256_gcm();
            let crypter = Crypter::new(cipher, Mode::Encrypt, &key, None);
            let _ = black_box(crypter);
            WorkUnits::bytes(32)
        });
    }

    // Benchmark: libsodium ChaCha20-Poly1305 IETF key generation
    {
        use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key};
        measure_throughput(
            "keygen/libsodium::chacha20poly1305_ietf",
            ReportAs::Bytes,
            budget,
            || {
                let key = Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
                let _ = black_box(key);
                WorkUnits::bytes(chacha20poly1305_ietf::KEYBYTES as u64)
            },
        );
    }

    // Benchmark: libsodium XChaCha20-Poly1305 IETF key generation
    {
        use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key};
        measure_throughput(
            "keygen/libsodium::xchacha20poly1305_ietf",
            ReportAs::Bytes,
            budget,
            || {
                let key = Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
                let _ = black_box(key);
                WorkUnits::bytes(xchacha20poly1305_ietf::KEYBYTES as u64)
            },
        );
    }
}

/// Benchmarks AEAD encryption (encrypt + authenticate). Each variant encrypts one token per call
/// and cycles the dataset for the budget; throughput is reported as bytes/s over the plaintext.
fn bench_encryption(budget: &BenchBudget, tokens: &BytesCowsAuto) {
    use ring::aead::{self, Aad, LessSafeKey, UnboundKey};

    // Collect token slices once so each cyclic call indexes a single token.
    let slices: Vec<&[u8]> = tokens.iter().collect();

    // Benchmark: ring ChaCha20-Poly1305 encryption
    {
        let key_bytes = [0u8; 32];
        let unbound_key = UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes).unwrap();
        let key = LessSafeKey::new(unbound_key);

        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "encryption/ring::chacha20poly1305",
            ReportAs::Bytes,
            budget,
            || {
                let token = slices[cursor % slices.len()];
                cursor += 1;
                let mut in_out = token.to_vec();
                in_out.reserve(aead::CHACHA20_POLY1305.tag_len());
                let nonce = next_ring_nonce(&mut nonce_counter);
                let _ = black_box(key.seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out));
                WorkUnits::new(1, token.len() as u64)
            },
        );
    }

    // Benchmark: ring AES-256-GCM encryption
    {
        let key_bytes = [0u8; 32];
        let unbound_key = UnboundKey::new(&aead::AES_256_GCM, &key_bytes).unwrap();
        let key = LessSafeKey::new(unbound_key);

        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "encryption/ring::aes256gcm",
            ReportAs::Bytes,
            budget,
            || {
                let token = slices[cursor % slices.len()];
                cursor += 1;
                let mut in_out = token.to_vec();
                // Reserve space for the authentication tag
                in_out.reserve(aead::AES_256_GCM.tag_len());
                let nonce = next_ring_nonce(&mut nonce_counter);
                let _ = black_box(key.seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out));
                WorkUnits::new(1, token.len() as u64)
            },
        );
    }

    // Benchmark: OpenSSL ChaCha20-Poly1305 encryption
    {
        use openssl::symm::{encrypt_aead, Cipher};

        let key = [0u8; 32];
        let cipher = Cipher::chacha20_poly1305();
        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "encryption/openssl::chacha20poly1305",
            ReportAs::Bytes,
            budget,
            || {
                let token = slices[cursor % slices.len()];
                cursor += 1;
                let iv = next_openssl_iv(&mut nonce_counter);
                let mut tag = vec![0u8; 16];
                let _ = black_box(encrypt_aead(cipher, &key, Some(&iv), &[], token, &mut tag));
                WorkUnits::new(1, token.len() as u64)
            },
        );
    }

    // Benchmark: OpenSSL AES-256-GCM encryption
    {
        use openssl::symm::{encrypt_aead, Cipher};

        let key = [0u8; 32];
        let cipher = Cipher::aes_256_gcm();
        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "encryption/openssl::aes256gcm",
            ReportAs::Bytes,
            budget,
            || {
                let token = slices[cursor % slices.len()];
                cursor += 1;
                let iv = next_openssl_iv(&mut nonce_counter);
                let mut tag = vec![0u8; 16];
                let _ = black_box(encrypt_aead(cipher, &key, Some(&iv), &[], token, &mut tag));
                WorkUnits::new(1, token.len() as u64)
            },
        );
    }

    // Benchmark: libsodium ChaCha20-Poly1305 IETF encryption
    {
        use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key};

        let key = Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "encryption/libsodium::chacha20poly1305_ietf",
            ReportAs::Bytes,
            budget,
            || {
                let token = slices[cursor % slices.len()];
                cursor += 1;
                let nonce = next_sodium_chacha20_nonce(&mut nonce_counter);
                let _ = black_box(chacha20poly1305_ietf::seal(token, None, &nonce, &key));
                WorkUnits::new(1, token.len() as u64)
            },
        );
    }

    // Benchmark: libsodium XChaCha20-Poly1305 IETF encryption
    {
        use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key};

        let key = Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "encryption/libsodium::xchacha20poly1305_ietf",
            ReportAs::Bytes,
            budget,
            || {
                let token = slices[cursor % slices.len()];
                cursor += 1;
                let nonce = next_sodium_xchacha20_nonce(&mut nonce_counter);
                let _ = black_box(xchacha20poly1305_ietf::seal(token, None, &nonce, &key));
                WorkUnits::new(1, token.len() as u64)
            },
        );
    }
}

/// Benchmarks AEAD decryption (verify + decrypt). Pre-encrypts every token once before the timed
/// loop; each measured call then decrypts one ciphertext and cycles the dataset for the budget.
/// Throughput is reported over the original plaintext lengths to match the encryption accounting.
fn bench_decryption(budget: &BenchBudget, tokens: &BytesCowsAuto) {
    use ring::aead::{self, Aad, LessSafeKey, UnboundKey};

    // Original plaintext lengths, used as the per-item byte work (the encrypted buffers carry
    // extra tag/overhead bytes that the original throughput accounting excluded).
    let plaintext_lengths: Vec<u64> = tokens.iter().map(|token| token.len() as u64).collect();

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
            let nonce = next_ring_nonce(&mut nonce_counter);
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
            let nonce = next_ring_nonce(&mut nonce_counter);
            key_aes
                .seal_in_place_append_tag(nonce, Aad::empty(), &mut in_out)
                .unwrap();
            encrypted_tokens_aes.push(in_out);
        }
    }

    // Benchmark: ring ChaCha20-Poly1305 decryption
    {
        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "decryption/ring::chacha20poly1305",
            ReportAs::Bytes,
            budget,
            || {
                let index = cursor % encrypted_tokens_chacha.len();
                cursor += 1;
                let mut in_out = encrypted_tokens_chacha[index].clone();
                let nonce = next_ring_nonce(&mut nonce_counter);
                let _ = black_box(key_chacha.open_in_place(nonce, Aad::empty(), &mut in_out));
                WorkUnits::new(1, plaintext_lengths[index])
            },
        );
    }

    // Benchmark: ring AES-256-GCM decryption
    {
        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "decryption/ring::aes256gcm",
            ReportAs::Bytes,
            budget,
            || {
                let index = cursor % encrypted_tokens_aes.len();
                cursor += 1;
                let mut in_out = encrypted_tokens_aes[index].clone();
                let nonce = next_ring_nonce(&mut nonce_counter);
                let _ = black_box(key_aes.open_in_place(nonce, Aad::empty(), &mut in_out));
                WorkUnits::new(1, plaintext_lengths[index])
            },
        );
    }

    // Prepare encrypted data for OpenSSL ChaCha20-Poly1305
    use openssl::symm::{decrypt_aead, encrypt_aead, Cipher};
    let cipher_chacha = Cipher::chacha20_poly1305();
    let mut encrypted_tokens_openssl_chacha: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let iv = next_openssl_iv(&mut nonce_counter);
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
            let iv = next_openssl_iv(&mut nonce_counter);
            let mut tag = vec![0u8; 16];
            let ciphertext =
                encrypt_aead(cipher_aes, &key_bytes, Some(&iv), &[], token, &mut tag).unwrap();
            encrypted_tokens_openssl_aes.push((ciphertext, tag));
        }
    }

    // Benchmark: OpenSSL ChaCha20-Poly1305 decryption
    {
        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "decryption/openssl::chacha20poly1305",
            ReportAs::Bytes,
            budget,
            || {
                let index = cursor % encrypted_tokens_openssl_chacha.len();
                cursor += 1;
                let (ciphertext, tag) = &encrypted_tokens_openssl_chacha[index];
                let iv = next_openssl_iv(&mut nonce_counter);
                let _ = black_box(decrypt_aead(
                    cipher_chacha,
                    &key_bytes,
                    Some(&iv),
                    &[],
                    ciphertext,
                    tag,
                ));
                WorkUnits::new(1, plaintext_lengths[index])
            },
        );
    }

    // Benchmark: OpenSSL AES-256-GCM decryption
    {
        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "decryption/openssl::aes256gcm",
            ReportAs::Bytes,
            budget,
            || {
                let index = cursor % encrypted_tokens_openssl_aes.len();
                cursor += 1;
                let (ciphertext, tag) = &encrypted_tokens_openssl_aes[index];
                let iv = next_openssl_iv(&mut nonce_counter);
                let _ = black_box(decrypt_aead(
                    cipher_aes,
                    &key_bytes,
                    Some(&iv),
                    &[],
                    ciphertext,
                    tag,
                ));
                WorkUnits::new(1, plaintext_lengths[index])
            },
        );
    }

    // Prepare encrypted data for libsodium ChaCha20-Poly1305 IETF
    use sodiumoxide::crypto::aead::chacha20poly1305_ietf::{self, Key as SodiumChaCha20Key};
    let key_sodium_chacha = SodiumChaCha20Key([0u8; chacha20poly1305_ietf::KEYBYTES]);
    let mut encrypted_tokens_sodium_chacha: Vec<Vec<u8>> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let nonce = next_sodium_chacha20_nonce(&mut nonce_counter);
            let ciphertext = chacha20poly1305_ietf::seal(token, None, &nonce, &key_sodium_chacha);
            encrypted_tokens_sodium_chacha.push(ciphertext);
        }
    }

    // Benchmark: libsodium ChaCha20-Poly1305 IETF decryption
    {
        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "decryption/libsodium::chacha20poly1305_ietf",
            ReportAs::Bytes,
            budget,
            || {
                let index = cursor % encrypted_tokens_sodium_chacha.len();
                cursor += 1;
                let ciphertext = &encrypted_tokens_sodium_chacha[index];
                let nonce = next_sodium_chacha20_nonce(&mut nonce_counter);
                let _ = black_box(chacha20poly1305_ietf::open(
                    ciphertext,
                    None,
                    &nonce,
                    &key_sodium_chacha,
                ));
                WorkUnits::new(1, plaintext_lengths[index])
            },
        );
    }

    // Prepare encrypted data for libsodium XChaCha20-Poly1305 IETF
    use sodiumoxide::crypto::aead::xchacha20poly1305_ietf::{self, Key as SodiumXChaCha20Key};
    let key_sodium_xchacha = SodiumXChaCha20Key([0u8; xchacha20poly1305_ietf::KEYBYTES]);
    let mut encrypted_tokens_sodium_xchacha: Vec<Vec<u8>> = Vec::new();
    {
        let mut nonce_counter: u64 = 0;
        for token in tokens.iter() {
            let nonce = next_sodium_xchacha20_nonce(&mut nonce_counter);
            let ciphertext = xchacha20poly1305_ietf::seal(token, None, &nonce, &key_sodium_xchacha);
            encrypted_tokens_sodium_xchacha.push(ciphertext);
        }
    }

    // Benchmark: libsodium XChaCha20-Poly1305 IETF decryption
    {
        let mut cursor = 0usize;
        let mut nonce_counter: u64 = 0;
        measure_throughput(
            "decryption/libsodium::xchacha20poly1305_ietf",
            ReportAs::Bytes,
            budget,
            || {
                let index = cursor % encrypted_tokens_sodium_xchacha.len();
                cursor += 1;
                let ciphertext = &encrypted_tokens_sodium_xchacha[index];
                let nonce = next_sodium_xchacha20_nonce(&mut nonce_counter);
                let _ = black_box(xchacha20poly1305_ietf::open(
                    ciphertext,
                    None,
                    &nonce,
                    &key_sodium_xchacha,
                ));
                WorkUnits::new(1, plaintext_lengths[index])
            },
        );
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    // Initialize libsodium
    sodiumoxide::init().expect("Failed to initialize libsodium");

    // Load the dataset defined by the environment variables
    let tape = load_dataset().unwrap_nice();

    let budget = BenchBudget::from_env(5.0, 10.0);

    // Profile key generation and cipher initialization overhead
    println!("# keygen");
    bench_key_generation(&budget);

    // Profile encryption operations
    println!("# encryption");
    bench_encryption(&budget, &tape);

    // Profile decryption operations
    println!("# decryption");
    bench_decryption(&budget, &tape);
}
