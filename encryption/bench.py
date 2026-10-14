"""AEAD benchmarks in Python: cryptography, pycryptodome, libsodium and StringZilla. Mirrors `encryption/bench.rs`.

StringZilla also contributes an AES-256-CTR row. Counter mode is unauthenticated, so it is not
comparable to the AEAD rows on security, only on what the cipher costs without a tag to accumulate.

Run from the repository root, configured only through `STRINGWARS_*` variables:

    STRINGWARS_DATASET=README.md uv run --group encryption encryption/bench.py
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from typing import Any

import nacl.bindings as libsodium
import stringzilla as sz
from Crypto.Cipher import AES, ChaCha20_Poly1305
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from stringwars import (
    Bytes,
    MeasureSpec,
    Settings,
    finish,
    log_dataset,
    log_timing_overhead,
    measure,
    pass_over,
    print_machine,
    print_settings,
    read_settings,
    resolve_dataset,
)

KEY = bytes(32)  # 256-bit key (all zeros — content is irrelevant to throughput)


def nonce_for(counter: int) -> bytes:
    """A 96-bit IETF nonce derived from a per-message counter, matching the Rust harness."""
    return counter.to_bytes(12, "little")


@dataclass(frozen=True)
class Cipher:
    """One cipher: its label, `encrypt(data, nonce)` returning an opaque blob, `decrypt(blob, nonce)` consuming
    it, and the call each direction's row names after the label where the row is a real method."""

    label: str
    encrypt: Callable[[Any, bytes], Any]
    decrypt: Callable[[Any, bytes], Any]
    encrypt_call: str = ""
    decrypt_call: str = ""


def bench_cipher(
    settings: Settings,
    name: str,
    items: list[Any],
    nonces: list[bytes],
    token_bytes: Bytes,
    operation: Callable[[Any, bytes], object],
) -> None:
    """One pass over the corpus; bytes/s is over the plaintext, sealed or not."""
    work = MeasureSpec(unit="bytes", elements=len(items), total_bytes=token_bytes)
    measure(settings, name, work, pass_over(operation, items, nonces))


# Nonces are supplied by the harness as a per-message counter.
def cryptography_aesgcm() -> Cipher:
    cipher = AESGCM(KEY)
    return Cipher(
        "cryptography.AESGCM", lambda d, n: cipher.encrypt(n, d, None), lambda b, n: cipher.decrypt(n, b, None)
    )


def cryptography_chacha() -> Cipher:
    cipher = ChaCha20Poly1305(KEY)
    return Cipher(
        "cryptography.ChaCha20Poly1305",
        lambda d, n: cipher.encrypt(n, d, None),
        lambda b, n: cipher.decrypt(n, b, None),
    )


def pycryptodome_aesgcm() -> Cipher:
    def encrypt(data: bytes, nonce: bytes) -> tuple[bytes, bytes]:
        return AES.new(KEY, AES.MODE_GCM, nonce=nonce).encrypt_and_digest(data)

    def decrypt(blob: tuple[bytes, bytes], nonce: bytes) -> bytes:
        ciphertext, tag = blob
        return AES.new(KEY, AES.MODE_GCM, nonce=nonce).decrypt_and_verify(ciphertext, tag)

    return Cipher("pycryptodome.AES-GCM", encrypt, decrypt)


def pycryptodome_chacha() -> Cipher:
    def encrypt(data: bytes, nonce: bytes) -> tuple[bytes, bytes]:
        return ChaCha20_Poly1305.new(key=KEY, nonce=nonce).encrypt_and_digest(data)

    def decrypt(blob: tuple[bytes, bytes], nonce: bytes) -> bytes:
        ciphertext, tag = blob
        return ChaCha20_Poly1305.new(key=KEY, nonce=nonce).decrypt_and_verify(ciphertext, tag)

    return Cipher("pycryptodome.ChaCha20Poly1305", encrypt, decrypt)


def pynacl_chacha() -> Cipher:
    return Cipher(
        "pynacl.chacha20poly1305_ietf",
        lambda d, n: libsodium.crypto_aead_chacha20poly1305_ietf_encrypt(d, None, n, KEY),
        lambda b, n: libsodium.crypto_aead_chacha20poly1305_ietf_decrypt(b, None, n, KEY),
    )


def stringzilla_aesgcm() -> Cipher:
    key = sz.Aes256GcmKey(KEY)
    return Cipher(
        "stringzilla.Aes256GcmKey",
        lambda d, n: key.encrypt(d, n),
        lambda b, n: key.decrypt(b[0], n, b[1]),
        ".encrypt",
        ".decrypt",
    )


def stringzilla_aesctr() -> Cipher:
    # Counter mode is its own inverse, so one call serves both directions.
    key = sz.Aes256CtrKey(KEY)
    return Cipher("stringzilla.Aes256CtrKey", lambda d, n: key.xor(d, n), lambda b, n: key.xor(b, n), ".xor", ".xor")


CIPHERS: list[Callable[[], Cipher]] = [
    cryptography_aesgcm,
    cryptography_chacha,
    pycryptodome_aesgcm,
    pycryptodome_chacha,
    pynacl_chacha,
    stringzilla_aesgcm,
    stringzilla_aesctr,
]


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    print_machine(
        {
            **{name: pkg_version(name) for name in ("cryptography", "pycryptodome", "pynacl")},
            "StringZilla": f"{sz.__version__} with {sz.cpu_capabilities_enabled().name}",
        }
    )
    settings = read_settings("encryption")
    print_settings(settings)
    dataset = resolve_dataset(settings)
    tokens = dataset.tokens
    log_dataset(dataset)
    log_timing_overhead(settings)

    nonces = [nonce_for(index) for index in range(len(tokens))]

    print("\n# encryption")
    for build in CIPHERS:
        cipher = build()
        name = f"encryption/{cipher.label}{cipher.encrypt_call}"
        bench_cipher(settings, name, tokens, nonces, dataset.token_bytes, cipher.encrypt)

    print("\n# decryption")
    for build in CIPHERS:
        cipher = build()
        name = f"decryption/{cipher.label}{cipher.decrypt_call}"
        if settings.selects(name):
            blobs = [cipher.encrypt(token, nonce) for token, nonce in zip(tokens, nonces, strict=True)]
            bench_cipher(settings, name, blobs, nonces, dataset.token_bytes, cipher.decrypt)

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
