"""Request signing. RSA keys sign with RSA-PSS/SHA-256; Ed25519 keys sign the message directly.

The PEM header does not tell the key type (PKCS#8 RSA keys also start with BEGIN PRIVATE KEY), so the
type comes from the parsed key, as the docs recommend. Key material is never logged or printed.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class Signer:
    def __init__(self, key_id: str, private_key: object):
        self.key_id = key_id
        self._key = private_key
        if isinstance(private_key, Ed25519PrivateKey):
            self.algorithm = "ed25519"
        elif isinstance(private_key, rsa.RSAPrivateKey):
            self.algorithm = "rsa-pss"
        else:
            raise ValueError("Unsupported key type: Kalshi accepts RSA or Ed25519 keys")

    @classmethod
    def from_file(cls, key_id: str, path: Path) -> "Signer":
        data = Path(path).expanduser().read_bytes()
        key = serialization.load_pem_private_key(data, password=None)
        return cls(key_id, key)

    def sign(self, timestamp_ms: str, method: str, path: str) -> str:
        msg = f"{timestamp_ms}{method.upper()}{path.split('?')[0]}".encode()
        if self.algorithm == "ed25519":
            sig = self._key.sign(msg)  # type: ignore[attr-defined]
        else:
            sig = self._key.sign(  # type: ignore[attr-defined]
                msg,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        return base64.b64encode(sig).decode()

    def headers(self, method: str, signed_path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self.sign(ts, method, signed_path),
        }

    def __repr__(self) -> str:  # never leak the key id
        return f"Signer(algorithm={self.algorithm})"
