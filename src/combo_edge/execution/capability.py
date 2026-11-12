"""The accept capability.

Only the execution module can mint one (``_mint`` is private to this package and test_safety.py
fails the build if it is referenced anywhere else). The HTTP client refuses any accept call that
doesn't carry a valid capability for exactly that RFQ and quote.
"""

from __future__ import annotations

import re
import time

_MINT = object()
_ACCEPT_RX = re.compile(r"^(?:/trade-api/v2)?/communications/rfqs/([^/]+)/quotes/([^/]+)/accept/?$")
CAPABILITY_TTL_S = 5.0


def accept_path(rfq_id: str, quote_id: str) -> str:
    """RFQ-scoped accept endpoint (PUT)."""
    return f"/communications/rfqs/{rfq_id}/quotes/{quote_id}/accept"


def is_accept_path(path: str) -> bool:
    p = path.split("?")[0]
    return _ACCEPT_RX.match(p) is not None or p.rstrip("/").endswith("/accept")


class AcceptCapability:
    __slots__ = ("_token", "rfq_id", "quote_id", "issued", "used")

    def __init__(self, token: object, rfq_id: str, quote_id: str):
        if token is not _MINT:
            raise PermissionError("Only the execution module may create an AcceptCapability")
        self._token = token
        self.rfq_id = rfq_id
        self.quote_id = quote_id
        self.issued = time.monotonic()
        self.used = False

    def valid_for(self, path: str) -> bool:
        m = _ACCEPT_RX.match(path.split("?")[0])
        if not m or self._token is not _MINT or self.used:
            return False
        if time.monotonic() - self.issued > CAPABILITY_TTL_S:
            return False
        return m.group(1) == self.rfq_id and m.group(2) == self.quote_id


def _mint(rfq_id: str, quote_id: str) -> AcceptCapability:
    return AcceptCapability(_MINT, rfq_id, quote_id)
