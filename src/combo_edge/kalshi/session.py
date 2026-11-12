"""Builds a connected session (client + optional websocket) for mock, demo or prod."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from combo_edge.kalshi.auth import Signer
from combo_edge.kalshi.client import KalshiClient
from combo_edge.kalshi.limiter import Limiter
from combo_edge.kalshi.transport import HttpxTransport, Transport
from combo_edge.kalshi.ws import WsManager
from combo_edge.settings import EnvConfig, env_config

log = logging.getLogger(__name__)


class MockSigner:
    algorithm = "mock"
    key_id = "mock"

    def headers(self, method: str, signed_path: str) -> dict[str, str]:
        return {"KALSHI-ACCESS-KEY": "mock", "KALSHI-ACCESS-TIMESTAMP": "0", "KALSHI-ACCESS-SIGNATURE": "mock"}


_MOCK_EXCHANGE: Any = None


def mock_exchange(**kw: Any) -> Any:
    """Process-wide mock exchange (so the dashboard's offline mode keeps state between requests)."""
    global _MOCK_EXCHANGE
    if _MOCK_EXCHANGE is None or kw:
        from combo_edge.kalshi.mock.exchange import MockExchange

        _MOCK_EXCHANGE = MockExchange(**kw)
    return _MOCK_EXCHANGE


def set_mock_exchange(ex: Any) -> None:
    global _MOCK_EXCHANGE
    _MOCK_EXCHANGE = ex


@dataclass
class Session:
    env: EnvConfig
    client: KalshiClient
    transport: Transport
    signer: Any
    ws: WsManager | None = None
    exchange: Any = None
    limits: dict = field(default_factory=dict)

    async def open_ws(self, *, user_filter_self: bool = True, channels: tuple[str, ...] = ("communications",), wait: float = 10.0) -> bool:
        if self.ws is None:
            self.ws = WsManager(self.env, self.transport, self.signer, channels=channels, user_filter_self=user_filter_self)
        return await self.ws.start(wait=wait)

    async def close(self) -> None:
        if self.ws:
            await self.ws.stop()
        await self.client.aclose()


def build_session(env: EnvConfig | None = None, *, exchange: Any = None, fraction: float = 0.6) -> Session:
    env = env or env_config()
    if env.name == "mock":
        from combo_edge.kalshi.mock.exchange import MockTransport

        ex = exchange or mock_exchange()
        transport: Transport = MockTransport(ex)
        signer: Any = MockSigner()
    else:
        if not env.has_keys:
            raise RuntimeError(
                f"{env.name.upper()} keys missing: set KALSHI_{env.name.upper()}_API_KEY_ID and "
                f"KALSHI_{env.name.upper()}_PRIVATE_KEY_PATH in .env"
            )
        transport = HttpxTransport()
        signer = Signer.from_file(env.key_id or "", env.key_path)  # type: ignore[arg-type]
        ex = None
    client = KalshiClient(env, transport, signer, Limiter(fraction=fraction))
    return Session(env, client, transport, signer, exchange=ex)
