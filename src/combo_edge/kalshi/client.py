"""Kalshi REST client: auth, rate limiting, retries for safe calls, an endpoint allowlist, and the
accept-capability check.

The client never names forbidden endpoints: anything not on the allowlist is refused. The accept
endpoint is only reachable with an ``AcceptCapability`` minted by the execution module.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Any

from combo_edge.execution.capability import AcceptCapability, is_accept_path
from combo_edge.kalshi.auth import Signer
from combo_edge.kalshi.limiter import Limiter
from combo_edge.kalshi.transport import Response, Transport
from combo_edge.settings import EnvConfig

log = logging.getLogger(__name__)

API_PREFIX = "/trade-api/v2"

# (method, path regex) relative to /trade-api/v2. Everything else is refused.
ALLOWED: list[tuple[str, re.Pattern[str]]] = [
    (m, re.compile(p))
    for m, p in [
        ("GET", r"^/exchange/status$"),
        ("GET", r"^/account/limits$"),
        ("GET", r"^/account/endpoint_costs$"),
        ("POST", r"^/account/api_usage_level/upgrade$"),
        ("GET", r"^/communications/id$"),
        ("GET", r"^/milestones$"),
        ("GET", r"^/milestones/[^/]+$"),
        ("GET", r"^/events$"),
        ("GET", r"^/events/[^/]+$"),
        ("GET", r"^/markets$"),
        ("GET", r"^/markets/[^/]+$"),
        ("GET", r"^/multivariate_event_collections$"),
        ("GET", r"^/multivariate_event_collections/[^/]+$"),
        ("POST", r"^/multivariate_event_collections/[^/]+$"),
        ("POST", r"^/communications/rfqs$"),
        ("GET", r"^/communications/rfqs$"),
        ("GET", r"^/communications/rfqs/[^/]+$"),
        ("DELETE", r"^/communications/rfqs/[^/]+$"),
        ("GET", r"^/communications/quotes$"),  # list only (GET); quote creation is never allowed
        ("GET", r"^/communications/rfqs/[^/]+/quotes/[^/]+$"),
        ("GET", r"^/portfolio/fills$"),
        ("GET", r"^/portfolio/balance$"),
        ("GET", r"^/portfolio/settlements$"),
        ("GET", r"^/portfolio/positions$"),
    ]
]

SAFE_METHODS = {"GET", "DELETE"}


class KalshiError(Exception):
    def __init__(self, status: int, code: str = "", message: str = "", details: str = "", path: str = ""):
        self.status, self.code, self.message, self.details, self.path = status, code, message, details, path
        super().__init__(f"HTTP {status} {code}: {message} {details}".strip() + (f" [{path}]" if path else ""))


class EndpointRefused(Exception):
    """Raised before any network call when an endpoint is not on the allowlist."""


class AcceptRefused(EndpointRefused):
    pass


def check_endpoint(method: str, path: str, capability: object | None = None) -> None:
    m = method.upper()
    p = path.split("?")[0]
    if p.startswith(API_PREFIX):
        p = p[len(API_PREFIX):]
    if is_accept_path(p):
        if m != "PUT" or not isinstance(capability, AcceptCapability) or not capability.valid_for(p):
            raise AcceptRefused("Accept refused: only the execution module may accept, with a valid capability")
        return
    for am, rx in ALLOWED:
        if am == m and rx.match(p):
            return
    raise EndpointRefused(f"Endpoint not allowed by combo-edge: {m} {p}")


class KalshiClient:
    def __init__(self, env: EnvConfig, transport: Transport, signer: Signer | None, limiter: Limiter | None = None):
        self.env = env
        self.transport = transport
        self.signer = signer
        self.limiter = limiter or Limiter()
        self.calls = 0

    # --- core -----------------------------------------------------------------------
    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: Any = None,
        capability: object | None = None,
        retries: int = 4,
        auth: bool = True,
    ) -> Any:
        method = method.upper()
        check_endpoint(method, path, capability)
        if isinstance(capability, AcceptCapability):
            capability.used = True
        url = self.env.rest_base + path
        signed_path = API_PREFIX + path
        safe = method in SAFE_METHODS and capability is None
        attempt = 0
        while True:
            await self.limiter.acquire(method, path)
            headers = {"Content-Type": "application/json"}
            if auth and self.signer is not None:
                headers.update(self.signer.headers(method, signed_path))
            self.calls += 1
            try:
                resp: Response = await self.transport.request(
                    method, url, params=_clean(params), json_body=json_body, headers=headers
                )
            except (OSError, asyncio.TimeoutError, ConnectionError) as e:
                if safe and attempt < retries:
                    attempt += 1
                    await asyncio.sleep(_backoff(attempt))
                    continue
                raise KalshiError(0, "network_error", str(e), path=path) from e
            except Exception as e:  # httpx errors
                if e.__class__.__module__.startswith("httpx") and safe and attempt < retries:
                    attempt += 1
                    await asyncio.sleep(_backoff(attempt))
                    continue
                if e.__class__.__module__.startswith("httpx"):
                    raise KalshiError(0, "network_error", str(e), path=path) from e
                raise
            if 200 <= resp.status < 300:
                return resp.data if resp.data is not None else {}
            code, msg, details = _err_fields(resp)
            retryable = resp.status >= 500 or (resp.status == 429 and code not in ("open_rfq_limit_exceeded",))
            if retryable and safe and attempt < retries:
                attempt += 1
                await asyncio.sleep(_backoff(attempt))
                continue
            raise KalshiError(resp.status, code, msg, details, path)

    async def get(self, path: str, **params: Any) -> Any:
        return await self.request("GET", path, params=params)

    async def paginate(self, path: str, key: str, *, max_pages: int = 50, **params: Any) -> list[dict]:
        out: list[dict] = []
        cursor = None
        for _ in range(max_pages):
            data = await self.get(path, **params, cursor=cursor)
            out.extend(data.get(key) or [])
            cursor = data.get("cursor")
            if not cursor:
                break
        return out

    # --- account ---------------------------------------------------------------------
    async def get_limits(self) -> dict:
        return await self.get("/account/limits")

    async def get_endpoint_costs(self) -> dict:
        return await self.get("/account/endpoint_costs")

    async def configure_limiter(self, fraction: float = 0.6) -> dict:
        limits = costs = None
        try:
            limits = await self.get_limits()
        except KalshiError as e:
            log.warning("limits unavailable: %s", e)
        try:
            costs = await self.get_endpoint_costs()
        except KalshiError as e:
            log.warning("endpoint costs unavailable: %s", e)
        self.limiter.fraction = fraction
        self.limiter.configure(limits, costs)
        return {"limits": limits, "costs": costs}

    async def upgrade_tier(self) -> dict:
        return await self.request("POST", "/account/api_usage_level/upgrade", json_body={})

    async def communications_id(self) -> str:
        return (await self.get("/communications/id")).get("communications_id", "")

    async def exchange_status(self) -> dict:
        return await self.get("/exchange/status")

    # --- market data -----------------------------------------------------------------
    async def milestones(self, **params: Any) -> list[dict]:
        params.setdefault("limit", 200)
        return await self.paginate("/milestones", "milestones", max_pages=10, **params)

    async def events(self, event_tickers: list[str], *, nested: bool = True) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(event_tickers), 40):
            chunk = ",".join(event_tickers[i : i + 40])
            out.extend(
                await self.paginate(
                    "/events", "events", max_pages=5, tickers=chunk, with_nested_markets=str(nested).lower(), limit=200
                )
            )
        return out

    async def event(self, event_ticker: str, *, nested: bool = True) -> dict:
        return await self.get(f"/events/{event_ticker}", with_nested_markets=str(nested).lower())

    async def market(self, ticker: str) -> dict:
        return (await self.get(f"/markets/{ticker}")).get("market", {})

    async def markets(self, tickers: list[str]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(tickers), 100):
            out.extend(await self.paginate("/markets", "markets", max_pages=3, tickers=",".join(tickers[i : i + 100]), limit=1000))
        return out

    async def collections(self, status: str = "open") -> list[dict]:
        return await self.paginate("/multivariate_event_collections", "multivariate_contracts", status=status, limit=200)

    async def create_combo(self, collection: str, legs: list[dict]) -> dict:
        """POST a combo. Counts against the 5,000/week creation cap even when the combo already exists."""
        body = {
            "selected_markets": [
                {"market_ticker": l["market_ticker"], "event_ticker": l["event_ticker"], "side": l["side"]} for l in legs
            ],
            "with_market_payload": False,
        }
        return await self.request("POST", f"/multivariate_event_collections/{collection}", json_body=body)

    # --- RFQs ------------------------------------------------------------------------
    async def create_rfq(self, market_ticker: str, *, target_cost: float, anonymous: bool) -> str:
        body = {
            "market_ticker": market_ticker,
            "target_cost_dollars": f"{target_cost:.2f}",
            "rest_remainder": False,
            "obscure_creator_id": bool(anonymous),
        }
        data = await self.request("POST", "/communications/rfqs", json_body=body)
        return data["id"]

    async def delete_rfq(self, rfq_id: str) -> None:
        try:
            await self.request("DELETE", f"/communications/rfqs/{rfq_id}")
        except KalshiError as e:
            if e.status == 404 or e.code in ("RFQ_CLOSED", "rfq_closed", "not_found"):
                return
            raise

    async def get_rfq(self, rfq_id: str) -> dict:
        return (await self.get(f"/communications/rfqs/{rfq_id}")).get("rfq", {})

    async def my_open_rfqs(self) -> list[dict]:
        return await self.paginate("/communications/rfqs", "rfqs", user_filter="self", status="open", limit=100)

    async def list_quotes(self, rfq_id: str) -> list[dict]:
        """Every quote on the RFQ (cancelled included), best price first on the RFQ's main side."""
        return await self.paginate("/communications/quotes", "quotes", rfq_id=rfq_id, limit=500)

    async def get_rfq_quote(self, rfq_id: str, quote_id: str) -> dict:
        return (await self.get(f"/communications/rfqs/{rfq_id}/quotes/{quote_id}")).get("quote", {})

    # --- portfolio -------------------------------------------------------------------
    async def fills(self, *, min_ts: int | None = None, ticker: str | None = None, order_id: str | None = None) -> list[dict]:
        return await self.paginate("/portfolio/fills", "fills", min_ts=min_ts, ticker=ticker, order_id=order_id, limit=1000)

    async def balance(self) -> dict:
        return await self.get("/portfolio/balance")

    async def settlements(self, *, min_ts: int | None = None) -> list[dict]:
        return await self.paginate("/portfolio/settlements", "settlements", min_ts=min_ts, limit=200)

    async def aclose(self) -> None:
        await self.transport.aclose()


def _clean(params: dict | None) -> dict | None:
    if not params:
        return None
    return {k: v for k, v in params.items() if v is not None and v != ""}


def _err_fields(resp: Response) -> tuple[str, str, str]:
    d = resp.data if isinstance(resp.data, dict) else {}
    err = d.get("error") if isinstance(d.get("error"), dict) else d
    if isinstance(d.get("error"), str):
        return (d.get("code") or "error", d["error"], d.get("details", ""))
    return (str(err.get("code", "")), str(err.get("message", resp.text[:300])), str(err.get("details", "")))


def _backoff(attempt: int) -> float:
    return min(8.0, 0.25 * 2**attempt) * (0.7 + 0.6 * random.random())
