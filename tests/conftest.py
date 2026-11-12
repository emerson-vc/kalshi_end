"""Shared fixtures. Network is blocked for every test; each test gets its own data dir and DB."""

from __future__ import annotations

import os
import socket

os.environ["KALSHI_ENV"] = "mock"  # before any combo_edge import reads .env
os.environ.pop("COMBO_EDGE_FORBID_LIVE", None)

import pytest  # noqa: E402

_LOCAL = {"127.0.0.1", "::1", "localhost", "testserver"}
_orig_connect = socket.socket.connect
_orig_getaddrinfo = socket.getaddrinfo


def _guard_connect(self, address):  # type: ignore[no-untyped-def]
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(address, (str, bytes)) or host in _LOCAL:
        return _orig_connect(self, address)
    raise RuntimeError(f"network access blocked in tests: {address!r}")


def _guard_getaddrinfo(host, *a, **kw):  # type: ignore[no-untyped-def]
    if host in _LOCAL or host is None:
        return _orig_getaddrinfo(host, *a, **kw)
    raise RuntimeError(f"DNS lookup blocked in tests: {host!r}")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(socket.socket, "connect", _guard_connect)
    monkeypatch.setattr(socket, "getaddrinfo", _guard_getaddrinfo)
    monkeypatch.setenv("COMBO_EDGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("KALSHI_ENV", "mock")
    from combo_edge.storage import db as dbmod
    from combo_edge.kalshi import session as sess

    dbmod.reset_default_db()
    sess.set_mock_exchange(None)
    yield
    dbmod.reset_default_db()
    sess.set_mock_exchange(None)


@pytest.fixture
def db(tmp_path):
    from combo_edge.storage.db import get_db

    return get_db()


@pytest.fixture
def settings(db):
    from combo_edge.settings import Settings

    return Settings(db)


@pytest.fixture
def exchange():
    from combo_edge.kalshi.mock.exchange import MockExchange
    from combo_edge.kalshi import session as sess

    ex = MockExchange(time_scale=0.05, seed=3)
    sess.set_mock_exchange(ex)
    return ex


@pytest.fixture
async def session(exchange):
    from combo_edge.kalshi.session import build_session
    from combo_edge.settings import env_config

    s = build_session(env_config("mock"), exchange=exchange)
    yield s
    await s.close()
