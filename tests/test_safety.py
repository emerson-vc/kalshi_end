"""Build-breaking safety checks.

* The accept endpoint (and minting the accept capability) appears only in the execution package —
  plus the mock exchange, which is the simulated *server* and has to route it.
* No resting-order, order-group, block-trade, batch-order, RFQ-batch, quote-create or confirm
  endpoint appears anywhere in the source.
* Every RFQ is created with rest_remainder: false.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "combo_edge"
EXECUTION = SRC / "execution"
MOCK_SERVER = SRC / "kalshi" / "mock" / "exchange.py"

FORBIDDEN = [
    r"/portfolio/orders", r"/portfolio/events/orders", r"order_groups", r"block-trade", r"block_trade",
    r"/batched", r"/confirm\b", r"rfq_batches", r"rfq-batches", r"/communications/quotes/[^\"'\s]*accept",
]


def _py_files():
    return [p for p in SRC.rglob("*.py")]


def _text_files():
    return [p for p in SRC.rglob("*") if p.is_file() and p.suffix in (".py", ".html", ".js", ".yaml", ".json") and "vendor" not in p.parts]


def test_accept_path_only_in_execution_module():
    offenders = []
    for p in _py_files():
        if EXECUTION in p.parents or p == MOCK_SERVER:
            continue
        t = p.read_text()
        if re.search(r"/accept\b", t) or "accept_path" in t.replace("is_accept_path", "") or "_mint" in t:
            offenders.append(str(p.relative_to(SRC)))
    assert offenders == [], f"accept path / capability minting outside execution: {offenders}"


def test_capability_minted_only_by_executor():
    users = [p for p in _py_files() if "_mint(" in p.read_text()]
    assert {p.name for p in users} <= {"capability.py", "executor.py"}, users


def test_no_forbidden_endpoints_anywhere():
    hits = []
    for p in _text_files():
        t = p.read_text(errors="ignore")
        for pat in FORBIDDEN:
            if re.search(pat, t):
                hits.append(f"{p.relative_to(SRC)}: {pat}")
    assert hits == [], hits


def test_no_quote_creation_calls():
    """No code POSTs to /communications/quotes (that is the maker-side quote-create endpoint)."""
    for p in _py_files():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and len(node.args) >= 2:
                a0, a1 = node.args[0], node.args[1]
                if isinstance(a0, ast.Constant) and a0.value == "POST" and isinstance(a1, (ast.Constant, ast.JoinedStr)):
                    s = a1.value if isinstance(a1, ast.Constant) else ast.unparse(a1)
                    assert "/communications/quotes" not in str(s), f"quote creation in {p}"


def test_rfqs_never_rest_remainder():
    t = (SRC / "kalshi" / "client.py").read_text()
    assert '"rest_remainder": False' in t
    for p in _py_files():
        assert '"rest_remainder": True' not in p.read_text()


def test_client_refuses_accept_without_capability():
    import pytest

    from combo_edge.kalshi.client import AcceptRefused, check_endpoint

    with pytest.raises(AcceptRefused):
        check_endpoint("PUT", "/communications/rfqs/a/quotes/b/accept")
