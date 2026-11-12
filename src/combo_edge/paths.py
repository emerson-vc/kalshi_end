"""Filesystem locations. Everything mutable lives under ``data/`` (gitignored at the repo root only)."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("COMBO_EDGE_HOME", PACKAGE_DIR.parent.parent)).resolve()
CONFIG_DIR = REPO_ROOT / "config"
CORE_CONFIG = CONFIG_DIR / "core.yaml"
MOCK_FIXTURES = PACKAGE_DIR / "kalshi" / "mock" / "fixtures"


def data_dir() -> Path:
    d = Path(os.environ.get("COMBO_EDGE_DATA_DIR", REPO_ROOT / "data")).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "combo_edge.sqlite3"


def kill_file() -> Path:
    return data_dir() / "KILL"


def nflverse_dir() -> Path:
    d = data_dir() / "nflverse"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
