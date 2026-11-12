"""Kill switch: dashboard button, ``combo-edge kill`` or ``touch data/KILL``.

It sets execution off, stops the running scan (scans poll ``is_killed``), cancels every open RFQ
this tool created, and pauses the schedulers. ``resume`` clears it; execution stays off until you
turn it back on.
"""

from __future__ import annotations

import time
from typing import Any

from combo_edge import paths
from combo_edge.storage.db import DB


def is_killed(db: DB | None = None) -> bool:
    if paths.kill_file().exists():
        return True
    return bool(db.get_state("killed", False)) if db is not None else False


def apply_kill_state(db: DB, reason: str) -> None:
    db.set_state("execution_mode", "off")
    db.set_state("killed", True)
    db.set_state("killed_reason", reason)
    db.set_state("killed_ts", time.time())
    db.set_state("schedulers_paused", True)
    try:
        paths.kill_file().write_text(reason + "\n")
    except OSError:
        pass
    db.log("error", "kill", f"kill switch: {reason}")


async def kill(db: DB, session: Any | None, reason: str = "kill switch") -> dict:
    apply_kill_state(db, reason)
    result = {"execution": "off", "schedulers": "paused", "cancelled": 0, "failed": 0}
    if session is not None:
        from combo_edge.pipeline.quotes import sweep_pending_cancels

        result.update(await sweep_pending_cancels(session, db))
    return result


def resume(db: DB) -> None:
    try:
        paths.kill_file().unlink()
    except FileNotFoundError:
        pass
    db.set_state("killed", False)
    db.set_state("schedulers_paused", False)
    db.log("info", "kill", "kill switch cleared (execution stays off)")
