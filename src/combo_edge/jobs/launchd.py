"""launchd agents (~/Library/LaunchAgents):

* recorder  — 24/7 read-only feed recorder (KeepAlive)
* requote   — every 10 minutes; ``combo-edge requote-due`` decides what's due (also runs the weekly fade check)
* dailyscan — every 15 minutes; ``combo-edge dailyscan-tick`` runs today's scan once, at the learned best time

Optional, off by default: a ``pmset`` wake schedule before the daily scan (needs sudo and power).
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from combo_edge import paths

LABEL = "com.comboedge.{job}"
JOBS = {
    "recorder": {"args": ["recorder"], "keepalive": True},
    "requote": {"args": ["requote-due"], "interval": 600},
    "dailyscan": {"args": ["dailyscan-tick"], "interval": 900},
}


def agents_dir() -> Path:
    d = Path.home() / "Library" / "LaunchAgents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plist_path(job: str) -> Path:
    return agents_dir() / f"{LABEL.format(job=job)}.plist"


def program_args(job: str) -> list[str]:
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    return [uv, "run", "--project", str(paths.REPO_ROOT), "combo-edge", *JOBS[job]["args"]]


def build_plist(job: str) -> dict:
    spec = JOBS[job]
    logs = paths.logs_dir()
    d = {
        "Label": LABEL.format(job=job),
        "ProgramArguments": program_args(job),
        "WorkingDirectory": str(paths.REPO_ROOT),
        "StandardOutPath": str(logs / f"{job}.out.log"),
        "StandardErrorPath": str(logs / f"{job}.err.log"),
        "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"), "PYTHONUNBUFFERED": "1"},
        "ProcessType": "Background",
    }
    if spec.get("keepalive"):
        d["KeepAlive"] = True
        d["RunAtLoad"] = True
        d["ThrottleInterval"] = 30
    else:
        d["StartInterval"] = spec["interval"]
        d["RunAtLoad"] = True
    return d


def install(job: str) -> str:
    if job not in JOBS:
        raise ValueError(f"unknown job {job}; choose from {', '.join(JOBS)}")
    p = plist_path(job)
    if p.exists():
        uninstall(job)
    with open(p, "wb") as f:
        plistlib.dump(build_plist(job), f)
    r = subprocess.run(["launchctl", "load", "-w", str(p)], capture_output=True, text=True)
    return f"installed {p.name}" + (f" (launchctl: {r.stderr.strip()})" if r.returncode else "")


def uninstall(job: str) -> str:
    p = plist_path(job)
    if not p.exists():
        return f"{job} not installed"
    subprocess.run(["launchctl", "unload", "-w", str(p)], capture_output=True, text=True)
    p.unlink(missing_ok=True)
    return f"uninstalled {p.name}"


def status(job: str) -> dict:
    p = plist_path(job)
    if not p.exists():
        return {"job": job, "installed": False, "running": False}
    r = subprocess.run(["launchctl", "list", LABEL.format(job=job)], capture_output=True, text=True)
    running = r.returncode == 0 and '"PID"' in r.stdout
    return {"job": job, "installed": True, "loaded": r.returncode == 0, "running": running}


def wake_command(hour: int, minute: int) -> str:
    """The pmset command for the optional wake schedule (needs sudo; we never run sudo ourselves)."""
    return f"sudo pmset repeat wakeorpoweron MTWRFSU {hour:02d}:{minute:02d}:00"


def wake_cancel_command() -> str:
    return "sudo pmset repeat cancel"


def python_exe() -> str:
    return sys.executable
