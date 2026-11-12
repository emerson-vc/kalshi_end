"""Build the stylesheet with the Tailwind standalone CLI (pytailwindcss; no Node).
The built app.css is committed, so the dashboard works without running this."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
TAILWIND_VERSION = "v4.3.5"


def build() -> str:
    src = HERE / "tailwind" / "input.css"
    out = HERE / "static" / "app.css"
    env = {**os.environ, "TAILWINDCSS_VERSION": TAILWIND_VERSION}
    r = subprocess.run(["tailwindcss", "-i", str(src), "-o", str(out), "--minify"], cwd=HERE, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return f"built {out} ({out.stat().st_size // 1024} KB)"
