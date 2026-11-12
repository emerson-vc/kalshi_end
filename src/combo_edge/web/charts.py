"""Server-rendered SVG charts (no JS charting library). Colors come from CSS variables so they
follow the light/dark theme."""

from __future__ import annotations

from datetime import datetime
from html import escape


def line_chart(points: list[tuple[float, float]], *, height: int = 180, money: bool = True) -> str:
    if len(points) < 2:
        return '<div class="empty">Not enough settled bets to draw a chart yet.</div>'
    w, h, pad = 640, height, 32
    xs = [p[0] for p in points]
    ys = [p[1] for p in points] + [0.0]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    if y1 == y0:
        y1 = y0 + 1
    sx = lambda x: pad + (x - x0) / ((x1 - x0) or 1) * (w - 2 * pad)  # noqa: E731
    sy = lambda y: h - pad - (y - y0) / (y1 - y0) * (h - 2 * pad)  # noqa: E731
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points))
    last = points[-1][1]
    color = "var(--pos)" if last >= 0 else "var(--neg)"
    fmt = (lambda v: f"${v:,.0f}") if money else (lambda v: f"{v:.2f}")
    return (
        f'<svg viewBox="0 0 {w} {h}" class="w-full h-auto" role="img" aria-label="Cumulative P&L">'
        f'<line x1="{pad}" x2="{w - pad}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="var(--grid)" stroke-dasharray="4 4"/>'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>'
        f'<text x="{pad}" y="14" class="chart-label">{escape(fmt(y1))}</text>'
        f'<text x="{pad}" y="{h - 8}" class="chart-label">{escape(fmt(y0))}</text>'
        f'<text x="{w - pad}" y="{h - 8}" text-anchor="end" class="chart-label">{datetime.fromtimestamp(x1).strftime("%b %-d")}</text>'
        f'<circle cx="{sx(points[-1][0]):.1f}" cy="{sy(last):.1f}" r="4" fill="{color}"/></svg>'
    )


def bar_chart(items: list[tuple[str, float | None]], *, pct: bool = True, height: int = 160) -> str:
    items = [(k, v) for k, v in items if v is not None]
    if not items:
        return '<div class="empty">No data yet.</div>'
    w, h, pad = 640, height, 28
    m = max(abs(v) for _, v in items) or 1
    bw = (w - 2 * pad) / len(items)
    mid = h / 2
    out = [f'<svg viewBox="0 0 {w} {h}" class="w-full h-auto" role="img">',
           f'<line x1="{pad}" x2="{w - pad}" y1="{mid}" y2="{mid}" stroke="var(--grid)"/>']
    for i, (k, v) in enumerate(items):
        bh = abs(v) / m * (h / 2 - pad)
        x = pad + i * bw + bw * 0.15
        y = mid - bh if v >= 0 else mid
        color = "var(--pos)" if v >= 0 else "var(--neg)"
        label = f"{v * 100:+.0f}%" if pct else f"{v:+.2f}"
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw * 0.7:.1f}" height="{max(bh, 1):.1f}" rx="3" fill="{color}"/>')
        out.append(f'<text x="{x + bw * 0.35:.1f}" y="{(y - 4) if v >= 0 else (y + bh + 12):.1f}" text-anchor="middle" class="chart-label">{escape(label)}</text>')
        out.append(f'<text x="{x + bw * 0.35:.1f}" y="{h - 4}" text-anchor="middle" class="chart-label">{escape(k[:10])}</text>')
    out.append("</svg>")
    return "".join(out)
