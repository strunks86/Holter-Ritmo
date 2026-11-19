"""Dibujo de tiras de ECG y gráficos con Pillow (sin matplotlib).

Se usa para el visor (miniaturas), el informe y las pruebas visuales.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paleta del informe (fondo claro, imprimible)
INK = (24, 32, 40)
MUTED = (110, 122, 136)
GRID_MINOR = (236, 226, 226)
GRID_MAJOR = (214, 196, 196)
ACCENT = (176, 42, 55)
CLASS_COLORS = {
    "N": (40, 110, 200),
    "S": (0, 150, 136),
    "V": (200, 60, 50),
    "F": (150, 90, 190),
    "Q": (120, 120, 120),
}
LEAD_COLORS = [(20, 60, 120), (10, 110, 90), (120, 60, 10)]

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def font(size: int) -> ImageFont.ImageFont:
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.load_default(size=size)
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _text(draw, xy, s, size=14, fill=INK, anchor="la"):
    draw.text(xy, s, font=font(size), fill=fill, anchor=anchor)


# ---------------------------------------------------------------------------
# Tira de ECG
# ---------------------------------------------------------------------------
def ecg_strip(sig: np.ndarray, fs: float, t0: float, width: int = 1800,
              height: int = 560, leads: list[str] | None = None,
              beats: list | None = None, scale: float = 10.0,
              title: str = "", paper: bool = True) -> Image.Image:
    """Dibuja una tira multi-derivación con rejilla de ECG.

    sig: (n, L) en mV para el intervalo [t0, t0 + n/fs].
    scale: mm por mV (10 mm/mV es lo estándar).
    """
    ss = 2  # supersampling
    W, H = width * ss, height * ss
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[:, None]
    n, L = sig.shape
    if leads is None:
        leads = [f"Canal {i + 1}" for i in range(L)]
    if n == 0:
        _text(d, (20, 20), "Sin señal", 20, MUTED)
        return img.resize((width, height), Image.LANCZOS)

    margin_l, margin_r, margin_t, margin_b = 64 * ss, 16 * ss, 34 * ss, 26 * ss
    if title:
        _text(d, (margin_l, 10 * ss), title, 15 * ss, INK)
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b
    band_h = plot_h / L
    dur = n / fs

    # Rejilla: 1 mm = 0.04 s horizontal y 0.1 mV vertical (25 mm/s, 10 mm/mV)
    px_per_s = plot_w / dur if dur > 0 else 1
    mm_px = px_per_s * 0.04
    if paper and mm_px >= 3 * ss:
        x = margin_l
        k = 0
        while x <= W - margin_r + 0.1:
            col = GRID_MAJOR if k % 5 == 0 else GRID_MINOR
            d.line([(x, margin_t), (x, H - margin_b)], fill=col, width=1)
            x += mm_px
            k += 1
    if paper:
        for bi in range(L):
            top = margin_t + bi * band_h
            mid = top + band_h * 0.62
            mmv = (band_h * 0.62) / 2.0  # px por 0.5 mV aproximadamente
            step = mmv / 5.0
            if step >= 2 * ss:
                k = 0
                y = mid
                while y > top:
                    col = GRID_MAJOR if k % 5 == 0 else GRID_MINOR
                    d.line([(margin_l, y), (W - margin_r, y)], fill=col, width=1)
                    y -= step
                    k += 1
                k = 0
                y = mid
                while y < top + band_h:
                    col = GRID_MAJOR if k % 5 == 0 else GRID_MINOR
                    d.line([(margin_l, y), (W - margin_r, y)], fill=col, width=1)
                    y += step
                    k += 1

    t = t0 + np.arange(n) / fs
    for bi in range(L):
        top = margin_t + bi * band_h
        mid = top + band_h * 0.62
        # 10 mm/mV
        px_per_mv = (band_h * 0.62) / 2.0 * scale / 5.0
        y = mid - (sig[:, bi] - np.median(sig[:, bi])) * px_per_mv
        y = np.clip(y, top + 2 * ss, top + band_h - 2 * ss)
        xs = margin_l + (t - t0) * px_per_s
        pts = list(zip(xs.tolist(), y.tolist()))
        d.line(pts, fill=LEAD_COLORS[bi % len(LEAD_COLORS)], width=max(1, ss), joint="curve")
        _text(d, (8 * ss, top + 4 * ss), str(leads[bi])[:12], 13 * ss, LEAD_COLORS[bi % 3])

    # Marcas de latidos
    if beats:
        for b in beats:
            idx = b["idx"] if isinstance(b, dict) else b.idx
            cls = b["cls"] if isinstance(b, dict) else b.cls
            bt = idx / fs
            if bt < t0 or bt > t0 + dur:
                continue
            x = margin_l + (bt - t0) * px_per_s
            color = CLASS_COLORS.get(cls, INK)
            d.line([(x, margin_t), (x, H - margin_b)], fill=color, width=max(1, ss))
            _text(d, (x, margin_t - 12 * ss), cls, 12 * ss, color, anchor="ma")

    # Barra de calibración 1 mV / 1 s
    if paper:
        x0 = W - margin_r - 120 * ss
        y0 = H - margin_b + 8 * ss
        d.line([(x0, y0), (x0 + 0.04 * px_per_s * 5, y0)], fill=INK, width=2 * ss)
        _text(d, (x0, y0 - 14 * ss), "0.2 s", 11 * ss, MUTED)
        d.line([(x0 + 130 * ss, y0), (x0 + 130 * ss, y0 - px_per_mv)], fill=INK, width=2 * ss)
        _text(d, (x0 + 140 * ss, y0 - px_per_mv / 2), "1 mV", 11 * ss, MUTED)

    # Eje de tiempo
    total_s = dur
    step = _nice_step(total_s / 8)
    tt = math.ceil(t0 / step) * step
    while tt <= t0 + total_s + 1e-9:
        x = margin_l + (tt - t0) * px_per_s
        _text(d, (x, H - margin_b + 8 * ss), _fmt_time(tt), 11 * ss, MUTED, anchor="ma")
        tt += step

    return img.resize((width, height), Image.LANCZOS)


def _nice_step(rough: float) -> float:
    if rough <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(rough))
    for m in (1, 2, 2.5, 5, 10):
        if rough <= m * mag:
            return m * mag
    return 10 * mag


def _fmt_time(s: float) -> str:
    s = max(0.0, float(s))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def trend_chart(x: np.ndarray, y: np.ndarray, width=1200, height=260,
                title="", ylabel="", color=(20, 60, 120),
                band: tuple[float, float] | None = None) -> Image.Image:
    ss = 2
    W, H = width * ss, height * ss
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ml, mr, mt, mb = 56 * ss, 18 * ss, 34 * ss, 30 * ss
    pw, ph = W - ml - mr, H - mt - mb
    if title:
        _text(d, (ml, 8 * ss), title, 14 * ss, INK)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0:
        return img.resize((width, height), Image.LANCZOS)
    y0 = float(np.nanmin(y)) if np.isfinite(y).any() else 0.0
    y1 = float(np.nanmax(y)) if np.isfinite(y).any() else 1.0
    if band:
        y0 = min(y0, band[0])
        y1 = max(y1, band[1])
    pad = (y1 - y0) * 0.08 or 1.0
    y0 -= pad
    y1 += pad
    x0, x1 = float(x.min()), float(x.max())
    if x1 <= x0:
        x1 = x0 + 1

    def px(v):
        return ml + (v - x0) / (x1 - x0) * pw

    def py(v):
        return mt + ph - (v - y0) / (y1 - y0) * ph

    for k in range(5):
        gv = y0 + (y1 - y0) * k / 4
        yy = py(gv)
        d.line([(ml, yy), (W - mr, yy)], fill=(228, 232, 236), width=1)
        _text(d, (ml - 8 * ss, yy), f"{gv:.0f}", 11 * ss, MUTED, anchor="rm")
    if band:
        d.rectangle([ml, py(band[1]), W - mr, py(band[0])], outline=(210, 120, 120), width=ss)
    pts = [(px(a), py(b)) for a, b in zip(x, y) if np.isfinite(b)]
    if len(pts) > 1:
        d.line(pts, fill=color, width=max(1, int(1.6 * ss)), joint="curve")
    # eje x
    for k in range(5):
        xv = x0 + (x1 - x0) * k / 4
        _text(d, (px(xv), H - mb + 8 * ss), _fmt_time(xv), 10 * ss, MUTED, anchor="ma")
    if ylabel:
        _text(d, (W - mr, mt - 6 * ss), ylabel, 10 * ss, MUTED, anchor="ra")
    return img.resize((width, height), Image.LANCZOS)


def scatter_chart(x: np.ndarray, y: np.ndarray, width=900, height=320,
                  title="", xlabel="", ylabel="", color=(20, 60, 120)) -> Image.Image:
    ss = 2
    W, H = width * ss, height * ss
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ml, mr, mt, mb = 56 * ss, 20 * ss, 34 * ss, 34 * ss
    pw, ph = W - ml - mr, H - mt - mb
    if title:
        _text(d, (ml, 8 * ss), title, 14 * ss, INK)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return img.resize((width, height), Image.LANCZOS)
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    for k in range(5):
        gv = y0 + (y1 - y0) * k / 4
        yy = mt + ph - (gv - y0) / (y1 - y0) * ph
        d.line([(ml, yy), (W - mr, yy)], fill=(228, 232, 236), width=1)
        _text(d, (ml - 8 * ss, yy), f"{gv:.2f}", 11 * ss, MUTED, anchor="rm")
    for a, b in zip(x, y):
        cx = ml + (a - x0) / (x1 - x0) * pw
        cy = mt + ph - (b - y0) / (y1 - y0) * ph
        r = 2.2 * ss
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    if xlabel:
        _text(d, (ml + pw / 2, H - 10 * ss), xlabel, 11 * ss, MUTED, anchor="mm")
    if ylabel:
        _text(d, (W - mr, mt - 6 * ss), ylabel, 10 * ss, MUTED, anchor="ra")
    return img.resize((width, height), Image.LANCZOS)


def bar_chart(labels: list[str], values: list[float], width=700, height=260,
              title="", colors: list | None = None) -> Image.Image:
    ss = 2
    W, H = width * ss, height * ss
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ml, mr, mt, mb = 20 * ss, 20 * ss, 40 * ss, 40 * ss
    pw, ph = W - ml - mr, H - mt - mb
    if title:
        _text(d, (ml, 10 * ss), title, 14 * ss, INK)
    vmax = max(values) if values else 1
    vmax = max(vmax, 1)
    bw = pw / max(1, len(values)) * 0.62
    colors = colors or [ACCENT] * len(values)
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = ml + pw * (i + 0.5) / len(values)
        h = v / vmax * (ph - 22 * ss)
        d.rectangle([cx - bw / 2, mt + ph - h, cx + bw / 2, mt + ph], fill=colors[i % len(colors)])
        _text(d, (cx, mt + ph - h - 4 * ss), f"{v:,.0f}", 12 * ss, INK, anchor="ms")
        _text(d, (cx, mt + ph + 6 * ss), lab, 12 * ss, MUTED, anchor="ma")
    d.line([(ml, mt + ph), (W - mr, mt + ph)], fill=(150, 150, 150), width=ss)
    return img.resize((width, height), Image.LANCZOS)


def save_png(img: Image.Image, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p, format="PNG", optimize=True)
    return p

