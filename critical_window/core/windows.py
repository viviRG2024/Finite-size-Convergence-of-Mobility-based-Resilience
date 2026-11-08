"""Window design and the geometry null N4.

Windows are rectangles with the aspect ratio of the extent; coverage f = window area / extent area,
so the side scale is sqrt(f). Random windows have centres uniform on the feasible region (the whole
window inside the extent).
"""
from __future__ import annotations

import numpy as np

F_GRID = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                   0.60, 0.70, 0.80, 0.90, 1.0])


def random_windows(ext, f: float, n: int, rng) -> np.ndarray:
    x0, y0, x1, y1 = ext
    s = np.sqrt(f)
    w, h = (x1 - x0) * s, (y1 - y0) * s
    cx = rng.uniform(x0 + w / 2, x1 - w / 2, n) if x1 - x0 > w else np.full(n, (x0 + x1) / 2)
    cy = rng.uniform(y0 + h / 2, y1 - h / 2, n) if y1 - y0 > h else np.full(n, (y0 + y1) / 2)
    return np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def nested_window(ext, f: float) -> np.ndarray:
    x0, y0, x1, y1 = ext
    s = np.sqrt(f)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * s, (y1 - y0) * s
    return np.array([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]])


def tiling(ext, f: float) -> np.ndarray:
    """Non-overlapping windows of coverage f on a regular lattice (m x m, m = floor(1/sqrt f)), centred."""
    m = int(np.floor(1 / np.sqrt(f) + 1e-9))
    if m < 2:
        return np.zeros((0, 4))
    x0, y0, x1, y1 = ext
    s = np.sqrt(f)
    w, h = (x1 - x0) * s, (y1 - y0) * s
    ox = x0 + ((x1 - x0) - m * w) / 2
    oy = y0 + ((y1 - y0) - m * h) / 2
    out = [[ox + i * w, oy + j * h, ox + (i + 1) * w, oy + (j + 1) * h] for i in range(m) for j in range(m)]
    return np.array(out)


def in_box(x, y, b) -> np.ndarray:
    return (x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3])


def counts(x, y, wins) -> np.ndarray:
    x = np.asarray(x)[None, :]
    y = np.asarray(y)[None, :]
    w = np.asarray(wins)
    return ((x >= w[:, [0]]) & (x <= w[:, [2]]) & (y >= w[:, [1]]) & (y <= w[:, [3]])).sum(1)


def mean_pairwise_overlap(wins) -> float:
    w = np.asarray(wins)
    if len(w) < 2:
        return np.nan
    ix = np.clip(np.minimum(w[:, None, 2], w[None, :, 2]) - np.maximum(w[:, None, 0], w[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(w[:, None, 3], w[None, :, 3]) - np.maximum(w[:, None, 1], w[None, :, 1]), 0, None)
    area = (w[:, 2] - w[:, 0]) * (w[:, 3] - w[:, 1])
    ov = ix * iy / area[:, None]
    iu = np.triu_indices(len(w), 1)
    return float(ov[iu].mean())


def expected_overlap(f: float, n_grid: int = 20001) -> float:
    """N4: expected overlap share of two independent uniformly placed windows of coverage f."""
    s = np.sqrt(f)
    D = 1.0 - s                       # feasible centre range per axis (extent side = 1)
    if D <= 1e-12:
        return 1.0
    d = np.linspace(0.0, D, n_grid)
    dens = 2.0 * (D - d) / D ** 2
    per_axis = np.trapezoid(np.clip(s - d, 0, None) * dens, d) / s if hasattr(np, "trapezoid") else \
        np.trapz(np.clip(s - d, 0, None) * dens, d) / s
    return float(per_axis ** 2)


def p_contains(ext, f: float, point) -> float:
    """N4: probability that a uniformly placed window of coverage f contains the given point."""
    x0, y0, x1, y1 = ext
    s = np.sqrt(f)
    p = 1.0
    for lo, hi, c in ((x0, x1, point[0]), (y0, y1, point[1])):
        L = hi - lo
        l = L * s
        if L - l <= 1e-9:
            p *= 1.0 if lo <= c <= hi else 0.0
            continue
        a_lo, a_hi = max(lo, c - l), min(c, hi - l)
        p *= max(0.0, a_hi - a_lo) / (L - l)
    return float(p)
