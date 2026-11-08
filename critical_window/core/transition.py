"""H2: smooth sampling curve vs a change of slope (threshold), calibrated against matched simulations (N2).

Per curve (one city, one probe): y = log(error statistic) at each usable coverage f, u = log(f / (1 - f)).
  M0 (smooth):        y = a + b u
  M2 (broken stick):  y = a + b1 u + b2 max(0, u - u_b), u_b searched over interior levels
dBIC = BIC(M0) - BIC(M2). Its null distribution comes from N2 (same layout, windows and probe; stationary
field with the city's sigma_z and xi_hat); the calibrated p-value replaces any fixed dBIC cut-off.
"""
from __future__ import annotations

import numpy as np


def _bic(rss: float, n: int, k: int) -> float:
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def dbic(f, e, min_side: int = 2):
    """Return (dBIC, breakpoint f_b) for one curve; NaN if fewer than 6 usable levels."""
    f = np.asarray(f, float)
    e = np.asarray(e, float)
    ok = np.isfinite(e) & (e > 0) & (f < 1) & (f > 0)
    f, y = f[ok], np.log(e[ok])
    n = f.size
    if n < 6:
        return np.nan, np.nan
    u = np.log(f / (1 - f))
    X0 = np.column_stack([np.ones(n), u])
    c0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    rss0 = float(((y - X0 @ c0) ** 2).sum())
    best = (np.inf, np.nan)
    for ub in np.sort(u)[min_side:-min_side]:
        X2 = np.column_stack([X0, np.maximum(0, u - ub)])
        c2, *_ = np.linalg.lstsq(X2, y, rcond=None)
        rss2 = float(((y - X2 @ c2) ** 2).sum())
        if rss2 < best[0]:
            best = (rss2, ub)
    fb = 1 / (1 + np.exp(-best[1]))
    return _bic(rss0, n, 2) - _bic(best[0], n, 4), float(fb)
