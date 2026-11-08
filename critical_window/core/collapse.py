"""Finite-size-scaling collapse quality for candidate scaling variables (X6, H3).

Each city contributes points (x, y): x = candidate window size (f, A, N or A / xi^2), y = log of the
normalised error. A common curve (cubic in log x) is fitted to all points; the collapse quality is
the share of the variance of y left unexplained (lower = better collapse).
"""
from __future__ import annotations

import numpy as np


def collapse_quality(curves, degree: int = 3) -> float:
    xs, ys = [], []
    for x, y in curves:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0)
        xs.append(np.log(x[m]))
        ys.append(y[m])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if x.size < degree + 3:
        return np.nan
    coef = np.polyfit(x, y, degree)
    res = y - np.polyval(coef, x)
    sst = ((y - y.mean()) ** 2).sum()
    return float((res ** 2).sum() / sst) if sst > 0 else np.nan
