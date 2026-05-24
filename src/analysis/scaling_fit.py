"""Scaling-law fits for ``CV(L) ~ L^alpha`` and ``H(L) ~ L^beta``.

Uses a *local* log-log linear fit restricted to L in ``[fit_min, fit_max]``,
optionally further capped by a window-dependent ``max_valid_L`` to avoid
finite-size artefacts.  This matches the published ``local_lc400_[200-1200m]``
configuration described in the paper.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def fit_local_power_law(
    grid_sizes: Sequence[float],
    values: Sequence[float],
    fit_min_l: float,
    fit_max_l: float,
    max_valid_l: Optional[float] = None,
    min_points: int = 4,
) -> dict:
    """Linear fit of ``log(values) ~ log(L)`` on the local L-window.

    Returns a dict with ``exponent``, ``intercept``, ``r_squared``, ``n_points``.
    Returns NaNs (and ``ok=False``) when fewer than ``min_points`` valid scales
    fall inside the fitting window.
    """
    L = np.asarray(grid_sizes, dtype=float)
    y = np.asarray(values, dtype=float)
    upper = fit_max_l if max_valid_l is None else min(fit_max_l, max_valid_l)

    mask = (np.isfinite(L) & np.isfinite(y) & (y > 0) & (L > 0)
            & (L >= fit_min_l) & (L <= upper))
    n = int(mask.sum())
    if n < min_points:
        return {"exponent": np.nan, "intercept": np.nan, "r_squared": np.nan,
                "n_points": n, "fit_min_l": float(fit_min_l),
                "fit_max_l": float(upper), "ok": False}

    log_L = np.log(L[mask])
    log_y = np.log(y[mask])
    slope, intercept = np.polyfit(log_L, log_y, 1)
    pred = slope * log_L + intercept
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "exponent": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "n_points": n,
        "fit_min_l": float(fit_min_l),
        "fit_max_l": float(upper),
        "ok": True,
    }


def compute_alpha_beta_per_hour(
    multi_scale: dict,
    fit_min_l: float = 200.0,
    fit_max_l: float = 1200.0,
    max_valid_l: Optional[float] = None,
    min_points: int = 4,
    hours: Optional[Sequence[int]] = None,
) -> dict:
    """Per-hour fits of ``CV(L) ~ L^alpha`` and ``H(L) ~ L^beta``.

    Parameters
    ----------
    multi_scale : dict
        Output of :func:`analyse_multi_scale`.
    hours : sequence of int, optional
        Hours to evaluate; defaults to all 24.
    """
    if hours is None:
        hours = list(range(24))
    grid_sizes = np.asarray(multi_scale["grid_sizes"], dtype=float)
    cv_by_scale = np.asarray(multi_scale["cv_by_scale"], dtype=float)
    h_by_scale = np.asarray(multi_scale["entropy_by_scale"], dtype=float)

    alpha_by_hour: dict = {}
    beta_by_hour: dict = {}
    for h in hours:
        cv_h = cv_by_scale[:, h]
        H_h = h_by_scale[:, h]
        alpha_by_hour[h] = fit_local_power_law(
            grid_sizes, cv_h,
            fit_min_l=fit_min_l, fit_max_l=fit_max_l,
            max_valid_l=max_valid_l, min_points=min_points,
        )
        beta_by_hour[h] = fit_local_power_law(
            grid_sizes, H_h,
            fit_min_l=fit_min_l, fit_max_l=fit_max_l,
            max_valid_l=max_valid_l, min_points=min_points,
        )
    return {"alpha_by_hour": alpha_by_hour, "beta_by_hour": beta_by_hour}
