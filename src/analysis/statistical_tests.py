"""Statistical tests: bootstrap CIs, permutation, KS.

These are the lightweight tests reported alongside ``alpha`` and ``beta``.
All randomness is controlled by a single ``rng`` argument so the pipeline
is bit-reproducible from a fixed seed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import ks_2samp, pearsonr

from .scaling_fit import fit_local_power_law


# ---------------------------------------------------------------------------
# Bootstrap CIs for CV-H correlation
# ---------------------------------------------------------------------------

def bootstrap_correlation_ci(
    cv: Sequence[float],
    H: Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, np.ndarray]:
    """Percentile bootstrap CI for the Pearson correlation rho(CV, H)."""
    rng = rng or np.random.default_rng()
    cv = np.asarray(cv, dtype=float)
    H = np.asarray(H, dtype=float)
    valid = np.isfinite(cv) & np.isfinite(H)
    cv = cv[valid]
    H = H[valid]
    if cv.size < 3:
        return float("nan"), float("nan"), np.array([])

    samples = np.empty(n_bootstrap)
    n = cv.size
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if cv[idx].std() == 0 or H[idx].std() == 0:
            samples[i] = 0.0
            continue
        rho, _ = pearsonr(cv[idx], H[idx])
        samples[i] = rho
    alpha = 1.0 - confidence
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi, samples


# ---------------------------------------------------------------------------
# Bootstrap CIs for scaling exponents alpha, beta
# ---------------------------------------------------------------------------

def bootstrap_scaling_exponents(
    grid_sizes: Sequence[float],
    cv_at_hour: Sequence[float],
    H_at_hour: Sequence[float],
    fit_min_l: float = 200.0,
    fit_max_l: float = 1200.0,
    max_valid_l: Optional[float] = None,
    min_points: int = 4,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Bootstrap CIs for ``alpha`` and ``beta`` by resampling scales."""
    rng = rng or np.random.default_rng()
    L = np.asarray(grid_sizes, dtype=float)
    cv = np.asarray(cv_at_hour, dtype=float)
    H = np.asarray(H_at_hour, dtype=float)
    n = L.size

    alpha = np.full(n_bootstrap, np.nan)
    beta = np.full(n_bootstrap, np.nan)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a = fit_local_power_law(L[idx], cv[idx], fit_min_l, fit_max_l,
                                max_valid_l=max_valid_l, min_points=min_points)
        b = fit_local_power_law(L[idx], H[idx], fit_min_l, fit_max_l,
                                max_valid_l=max_valid_l, min_points=min_points)
        if a["ok"]:
            alpha[i] = a["exponent"]
        if b["ok"]:
            beta[i] = b["exponent"]

    def _ci(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return (float("nan"), float("nan"))
        lo = float(np.percentile(arr, 100 * (1 - confidence) / 2))
        hi = float(np.percentile(arr, 100 * (1 - (1 - confidence) / 2)))
        return (lo, hi)

    return {"alpha_ci": _ci(alpha), "beta_ci": _ci(beta),
            "n_valid_alpha": int(np.isfinite(alpha).sum()),
            "n_valid_beta": int(np.isfinite(beta).sum())}


# ---------------------------------------------------------------------------
# Permutation test: optimal scale difference
# ---------------------------------------------------------------------------

def permutation_test_optimal_scale(
    grid_sizes: Sequence[float],
    correlations_a: Sequence[float],
    correlations_b: Sequence[float],
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Permutation test for ``|L*_a - L*_b|`` under H0 of label exchangeability."""
    rng = rng or np.random.default_rng()
    L = np.asarray(grid_sizes, dtype=float)
    ca = np.asarray(correlations_a, dtype=float)
    cb = np.asarray(correlations_b, dtype=float)
    if L.size != ca.size or L.size != cb.size:
        raise ValueError("grid_sizes and correlation arrays must align")

    L_star_a = float(L[np.argmax(np.abs(ca))])
    L_star_b = float(L[np.argmax(np.abs(cb))])
    observed = abs(L_star_a - L_star_b)

    null = np.empty(n_permutations)
    for i in range(n_permutations):
        if rng.random() < 0.5:
            ca2, cb2 = cb, ca
        else:
            ca2, cb2 = ca, cb
        null[i] = abs(L[np.argmax(np.abs(ca2))] - L[np.argmax(np.abs(cb2))])
    p = float(np.mean(null >= observed))
    return {"observed_diff_m": float(observed), "p_value": p,
            "L_star_a_m": L_star_a, "L_star_b_m": L_star_b}


# ---------------------------------------------------------------------------
# KS test on rho(L)
# ---------------------------------------------------------------------------

def ks_test_correlation_curves(
    correlations_a: Sequence[float],
    correlations_b: Sequence[float],
) -> dict:
    """Two-sample Kolmogorov-Smirnov test on rho(L) curves."""
    stat, p = ks_2samp(np.asarray(correlations_a), np.asarray(correlations_b))
    return {"ks_statistic": float(stat), "p_value": float(p)}
