"""Sufficiency criterion: representativeness error, tolerance and critical window."""
from __future__ import annotations

import numpy as np


def p90(errors) -> float:
    e = np.asarray(errors, float)
    e = e[np.isfinite(e)]
    return float(np.quantile(e, 0.9)) if e.size else np.nan


def tolerance(theta_extent_by_city, frac: float = 0.25) -> float:
    v = np.asarray(theta_extent_by_city, float)
    v = v[np.isfinite(v)]
    q1, q3 = np.quantile(v, [0.25, 0.75])
    return float(frac * (q3 - q1))


def critical_f(fs, stat, eps: float) -> dict:
    """Smallest coverage f* with stat(f) <= eps at f* and at every larger f (f = 1 excluded).

    fs, stat: coverage levels and the error statistic (e.g. P90), NaN where the level is not usable.
    Returns f_star (log-log interpolation between grid levels) and a censoring flag:
      'ok' | 'below' (already met at the smallest usable f) | 'above' (not met below f = 1) | 'none'.
    """
    fs = np.asarray(fs, float)
    st = np.asarray(stat, float)
    use = np.isfinite(st) & (fs < 1.0)
    fs, st = fs[use], st[use]
    if fs.size == 0:
        return dict(f_star=np.nan, flag="none")
    order = np.argsort(fs)
    fs, st = fs[order], st[order]
    ok = st <= eps
    if ok.all():
        return dict(f_star=float(fs[0]), flag="below")
    if not ok[-1]:
        return dict(f_star=float(fs[-1]), flag="above")
    k = int(np.flatnonzero(~ok).max()) + 1           # first level of the final run of passing levels
    f0, f1, s0, s1 = fs[k - 1], fs[k], st[k - 1], st[k]
    if s1 <= 0 or s0 <= 0 or s0 == s1:
        return dict(f_star=float(f1), flag="ok")
    lf = np.log(f0) + (np.log(eps) - np.log(s0)) * (np.log(f1) - np.log(f0)) / (np.log(s1) - np.log(s0))
    return dict(f_star=float(np.clip(np.exp(lf), f0, f1)), flag="ok")


def interp_at(fs, values, f_star: float) -> float:
    """Interpolate a per-level quantity (e.g. median N in the window) at f* on log f."""
    fs = np.asarray(fs, float)
    v = np.asarray(values, float)
    m = np.isfinite(v) & (fs > 0)
    if not np.isfinite(f_star) or m.sum() < 2:
        return np.nan
    return float(np.interp(np.log(f_star), np.log(fs[m]), v[m]))


def holm(pvals) -> np.ndarray:
    """Holm-adjusted p-values."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[i]))
        adj[i] = running
    return adj


def perm_pvalue(obs: float, null) -> float:
    """One-sided permutation p-value P(null >= obs) with the +1 correction."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan
    return float((1 + np.sum(null >= obs)) / (1 + null.size))


def fstar_stats(q) -> dict:
    """Median and quartiles of f* over the cities with a usable level, counting the cities whose error is not within the
    tolerance at the largest usable level ('above'; f_star holds that level, 0.9 in the main analysis).

    Kaplan-Meier estimate with those cities as right-censored at their largest usable level; when the curve sits exactly
    on a quantile over an interval, the midpoint is taken (the usual convention, which for a common censoring level of 0.9
    equals the ordinary median with the 'above' cities ranked last). 'below' counts at the smallest usable level; cities
    without any usable level ('none') are left out. A quantile that is not reached is returned as inf, with a lower bound.
    """
    fl = q.flag.to_numpy()
    keep = fl != "none"
    x = q.f_star.to_numpy(float)[keep]
    ev = (fl != "above")[keep]
    n = int(keep.sum())
    out = dict(n=n, n_ok=int((fl == "ok").sum()), n_above=int((fl == "above").sum()), n_below=int((fl == "below").sum()),
               n_none=int((fl == "none").sum()))
    if n == 0:
        return {**out, "median": np.nan, "median_lower_bound": np.nan, "q25": np.nan, "q75": np.nan, "share_le_0_8": np.nan}
    times = np.unique(x[ev])
    S, s_ = [], 1.0
    for tt in times:
        at_risk = np.sum(x >= tt)
        d = np.sum((x == tt) & ev)
        s_ *= 1 - d / at_risk
        S.append(s_)
    S = np.array(S)
    end = float(x.max())

    def qt(p):
        target = 1 - p
        hit = np.where(S <= target + 1e-12)[0]
        if hit.size == 0:
            return np.inf, end
        i = hit[0]
        if np.isclose(S[i], target):
            if i + 1 < times.size:
                return float((times[i] + times[i + 1]) / 2), None
            return np.inf, float((times[i] + end) / 2)
        return float(times[i]), None

    med, lb = qt(0.5)
    q25, _ = qt(0.25)
    q75, _ = qt(0.75)
    s08 = S[times <= 0.8][-1] if np.any(times <= 0.8) else 1.0
    return {**out, "median": med, "median_lower_bound": med if lb is None else lb, "q25": q25, "q75": q75,
            "share_le_0_8": float(1 - s08)}
