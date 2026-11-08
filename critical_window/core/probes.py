"""Traffic diagnostics (probes) computed in a window.

Columns of the flow matrix F (n_sites x T) are realisations that share one layout: the observed flows,
N1 permutations, N2 simulations or bootstrap resamples. Every probe is returned per column, so a
whole null distribution costs one call.

grid_probe / multi_scale / fit_loglog are identical to clean_code/src/analysis/multi_scale.py and
scaling_fit.py (checked in tests/test_core.py, T1).
"""
from __future__ import annotations

import numpy as np

GRID_SIZES = np.array([10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
                       750, 800, 850, 900, 950, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
                       5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000], float)
LOCAL_L = GRID_SIZES[(GRID_SIZES >= 200) & (GRID_SIZES <= 1200)]          # 17 scales, 200..1000 m
OFFSETS = ((1, 0), (0, 1), (1, 1), (-1, 1))                                # each 8-neighbour pair once
MIN_SHORT_SIDE_EXPONENT = 4000.0                                           # L_max <= short side / 4


def grid_probe(x, y, F, bbox, L):
    """CV over active cells and node-strength entropy H at cell size L -> (cv[T], H[T], n_cells, n_nodes[T])."""
    x0, y0, x1, y1 = bbox
    T = F.shape[1]
    nan = np.full(T, np.nan)
    if x.size == 0:
        return nan, nan.copy(), 0, np.zeros(T, int)
    nx = max(int(np.ceil((x1 - x0) / L)), 1)
    ny = max(int(np.ceil((y1 - y0) / L)), 1)
    gx = np.clip(((x - x0) / L).astype(np.int64), 0, nx - 1)
    gy = np.clip(((y - y0) / L).astype(np.int64), 0, ny - 1)
    cid = gy * nx + gx
    cells, inv = np.unique(cid, return_inverse=True)
    nc = cells.size
    cf = np.empty((nc, T))
    for t in range(T):
        cf[:, t] = np.bincount(inv, weights=F[:, t], minlength=nc)
    act = cf > 0
    na = act.sum(0)
    cv = np.full(T, np.nan)
    for t in range(T):
        if na[t] > 1:
            v = cf[act[:, t], t]
            m = v.mean()
            if m > 0:
                cv[t] = v.std(ddof=1) / m
    cgx = cells % nx
    cgy = cells // nx
    ei, ej = [], []
    for dx, dy in OFFSETS:
        ngx, ngy = cgx + dx, cgy + dy
        ok = (ngx >= 0) & (ngx < nx) & (ngy >= 0) & (ngy < ny)
        nid = ngy * nx + ngx
        pos = np.searchsorted(cells, nid)
        pos = np.minimum(pos, nc - 1)
        ok &= cells[pos] == nid
        ei.append(np.nonzero(ok)[0])
        ej.append(pos[ok])
    ei = np.concatenate(ei)
    ej = np.concatenate(ej)
    H = np.full(T, np.nan)
    nn = np.zeros(T, int)
    if ei.size:
        w = np.sqrt(cf[ei] * cf[ej])
        for t in range(T):
            s = np.bincount(ei, w[:, t], minlength=nc) + np.bincount(ej, w[:, t], minlength=nc)
            s = s[s > 0]
            nn[t] = s.size
            if s.size:
                p = s / s.sum()
                H[t] = -np.sum(p * np.log(p))
    return cv, H, nc, nn


def multi_scale(x, y, F, bbox, Ls):
    T = F.shape[1]
    CV = np.full((len(Ls), T), np.nan)
    HH = np.full((len(Ls), T), np.nan)
    NC = np.zeros(len(Ls), int)
    NN = np.zeros((len(Ls), T), int)
    for k, L in enumerate(Ls):
        CV[k], HH[k], NC[k], NN[k] = grid_probe(x, y, F, bbox, L)
    return CV, HH, NC, NN


def fit_loglog(Ls, V, lmin=200.0, lmax=1200.0, min_pts=4, r2min=None, max_slope=2.0):
    """Per-column log-log slope of V (nL x T) within [lmin, lmax]; r2min=None -> no quality filter."""
    Ls = np.asarray(Ls, float)
    T = V.shape[1]
    out = np.full(T, np.nan)
    inr = (Ls >= lmin) & (Ls <= lmax)
    lx = np.log(Ls)
    for t in range(T):
        v = V[:, t]
        m = inr & np.isfinite(v) & (v > 0)
        if m.sum() < min_pts:
            continue
        xx, yy = lx[m], np.log(v[m])
        b, a = np.polyfit(xx, yy, 1)
        if r2min is not None:
            pred = a + b * xx
            sst = ((yy - yy.mean()) ** 2).sum()
            r2 = 1 - ((yy - pred) ** 2).sum() / sst if sst > 0 else 0.0
            if r2 < r2min or abs(b) > max_slope:
                continue
        out[t] = b
    return out


# ------------------------------------------------------------------------------------------------
# Probes used in the window analysis (fixed resolution) and exponents
# ------------------------------------------------------------------------------------------------
def level_probes(x, y, F, bbox, L0=500.0) -> dict:
    """Window-level probes per column: CV and H at L0 (H also normalised by ln n_nodes), log-flow mean and SD."""
    cv, H, nc, nn = grid_probe(x, y, F, bbox, L0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hn = np.where(nn > 1, H / np.log(np.maximum(nn, 2)), np.nan)
        lf = np.where(F > 0, np.log(np.where(F > 0, F, 1.0)), np.nan)
    n_pos = np.isfinite(lf).sum(0)
    mean_log = np.where(n_pos > 0, np.nansum(lf, 0) / np.maximum(n_pos, 1), np.nan)
    dev = np.where(np.isfinite(lf), lf - mean_log, 0.0)
    sd_log = np.where(n_pos > 1, np.sqrt((dev ** 2).sum(0) / np.maximum(n_pos - 1, 1)), np.nan)
    m = F.mean(0)
    site_cv = np.where((F.shape[0] > 1) & (m > 0), F.std(0, ddof=1) / np.where(m > 0, m, 1.0), np.nan)
    return dict(cv500=cv, h500=H, hn500=hn, mean_log=mean_log, sd_log=sd_log, site_cv=site_cv, n_nodes500=nn)


def exponents(x, y, F, bbox) -> dict:
    """alpha, beta from the 200-1000 m local fit (no R^2 filter); NaN when the short side is < 4 km."""
    T = F.shape[1]
    short = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
    if short < MIN_SHORT_SIDE_EXPONENT or x.size < 2:
        return dict(alpha=np.full(T, np.nan), beta=np.full(T, np.nan))
    CV, HH, _, _ = multi_scale(x, y, F, bbox, LOCAL_L)
    return dict(alpha=fit_loglog(LOCAL_L, CV), beta=fit_loglog(LOCAL_L, HH))


# ------------------------------------------------------------------------------------------------
# Correlation length
# ------------------------------------------------------------------------------------------------
def default_bins(max_dist: float) -> np.ndarray:
    fine = np.arange(0.0, 1000.0 + 1e-9, 50.0)
    coarse = np.arange(1250.0, max(max_dist, 1250.0) + 1e-9, 250.0)
    return np.concatenate([fine, coarse])


def correlogram(x, y, z, max_dist: float, rng=None, max_pts: int = 3000, min_pairs: int = 30):
    """Empirical correlogram rho(h) = 1 - gamma(h) / var(z) of values z at points (x, y)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    ok = np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if z.size > max_pts:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(z.size, max_pts, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    bins = default_bins(max_dist)
    if z.size < 3:
        return bins, np.full(bins.size - 1, np.nan), np.zeros(bins.size - 1, int)
    i, j = np.triu_indices(z.size, 1)
    h = np.hypot(x[i] - x[j], y[i] - y[j])
    g = 0.5 * (z[i] - z[j]) ** 2
    k = np.digitize(h, bins) - 1
    use = (k >= 0) & (k < bins.size - 1)
    cnt = np.bincount(k[use], minlength=bins.size - 1)
    gam = np.bincount(k[use], g[use], minlength=bins.size - 1) / np.maximum(cnt, 1)
    rho = 1.0 - gam / np.var(z)
    rho[cnt < min_pairs] = np.nan
    return bins, rho, cnt


def xi_bias_curve(x, y, bbox, rng, factors=(0.04, 0.06, 0.08, 0.11, 0.14, 0.18, 0.22, 0.27, 0.33, 0.4),
                  reps: int = 20, max_pts: int = 1500):
    """Mean estimated correlation length for true values xi = factor x short side, on this layout.

    Used to correct the finite-extent bias of corr_length (the empirical correlogram falls too fast when
    xi is not small compared with the extent). Gaussian correlation, as in the N2 null.
    """
    from .synth import gaussian_field
    short = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
    true, mean_est = [], []
    for fac in factors:
        xi = fac * short
        est = [corr_length(x, y, gaussian_field(x, y, bbox, xi, rng), short / 2, rng=rng, max_pts=max_pts)
               for _ in range(reps)]
        true.append(xi)
        mean_est.append(np.nanmean(est) if np.isfinite(est).any() else np.nan)
    return np.array(true), np.array(mean_est)


def xi_correct(xi_hat: float, true, mean_est) -> float:
    """Invert the bias curve: the true xi whose expected estimate equals xi_hat (NaN outside the curve)."""
    m = np.isfinite(mean_est)
    t, e = np.asarray(true)[m], np.asarray(mean_est)[m]
    if not np.isfinite(xi_hat) or t.size < 2:
        return np.nan
    order = np.argsort(e)
    e, t = np.maximum.accumulate(e[order]), t[order]
    if xi_hat > e[-1]:
        return np.nan
    if xi_hat < e[0]:
        return float(xi_hat * t[0] / e[0])
    return float(np.interp(xi_hat, e, t))


def corr_length(x, y, z, max_dist: float, rng=None, max_pts: int = 3000) -> float:
    """First distance where the correlogram falls below 1/e (linear interpolation); NaN if never within max_dist."""
    bins, rho, _ = correlogram(x, y, z, max_dist, rng=rng, max_pts=max_pts)
    mid = 0.5 * (bins[:-1] + bins[1:])
    thr = np.exp(-1.0)
    prev = None
    for q in range(rho.size):
        if not np.isfinite(rho[q]):
            continue
        if rho[q] < thr:
            if prev is None:
                return float(mid[q])
            (mp, rp) = prev
            return float(mp + (rp - thr) / (rp - rho[q]) * (mid[q] - mp))
        prev = (mid[q], rho[q])
    return np.nan
