"""E3: improved grid probes (CV and normalised entropy H at 500 m) and their critical coverage.

    python -m critical_window.tools.e3_grid run      [--main main34] [--x0 DIR] [--workers 2]
    python -m critical_window.tools.e3_grid summary  [--main main34]

Variants (cells of 500 m):
  sum        cells carry the sum of their stations' flows (earlier algorithm; = cv500 / hn500 of the main run)
  mean       cells carry the mean flow of their stations
  mean_k8    'mean' averaged over 8 grid origins (one fixed set of offsets, the same for every window and the city)
  nearest    every 500 m cell whose centre lies within 500 m of a station takes the flow of the nearest station
For each variant and city: whole-extent value, P90 window error per coverage level (main-run windows, same usability
rule), critical coverage at 0.25 x IQR across cities, grid-origin floor (P90 change of the whole-extent value over 50
origin shifts), and the whole-extent value against 200 shuffles of the flows among stations.
Writes out/e3/.
"""
from __future__ import annotations

import argparse
import json
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from ..common.cfg import load_config
from ..core import criteria, probes, windows
from ..main.data import load_city

OUT = Path(__file__).resolve().parents[1] / "out"
L = 500.0
VARIANTS = ("sum", "mean", "mean_k8", "mean_k32", "sum_k32", "nearest")
ALL_OFFSETS = [(0.0, 0.0)] + [tuple(v) for v in np.random.default_rng(20261001).uniform(0, L, (31, 2))]
K_OFFSETS = ALL_OFFSETS[:8]
N_SHIFT, N_SHUFFLE = 50, 200


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def _cv_hn(cells, cf, nx, ny):
    """CV over active cells and node-strength entropy / ln(n) on queen adjacency, per column of cf (cells x T)."""
    nc, T = cf.shape
    cv = np.full(T, np.nan)
    hn = np.full(T, np.nan)
    if nc < 2:
        return cv, hn
    for t in range(T):
        v = cf[:, t]
        v = v[v > 0]
        if v.size > 1 and v.mean() > 0:
            cv[t] = v.std(ddof=1) / v.mean()
    cgx, cgy = cells % nx, cells // nx
    ei, ej = [], []
    for dx, dy in probes.OFFSETS:
        ngx, ngy = cgx + dx, cgy + dy
        ok = (ngx >= 0) & (ngx < nx) & (ngy >= 0) & (ngy < ny)
        nid = ngy * nx + ngx
        pos = np.minimum(np.searchsorted(cells, nid), nc - 1)
        ok &= cells[pos] == nid
        ei.append(np.nonzero(ok)[0])
        ej.append(pos[ok])
    ei, ej = np.concatenate(ei), np.concatenate(ej)
    if ei.size:
        w = np.sqrt(np.maximum(cf[ei], 0) * np.maximum(cf[ej], 0))
        for t in range(T):
            s = np.bincount(ei, w[:, t], minlength=nc) + np.bincount(ej, w[:, t], minlength=nc)
            s = s[s > 0]
            if s.size > 1:
                p = s / s.sum()
                hn[t] = -np.sum(p * np.log(p)) / np.log(s.size)
    return cv, hn


def _grid(x, y, bbox, off=(0.0, 0.0)):
    x0, y0 = bbox[0] - off[0], bbox[1] - off[1]
    nx = max(int(np.ceil((bbox[2] - x0) / L)), 1)
    ny = max(int(np.ceil((bbox[3] - y0) / L)), 1)
    gx = np.clip(((x - x0) / L).astype(np.int64), 0, nx - 1)
    gy = np.clip(((y - y0) / L).astype(np.int64), 0, ny - 1)
    return gy * nx + gx, nx, ny, x0, y0


def variant_values(x, y, F, bbox, variant, off=(0.0, 0.0)):
    """cv, hn per column of F (stations x T) for one variant."""
    if "_k" in variant:                      # <sum|mean>_k<K>: average over the first K grid origins
        agg, k = variant.split("_k")
        res = [variant_values(x, y, F, bbox, agg, (o[0] + off[0], o[1] + off[1])) for o in ALL_OFFSETS[:int(k)]]
        return np.nanmean([r[0] for r in res], 0), np.nanmean([r[1] for r in res], 0)
    cid, nx, ny, x0, y0 = _grid(x, y, bbox, off)
    if variant in ("sum", "mean"):
        cells, inv = np.unique(cid, return_inverse=True)
        cf = np.stack([np.bincount(inv, weights=F[:, t], minlength=cells.size) for t in range(F.shape[1])], 1)
        if variant == "mean":
            cf = cf / np.bincount(inv, minlength=cells.size)[:, None]
        return _cv_hn(cells, cf, nx, ny)
    # nearest: all cells of the window grid whose centre is within L of a station
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    cx, cy = x0 + (gx.ravel() + 0.5) * L, y0 + (gy.ravel() + 0.5) * L
    inside = (cx >= bbox[0]) & (cx <= bbox[2]) & (cy >= bbox[1]) & (cy <= bbox[3])
    d, j = cKDTree(np.column_stack([x, y])).query(np.column_stack([cx, cy]))
    keep = inside & (d <= L)
    cells = (gy.ravel() * nx + gx.ravel())[keep]
    order = np.argsort(cells)
    cells, jj = cells[order], j[keep][order]
    return _cv_hn(cells, F[jj], nx, ny)


def city_job(a):
    c, x0, main_tag = a
    m = load_config()["main"]
    cd = load_city(c, Path(x0), m)
    x, y, F = cd.x, cd.y, cd.F[:, None]
    ext = cd.extent
    fs = np.array([f for f in m["f_grid"] if f < 1.0])
    rec = {}
    rng = _rng("e3", c)
    shuffles = np.column_stack([cd.F] + [cd.F[rng.permutation(cd.F.size)] for _ in range(N_SHUFFLE)])
    shifts = rng.uniform(0, L, (N_SHIFT, 2))
    for v in VARIANTS:
        cvE, hnE = variant_values(x, y, F, ext, v)
        cvS, hnS = variant_values(x, y, shuffles, ext, v)
        rec[f"{v}|E"] = (float(cvE[0]), float(hnE[0]))
        rec[f"{v}|shuffle"] = (cvS[1:].tolist(), hnS[1:].tolist())
        fl = [variant_values(x, y, F, ext, v, off=tuple(s_)) for s_ in (shifts[:20] if "_k" in v else shifts)]
        rec[f"{v}|floor"] = (float(np.quantile([abs(f_[0][0] - cvE[0]) for f_ in fl], 0.9)),
                             float(np.quantile([abs(f_[1][0] - hnE[0]) for f_ in fl], 0.9)))
        p90 = {"cv": [], "hn": []}
        for f in fs:
            wins = windows.random_windows(ext, f, int(m["n_random"]), _rng(c, "main", "windows", f))
            vals_cv, vals_hn = [], []
            for w in wins:
                inb = windows.in_box(x, y, w)
                if inb.sum() < int(m["n_w"]):
                    continue
                cv, hn = variant_values(x[inb], y[inb], F[inb], tuple(w), v)
                vals_cv.append(cv[0])
                vals_hn.append(hn[0])
            if len(vals_cv) >= int(m["min_usable_windows"]):
                p90["cv"].append(float(np.nanquantile(np.abs(np.array(vals_cv) - cvE[0]), 0.9)))
                p90["hn"].append(float(np.nanquantile(np.abs(np.array(vals_hn) - hnE[0]), 0.9)))
            else:
                p90["cv"].append(np.nan)
                p90["hn"].append(np.nan)
        rec[f"{v}|p90"] = (p90["cv"], p90["hn"])
    rec["levels"] = fs.tolist()
    rec["n_stations"] = int(cd.F.size)
    od = OUT / "e3" / "cities"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(od / f"{c}.json", "w"))
    return c


def run(main_tag, x0, workers):
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    todo = [c for c in cities if not (OUT / "e3" / "cities" / f"{c}.json").exists()]
    with Pool(workers) as pool:
        for c in pool.imap_unordered(city_job, [(c, x0, main_tag) for c in todo]):
            print(c, flush=True)


def summary(main_tag):
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    recs = {c: json.load(open(OUT / "e3" / "cities" / f"{c}.json")) for c in cities if (OUT / "e3" / "cities" / f"{c}.json").exists()}
    rows, srows = [], []
    for v in VARIANTS:
        for k, name in ((0, "cv"), (1, "hn")):
            E = {c: r[f"{v}|E"][k] for c, r in recs.items()}
            vals = np.array([e for e in E.values() if np.isfinite(e)])
            iqr = float(np.subtract(*np.quantile(vals, [0.75, 0.25])))
            eps = 0.25 * iqr
            fl, okc, out95, obs, shm = [], 0, 0, [], []
            for c, r in recs.items():
                fs = np.array(r["levels"])
                st = np.array(r[f"{v}|p90"][k], float)
                cr = criteria.critical_f(fs, st, eps)
                sh = np.array(r[f"{v}|shuffle"][k], float)
                lo, hi = np.nanquantile(sh, [0.025, 0.975])
                outside = bool(E[c] < lo or E[c] > hi)
                rows.append(dict(variant=v, probe=name, city=c, n_stations=r["n_stations"], theta_E=E[c], f_star=cr["f_star"], flag=cr["flag"],
                                 floor=r[f"{v}|floor"][k], floor_over_eps=r[f"{v}|floor"][k] / eps if eps > 0 else np.nan,
                                 shuffled_median=float(np.nanmedian(sh)), outside_shuffle95=outside))
            d = pd.DataFrame([x for x in rows if x["variant"] == v and x["probe"] == name])
            okd = d[d.flag == "ok"]
            fin = d[np.isfinite(d.theta_E) & np.isfinite(d.shuffled_median)]
            st = criteria.fstar_stats(d)
            srows.append(dict(variant=v, probe=name, n_cities=len(d), eps=eps, n_ok=len(okd), f_median=float(okd.f_star.median()) if len(okd) else np.nan,
                              f_median_all=st["median"], f_median_all_lower_bound=st["median_lower_bound"],
                              n_above=int((d.flag == "above").sum()), n_none=int((d.flag == "none").sum()),
                              floor_over_iqr_median=float(np.median(d.floor) / iqr), n_floor_above_eps=int((d.floor > eps).sum()),
                              n_outside_shuffle95=int(d.outside_shuffle95.sum()),
                              spearman_obs_vs_shuffled=float(stats.spearmanr(fin.theta_E, fin.shuffled_median)[0]) if len(fin) > 4 else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "e3" / "e3_cities.csv", index=False)
    s = pd.DataFrame(srows)
    s.to_csv(OUT / "e3" / "e3_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string(index=False))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "summary"])
    ap.add_argument("--main", default="main34")
    ap.add_argument("--x0", default=str(OUT / "x0"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    (OUT / "e3").mkdir(parents=True, exist_ok=True)
    if a.what == "run":
        run(a.main, a.x0, a.workers)
    else:
        summary(a.main)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
