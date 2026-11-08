"""E8: small summary tables.

    python -m critical_window.tools.e8_tables run      [--main main34] [--x0 DIR] [--workers 2]
    python -m critical_window.tools.e8_tables summary  [--main main34]

run (per city, out/e8/cities/<city>.npz + .json): signed window values of sd_log, mean_log, site_cv and of CV / H with
  32 grid origins (E3 'sum_k32') for the 50 random windows of every level of the main run (usable: >= 20 stations);
  which flow centres each window contains; whole-extent values from 200 day-bootstrap replicates and from each
  single city weekday.
  Flow centres: the flow-density peaks of the main run (X9: 100 m grid, Gaussian kernel, bandwidth max(xi, 250 m),
  local maxima >= 50% of the highest), merged when closer than 2 km to a higher peak.
summary (out/e8/):
  t_dates.csv     per city: dates, weekdays used, weekend days, hours
  t_cells.csv     per city: stations, active 500 m cells (one grid origin and the mean over 32), stations per cell
  t_overlap.csv   per level: mean pairwise overlap of the random windows (observed, analytic); P90 error of the
                        non-overlapping tiles relative to the random windows
  t_centres.csv   per level and probe: P90 error of windows that contain the main centre, a secondary centre only,
                        or no centre; t_centres_fstar.csv: f* from each group (cities with >= 2 centres)
  t_ranking.csv   Spearman correlation between window values and whole-extent values across cities, one random
                        window per city (1000 draws), at f = 0.4 ... 0.9 and at each city's f*; across dates: one
                        bootstrap replicate or one single weekday per city against the whole-period value
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

from ..common.cfg import load_config
from ..core import criteria, windows
from ..main.data import load_city
from .e2_formula import probe_values
from .e3_grid import variant_values

OUT = Path(__file__).resolve().parents[1] / "out"
PROBES = ("sd_log", "mean_log", "site_cv", "cv_k32", "hn_k32")
MERGE_M = 2000.0
N_BOOT, N_DRAW = 200, 1000
RANK_LEVELS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def values(x, y, F, bbox):
    """The five probes per column of F (stations x T); station probes on stations with flow > 0."""
    out = {}
    T = F.shape[1]
    sp = {p: np.full(T, np.nan) for p in ("sd_log", "mean_log", "site_cv")}
    for t in range(T):
        pos = F[:, t] > 0
        if pos.sum() >= 2:
            v = probe_values(F[pos, t], np.ones((1, pos.sum()), bool))
            for p in sp:
                sp[p][t] = v[p][0]
    out.update(sp)
    cv, hn = variant_values(x, y, F, bbox, "sum_k32")
    out["cv_k32"], out["hn_k32"] = cv, hn
    return out


def distinct_centres(centres):
    keep = []
    for cx, cy, w in centres:
        if all(np.hypot(cx - kx, cy - ky) >= MERGE_M for kx, ky, _ in keep):
            keep.append((cx, cy, w))
    return keep


def city_job(a):
    c, x0, main_tag, m = a
    try:
        cd = load_city(c, Path(x0), m)
        js = json.load(open(OUT / main_tag / c / "city.json", encoding="utf-8"))
        cen = distinct_centres(js["x9"]["centres"])
        x, y, F = cd.x, cd.y, cd.F
        E = {p: float(v[0]) for p, v in values(x, y, F[:, None], cd.extent).items()}
        fs = np.array([f for f in m["f_grid"] if f < 1.0])
        arr = {}
        for i, f in enumerate(fs):
            wins = windows.random_windows(cd.extent, f, int(m["n_random"]), _rng(c, "main", "windows", f))
            V = {p: np.full(len(wins), np.nan) for p in PROBES}
            n = np.zeros(len(wins), int)
            C = np.zeros((len(wins), max(len(cen), 1)), bool)
            for k, w in enumerate(wins):
                inb = windows.in_box(x, y, w)
                n[k] = inb.sum()
                for j, (cx, cy, _) in enumerate(cen):
                    C[k, j] = w[0] <= cx <= w[2] and w[1] <= cy <= w[3]
                if n[k] < int(m["n_w"]):
                    continue
                v = values(x[inb], y[inb], F[inb][:, None], tuple(w))
                for p in PROBES:
                    V[p][k] = v[p][0]
            arr[f"n_{i}"] = n
            arr[f"centres_{i}"] = C
            for p in PROBES:
                arr[f"{p}_{i}"] = V[p]
        # dates: day bootstrap and single weekdays (whole extent)
        B = cd.bootstrap(N_BOOT, _rng("e8", c, "boot"))
        vb = values(x, y, B, cd.extent)
        Xd = np.where(np.isfinite(cd.X), cd.X, 0.0)
        vd = values(x, y, Xd, cd.extent)
        rec = dict(city=c, n_stations=int(F.size), theta_E=E, levels=fs.tolist(), centres=[list(t) for t in cen],
                   days=[str(d) for d in cd.days], n_valid_by_day=np.isfinite(cd.X).sum(0).tolist(),
                   boot={p: vb[p].tolist() for p in PROBES}, single_day={p: vd[p].tolist() for p in PROBES})
        od = OUT / "e8" / "cities"
        od.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(od / f"{c}.npz", **arr)
        json.dump(rec, open(od / f"{c}.json", "w"))
        return c, "ok"
    except Exception as exc:
        import traceback
        return c, f"error: {exc}\n{traceback.format_exc()}"


def run(main_tag, x0, workers):
    m = load_config()["main"]
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    todo = [(c, x0, main_tag, m) for c in cities if not (OUT / "e8" / "cities" / f"{c}.json").exists()]
    with Pool(workers) as pool:
        for c, st in pool.imap_unordered(city_job, todo):
            print(c, st.splitlines()[0], flush=True)
            if st.startswith("error"):
                print(st, flush=True)


def summary(main_tag, x0):
    m = load_config()["main"]
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    R = {c: json.load(open(OUT / "e8" / "cities" / f"{c}.json")) for c in cities if (OUT / "e8" / "cities" / f"{c}.json").exists()}
    A = {c: np.load(OUT / "e8" / "cities" / f"{c}.npz") for c in R}
    od = OUT / "e8"
    # ---- dates
    rows = []
    for c in R:
        d = pd.read_csv(Path(x0) / "per_city" / c / "days.csv")
        cw = d[d.city_weekday]
        we = d[(d.weekday >= 5) & ~d.holiday & (d.share_stations_valid >= 0.5)]
        rows.append(dict(city=c, first_date=str(d.date.min()), last_date=str(d.date.max()), n_calendar_days=len(d),
                         n_city_weekdays=len(cw), n_weekend_days=len(we), n_holidays=int(d.holiday.sum()),
                         hours="07:00-19:59", n_stations=R[c]["n_stations"]))
    pd.DataFrame(rows).to_csv(od / "t_dates.csv", index=False)
    # ---- cells
    from .e3_grid import ALL_OFFSETS, _grid
    rows = []
    for c in R:
        cd = load_city(c, Path(x0), m)
        n1 = np.unique(_grid(cd.x, cd.y, cd.extent)[0]).size
        n32 = [np.unique(_grid(cd.x, cd.y, cd.extent, o)[0]).size for o in ALL_OFFSETS]
        cnt = np.bincount(np.unique(_grid(cd.x, cd.y, cd.extent)[0], return_inverse=True)[1])
        rows.append(dict(city=c, n_stations=int(cd.F.size), area_km2=cd.area_km2, active_cells_500m=n1, active_cells_500m_mean32=float(np.mean(n32)),
                         stations_per_active_cell=float(cd.F.size / n1), share_cells_one_station=float((cnt == 1).mean()),
                         cells_in_extent=int(np.ceil((cd.extent[2] - cd.extent[0]) / 500) * np.ceil((cd.extent[3] - cd.extent[1]) / 500))))
    pd.DataFrame(rows).to_csv(od / "t_cells.csv", index=False)
    # ---- overlap and tiles
    rows = []
    for c in R:
        a = np.load(OUT / main_tag / c / "city.npz")
        js = json.load(open(OUT / main_tag / c / "city.json", encoding="utf-8"))
        for i, f in enumerate(js["levels"]):
            r = dict(city=c, f=f, overlap=float(a["lev__overlap"][i]), overlap_analytic=float(a["lev__overlap_analytic"][i]), n_tiles=int(a["lev__n_tiles"][i]))
            for p in ("sd_log", "mean_log", "site_cv", "cv500", "hn500"):
                r[f"tiles_over_random_{p}"] = float(a[f"{p}__p90_tiles"][i] / a[f"{p}__p90_obs"][i]) if a[f"{p}__p90_obs"][i] > 0 else np.nan
            rows.append(r)
    ov = pd.DataFrame(rows)
    ov.groupby("f").agg(cities=("city", "count"), overlap_median=("overlap", "median"), overlap_analytic=("overlap_analytic", "median"),
                        cities_with_tiles=("n_tiles", lambda v: int((v >= 2).sum())),
                        **{f"tiles_over_random_{p}": (f"tiles_over_random_{p}", "median") for p in ("sd_log", "mean_log", "site_cv", "cv500", "hn500")}
                        ).reset_index().to_csv(od / "t_overlap.csv", index=False)
    # ---- centres
    crow, frow = [], []
    eps = {}
    for p in PROBES:
        th = np.array([R[c]["theta_E"][p] for c in R], float)
        eps[p] = criteria.tolerance(th)
    for c in R:
        fs = np.array(R[c]["levels"])
        ncen = len(R[c]["centres"])
        for p in PROBES:
            st = {g: np.full(fs.size, np.nan) for g in ("main", "secondary_only", "none", "all")}
            for i, f in enumerate(fs):
                v = A[c][f"{p}_{i}"]
                Cm = A[c][f"centres_{i}"]
                ok = np.isfinite(v)
                e = np.abs(v - R[c]["theta_E"][p])
                grp = {"main": ok & Cm[:, 0], "secondary_only": ok & ~Cm[:, 0] & Cm[:, 1:].any(1) if Cm.shape[1] > 1 else np.zeros(ok.size, bool),
                       "none": ok & ~Cm.any(1), "all": ok}
                for g, mk in grp.items():
                    if mk.sum() >= 5:
                        st[g][i] = float(np.quantile(e[mk], 0.9))
                crow.append(dict(city=c, probe=p, f=f, n_centres=ncen, **{f"n_{g}": int(mk.sum()) for g, mk in grp.items()},
                                 **{f"p90_{g}": st[g][i] for g in grp}))
            r = dict(city=c, probe=p, n_centres=ncen)
            for g in st:
                cr = criteria.critical_f(fs, st[g], eps[p])
                r[f"f_{g}"], r[f"flag_{g}"] = cr["f_star"], cr["flag"]
            frow.append(r)
    ce = pd.DataFrame(crow)
    ce.to_csv(od / "t_centres_levels.csv", index=False)
    fr = pd.DataFrame(frow)
    fr.to_csv(od / "t_centres_fstar.csv", index=False)
    cs = []
    for (p, f), g in ce.groupby(["probe", "f"]):
        if not any(np.isclose(f, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])):
            continue
        poly = g[g.n_centres >= 2]
        cs.append(dict(probe=p, f=f, cities=int(g.p90_all.notna().sum()), poly_cities=int(len(poly)),
                       ratio_main_over_none=float(np.nanmedian(g.p90_main / g.p90_none)),
                       ratio_secondary_over_main_poly=float(np.nanmedian(poly.p90_secondary_only / poly.p90_main)) if len(poly) else np.nan,
                       n_poly_with_both=int((poly.p90_secondary_only.notna() & poly.p90_main.notna()).sum())))
    pd.DataFrame(cs).to_csv(od / "t_centres.csv", index=False)
    # ---- ranking
    rk = []
    cw = pd.read_csv(OUT / main_tag / "critical_windows.csv")
    e3 = pd.read_csv(OUT / "e3" / "e3_cities.csv")
    fstar = {}
    for p in PROBES:
        if p in ("cv_k32", "hn_k32"):
            q = e3[(e3.variant == "sum_k32") & (e3.probe == p[:2])].set_index("city")
        else:
            q = cw[cw.probe == p].set_index("city")
        fstar[p] = {c: (float(q.loc[c, "f_star"]) if q.loc[c, "flag"] in ("ok", "below") else (0.9 if q.loc[c, "flag"] == "above" else np.nan)) for c in R if c in q.index}
    cl = list(R)
    for p in PROBES:
        thE = np.array([R[c]["theta_E"][p] for c in cl], float)
        targets = [(f"f={f}", {c: f for c in cl}) for f in RANK_LEVELS] + [("f*", fstar[p])]
        for lab, fmap in targets:
            rng = _rng("e8", "rank", p, lab)
            pools = []
            for c in cl:
                fs = np.array(R[c]["levels"])
                f = fmap.get(c, np.nan)
                if not np.isfinite(f):
                    pools.append(np.array([]))
                    continue
                i = int(np.argmin(np.abs(fs - f))) if lab != "f*" else int(np.searchsorted(fs, f - 1e-9))
                i = min(i, fs.size - 1)
                v = A[c][f"{p}_{i}"]
                pools.append(v[np.isfinite(v)])
            rho = []
            for _ in range(N_DRAW):
                vals = np.array([rng.choice(pl) if pl.size else np.nan for pl in pools])
                ok = np.isfinite(vals) & np.isfinite(thE)
                if ok.sum() >= 8:
                    rho.append(stats.spearmanr(vals[ok], thE[ok])[0])
            rho = np.array(rho)
            ncity = int(sum(pl.size > 0 for pl in pools))
            rk.append(dict(probe=p, comparison="window location", at=lab, cities=ncity, rho_median=float(np.median(rho)) if rho.size else np.nan,
                           rho_p05=float(np.quantile(rho, 0.05)) if rho.size else np.nan))
        for lab, key in (("day bootstrap", "boot"), ("single weekday", "single_day")):
            rng = _rng("e8", "rank", p, lab)
            pools = [np.array(R[c][key][p], float) for c in cl]
            pools = [pl[np.isfinite(pl)] for pl in pools]
            rho = []
            for _ in range(N_DRAW):
                vals = np.array([rng.choice(pl) if pl.size else np.nan for pl in pools])
                ok = np.isfinite(vals) & np.isfinite(thE)
                if ok.sum() >= 8:
                    rho.append(stats.spearmanr(vals[ok], thE[ok])[0])
            rho = np.array(rho)
            rk.append(dict(probe=p, comparison="dates", at=lab, cities=int(sum(pl.size > 0 for pl in pools)),
                           rho_median=float(np.median(rho)), rho_p05=float(np.quantile(rho, 0.05))))
    rk = pd.DataFrame(rk)
    rk.to_csv(od / "t_ranking.csv", index=False)
    pd.set_option("display.width", 250)
    for f in ("t_dates", "t_cells", "t_overlap", "t_centres", "t_ranking"):
        print(f)
        print(pd.read_csv(od / f"{f}.csv").round(3).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "summary"])
    ap.add_argument("--main", default="main34")
    ap.add_argument("--x0", default=str(OUT / "x0"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    (OUT / "e8").mkdir(parents=True, exist_ok=True)
    if a.what == "run":
        run(a.main, a.x0, a.workers)
    else:
        summary(a.main, a.x0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
