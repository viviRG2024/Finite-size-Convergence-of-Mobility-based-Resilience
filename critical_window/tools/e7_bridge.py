"""E7: from the earlier analysis to the revised one, one change at a time, on the real IUTF flows of the 34 cities.

    python -m critical_window.tools.e7_bridge run      [--main main34] [--x0 DIR] [--workers 2]
    python -m critical_window.tools.e7_bridge run-glob [--main main34] [--x0 DIR] [--workers 2]
    python -m critical_window.tools.e7_bridge summary  [--main main34]

run-glob adds the earlier designs with the exponents fitted over all 39 grid sizes (10 m - 10 km, no R^2 filter;
the fit described in the earlier Methods), observed and shuffled, to each city file.

Flows: per valid station, the mean flow of each hour 0-23 over the clean city weekdays (outlier and stuck days removed);
stations with all 24 hours. Exponents per hour: log-log slope of CV(L) (alpha) and of H(L) (beta) over the 17 grid sizes
200-1000 m (cell sums, raw node-strength entropy), fitted as in the earlier code; as in the earlier analysis, hourly
values with R^2 < 0.3 (alpha) or R^2 < 0.6 (beta) are discarded.
Earlier design (per city): extent = bounding box of the road file (data/layouts), linear scale factor SF =
0.1, 0.2, 0.4, 0.6, 0.8, 1 (window side = SF x extent side), one centred window and 20 random windows inside the extent.
Per window: median over hours of beta and alpha, CV_temp(beta) = SD / |mean| of the 24 hourly betas, SD of the hourly
alphas. The same with the station box as extent, and both with the 24-hour series shuffled among stations (10 shuffles).
New criterion on the same exponents: station box, the 19 area levels and 50 random windows of the main run (>= 20
stations), P90 of |beta(W) - beta(E)| against eps = 0.25 x IQR of beta(E) across cities -> f*.
Writes out/e7/cities/<city>.json; the summary assembles the bridge table with the E1, E3 and E4 results.
"""
from __future__ import annotations

import argparse
import json
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.cfg import load_config
from ..core import criteria, probes, windows
from ..main.data import _stuck_days, load_city
from .e6_robust import road_bbox_utm

OUT = Path(__file__).resolve().parents[1] / "out"
SF = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
N_RAND, N_SHUF = 20, 10
THRESH = (0.015, 0.02, 0.025, 0.03, 0.04)


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def hourly_profile(c, x0, m):
    """(x, y, F24) for the valid stations with all 24 hourly means on clean city weekdays."""
    cd = load_city(c, Path(x0), m)
    pdir = Path(x0) / "per_city" / c
    h = pd.read_parquet(pdir / "station_hourly.parquet")
    stuck = _stuck_days(h, int(m["stuck_run_hours"]))
    ts = pd.to_datetime(h.hour_start)
    h = h.assign(date=ts.dt.strftime("%Y-%m-%d"), hour=ts.dt.hour)
    daily = pd.read_parquet(Path(x0) / "station_daily_all.parquet", filters=[("city", "==", c)])
    daily["date"] = daily["date"].astype(str)
    clean = daily[daily.city_weekday & ~daily.outlier]
    ok = set(zip(clean.station_id, clean.date)) - set(stuck)
    h = h[[(s, d) in ok for s, d in zip(h.station_id, h.date)]]
    P = h.groupby(["station_id", "hour"]).flow_vehh.mean().unstack()
    P = P.reindex(index=cd.station_ids, columns=range(24))
    keep = P.notna().all(1).to_numpy()
    return cd.x[keep], cd.y[keep], P.to_numpy(float)[keep], cd


def exps(x, y, F, bbox):
    """Hourly alpha and beta, fitted as in the earlier code; NaN where R^2 < 0.3 (alpha) or R^2 < 0.6 (beta), as in the
    earlier analysis."""
    if x.size < 2:
        n = F.shape[1]
        return np.full(n, np.nan), np.full(n, np.nan)
    CV, HH, _, _ = probes.multi_scale(x, y, F, bbox, probes.LOCAL_L)
    return probes.fit_loglog(probes.LOCAL_L, CV, r2min=0.3), probes.fit_loglog(probes.LOCAL_L, HH, r2min=0.6)


def win_summary(a, b):
    fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]
    return dict(alpha=float(np.median(fa)) if fa.size else np.nan, beta=float(np.median(fb)) if fb.size else np.nan,
                cvtemp_beta=float(np.std(fb, ddof=1) / abs(np.mean(fb))) if fb.size > 2 and abs(np.mean(fb)) > 1e-12 else np.nan,
                sd_alpha=float(np.std(fa, ddof=1)) if fa.size > 2 else np.nan, n_hours_beta=int(fb.size))


def old_design(c, x, y, F, ext, tag, perms):
    """Per SF: summaries of the centred window and the 20 random windows, observed and for each shuffle."""
    out = {}
    for sf in SF:
        wins = [("nested", windows.nested_window(ext, sf * sf)[0])]
        if sf < 1:
            wins += [("random", w) for w in windows.random_windows(ext, sf * sf, N_RAND, _rng(c, "e7", tag, sf))]
        rec = []
        for kind, w in wins:
            inb = windows.in_box(x, y, w)
            r = dict(kind=kind, n=int(inb.sum()))
            for j, p in enumerate([None] + perms):
                Fp = F if p is None else F[p]
                s = win_summary(*exps(x[inb], y[inb], Fp[inb], tuple(w)))
                r["obs" if p is None else f"shuf{j - 1}"] = s
            rec.append(r)
        out[str(sf)] = rec
    return out


def city_job(a):
    c, x0, m = a
    try:
        x, y, F, cd = hourly_profile(c, x0, m)
        qc = json.load(open(Path(x0) / "per_city" / c / "qc.json", encoding="utf-8"))
        rb = road_bbox_utm(c, qc["epsg"])
        rng = _rng("e7", c, "perm")
        perms = [rng.permutation(F.shape[0]) for _ in range(N_SHUF)]
        rec = dict(city=c, n_stations=int(F.shape[0]), road_bbox=list(rb), station_box=list(map(float, cd.extent)),
                   road_area_km2=(rb[2] - rb[0]) * (rb[3] - rb[1]) / 1e6, area_km2=cd.area_km2)
        kr = windows.in_box(x, y, rb)
        rec["old_road"] = old_design(c, x[kr], y[kr], F[kr], rb, "road", [p for p in perms] if kr.all() else
                                     [np.argsort(np.argsort(p[kr])) for p in perms])
        rec["old_station"] = old_design(c, x, y, F, cd.extent, "station", perms)
        # new criterion on the exponents: main windows, station box
        E = win_summary(*exps(x, y, F, cd.extent))
        rec["theta_E"] = E
        fs = [f for f in m["f_grid"] if f < 1.0]
        lev = []
        for f in fs:
            vals = []
            for w in windows.random_windows(cd.extent, f, int(m["n_random"]), _rng(c, "main", "windows", f)):
                inb = windows.in_box(x, y, w)
                if inb.sum() < int(m["n_w"]):
                    continue
                vals.append(win_summary(*exps(x[inb], y[inb], F[inb], tuple(w))))
            lev.append(dict(f=f, n_usable=len(vals), alpha=[v["alpha"] for v in vals], beta=[v["beta"] for v in vals]))
        rec["new_levels"] = lev
        od = OUT / "e7" / "cities"
        od.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(od / f"{c}.json", "w"))
        return c, "ok"
    except Exception as exc:
        import traceback
        return c, f"error: {exc}\n{traceback.format_exc()}"


def run(main_tag, x0, workers):
    m = load_config()["main"]
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    todo = [(c, x0, m) for c in cities if not (OUT / "e7" / "cities" / f"{c}.json").exists()]
    with Pool(workers) as pool:
        for c, st in pool.imap_unordered(city_job, todo):
            print(c, st.splitlines()[0], flush=True)
            if st.startswith("error"):
                print(st, flush=True)


def exps_glob(x, y, F, bbox):
    """Hourly alpha, beta from the 39-scale fit of the earlier Methods (10 m - 10 km, no R^2 filter)."""
    if x.size < 2:
        n = F.shape[1]
        return np.full(n, np.nan), np.full(n, np.nan)
    CV, HH, _, _ = probes.multi_scale(x, y, F, bbox, probes.GRID_SIZES)
    return (probes.fit_loglog(probes.GRID_SIZES, CV, lmin=0, lmax=1e9, min_pts=2),
            probes.fit_loglog(probes.GRID_SIZES, HH, lmin=0, lmax=1e9, min_pts=2))


def old_design_glob(c, x, y, F, ext, tag, perms):
    out = {}
    for sf in SF:
        wins = [("nested", windows.nested_window(ext, sf * sf)[0])]
        if sf < 1:
            wins += [("random", w) for w in windows.random_windows(ext, sf * sf, N_RAND, _rng(c, "e7", tag, sf))]
        rec = []
        for kind, w in wins:
            inb = windows.in_box(x, y, w)
            r = dict(kind=kind, n=int(inb.sum()))
            for j, p in enumerate([None] + perms):
                Fp = F if p is None else F[p]
                r["obs" if p is None else f"shuf{j - 1}"] = win_summary(*exps_glob(x[inb], y[inb], Fp[inb], tuple(w)))
            rec.append(r)
        out[str(sf)] = rec
    return out


def glob_job(a):
    c, x0, m = a
    try:
        fp = OUT / "e7" / "cities" / f"{c}.json"
        rec = json.load(open(fp))
        if "old_road_glob" in rec:
            return c, "ok (exists)"
        x, y, F, cd = hourly_profile(c, x0, m)
        rb = tuple(rec["road_bbox"])
        rng = _rng("e7", c, "perm")
        perms = [rng.permutation(F.shape[0]) for _ in range(N_SHUF)]
        kr = windows.in_box(x, y, rb)
        rec["old_road_glob"] = old_design_glob(c, x[kr], y[kr], F[kr], rb, "road", perms if kr.all() else [np.argsort(np.argsort(p[kr])) for p in perms])
        rec["old_station_glob"] = old_design_glob(c, x, y, F, cd.extent, "station", perms)
        json.dump(rec, open(fp, "w"))
        return c, "ok"
    except Exception as exc:
        import traceback
        return c, f"error: {exc}\n{traceback.format_exc()}"


def run_glob(main_tag, x0, workers):
    m = load_config()["main"]
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    with Pool(workers) as pool:
        for c, st in pool.imap_unordered(glob_job, [(c, x0, m) for c in cities]):
            print(c, st.splitlines()[0], flush=True)
            if st.startswith("error"):
                print(st, flush=True)


def _sf_star(sfs, vals, thr):
    """Smallest SF from which the value stays <= thr up to SF = 1 (NaN if it is above thr at SF = 1)."""
    v = np.array(vals, float)
    ok = np.isfinite(v) & (v <= thr)
    if not ok[-1]:
        return np.nan
    k = len(ok) - 1
    while k > 0 and ok[k - 1]:
        k -= 1
    return float(sfs[k])


def summary(main_tag):
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    R = {c: json.load(open(OUT / "e7" / "cities" / f"{c}.json")) for c in cities if (OUT / "e7" / "cities" / f"{c}.json").exists()}
    od = OUT / "e7"
    rows = []
    for c, r in R.items():
        for design in ("old_road", "old_station", "old_road_glob", "old_station_glob"):
            if design not in r:
                continue
            for sf in SF:
                ws = r[design][str(sf)]
                for src in ["obs"] + [f"shuf{j}" for j in range(N_SHUF)]:
                    nest = [w[src] for w in ws if w["kind"] == "nested"][0]
                    rnd = [w[src] for w in ws if w["kind"] == "random"]
                    rows.append(dict(city=c, design=design, sf=sf, f_area=sf * sf, source="observed" if src == "obs" else "shuffled", rep=src,
                                     n_nested=[w["n"] for w in ws if w["kind"] == "nested"][0],
                                     n_random_median=float(np.median([w["n"] for w in ws if w["kind"] == "random"])) if rnd else np.nan,
                                     cvtemp_beta_nested=nest["cvtemp_beta"], sd_alpha_nested=nest["sd_alpha"],
                                     cvtemp_beta_random_median=float(np.nanmedian([w["cvtemp_beta"] for w in rnd])) if rnd else np.nan,
                                     sd_beta_across_random=float(np.nanstd([w["beta"] for w in rnd], ddof=1)) if rnd else np.nan,
                                     sd_alpha_across_random=float(np.nanstd([w["alpha"] for w in rnd], ddof=1)) if rnd else np.nan,
                                     beta_nested=nest["beta"], alpha_nested=nest["alpha"]))
    d = pd.DataFrame(rows)
    d.to_csv(od / "old_design_cities.csv", index=False)
    # per city: shuffled = median over the shuffles
    keys = ["cvtemp_beta_nested", "sd_alpha_nested", "cvtemp_beta_random_median", "sd_beta_across_random", "sd_alpha_across_random"]
    dc = d.groupby(["design", "source", "city", "sf"])[keys + ["n_nested", "n_random_median"]].median().reset_index()
    cur = dc.groupby(["design", "source", "sf"])[keys + ["n_random_median"]].median().reset_index()
    cur["f_area"] = cur.sf ** 2
    cur.to_csv(od / "old_design_curves.csv", index=False)
    # old criterion per city: SF* by CV_temp(beta) of the centred window at each threshold
    crit = []
    for (design, source, c), g in dc.groupby(["design", "source", "city"]):
        g = g.sort_values("sf")
        sfs = g.sf.to_numpy()
        for thr in THRESH:
            s = _sf_star(sfs, g.cvtemp_beta_nested.to_numpy(), thr)
            crit.append(dict(design=design, source=source, city=c, threshold=thr, sf_star=s))
    crit = pd.DataFrame(crit)
    passing = []
    for (design, source, thr), g in crit.groupby(["design", "source", "threshold"]):
        for sf in SF[:-1]:
            passing.append(dict(design=design, source=source, threshold=thr, sf=sf, share_cities_passing=float((g.sf_star <= sf).mean())))
    passing = pd.DataFrame(passing)
    passing.to_csv(od / "old_criterion_passing.csv", index=False)
    # new criterion on the exponents
    m = load_config()["main"]
    ne = []
    th = {k: np.array([R[c]["theta_E"][k] for c in R], float) for k in ("alpha", "beta")}
    eps = {k: criteria.tolerance(th[k]) for k in th}
    for c, r in R.items():
        fs = np.array([lv["f"] for lv in r["new_levels"]])
        for k in ("alpha", "beta"):
            st = []
            for lv in r["new_levels"]:
                e = np.abs(np.array(lv[k], float) - r["theta_E"][k])
                e = e[np.isfinite(e)]
                st.append(float(np.quantile(e, 0.9)) if e.size >= int(m["min_usable_windows"]) else np.nan)
            cr = criteria.critical_f(fs, np.array(st), eps[k])
            ne.append(dict(city=c, probe=k, theta_E=r["theta_E"][k], eps=eps[k], f_star=cr["f_star"], flag=cr["flag"]))
    ne = pd.DataFrame(ne)
    ne.to_csv(od / "new_criterion_exponents.csv", index=False)
    # bridge table
    br = []

    def add(step, change, measure, value, n_ok=None, note=""):
        br.append(dict(step=step, change=change, measure=measure, value=value, n_cities=n_ok, note=note))

    for design, lab in (("old_road", "road-file box (earlier), code fit 200-1200 m"), ("old_station", "station box, code fit"),
                        ("old_road_glob", "road-file box (earlier), 39-scale fit"), ("old_station_glob", "station box, 39-scale fit")):
        for source in ("observed", "shuffled"):
            g = cur[(cur.design == design) & (cur.source == source)].set_index("sf")
            if g.empty:
                continue
            add(f"{lab}, {source}", "earlier criterion", "median CV_temp(beta), centred window, SF 0.1 / 0.2 / 0.4 / 0.8",
                " / ".join(f"{g.loc[s, 'cvtemp_beta_nested'] * 100:.2f}%" for s in (0.1, 0.2, 0.4, 0.8)))
            add(f"{lab}, {source}", "earlier criterion", "median SD of beta across random windows, SF 0.2 / 0.4 / 0.6 / 0.8",
                " / ".join(f"{g.loc[s, 'sd_beta_across_random']:.3f}" for s in (0.2, 0.4, 0.6, 0.8)))
            cc = crit[(crit.design == design) & (crit.source == source) & np.isclose(crit.threshold, 0.015)]
            add(f"{lab}, {source}", "earlier criterion", "median SF* at CV_temp(beta) <= 1.5% (area = SF^2)",
                f"{np.nanmedian(cc.sf_star):.2f} (area {np.nanmedian(cc.sf_star) ** 2:.2f})", int(cc.sf_star.notna().sum()))
    def med(g):
        st = criteria.fstar_stats(g)
        if not st["n"]:
            return "n/a"
        return f"{st['median']:.2f}" if np.isfinite(st["median"]) else f"> {st['median_lower_bound']:.2f}"

    note = "median over cities with a usable level; cities above 0.9 counted as above every found f*"
    for k in ("beta", "alpha"):
        g = ne[ne.probe == k]
        add("new criterion on the earlier exponent", f"P90 |{k}(W) - {k}(E)| <= 0.25 IQR", "median f* (area)", med(g),
            int((g.flag != "none").sum()), f"{(g.flag == 'ok').sum()} found, {(g.flag == 'above').sum()} above 0.9, "
            f"{(g.flag == 'none').sum()} none; {note}")
    cw = pd.read_csv(OUT / main_tag / "critical_windows.csv")
    for p in ("sd_log", "mean_log", "site_cv"):
        g = cw[cw.probe == p]
        add("new probes", p, "median f* (area)", med(g), int((g.flag != "none").sum()),
            f"{(g.flag == 'ok').sum()} found, {(g.flag == 'above').sum()} above 0.9; {note}")
    e3c = pd.read_csv(OUT / "e3" / "e3_cities.csv")
    for v in ("sum", "sum_k32"):
        for p in ("cv", "hn"):
            g = e3c[(e3c.variant == v) & (e3c.probe == p)]
            add("grid probes", f"{p.upper()} 500 m, {v}", "median f* (area)", med(g), int((g.flag != "none").sum()),
                f"{(g.flag == 'ok').sum()} found, {(g.flag == 'above').sum()} above 0.9; {note}")
    if (OUT / "e4b" / "critical.csv").exists():
        e4 = pd.read_csv(OUT / "e4b" / "critical.csv")
        for k, g in e4.groupby("metric", sort=False):
            add("network resilience", k, "median f* (area)", med(g), int((g.flag != "none").sum()),
                f"{(g.flag == 'ok').sum()} found, {(g.flag == 'above').sum()} above 0.9; {note}")
    br = pd.DataFrame(br)
    br.to_csv(od / "bridge_table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(cur.round(4).to_string(index=False))
    print(passing[np.isclose(passing.threshold, 0.015) | np.isclose(passing.threshold, 0.04)].round(3).to_string(index=False))
    print(br.to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "run-glob", "summary"])
    ap.add_argument("--main", default="main34")
    ap.add_argument("--x0", default=str(OUT / "x0"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    (OUT / "e7").mkdir(parents=True, exist_ok=True)
    if a.what == "run":
        run(a.main, a.x0, a.workers)
    elif a.what == "run-glob":
        run_glob(a.main, a.x0, a.workers)
    else:
        summary(a.main)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
