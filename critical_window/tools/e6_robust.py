"""E6: robustness of the critical coverage to the window shape, the observation extent, the time window, rain, the
outlier rule, the grid size and the tolerance.

    python -m critical_window.tools.e6_robust run      [--main main34] [--x0 DIR] [--weather DIR] [--workers 2] [--variants a,b]
    python -m critical_window.tools.e6_robust summary  [--main main34]

Variants (one change from the main design each):
  main             the main design (rectangles with the extent's aspect ratio, weekday daytime totals); check against E1/E3
  square           square windows of the same area, inside the extent (levels where a square fits)
  circle           circular windows of the same area, inside the extent (levels where a circle fits)
  trim95           extent = box of the central 95% of stations on each axis (stations outside it dropped)
  buffer500        extent = station box enlarged by 500 m on each side (coverage relative to the larger box)
  road_bbox        extent = bounding box of the road file of the earlier study (data/layouts; stations
                   outside it dropped), coverage relative to it
  hull250          extent = convex hull of the stations + 250 m; coverage relative to its area; rectangles with the
                   aspect ratio of the hull's box, centre uniform in that box and accepted if inside the hull
  peak_hour        flow = mean 08:00-08:59 flow on city weekdays
  midday_hour      flow = mean 12:00-12:59 flow on city weekdays
  all_day          flow = mean 00:00-23:59 total on city weekdays (>= 22 valid hours, total = mean hour x 24)
  weekend          flow = mean 07:00-19:59 total on Saturdays and Sundays (no holidays, >= 50% of the stations valid)
  dry_days         weekday daytime totals on the days without an hour >= 0.2 mm/h in 07:00-19:59 (ERA5)
  no_outlier_rule  weekday daytime totals with the outlier days kept
  grid250/grid1000 CV and H with 250 m / 1000 m cells (station probes as in main)
Hour-based variants apply the main outlier rule (5 x 1.4826 x MAD per station, >= 5 days) to their own daily values;
all variants drop the stuck station-days of the main run. Stations: the valid stations of the main run.
Probes: sd_log, mean_log, site_cv (stations with flow > 0) and cv, hn = CV and H / ln(n) of cell sums averaged over
32 grid origins (E3 'sum_k32'). Window usability as in the main run (>= 20 stations, >= 25 usable windows per level).
Per variant and probe: eps = 0.25 x IQR of the variant's whole-extent values across cities; f* as in the main run.
The summary also recomputes f* of 'main' for eps = 0.1, 0.5, 1 x IQR and window shares q = 0.8, 0.95, and gives the
critical side fraction s* = sqrt(f*).
Writes out/e6/.
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
from ..main.data import _stuck_days, load_city
from ..resil import rain
from .e2_formula import probe_values
from .e3_grid import variant_values

OUT = Path(__file__).resolve().parents[1] / "out"
VARIANTS = ("main", "square", "circle", "trim95", "buffer500", "peak_hour", "midday_hour", "all_day", "weekend", "dry_days",
            "no_outlier_rule", "grid250", "grid1000", "road_bbox", "hull250")
P0_LAYOUTS = Path(__file__).resolve().parents[1] / "data" / "layouts"
STATION_PROBES = ("sd_log", "mean_log", "site_cv")
GRID_PROBES = ("cv", "hn")
QS = (0.8, 0.9, 0.95)


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


# ------------------------------------------------------------------------------------------------ station flows
def _outlier_mask(st_ids, vals, k=5.0, min_days=5):
    df = pd.DataFrame(dict(s=st_ids, v=vals))
    med = df.groupby("s").v.transform("median")
    mad = (df.v - med).abs().groupby(df.s).transform("median")
    cnt = df.groupby("s").v.transform("count")
    return ((df.v - med).abs() > k * 1.4826 * mad) & (cnt >= min_days) & (mad > 0)


def variant_flows(c, variant, x0, m, weather):
    """(x, y, F, extent) of the valid stations for a flow variant; extent = station box (main definition)."""
    cd = load_city(c, Path(x0), m)
    if variant not in ("peak_hour", "midday_hour", "all_day", "weekend", "dry_days", "no_outlier_rule"):
        return cd.x, cd.y, cd.F, cd.extent, int(cd.days.size)
    pdir = Path(x0) / "per_city" / c
    days = pd.read_csv(pdir / "days.csv")
    days["date"] = days["date"].astype(str)
    hourly = pd.read_parquet(pdir / "station_hourly.parquet")
    stuck = _stuck_days(hourly, int(m["stuck_run_hours"]))
    sid = {s: i for i, s in enumerate(cd.station_ids)}
    if variant in ("weekend", "dry_days", "no_outlier_rule"):
        daily = pd.read_parquet(Path(x0) / "station_daily_all.parquet", filters=[("city", "==", c)])
        daily["date"] = daily["date"].astype(str)
        daily = daily[daily.station_id.isin(sid)]
        if variant == "weekend":
            ok_days = set(days.loc[(days.weekday >= 5) & ~days.holiday & (days.share_stations_valid >= 0.5), "date"])
            daily = daily[daily.date.isin(ok_days)]
            daily = daily[~_outlier_mask(daily.station_id.to_numpy(), daily.flow_veh.to_numpy()).to_numpy()]
        elif variant == "dry_days":
            qc = json.load(open(pdir / "qc.json", encoding="utf-8"))
            cw = days.loc[days.city_weekday, "date"].tolist()
            ser = rain.city_rain_series(Path(weather) / c / "weather", qc, cw)
            dry = []
            for d in cw:
                ts = [pd.Timestamp(d) + pd.Timedelta(hours=h) for h in range(7, 20)]
                v = np.array([ser.get(t, np.nan) for t in ts], float)
                if np.isfinite(v).all() and (v < 0.2).all():
                    dry.append(d)
            daily = daily[daily.city_weekday & ~daily.outlier & daily.date.isin(dry)]
        else:
            daily = daily[daily.city_weekday]
        vals = daily[["station_id", "date", "flow_veh"]].rename(columns={"flow_veh": "v"})
    else:
        ts = pd.to_datetime(hourly.hour_start)
        h = hourly.assign(date=ts.dt.strftime("%Y-%m-%d"), hour=ts.dt.hour)
        h = h[h.station_id.isin(sid) & h.date.isin(set(days.loc[days.city_weekday, "date"]))]
        if variant in ("peak_hour", "midday_hour"):
            hh = 8 if variant == "peak_hour" else 12
            vals = h[h.hour == hh][["station_id", "date", "flow_vehh"]].rename(columns={"flow_vehh": "v"})
        else:
            g = h.groupby(["station_id", "date"]).flow_vehh.agg(["mean", "count"]).reset_index()
            g = g[g["count"] >= 22]
            vals = g.assign(v=g["mean"] * 24)[["station_id", "date", "v"]]
        vals = vals[~_outlier_mask(vals.station_id.to_numpy(), vals.v.to_numpy()).to_numpy()]
    if stuck and len(vals):
        vals = vals[np.array([(s, d) not in stuck for s, d in zip(vals.station_id, vals.date)], bool)]
    if len(vals) == 0:
        return None
    Fm = vals.groupby("station_id").v.mean()
    keep = np.array([s in Fm.index for s in cd.station_ids])
    if keep.sum() < 2:
        return None
    F = Fm.reindex(cd.station_ids[keep]).to_numpy(float)
    return cd.x[keep], cd.y[keep], F, cd.extent, int(vals.date.nunique())


def road_bbox_utm(c, epsg):
    """Road-file bounding box of the earlier study (local equirectangular metres in data/layouts) in UTM."""
    from ..common.geo import utm_forward
    mt = json.load(open(P0_LAYOUTS / f"{c}_meta.json"))
    R = 6371008.8
    k = np.cos(np.radians(mt["lat0"]))
    bx = np.array(mt["road_bbox"], float)
    xs, ys = np.array([bx[0], bx[2], bx[2], bx[0]]), np.array([bx[1], bx[1], bx[3], bx[3]])
    lon = mt["lon0"] + np.degrees(xs / (R * k))
    lat = mt["lat0"] + np.degrees(ys / R)
    ux, uy = utm_forward(lon, lat, int(epsg) % 100, int(epsg) // 100 == 327)
    return (float(ux.min()), float(uy.min()), float(ux.max()), float(uy.max()))


# ------------------------------------------------------------------------------------------------ windows
def level_windows(c, variant, ext, f, n, x, y, hull=None):
    """Membership masks (n x stations) and window boxes for one level; None where the shape does not fit."""
    W, H = ext[2] - ext[0], ext[3] - ext[1]
    A = W * H
    if variant == "hull250":
        from shapely import contains_xy
        s_ = np.sqrt(f * hull.area / A)
        w, h = W * s_, H * s_
        rng = _rng(c, "e6", variant, f)
        cx, cy = [], []
        for _ in range(200):
            px = rng.uniform(ext[0] + w / 2, ext[2] - w / 2, 4 * n)
            py = rng.uniform(ext[1] + h / 2, ext[3] - h / 2, 4 * n)
            ok = contains_xy(hull, px, py)
            cx += px[ok].tolist()
            cy += py[ok].tolist()
            if len(cx) >= n:
                break
        cx, cy = np.array(cx[:n]), np.array(cy[:n])
        boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        return np.array([windows.in_box(x, y, b) for b in boxes]), boxes
    if variant in ("square", "circle"):
        rng = _rng(c, "e6", variant, f)
        if variant == "square":
            a = np.sqrt(f * A)
            if a > min(W, H):
                return None, None
            cx = rng.uniform(ext[0] + a / 2, ext[2] - a / 2, n)
            cy = rng.uniform(ext[1] + a / 2, ext[3] - a / 2, n)
            boxes = np.column_stack([cx - a / 2, cy - a / 2, cx + a / 2, cy + a / 2])
            M = np.array([windows.in_box(x, y, b) for b in boxes])
        else:
            r = np.sqrt(f * A / np.pi)
            if 2 * r > min(W, H):
                return None, None
            cx = rng.uniform(ext[0] + r, ext[2] - r, n)
            cy = rng.uniform(ext[1] + r, ext[3] - r, n)
            boxes = np.column_stack([cx - r, cy - r, cx + r, cy + r])
            M = (x[None, :] - cx[:, None]) ** 2 + (y[None, :] - cy[:, None]) ** 2 <= r ** 2
        return M, boxes
    seed = _rng(c, "main", "windows", f) if variant not in ("trim95", "buffer500", "road_bbox") else _rng(c, "e6", variant, f)
    boxes = windows.random_windows(ext, f, n, seed)
    return np.array([windows.in_box(x, y, b) for b in boxes]), boxes


# ------------------------------------------------------------------------------------------------ per city and variant
def job(a):
    c, variant, x0, weather, m = a
    od = OUT / "e6" / "cities" / variant
    od.mkdir(parents=True, exist_ok=True)
    try:
        r = variant_flows(c, variant, x0, m, weather)
        if r is None:
            json.dump(dict(city=c, variant=variant, status="no data"), open(od / f"{c}.json", "w"))
            return c, variant, "no data"
        x, y, F, ext, ndays = r
        if variant == "trim95":
            q = (np.quantile(x, 0.025), np.quantile(y, 0.025), np.quantile(x, 0.975), np.quantile(y, 0.975))
            k = windows.in_box(x, y, q)
            x, y, F, ext = x[k], y[k], F[k], tuple(map(float, q))
        elif variant == "buffer500":
            ext = (ext[0] - 500, ext[1] - 500, ext[2] + 500, ext[3] + 500)
        elif variant == "road_bbox":
            qc = json.load(open(Path(x0) / "per_city" / c / "qc.json", encoding="utf-8"))
            ext = road_bbox_utm(c, qc["epsg"])
            k = windows.in_box(x, y, ext)
            x, y, F = x[k], y[k], F[k]
        hull = None
        if variant == "hull250":
            from shapely import MultiPoint
            hull = MultiPoint(np.column_stack([x, y])).convex_hull.buffer(250.0)
            ext = tuple(map(float, hull.bounds))
        cell = 250.0 if variant == "grid250" else 1000.0 if variant == "grid1000" else 500.0
        pos = F > 0
        xs, ys, Fs = x[pos], y[pos], F[pos]
        E = {k: float(v[0]) for k, v in probe_values(Fs, np.ones((1, Fs.size), bool)).items()}
        gE = _grid(x, y, F, ext, cell)
        E.update(cv=float(gE[0]), hn=float(gE[1]))
        fs = np.array([f for f in m["f_grid"] if f < 1.0])
        rec = dict(city=c, variant=variant, status="ok", n_stations=int(F.size), n_days=ndays, extent=list(map(float, ext)),
                   levels=fs.tolist(), theta_E=E, cell_m=cell, area_km2=(hull.area if hull is not None else (ext[2] - ext[0]) * (ext[3] - ext[1])) / 1e6)
        n_w, min_use = int(m["n_w"]), int(m["min_usable_windows"])
        curves = {p: {f"q{int(q * 100)}": [] for q in QS} | {"bias": [], "n_usable": []} for p in STATION_PROBES + GRID_PROBES}
        for f in fs:
            M, boxes = level_windows(c, variant, ext, f, int(m["n_random"]), x, y, hull)
            res = {}
            if M is not None:
                cnt_pos = M[:, pos].sum(1)
                use = cnt_pos >= n_w
                if use.sum() >= min_use:
                    V = probe_values(Fs, M[use][:, pos])
                    for p in STATION_PROBES:
                        res[p] = V[p] - E[p]
                cnt = M.sum(1)
                useg = cnt >= n_w
                if useg.sum() >= min_use:
                    gv = np.array([_grid(x[M[i]], y[M[i]], F[M[i]], tuple(boxes[i]), cell) for i in np.flatnonzero(useg)])
                    res["cv"], res["hn"] = gv[:, 0] - E["cv"], gv[:, 1] - E["hn"]
            for p in STATION_PROBES + GRID_PROBES:
                e = res.get(p)
                e = e[np.isfinite(e)] if e is not None else None
                ok = e is not None and e.size >= min_use
                for q in QS:
                    curves[p][f"q{int(q * 100)}"].append(float(np.quantile(np.abs(e), q)) if ok else np.nan)
                curves[p]["bias"].append(float(np.median(e)) if ok else np.nan)
                curves[p]["n_usable"].append(int(e.size) if e is not None else 0)
        rec["curves"] = curves
        json.dump(rec, open(od / f"{c}.json", "w"))
        return c, variant, "ok"
    except Exception as exc:
        import traceback
        return c, variant, f"error: {exc}\n{traceback.format_exc()}"


def _grid(x, y, F, bbox, cell):
    """sum_k32 CV and H / ln(n) at the given cell size (E3 definition, 32 grid origins)."""
    from . import e3_grid
    old = e3_grid.L
    try:
        e3_grid.L = cell
        offs = e3_grid.ALL_OFFSETS
        if cell != 500.0:                        # same relative origins as at 500 m
            e3_grid.ALL_OFFSETS = [(o[0] * cell / 500.0, o[1] * cell / 500.0) for o in offs]
        cv, hn = variant_values(x, y, F[:, None], bbox, "sum_k32")
    finally:
        e3_grid.L = old
        e3_grid.ALL_OFFSETS = offs
    return float(cv[0]), float(hn[0])


def run(main_tag, x0, weather, workers, variants):
    m = load_config()["main"]
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    todo = [(c, v, x0, weather, m) for v in variants for c in cities if not (OUT / "e6" / "cities" / v / f"{c}.json").exists()]
    with Pool(workers) as pool:
        for c, v, st in pool.imap_unordered(job, todo):
            print(v, c, st.splitlines()[0], flush=True)
            if st.startswith("error"):
                print(st, flush=True)


# ------------------------------------------------------------------------------------------------ summary
def _law(fs, st, eps):
    ok = np.isfinite(st) & (st > 0)
    if ok.sum() < 4:
        return np.nan
    return float(np.polyfit(np.log((1 - fs[ok]) / fs[ok]), np.log(st[ok] / eps), 1)[0])


GEOM_VARIANTS = ("main", "square", "circle", "trim95", "buffer500", "road_bbox", "hull250")


def window_counts(c, variant, x0, m):
    """Median number of stations (flow > 0) in the usable windows per level, for the geometric variants."""
    cd = load_city(c, Path(x0), m)
    x, y, F, ext = cd.x, cd.y, cd.F, cd.extent
    hull = None
    if variant == "trim95":
        q = (np.quantile(x, 0.025), np.quantile(y, 0.025), np.quantile(x, 0.975), np.quantile(y, 0.975))
        k = windows.in_box(x, y, q)
        x, y, F, ext = x[k], y[k], F[k], tuple(map(float, q))
    elif variant == "buffer500":
        ext = (ext[0] - 500, ext[1] - 500, ext[2] + 500, ext[3] + 500)
    elif variant == "road_bbox":
        qc = json.load(open(Path(x0) / "per_city" / c / "qc.json", encoding="utf-8"))
        ext = road_bbox_utm(c, qc["epsg"])
        k = windows.in_box(x, y, ext)
        x, y, F = x[k], y[k], F[k]
    elif variant == "hull250":
        from shapely import MultiPoint
        hull = MultiPoint(np.column_stack([x, y])).convex_hull.buffer(250.0)
        ext = tuple(map(float, hull.bounds))
    pos = F > 0
    out = []
    for f in [f for f in m["f_grid"] if f < 1.0]:
        M, _ = level_windows(c, variant, ext, f, int(m["n_random"]), x, y, hull)
        if M is None:
            out.append(np.nan)
            continue
        n = M[:, pos].sum(1)
        n = n[n >= int(m["n_w"])]
        out.append(float(np.median(n)) if n.size >= int(m["min_usable_windows"]) else np.nan)
    return np.array(out), int(pos.sum())


def summary(main_tag, x0=None):
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    od = OUT / "e6"
    R = {}
    for v in VARIANTS:
        for c in cities:
            p = od / "cities" / v / f"{c}.json"
            if p.exists():
                r = json.load(open(p))
                if r.get("status") == "ok":
                    R[(v, c)] = r
    rows = []
    for v in VARIANTS:
        rv = {c: R[(v, c)] for c in cities if (v, c) in R}
        if not rv:
            continue
        for p in STATION_PROBES + GRID_PROBES:
            th = np.array([r["theta_E"][p] for r in rv.values()], float)
            eps = criteria.tolerance(th)
            for c, r in rv.items():
                fs = np.array(r["levels"])
                st = np.array(r["curves"][p]["q90"], float)
                cr = criteria.critical_f(fs, st, eps)
                rows.append(dict(variant=v, probe=p, city=c, eps=eps, theta_E=r["theta_E"][p], f_star=cr["f_star"], flag=cr["flag"],
                                 slope=_law(fs, st, eps), n_stations=r["n_stations"], n_days=r["n_days"]))
    d = pd.DataFrame(rows)
    # stations in the window at f* (geometric variants): is the requirement stable as a station count?
    m = load_config()["main"]
    x0 = str(x0 or OUT / "x0")                        # station tables of X0 (window station counts)
    nstar = []
    for v in GEOM_VARIANTS:
        for c in cities:
            if (v, c) not in R:
                continue
            cnt, N = window_counts(c, v, x0, m)
            fs = np.array(R[(v, c)]["levels"])
            for _, r in d[(d.variant == v) & (d.city == c) & (d.flag == "ok")].iterrows():
                n_ = criteria.interp_at(fs, cnt, r.f_star)
                nstar.append(dict(variant=v, probe=r.probe, city=c, f_star=r.f_star, n_star=n_, share_star=n_ / N if N else np.nan))
    ns = pd.DataFrame(nstar)
    ns.to_csv(od / "e6_nstar.csv", index=False)
    base_n = ns[ns.variant == "main"].set_index(["probe", "city"])
    nsum = []
    for (v, p), g in ns.groupby(["variant", "probe"], sort=False):
        both = [c for c in g.city if (p, c) in base_n.index]
        gi = g.set_index("city")
        ratio = np.array([gi.loc[c, "n_star"] / base_n.loc[(p, c), "n_star"] for c in both])
        nsum.append(dict(variant=v, probe=p, cities=len(g), f_median=float(g.f_star.median()), n_star_median=float(g.n_star.median()),
                         share_star_median=float(g.share_star.median()), n_both=len(both),
                         median_ratio_n_star_vs_main=float(np.nanmedian(ratio)) if len(both) else np.nan,
                         median_ratio_f_star_vs_main=float(np.nanmedian([gi.loc[c, "f_star"] / d[(d.variant == "main") & (d.probe == p) & (d.city == c)].f_star.iloc[0] for c in both])) if len(both) else np.nan))
    pd.DataFrame(nsum).to_csv(od / "e6_nstar_summary.csv", index=False)
    # tolerance and window-share variants of 'main'
    tol = []
    for c in cities:
        if ("main", c) not in R:
            continue
        r = R[("main", c)]
        fs = np.array(r["levels"])
        for p in STATION_PROBES + GRID_PROBES:
            th = np.array([R[("main", cc)]["theta_E"][p] for cc in cities if ("main", cc) in R], float)
            iqr = criteria.tolerance(th, 1.0)
            for q in QS:
                for mult in (0.1, 0.25, 0.5, 1.0):
                    if q != 0.9 and mult != 0.25:
                        continue
                    cr = criteria.critical_f(fs, np.array(r["curves"][p][f"q{int(q * 100)}"], float), mult * iqr)
                    tol.append(dict(variant=f"q{int(q * 100)}_eps{mult}", probe=p, city=c, eps=mult * iqr, f_star=cr["f_star"], flag=cr["flag"]))
    d = pd.concat([d, pd.DataFrame(tol)], ignore_index=True)
    d.to_csv(od / "e6_cities.csv", index=False)
    base = d[d.variant == "main"].set_index(["probe", "city"])
    srows = []
    for (v, p), g in d.groupby(["variant", "probe"], sort=False):
        ok = g[g.flag == "ok"]
        b = base.loc[p] if p in base.index.get_level_values(0) else None
        both = [c for c in ok.city if b is not None and c in b.index and b.loc[c, "flag"] == "ok"]
        gi = g.set_index("city")
        dv = np.array([gi.loc[c, "f_star"] - b.loc[c, "f_star"] for c in both])
        th_both = [c for c in g.city if b is not None and c in b.index and np.isfinite(gi.loc[c, "theta_E"])] if g.theta_E.notna().any() else []
        srows.append(dict(variant=v, probe=p, cities=len(g), eps=float(g.eps.iloc[0]) if g.eps.nunique() == 1 else np.nan, n_ok=len(ok),
                          f_median=float(ok.f_star.median()) if len(ok) else np.nan, s_median=float(np.sqrt(ok.f_star).median()) if len(ok) else np.nan,
                          f_median_all=criteria.fstar_stats(g)["median"],
                          n_above=int((g.flag == "above").sum()), n_below=int((g.flag == "below").sum()), n_none=int((g.flag == "none").sum()),
                          n_ok_both=len(both), median_abs_change=float(np.median(np.abs(dv))) if len(dv) else np.nan,
                          share_change_le_0p1=float(np.mean(np.abs(dv) <= 0.1)) if len(dv) else np.nan,
                          spearman_fstar_vs_main=float(stats.spearmanr([gi.loc[c, "f_star"] for c in both], [b.loc[c, "f_star"] for c in both])[0]) if len(both) >= 5 else np.nan,
                          spearman_theta_vs_main=float(stats.spearmanr([gi.loc[c, "theta_E"] for c in th_both], [b.loc[c, "theta_E"] for c in th_both])[0]) if len(th_both) >= 5 else np.nan,
                          slope_median=float(g.slope.median()) if "slope" in g else np.nan))
    s = pd.DataFrame(srows)
    s.to_csv(od / "e6_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string(index=False))
    print(pd.DataFrame(nsum).round(3).to_string(index=False))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "summary"])
    ap.add_argument("--main", default="main34")
    ap.add_argument("--x0", default=str(OUT / "x0"))
    ap.add_argument("--weather", default="data/input/IUTFD")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args(argv)
    (OUT / "e6").mkdir(parents=True, exist_ok=True)
    if a.what == "run":
        run(a.main, a.x0, a.weather, a.workers, [v for v in a.variants.split(",") if v])
    else:
        summary(a.main, a.x0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
