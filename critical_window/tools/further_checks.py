"""Further checks (exploratory, after E1-E10).

    python -m critical_window.tools.further_checks ranking     [--main main34]     (network metrics)
    python -m critical_window.tools.further_checks segments    [--main main34]    
    python -m critical_window.tools.further_checks correlates  [--main main34]    
    python -m critical_window.tools.further_checks units200    [--main main34]     (the "law of 200 units" of the earlier SI 4.2)

ranking     City ranking of the road-network metrics from one random window per city against the whole-city ranking:
            Spearman rho across cities, one window drawn per city (the 20 windows per level of E4b), 1000 draws, median
            and 5th percentile; at f = 0.4 ... 0.9 and at each city's own f* (first level at or above it; 0.9 when f*
            is above 0.9). Same procedure as the traffic diagnostics in E8 (Fig. 2d).
segments    Whole-extent CV and H of the station flows at the 39 grid sizes of the earlier Methods (10 m - 10 km, one
            grid origin at the extent corner as in the earlier code). Per city: one power law against two power laws
            joined at a break (continuous in log-log, break at an interior size with >= 3 sizes on each side), least
            squares in log-log; BIC (single: 2 parameters, broken: 4). Over 200-1000 m (17 sizes) and over all sizes
            with a defined value.
correlates  Spearman correlation across cities between f* and the sensor coverage and observation period: number of
            stations, extent area, station density, share of empty 500 m cells in the extent, box-counting dimension of
            the stations (E5), weekdays used, calendar span of the data, correlation length. Cities whose f* is above
            0.9 enter with 0.95 (ranked last); cities without a usable level are left out.
units200    The earlier SI 4.2 claimed that windows stabilise once they hold about 200 "organisational units". Checked with
            the new results: the number of stations in the usable windows at f* of sigma_z (E6, main) across cities; the
            error of f* predicted by a fixed station count K (f = K / N, capped at 1) for K = 100, 200, 400; and the number of
            correlation areas xi^2 in the window at f*.
Outputs in out/checks/.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..common.cfg import load_config, repo_path
from ..core import probes
from ..main.data import load_city

N_DRAW = 1000
RANK_LEVELS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
NET = ("R_random", "R_btw", "eff_loss", "dw_eff_loss", "acc_loss")


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def ranking(out_root: Path, od: Path):
    cr = pd.read_csv(out_root / "e4b" / "critical.csv")
    files = sorted((out_root / "e4b" / "network").glob("*.csv"))
    D = {f.stem: pd.read_csv(f) for f in files}
    cities = sorted(D)
    rows = []
    for k in NET:
        whole = np.array([D[c][D[c].f >= 0.999][k].iloc[0] if (D[c].f >= 0.999).any() else np.nan for c in cities], float)
        q = cr[cr.metric == k].set_index("city")
        fst = {c: (float(q.loc[c, "f_star"]) if q.loc[c, "flag"] in ("ok", "below") else (0.9 if q.loc[c, "flag"] == "above" else np.nan))
               for c in cities if c in q.index}
        targets = [(f"f={f:g}", {c: f for c in cities}) for f in RANK_LEVELS] + [("f*", fst)]
        for lab, fmap in targets:
            pools = []
            for c in cities:
                f = fmap.get(c, np.nan)
                d = D[c][D[c].f < 0.999]
                lv = np.array(sorted(d.f.unique()))
                if not np.isfinite(f) or lv.size == 0:
                    pools.append(np.array([]))
                    continue
                i = int(np.argmin(np.abs(lv - f))) if lab != "f*" else min(int(np.searchsorted(lv, f - 1e-9)), lv.size - 1)
                v = d[np.isclose(d.f, lv[i])][k].to_numpy(float)
                pools.append(v[np.isfinite(v)])
            rng = _rng("checks", "rank", k, lab)
            rho = []
            for _ in range(N_DRAW):
                vals = np.array([rng.choice(pl) if pl.size else np.nan for pl in pools])
                ok = np.isfinite(vals) & np.isfinite(whole)
                if ok.sum() >= 8:
                    rho.append(stats.spearmanr(vals[ok], whole[ok])[0])
            rho = np.array(rho)
            rows.append(dict(metric=k, at=lab, cities=int(sum(pl.size > 0 for pl in pools)),
                             rho_median=float(np.median(rho)) if rho.size else np.nan,
                             rho_p05=float(np.quantile(rho, 0.05)) if rho.size else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(od / "ranking_network.csv", index=False)
    print(df.round(3).to_string(index=False))


def _fit_single(x, y):
    b, a = np.polyfit(x, y, 1)
    rss = float(np.sum((y - (a + b * x)) ** 2))
    return b, rss


def _fit_broken(x, y, min_side=3):
    best = (np.inf, np.nan, np.nan, np.nan)
    for k in range(min_side - 1, x.size - min_side):
        xk = x[k]
        A = np.column_stack([np.ones_like(x), np.minimum(x - xk, 0), np.maximum(x - xk, 0)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rss = float(np.sum((y - A @ coef) ** 2))
        if rss < best[0]:
            best = (rss, float(np.exp(xk)), float(coef[1]), float(coef[2]))
    return best


def segments(out_root: Path, od: Path, main_tag: str, x0: Path, m: dict):
    cities = json.load(open(out_root / main_tag / "summary.json", encoding="utf-8"))["cities"]
    rows, curves = [], []
    Ls = probes.GRID_SIZES
    for c in cities:
        cd = load_city(c, x0, m)
        F = cd.F[:, None]
        CV, HH, NC, NN = probes.multi_scale(cd.x, cd.y, F, cd.extent, Ls)
        for L, cv, h, nc, nn in zip(Ls, CV[:, 0], HH[:, 0], NC, NN[:, 0]):
            curves.append(dict(city=c, L=float(L), CV=float(cv), H=float(h), n_cells=int(nc), n_nodes=int(nn)))
        for name, V in (("CV", CV[:, 0]), ("H", HH[:, 0])):
            for rng_lab, sel in (("200-1000 m", (Ls >= 200) & (Ls <= 1000)), ("all sizes", np.ones(Ls.size, bool))):
                ok = sel & np.isfinite(V) & (V > 0)
                if ok.sum() < 8:
                    rows.append(dict(city=c, measure=name, range=rng_lab, n=int(ok.sum())))
                    continue
                x, y = np.log(Ls[ok]), np.log(V[ok])
                b, rss1 = _fit_single(x, y)
                rss2, Lb, b1, b2 = _fit_broken(x, y)
                n = x.size
                rss1, rss2 = max(rss1, 1e-12), max(rss2, 1e-12)
                bic1 = n * np.log(rss1 / n) + 2 * np.log(n)
                bic2 = n * np.log(rss2 / n) + 4 * np.log(n)
                r2 = 1 - rss1 / float(np.sum((y - y.mean()) ** 2))
                rows.append(dict(city=c, measure=name, range=rng_lab, n=int(n), L_min=float(Ls[ok].min()), L_max=float(Ls[ok].max()),
                                 slope=float(b), r2_single=float(r2), dBIC=float(bic1 - bic2), break_m=Lb, slope_below=b1, slope_above=b2,
                                 short_side_km=cd.short_side / 1000))
        print(c, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(od / "segments.csv", index=False)
    pd.DataFrame(curves).to_csv(od / "segments_curves.csv", index=False)
    sm = []
    for (name, rng_lab), g in df[df.dBIC.notna()].groupby(["measure", "range"]):
        for lab, gg in (("all cities", g), ("short side >= 4 km", g[g.short_side_km >= 4])):
            sm.append(dict(measure=name, range=rng_lab, cities=lab, n=int(len(gg)), r2_single_median=float(gg.r2_single.median()),
                           share_broken_preferred=float(np.mean(gg.dBIC > 10)), share_dBIC_above_2=float(np.mean(gg.dBIC > 2)),
                           break_m_median=float(gg[gg.dBIC > 10].break_m.median()) if (gg.dBIC > 10).any() else np.nan,
                           slope_change_median=float((gg.slope_above - gg.slope_below).median())))
    sm = pd.DataFrame(sm)
    sm.to_csv(od / "segments_summary.csv", index=False)
    print(sm.round(3).to_string(index=False))


def correlates(out_root: Path, od: Path, main_tag: str, x0: Path, m: dict):
    t2 = pd.read_csv(out_root / main_tag / "table2_cities.csv").set_index("city")
    dates = pd.read_csv(out_root / "e8" / "t_dates.csv").set_index("city")
    geo = pd.read_csv(out_root / "e5" / "geometry.csv").set_index("city")
    att = []
    for c in t2.index:
        cd = load_city(c, x0, m)
        x0_, y0_, x1_, y1_ = cd.extent
        nx, ny = max(int(np.ceil((x1_ - x0_) / 500)), 1), max(int(np.ceil((y1_ - y0_) / 500)), 1)
        occ = np.zeros((ny, nx), bool)
        occ[np.clip(((cd.y - y0_) / 500).astype(int), 0, ny - 1), np.clip(((cd.x - x0_) / 500).astype(int), 0, nx - 1)] = True
        att.append(dict(city=c, stations=int(t2.loc[c, "n_stations"]), area_km2=float(t2.loc[c, "area_km2"]),
                        density_per_km2=float(t2.loc[c, "n_stations"] / t2.loc[c, "area_km2"]), empty_cells_500m=float(1 - occ.mean()),
                        D_station=float(geo.loc[c, "D_station"]), weekdays=int(t2.loc[c, "n_days"]),
                        span_days=int(dates.loc[c, "n_calendar_days"]),
                        xi_m=float(t2.loc[c, "xi_m"]) if t2.loc[c, "xi_identifiable"] else np.nan))
    att = pd.DataFrame(att).set_index("city")
    att.to_csv(od / "city_attributes.csv")
    # f* per metric (above 0.9 -> 0.95, none -> left out)
    cw = pd.read_csv(out_root / main_tag / "critical_windows.csv")
    e3 = pd.read_csv(out_root / "e3" / "e3_cities.csv")
    net = pd.read_csv(out_root / "e4b" / "critical.csv")
    fst = {}
    q = cw[cw.probe == "sd_log"].set_index("city")
    fst["sd_log"] = q
    for p in ("cv", "hn"):
        fst[p] = e3[(e3.variant == "sum_k32") & (e3.probe == p)].set_index("city")
    for k in ("R_random", "R_btw", "eff_loss"):
        fst[k] = net[net.metric == k].set_index("city")
    rows = []
    for k, q in fst.items():
        v = q.f_star.where(q.flag.isin(["ok", "below"]), np.where(q.flag == "above", 0.95, np.nan)).astype(float)
        for a_ in att.columns:
            d = pd.concat([v.rename("f"), att[a_]], axis=1).dropna()
            r, pv = stats.spearmanr(d.f, d[a_]) if len(d) >= 8 else (np.nan, np.nan)
            rows.append(dict(metric=k, attribute=a_, n=int(len(d)), spearman=float(r), p=float(pv)))
    df = pd.DataFrame(rows)
    df.to_csv(od / "correlates.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.pivot(index="attribute", columns="metric", values="spearman").round(2).to_string())
    print(df.pivot(index="attribute", columns="metric", values="p").round(3).to_string())


def units200(out_root: Path, od: Path, main_tag: str):
    ns = pd.read_csv(out_root / "e6" / "e6_nstar.csv")
    m = ns[(ns.variant == "main") & (ns.probe == "sd_log")].set_index("city")
    t2 = pd.read_csv(out_root / main_tag / "table2_cities.csv").set_index("city")
    cw = pd.read_csv(out_root / main_tag / "critical_windows.csv")
    s = cw[cw.probe == "sd_log"].set_index("city")
    ok = s[s.flag == "ok"].index
    d = pd.DataFrame({"stations": t2.n_stations, "area_km2": t2.area_km2, "xi_m": t2.xi_m.where(t2.xi_identifiable),
                      "f_star": s.f_star, "n_star": m.n_star}).loc[ok]
    d["xi2_areas_at_fstar"] = d.f_star * d.area_km2 * 1e6 / d.xi_m ** 2
    for K in (100, 200, 400):
        d[f"f_fixed_{K}"] = np.minimum(K / d.stations, 1.0)
    d.to_csv(od / "units200_cities.csv")
    rows = [dict(quantity="stations in the window at f*", n=int(d.n_star.notna().sum()), median=float(d.n_star.median()),
                 q25=float(d.n_star.quantile(0.25)), q75=float(d.n_star.quantile(0.75)), min=float(d.n_star.min()), max=float(d.n_star.max())),
            dict(quantity="correlation areas xi^2 in the window at f*", n=int(d.xi2_areas_at_fstar.notna().sum()),
                 median=float(d.xi2_areas_at_fstar.median()), q25=float(d.xi2_areas_at_fstar.quantile(0.25)),
                 q75=float(d.xi2_areas_at_fstar.quantile(0.75)), min=float(d.xi2_areas_at_fstar.min()), max=float(d.xi2_areas_at_fstar.max()))]
    for K in (100, 200, 400):
        e = (d[f"f_fixed_{K}"] - d.f_star).abs()
        rows.append(dict(quantity=f"f* from a fixed count of {K} stations", n=int(len(e)), mean_abs_error=float(e.mean()),
                         share_within_0p1=float(np.mean(e <= 0.1))))
    df = pd.DataFrame(rows)
    df.to_csv(od / "units200.csv", index=False)
    print(df.round(3).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ranking", "segments", "correlates", "units200"])
    ap.add_argument("--main", default="main34")
    ap.add_argument("--x0", default="")
    a = ap.parse_args(argv)
    cfg = load_config()
    out_root = repo_path(cfg["paths"]["out_dir"])
    od = out_root / "checks"
    od.mkdir(parents=True, exist_ok=True)
    x0 = Path(a.x0) if a.x0 else out_root / "x0"
    if a.cmd == "ranking":
        ranking(out_root, od)
    elif a.cmd == "segments":
        segments(out_root, od, a.main, x0, cfg["main"])
    elif a.cmd == "units200":
        units200(out_root, od, a.main)
    else:
        correlates(out_root, od, a.main, x0, cfg["main"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
