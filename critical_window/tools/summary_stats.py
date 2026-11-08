"""Numbers reported in the text that are not read directly from a figure or table.

    python -m critical_window.tools.summary_stats pairs      [--main main34]   # pairwise order of cities (rankings)
    python -m critical_window.tools.summary_stats centres    [--main main34]   # flow centres and critical coverage
    python -m critical_window.tools.summary_stats error_law                    # fits of the error law (Fig. 5a,b)
    python -m critical_window.tools.summary_stats cleaning   [--main main34]   # effect of the data-cleaning rules
    python -m critical_window.tools.summary_stats all        [--main main34]

Run after the main analysis and the stages E3, E4b, E8 and E10 (see run_all.sh). Outputs in out/checks/.

pairs      For one random usable window per city (1,000 draws), the share of city pairs whose order in the windows matches
           the whole-city order, for all pairs and for the pairs whose whole-city values differ by more than twice the
           tolerance (eps = 0.25 x IQR of the whole-city values across cities); and the median absolute error of the
           difference between two cities, in units of eps. Windows at f = 0.4 ... 0.9 and at each city's own f* (first
           level at or above it; 0.9 when f* is above 0.9); for the traffic diagnostics also one single weekday per
           city instead of all weekdays. Traffic diagnostics: windows of E8 (sigma_z; CV and H on 500 m cells averaged
           over 32 grid origins). Road-network metrics: the 20 windows per level of E4b.
           -> pairs_traffic.csv, pairs_network.csv
centres    Number of flow centres per city (E8) against the critical coverage: Spearman correlations with f* and with
           the number of stations, the partial correlation controlling for the number of stations (from the three
           Spearman coefficients), and a Mann-Whitney test between cities with one and with several centres. Cities whose
           error is not within the tolerance below f = 0.9 enter with f* = 0.95. Also, for sigma_z, the critical coverage
           computed from the windows containing the main centre compared with that from all windows (cities where both
           are found), and the number of cities in which the windows containing only secondary centres, or no centre,
           reach the tolerance (E8; a group needs at least five windows at a level).
           -> centres_fstar.csv (per city), centres_summary.csv (per diagnostic), centres_groups.json
error_law  For every city and metric of Fig. 5a,b: the window error divided by its value at f = 0.5, regressed on
           (1 - f) / f on logarithmic axes through the origin, excluding f = 0.5 (equal to one by construction):
           exponent b and R^2 (1 - residual / total sum of squares about the mean). Pooled over the cities of each
           family: the exponent of a single power law and the median and 90th percentile of the deviation from it,
           exp(|ln(y / y_fit)|) - 1, in percent; all levels and f >= 0.2.
           -> error_law_fits.csv (per city and metric), error_law_summary.csv
cleaning   Share of readings above 3,600 vehicles per hour per lane (of the readings without an error flag), share of
           station-days on the analysis weekdays removed by the outlier rule, their ratio to the station's median,
           the share of stations flagged on a city-day with at least one flag, the share flagged on days with a rainfall
           event (threshold 0.5 mm/h) and on other days in the cities with events, and the station-days removed as stuck.
           -> cleaning_cities.csv (per city), cleaning_summary.json
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..common.cfg import load_config, repo_path

OUT = repo_path(load_config()["paths"]["out_dir"])
LEVELS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_DRAWS = 1000


def _cities(main: str) -> list:
    return json.load(open(OUT / main / "summary.json", encoding="utf-8"))["cities"]


def _od() -> Path:
    od = OUT / "checks"
    od.mkdir(parents=True, exist_ok=True)
    return od


def _flag_f(f_star, flag) -> float:
    """f* for the ranking comparisons: the value when found ('ok', 'below'), 0.9 when above the largest level."""
    if flag in ("ok", "below"):
        return float(f_star)
    return 0.9 if flag == "above" else np.nan


# ------------------------------------------------------------------------------------------------ pairs
def _pair_stats(vals, whole, eps):
    ok = np.isfinite(vals) & np.isfinite(whole)
    v, t = vals[ok], whole[ok]
    i, j = np.triu_indices(v.size, 1)
    dt, dv = t[i] - t[j], v[i] - v[j]
    same = np.sign(dt) == np.sign(dv)
    big = np.abs(dt) > 2 * eps
    return same.mean(), same[big].mean() if big.any() else np.nan, big.sum(), np.median(np.abs(dv - dt)) / eps


def _draws(pools, whole, eps, rng):
    res = np.array([_pair_stats(np.array([rng.choice(pl) if pl.size else np.nan for pl in pools]), whole, eps)
                    for _ in range(N_DRAWS)])
    return dict(all_pairs_same_order=np.median(res[:, 0]), distinct_pairs_same_order=np.median(res[:, 1]),
                distinct_pairs_same_order_p05=np.quantile(res[:, 1], 0.05), n_distinct_pairs=int(np.median(res[:, 2])),
                median_abs_error_of_difference_eps=np.median(res[:, 3]))


def pairs(main: str) -> None:
    # traffic diagnostics (E8 windows)
    R, A = {}, {}
    for c in _cities(main):
        p = OUT / "e8" / "cities" / f"{c}.json"
        if p.exists():
            R[c] = json.load(open(p))
            A[c] = np.load(OUT / "e8" / "cities" / f"{c}.npz")
    cw = pd.read_csv(OUT / main / "critical_windows.csv")
    e3 = pd.read_csv(OUT / "e3" / "e3_cities.csv")
    cl = list(R)
    rows = []
    for p in ("sd_log", "cv_k32", "hn_k32"):
        q = cw[cw.probe == p].set_index("city") if p == "sd_log" else e3[(e3.variant == "sum_k32") & (e3.probe == p[:2])].set_index("city")
        fst = {c: _flag_f(q.loc[c, "f_star"], q.loc[c, "flag"]) for c in cl if c in q.index}
        whole = np.array([R[c]["theta_E"][p] for c in cl], float)
        eps = 0.25 * np.subtract(*np.nanpercentile(whole, [75, 25]))
        for lab, fmap in [(f"f={f}", {c: f for c in cl}) for f in LEVELS] + [("f*", fst)]:
            pools = []
            for c in cl:
                fs, f = np.array(R[c]["levels"]), fmap.get(c, np.nan)
                if not np.isfinite(f):
                    pools.append(np.array([]))
                    continue
                i = int(np.argmin(np.abs(fs - f))) if lab != "f*" else min(int(np.searchsorted(fs, f - 1e-9)), fs.size - 1)
                v = A[c][f"{p}_{i}"]
                pools.append(v[np.isfinite(v)])
            rng = np.random.default_rng(zlib.crc32(f"pairs|{p}|{lab}".encode()))
            rows.append(dict(probe=p, comparison="window location", at=lab, **_draws(pools, whole, eps, rng)))
        pools = [np.array(R[c]["single_day"][p], float) for c in cl]
        pools = [pl[np.isfinite(pl)] for pl in pools]
        rng = np.random.default_rng(zlib.crc32(f"pairs|{p}|day".encode()))
        rows.append(dict(probe=p, comparison="dates", at="single weekday", **_draws(pools, whole, eps, rng)))
    d = pd.DataFrame(rows)
    d.to_csv(_od() / "pairs_traffic.csv", index=False)
    print(d.round(3).to_string(index=False))
    # road-network metrics (E4b windows)
    cr = pd.read_csv(OUT / "e4b" / "critical.csv")
    D = {f.stem: pd.read_csv(f) for f in sorted((OUT / "e4b" / "network").glob("*.csv"))}
    cl = sorted(D)
    rows = []
    for k in ("R_random", "R_btw", "eff_loss"):
        whole = np.array([D[c][D[c].f >= 0.999][k].iloc[0] for c in cl], float)
        eps = 0.25 * np.subtract(*np.nanpercentile(whole, [75, 25]))
        q = cr[cr.metric == k].set_index("city")
        fst = {c: _flag_f(q.loc[c, "f_star"], q.loc[c, "flag"]) for c in cl if c in q.index}
        for lab, fmap in [(f"f={f:g}", {c: f for c in cl}) for f in LEVELS] + [("f*", fst)]:
            pools = []
            for c in cl:
                f, d = fmap.get(c, np.nan), D[c][D[c].f < 0.999]
                lv = np.array(sorted(d.f.unique()))
                if not np.isfinite(f) or lv.size == 0:
                    pools.append(np.array([]))
                    continue
                i = int(np.argmin(np.abs(lv - f))) if lab != "f*" else min(int(np.searchsorted(lv, f - 1e-9)), lv.size - 1)
                v = d[np.isclose(d.f, lv[i])][k].to_numpy(float)
                pools.append(v[np.isfinite(v)])
            rng = np.random.default_rng(zlib.crc32(f"pairs|{k}|{lab}".encode()))
            rows.append(dict(metric=k, at=lab, **_draws(pools, whole, eps, rng)))
    d = pd.DataFrame(rows)
    d.to_csv(_od() / "pairs_network.csv", index=False)
    print(d.round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ centres
def centres(main: str) -> None:
    t = pd.read_csv(OUT / "e8" / "t_centres_fstar.csv")
    cw = pd.read_csv(OUT / main / "critical_windows.csv")
    nc = t.groupby("city").n_centres.first()
    cities = _cities(main)
    st = pd.Series({c: json.load(open(OUT / main / c / "city.json"))["n_stations"] for c in cities})
    per = pd.DataFrame(dict(n_centres=nc.reindex(cities), n_stations=st))
    rows = []
    for p in ("sd_log", "cv500", "hn500", "mean_log", "site_cv"):
        q = cw[(cw.probe == p) & cw.flag.isin(["ok", "above", "below"])].set_index("city")
        f = pd.Series(np.where(q.flag == "above", 0.95, q.f_star), index=q.index, dtype=float)
        per[f"f_star_{p}"] = f.reindex(cities)
        x, s = per.n_centres.reindex(f.index), per.n_stations.reindex(f.index)
        r_cf, p_cf = stats.spearmanr(x, f)
        r_cs = stats.spearmanr(x, s)[0]
        r_sf = stats.spearmanr(s, f)[0]
        partial = (r_cf - r_cs * r_sf) / np.sqrt((1 - r_cs ** 2) * (1 - r_sf ** 2))
        one, many = f[x == 1], f[x > 1]
        mw = stats.mannwhitneyu(one, many, alternative="two-sided").pvalue
        rows.append(dict(probe=p, n_cities=int(f.size), rho_centres_fstar=r_cf, p_centres_fstar=p_cf,
                         rho_centres_stations=r_cs, rho_stations_fstar=r_sf, partial_rho_given_stations=partial,
                         n_one_centre=int(one.size), median_fstar_one_centre=one.median(), n_several_centres=int(many.size),
                         median_fstar_several_centres=many.median(), mannwhitney_p=mw))
    od = _od()
    per.rename_axis("city").reset_index().to_csv(od / "centres_fstar.csv", index=False)
    s = pd.DataFrame(rows)
    s.to_csv(od / "centres_summary.csv", index=False)
    print(s.round(3).to_string(index=False))
    # sigma_z: f* from the windows containing the main centre, only secondary centres or no centre (E8)
    g = t[t.probe == "sd_log"].set_index("city")
    both = g[(g.flag_main == "ok") & (g.flag_all == "ok")]
    dm = both.f_main - both.f_all
    multi = g[g.n_centres > 1]
    groups = dict(
        main_vs_all=dict(n_cities=int(both.shape[0]), n_lower=int((dm < -1e-9).sum()), n_within_0p05=int((dm.abs() <= 0.05).sum()),
                         median_difference=float(dm.median()), q25_difference=float(dm.quantile(0.25)),
                         q75_difference=float(dm.quantile(0.75))),
        secondary_only=dict(n_cities_several_centres=int(multi.shape[0]), n_reaching_tolerance=int((multi.flag_secondary_only == "ok").sum()),
                            n_not_reaching=int((multi.flag_secondary_only == "above").sum()),
                            n_too_few_windows=int((multi.flag_secondary_only == "none").sum())),
        no_centre=dict(n_cities=int(g.flag_none.isin(["ok", "above", "below"]).sum()), n_reaching_tolerance=int((g.flag_none == "ok").sum())),
        n_centres_distribution={str(k): int(v) for k, v in nc.value_counts().sort_index().items()})
    json.dump(groups, open(od / "centres_groups.json", "w"), indent=1)
    print(json.dumps(groups, indent=1))


# ------------------------------------------------------------------------------------------------ error law
def _normalised_errors() -> pd.DataFrame:
    """The values plotted in Fig. 5a,b: P90 / tolerance divided by its value at f = 0.5, against (1 - f) / f."""
    from ..figures import common as C
    tc, _, _ = C.traffic_curves()
    nc, _, _, nlv = C.network_curves()
    tl = C.levels()
    rows = []
    for fam, keys, lv, cur in (("traffic", C.TRAFFIC, tl, tc), ("network", C.NETWORK, nlv, nc)):
        i5 = int(np.argmin(np.abs(lv - 0.5)))
        u = (1 - lv) / lv
        for k in keys:
            for c, y in cur[k].items():
                if not np.isfinite(y[i5]) or y[i5] <= 0:
                    continue
                v = y / y[i5]
                ok = np.isfinite(v) & (v > 0)
                rows += [dict(family=fam, metric=k, city=c, f=f, u=uu, ratio=vv) for f, uu, vv in zip(lv[ok], u[ok], v[ok])]
    return pd.DataFrame(rows)


def _fit_origin(x, y):
    b = (x * y).sum() / (x * x).sum()
    res = y - b * x
    return b, 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum(), res


def error_law() -> None:
    d = _normalised_errors()
    d = d[~np.isclose(d.f, 0.5)]
    fits = []
    for (fam, k, c), g in d.groupby(["family", "metric", "city"], sort=False):
        if len(g) < 3:
            continue
        b, r2, _ = _fit_origin(np.log(g.u.to_numpy()), np.log(g.ratio.to_numpy()))
        fits.append(dict(family=fam, metric=k, city=c, b=b, r2=r2, n_levels=len(g)))
    fits = pd.DataFrame(fits)
    rows = []
    for (fam, k), g in fits.groupby(["family", "metric"], sort=False):
        rows.append(dict(family=fam, metric=k, scope="per city", n_cities=len(g), median_b=g.b.median(), median_r2=g.r2.median()))
    for fam, g in d.groupby("family", sort=False):
        for fmin in (0.0, 0.2):
            h = g[g.f >= fmin - 1e-9]
            b, r2, res = _fit_origin(np.log(h.u.to_numpy()), np.log(h.ratio.to_numpy()))
            dev = 100 * (np.exp(np.abs(res)) - 1)
            rows.append(dict(family=fam, metric="all", scope=f"pooled, f >= {fmin:g}", n_cities=h.city.nunique(), n_points=len(h),
                             pooled_b=b, median_deviation_pct=np.median(dev), p90_deviation_pct=np.quantile(dev, 0.9)))
    od = _od()
    fits.to_csv(od / "error_law_fits.csv", index=False)
    s = pd.DataFrame(rows)
    s.to_csv(od / "error_law_summary.csv", index=False)
    print(s.round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ cleaning
def cleaning(main: str) -> None:
    cities = _cities(main)
    x0 = OUT / "x0"
    q = pd.read_csv(x0 / "qc_city.csv").set_index("city").loc[cities]
    readable = q.n_records - q.n_error_flag
    D = pd.concat([pd.read_parquet(x0 / "per_city" / c / "station_daily.parquet").assign(city=c) for c in cities])
    w = D[D.city_weekday].copy()
    w["date"] = pd.to_datetime(w.date).dt.strftime("%Y-%m-%d")
    w["ratio"] = w.flow_veh / w.groupby(["city", "station_id"]).flow_veh.transform("median")
    fl = w[w.outlier]
    cd = w.groupby(["city", "date"]).outlier.agg(["mean", "sum"])
    cd = cd[cd["sum"] > 0]
    ev = pd.read_csv(OUT / "e10" / "thr0p5" / "events.csv")
    rain_days = set(zip(ev.city, pd.to_datetime(ev.day).dt.strftime("%Y-%m-%d")))
    wr = w[w.city.isin(set(ev.city))]
    is_rain = np.array([(c, d) in rain_days for c, d in zip(wr.city, wr.date)])
    stuck = {c: json.load(open(OUT / main / c / "city.json"))["n_stuck_days"] for c in cities}
    per = pd.DataFrame(dict(readings_without_error_flag=readable, readings_above_cap=q.n_above_cap,
                            share_above_cap_pct=100 * q.n_above_cap / readable,
                            station_days_analysis=w.groupby("city").size().reindex(cities),
                            station_days_outlier=w.groupby("city").outlier.sum().reindex(cities),
                            station_days_stuck=pd.Series(stuck)))
    summary = dict(
        share_readings_above_cap_pct=float(100 * q.n_above_cap.sum() / readable.sum()),
        max_city_share_above_cap_pct=float(per.share_above_cap_pct.max()), max_city_above_cap=str(per.share_above_cap_pct.idxmax()),
        station_days_analysis=int(len(w)), station_days_outlier=int(w.outlier.sum()), share_outlier_pct=float(100 * w.outlier.mean()),
        outlier_share_below_median_pct=float(100 * (fl.ratio < 1).mean()),
        outlier_median_ratio_below=float(fl.ratio[fl.ratio < 1].median()), outlier_median_ratio_above=float(fl.ratio[fl.ratio > 1].median()),
        city_days_with_outlier=int(len(cd)), median_share_flagged_on_such_days_pct=float(100 * cd["mean"].median()),
        p90_share_flagged_on_such_days_pct=float(100 * cd["mean"].quantile(0.9)),
        n_cities_with_rain_events=int(wr.city.nunique()), share_outlier_rain_days_pct=float(100 * wr.outlier[is_rain].mean()),
        share_outlier_other_days_pct=float(100 * wr.outlier[~is_rain].mean()),
        station_days_stuck=int(sum(stuck.values())), share_stuck_pct=float(100 * sum(stuck.values()) / len(w)))
    od = _od()
    per.rename_axis("city").reset_index().to_csv(od / "cleaning_cities.csv", index=False)
    json.dump(summary, open(od / "cleaning_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["pairs", "centres", "error_law", "cleaning", "all"])
    ap.add_argument("--main", default="main34")
    a = ap.parse_args(argv)
    if a.what in ("pairs", "all"):
        pairs(a.main)
    if a.what in ("centres", "all"):
        centres(a.main)
    if a.what in ("error_law", "all"):
        error_law()
    if a.what in ("cleaning", "all"):
        cleaning(a.main)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
