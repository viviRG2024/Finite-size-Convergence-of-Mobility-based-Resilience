"""E4: road-network metrics on 12 coverage levels, two flow-aware metrics, the common coverage law on the network
metrics, and whether the critical window of the traffic probes carries over to the network metrics.

    python -m critical_window.tools.e4_network run      [--main main34] [--x0 DIR] [--iutfd DIR] [--workers 2]
    python -m critical_window.tools.e4_network summary  [--main main34] [--x11 resilience34/network]
    python -m critical_window.tools.e4_network noise    [--main main34] [--workers 2]

--kind e4b (the default) gives the road-network metrics of the paper (resil/network_e4b.py; outputs in out/e4b/).
--kind e4 gives the earlier settings with higher Monte-Carlo noise (resil/network_e4.py; out/e4/). It is kept for the
noise check that motivated E4b (noise --kind e4); its summary needs the per-window rows of an earlier run (--x11).

noise   : Monte-Carlo noise of the sampled metrics (betweenness and efficiency from sampled sources, random removal
          orders, sampled accessibility origins): each metric recomputed with 8 seeds on the whole graph and on the first
          3 windows at f = 0.5; SD over seeds against eps -> out/e4/noise.csv, noise_summary.csv

run     : per city, out/e4/network/<city>.csv (resil/network_e4.city_e4)
summary : merges with the X11 rows (R_random, R_btw, eff_loss on the 8 X11 levels and the whole graph) and writes out/e4/
  levels.csv       P90 error, signed bias and relative bias per city, metric and level
  critical.csv     f* on the 12 levels per city and metric, with the X11 f* (8 levels) for comparison
  law_fits.csv     slope b, C/eps (error at f = 0.5 relative to eps), R^2 per city and metric
  law_summary.csv
  bias.csv         median relative bias per metric and level across cities, share of cities with a positive bias
  transfer.csv     P90/eps of each network metric at the f* of each traffic probe; share of cities within tolerance
  fstar_corr.csv   Spearman correlations of f* (and A* = f* x area) between traffic probes and network metrics
  onepoint.csv     f* predicted from the error at f = 0.5 and an exponent b (leave-one-out median, 0.5, 0.7)
  summary.json     eps per metric and headline numbers
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..common.cfg import load_config, repo_path
from ..core import criteria
from ..main.data import load_city
from ..resil import network, network_e4, network_e4b

OUT = Path(__file__).resolve().parents[1] / "out"
METRICS = ("R_random", "R_btw", "eff_loss", "dw_eff_loss", "acc_loss")
X11_METRICS = ("R_random", "R_btw", "eff_loss")
LEVELS = np.array(network_e4.ALL_LEVELS)
MIN_WIN = 10                       # usable level: >= 10 windows with a finite value (as X11)


def _job(a):
    city, x0, iutfd, m, od, kind = a
    t0 = time.time()
    if (Path(od) / f"{city}.csv").exists():                 # done meanwhile by another process
        return city, "ok (exists)", 0
    try:
        inv = pd.read_csv(Path(x0) / "network_inventory.csv").set_index("city")
        if not isinstance(inv.loc[city, "road_file"], str):
            return city, "no road network", 0
        path = Path(iutfd) / city / inv.loc[city, "road_file"].replace("\\", "/")
        qc = json.load(open(Path(x0) / "per_city" / city / "qc.json", encoding="utf-8"))
        cd = load_city(city, Path(x0), m)
        seg = network_e4.load_segments_xy(path, inv.loc[city, "road_source"], int(qc["epsg"]))
        rows = network_e4.city_e4(city, seg, cd) if kind == "e4" else network_e4b.city_e4b(city, seg, cd)
        pd.DataFrame(rows).assign(city=city, area_km2=cd.area_km2).to_csv(Path(od) / f"{city}.csv", index=False)
        return city, "ok", round(time.time() - t0)
    except Exception as exc:
        import traceback
        return city, f"error: {exc}\n{traceback.format_exc()}", round(time.time() - t0)


def run(main_tag, x0, iutfd, workers, x11_dir, kind="e4", order="large"):
    cfg = load_config()
    m = cfg["main"]
    od = OUT / kind / "network"
    od.mkdir(parents=True, exist_ok=True)
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    sp = OUT / x11_dir / "network_critical.csv"                      # only used to order the cities by graph size
    size = pd.read_csv(sp).set_index("city")["n_nodes_E"] if sp.exists() else pd.Series(dtype=float)
    todo = sorted([c for c in cities if not (od / f"{c}.csv").exists()], key=lambda c: -size.get(c, 0))   # largest first
    if order == "small":
        todo = todo[::-1]
    with Pool(workers) as pool:
        for c, st, s in pool.imap_unordered(_job, [(c, str(x0), str(iutfd), m, str(od), kind) for c in todo]):
            print(f"[{time.strftime('%H:%M:%S')}] {c}: {st.splitlines()[0]} ({s} s)", flush=True)
            if st.startswith("error"):
                print(st, flush=True)


def merged(city, x11_dir, kind="e4"):
    """E4 rows with R_random, R_btw, eff_loss taken from X11 on the X11 levels and the whole graph (E4b: own rows)."""
    if kind != "e4":
        e = pd.read_csv(OUT / kind / "network" / f"{city}.csv")
        e["fk"] = e.f.round(3)
        return e
    e = pd.read_csv(OUT / "e4" / "network" / f"{city}.csv")
    x = pd.read_csv(OUT / x11_dir / f"{city}.csv")
    e["fk"], x["fk"] = e.f.round(3), x.f.round(3)
    e, x = e.set_index(["fk", "pos"]), x.set_index(["fk", "pos"])
    old = e.index.get_level_values(0).isin(list(np.round(network.LEVELS, 3)) + [1.0])
    nn = x["n_nodes"].reindex(e.index[old])
    if not np.array_equal(nn.to_numpy(float), e.loc[old, "n_nodes"].to_numpy(float), equal_nan=True):
        raise ValueError(f"{city}: window graphs differ from X11")
    for k in X11_METRICS:
        if k not in e:
            e[k] = np.nan
        e.loc[old, k] = x[k].reindex(e.index[old]).to_numpy()
    return e.reset_index()


def _p90_curves(d, k):
    E = float(d.loc[d.fk == 1.0, k].iloc[0])
    p90, bias, rbias, nuse = [], [], [], []
    for f in LEVELS:
        w = d.loc[np.isclose(d.f, f), k].dropna().to_numpy()
        ok = w.size >= MIN_WIN and np.isfinite(E)
        p90.append(float(np.quantile(np.abs(w - E), 0.9)) if ok else np.nan)
        bias.append(float(np.median(w - E)) if ok else np.nan)
        rbias.append(float(np.median(w - E) / E) if ok and E != 0 else np.nan)
        nuse.append(int(w.size))
    return E, np.array(p90), np.array(bias), np.array(rbias), np.array(nuse)


def _law(fs, st, eps):
    ok = np.isfinite(st) & (st > 0) & (fs < 1)
    if ok.sum() < 4:
        return None
    u = np.log((1 - fs[ok]) / fs[ok])
    y = np.log(st[ok] / eps)
    b, a = np.polyfit(u, y, 1)
    r2 = 1 - np.sum((y - (a + b * u)) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(n_levels=int(ok.sum()), slope=float(b), C_over_eps=float(np.exp(a)), r2=float(r2))


def _at(fs, st, f0):
    """P90 at coverage f0, interpolated on log f / log P90 between usable levels (NaN outside them)."""
    ok = np.isfinite(st) & (st > 0)
    if not np.isfinite(f0) or ok.sum() < 2 or f0 < fs[ok].min() or f0 > fs[ok].max():
        return np.nan
    return float(np.exp(np.interp(np.log(f0), np.log(fs[ok]), np.log(st[ok]))))


def summary(main_tag, x11_dir, kind="e4"):
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    cities = [c for c in cities if (OUT / kind / "network" / f"{c}.csv").exists()]
    D = {c: merged(c, x11_dir, kind) for c in cities}
    area = {c: float(D[c].area_km2.iloc[0]) for c in cities}
    whole = {k: np.array([float(D[c].loc[D[c].fk == 1.0, k].iloc[0]) for c in cities]) for k in METRICS}
    eps = {k: criteria.tolerance(whole[k]) for k in METRICS}
    od = OUT / kind
    lev, crit, fits, curves = [], [], [], {}
    xp = OUT / x11_dir / "network_critical.csv"                      # earlier X11 run, only for the comparison columns
    x11c = pd.read_csv(xp).set_index("city") if xp.exists() else pd.DataFrame()
    for c in cities:
        for k in METRICS:
            E, p90, bias, rbias, nuse = _p90_curves(D[c], k)
            curves[(c, k)] = p90
            for f, a_, b_, r_, n_ in zip(LEVELS, p90, bias, rbias, nuse):
                lev.append(dict(city=c, metric=k, f=f, n_usable=n_, p90=a_, p90_over_eps=a_ / eps[k], bias=b_, rel_bias=r_))
            cr = criteria.critical_f(LEVELS, p90, eps[k])
            r = dict(city=c, metric=k, area_km2=area[c], theta_E=E, f_star=cr["f_star"], flag=cr["flag"],
                     A_star=cr["f_star"] * area[c] if cr["flag"] == "ok" else np.nan)
            if k in X11_METRICS and c in x11c.index:
                r.update(f_star_x11=float(x11c.loc[c, f"f_star_{k}"]), flag_x11=x11c.loc[c, f"flag_{k}"])
            crit.append(r)
            lw = _law(LEVELS, p90, eps[k])
            if lw:
                fits.append(dict(city=c, metric=k, **lw))
    lev, crit, fits = pd.DataFrame(lev), pd.DataFrame(crit), pd.DataFrame(fits)
    lev.to_csv(od / "levels.csv", index=False)
    crit.to_csv(od / "critical.csv", index=False)
    fits.to_csv(od / "law_fits.csv", index=False)
    law = fits.groupby("metric").agg(cities=("city", "count"), slope_median=("slope", "median"),
                                     slope_q25=("slope", lambda x: x.quantile(0.25)), slope_q75=("slope", lambda x: x.quantile(0.75)),
                                     r2_median=("r2", "median"), share_r2_ge_0p9=("r2", lambda x: (x >= 0.9).mean()),
                                     C_median=("C_over_eps", "median")).reindex(METRICS)
    law.to_csv(od / "law_summary.csv")
    # bias by level
    bias = lev.groupby(["metric", "f"]).agg(cities=("rel_bias", lambda x: int(x.notna().sum())), rel_bias_median=("rel_bias", "median"),
                                            share_positive=("rel_bias", lambda x: float((x.dropna() > 0).mean()) if x.notna().any() else np.nan),
                                            bias_over_p90_median=("bias", lambda x: np.nan)).reset_index()
    bop = lev.assign(r=np.abs(lev.bias) / lev.p90).groupby(["metric", "f"]).r.median().reset_index()
    bias["bias_over_p90_median"] = bop.r.to_numpy()
    bias.to_csv(od / "bias.csv", index=False)
    # critical coverage of the traffic probes
    cw = pd.read_csv(OUT / main_tag / "critical_windows.csv")
    e3 = pd.read_csv(OUT / "e3" / "e3_cities.csv")
    probes = {}
    for p in ("sd_log", "mean_log", "site_cv"):
        q = cw[cw.probe == p].set_index("city")
        probes[p] = {c: (float(q.loc[c, "f_star"]), q.loc[c, "flag"]) for c in cities if c in q.index}
    for p, lab in (("cv", "cv_k32"), ("hn", "hn_k32")):
        q = e3[(e3.variant == "sum_k32") & (e3.probe == p)].set_index("city")
        probes[lab] = {c: (float(q.loc[c, "f_star"]), q.loc[c, "flag"]) for c in cities if c in q.index}
    tr, corr = [], []
    ck = crit.set_index(["city", "metric"])
    for p, fp in probes.items():
        for k in METRICS:
            vals = []
            for c in cities:
                f0, fl = fp.get(c, (np.nan, "none"))
                if fl != "ok":
                    continue
                v = _at(LEVELS, curves[(c, k)], f0)
                if np.isfinite(v):
                    vals.append(v / eps[k])
            vals = np.array(vals)
            tr.append(dict(probe=p, metric=k, cities=int(vals.size), share_within=float((vals <= 1).mean()) if vals.size else np.nan,
                           p90_over_eps_median=float(np.median(vals)) if vals.size else np.nan))
            both = [c for c in cities if fp.get(c, (0, "none"))[1] == "ok" and ck.loc[(c, k), "flag"] == "ok"]
            if len(both) >= 5:
                a1 = np.array([fp[c][0] for c in both])
                a2 = np.array([ck.loc[(c, k), "f_star"] for c in both])
                ar = np.array([area[c] for c in both])
                corr.append(dict(probe=p, metric=k, cities=len(both), spearman_fstar=float(stats.spearmanr(a1, a2)[0]),
                                 spearman_Astar=float(stats.spearmanr(a1 * ar, a2 * ar)[0]),
                                 median_fstar_probe=float(np.median(a1)), median_fstar_metric=float(np.median(a2))))
    tr, corr = pd.DataFrame(tr), pd.DataFrame(corr)
    tr.to_csv(od / "transfer.csv", index=False)
    corr.to_csv(od / "fstar_corr.csv", index=False)
    # one-point calibration: error at f = 0.5 and an exponent b
    i5 = int(np.flatnonzero(np.isclose(LEVELS, 0.5))[0])
    one = []
    for k in METRICS:
        fk = fits[fits.metric == k].set_index("city")
        for c in cities:
            C = curves[(c, k)][i5] / eps[k]
            if not np.isfinite(C) or C <= 0:
                continue
            b_loo = float(fk.drop(index=c, errors="ignore").slope.median())
            row = dict(city=c, metric=k, C_over_eps=C, b_loo=b_loo, f_obs=ck.loc[(c, k), "f_star"], flag_obs=ck.loc[(c, k), "flag"])
            for lab, b in (("loo", b_loo), ("b0p5", 0.5), ("b0p7", 0.7)):
                row[f"f_pred_{lab}"] = 1 / (1 + (1 / C) ** (1 / b))
            one.append(row)
    one = pd.DataFrame(one)
    one.to_csv(od / "onepoint.csv", index=False)
    osum = []
    for (k), g in one.groupby("metric"):
        ok = g[g.flag_obs == "ok"]
        ab = g[g.flag_obs == "above"]
        for lab in ("loo", "b0p5", "b0p7"):
            d = ok[f"f_pred_{lab}"] - ok.f_obs
            osum.append(dict(metric=k, b=lab, n_ok=len(ok), mae=float(d.abs().mean()), share_within_0p1=float((d.abs() <= 0.1).mean()),
                             spearman=float(stats.spearmanr(ok.f_obs, ok[f"f_pred_{lab}"])[0]) if len(ok) >= 5 else np.nan,
                             n_above=len(ab), share_above_pred_gt_0p8=float((ab[f"f_pred_{lab}"] > 0.8).mean()) if len(ab) else np.nan))
    osum = pd.DataFrame(osum)
    osum.to_csv(od / "onepoint_summary.csv", index=False)
    # headline
    head = dict(eps=eps, cities=len(cities))
    for k in METRICS:
        g = crit[crit.metric == k]
        okg = g[g.flag == "ok"]
        st = criteria.fstar_stats(g)
        head[k] = dict(n_ok=int(len(okg)), f_median=float(okg.f_star.median()) if len(okg) else None,
                       f_median_all=st["median"], f_median_all_lower_bound=st["median_lower_bound"],
                       n_above=int((g.flag == "above").sum()), n_below=int((g.flag == "below").sum()), n_none=int((g.flag == "none").sum()))
        if k in X11_METRICS and "flag_x11" in g:
            both = g[(g.flag == "ok") & (g.flag_x11 == "ok")]
            head[k].update(x11_n_ok=int((g.flag_x11 == "ok").sum()), x11_f_median=float(g[g.flag_x11 == "ok"].f_star_x11.median()),
                           mad_vs_x11=float((both.f_star - both.f_star_x11).abs().median()) if len(both) else None)
    json.dump(head, open(od / "summary.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(json.dumps(head, indent=1))
    print(law.round(3).to_string())
    print(bias[np.isclose(bias.f, 0.4) | np.isclose(bias.f, 0.2) | np.isclose(bias.f, 0.7)].round(3).to_string(index=False))
    print(tr.round(3).to_string(index=False))
    print(corr.round(3).to_string(index=False))
    print(osum.round(3).to_string(index=False))


N_SEEDS = 8


def _noise_job(a):
    city, x0, iutfd, m, kind = a
    try:
        inv = pd.read_csv(Path(x0) / "network_inventory.csv").set_index("city")
        path = Path(iutfd) / city / inv.loc[city, "road_file"].replace("\\", "/")
        qc = json.load(open(Path(x0) / "per_city" / city / "qc.json", encoding="utf-8"))
        cd = load_city(city, Path(x0), m)
        seg = network_e4.load_segments_xy(path, inv.loc[city, "road_source"], int(qc["epsg"]))
        from ..core import windows as W
        boxes = [("E", cd.extent)] + [(f"w{k}", b) for k, b in enumerate(W.random_windows(cd.extent, 0.5, 50, network_e4._rng(city, "main", "windows", 0.5))[:3])]
        rows = []
        for lab, box in boxes:
            n, E, L, xy = network_e4._window_graph(seg, box)
            if n < network.MIN_NODES:
                continue
            inb = W.in_box(cd.x, cd.y, box)
            for k in range(N_SEEDS if kind == "e4" else 4):
                r = dict(city=city, graph=lab, seed=k, n_nodes=n)
                if kind == "e4":
                    r.update({kk: v for kk, v in network.robustness(n, E, L, network_e4._rng(city, "noise", lab, k)).items() if kk in X11_METRICS})
                    r.update(network_e4.extra_metrics(n, E, L, xy, cd.x[inb], cd.y[inb], cd.F[inb], network_e4._rng(city, "noise-x", lab, k)))
                else:
                    r.update({kk: v for kk, v in network_e4b.metrics(n, E, L, xy, cd.x[inb], cd.y[inb], cd.F[inb],
                                                                     network_e4._rng(city, "noise-b", lab, k)).items() if kk in METRICS})
                rows.append(r)
        return city, rows, "ok"
    except Exception as exc:
        import traceback
        return city, [], f"error: {exc}\n{traceback.format_exc()}"


def noise(main_tag, x0, iutfd, workers, kind="e4"):
    m = load_config()["main"]
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    eps = json.load(open(OUT / "e4" / "summary.json"))["eps"]           # E4 tolerances, for comparison
    out = []
    with Pool(workers) as pool:
        for c, rows, st in pool.imap_unordered(_noise_job, [(c, str(x0), str(iutfd), m, kind) for c in cities]):
            print(c, st.splitlines()[0], flush=True)
            out += rows
    d = pd.DataFrame(out)
    d.to_csv(OUT / kind / "noise.csv", index=False)
    sd = d.groupby(["city", "graph"])[list(METRICS)].std(ddof=1).reset_index()
    rows = []
    for k in METRICS:
        for g, lab in ((sd.graph == "E", "whole"), (sd.graph != "E", "windows f=0.5")):
            v = sd.loc[g, k].dropna() / eps[k]
            rows.append(dict(metric=k, graphs=lab, n=int(v.size), sd_over_eps_median=float(v.median()), sd_over_eps_q90=float(v.quantile(0.9)),
                             floor_p90_over_eps_median=float(1.645 * np.sqrt(2) * v.median())))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / kind / "noise_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "summary", "noise"])
    ap.add_argument("--kind", default="e4b", choices=["e4", "e4b"], help="e4b: the metrics of the paper (default); e4: earlier settings, higher Monte-Carlo noise")
    ap.add_argument("--order", default="large", choices=["large", "small"], help="run the largest (default) or smallest graphs first")
    ap.add_argument("--main", default="main34")
    ap.add_argument("--x0", default=str(OUT / "x0"))
    ap.add_argument("--iutfd", default="")
    ap.add_argument("--x11", default="resilience34/network")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    (OUT / "e4").mkdir(parents=True, exist_ok=True)
    iutfd = Path(a.iutfd) if a.iutfd else repo_path(load_config()["paths"]["iutfd_dir"])
    (OUT / a.kind).mkdir(parents=True, exist_ok=True)
    if a.what == "run":
        run(a.main, a.x0, iutfd, a.workers, a.x11, a.kind, a.order)
    elif a.what == "noise":
        noise(a.main, a.x0, iutfd, a.workers, a.kind)
    else:
        summary(a.main, a.x11, a.kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
