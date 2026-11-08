"""E2: test the representativeness formula on the station-level probes, and the common coverage law on all metrics.

    python -m critical_window.tools.e2_formula windows   [--main main34] [--x0 DIR]   # per-window errors (cached)
    python -m critical_window.tools.e2_formula formula   [--main main34]              # f* observed vs predicted
    python -m critical_window.tools.e2_formula law       [--main main34] [--net resilience34/network]   (earlier network run;
                                                           not used in the paper: see tools/e4_network.py summary)

Formula (no fitted parameter):  err_q(f) = z_q * s * sqrt(1/n(f) - 1/N),   f* = 1 / (1 + N (eps / (z_q s))^2) when n = f N.
  s   = SD over stations of the influence function of the probe (mean_log: z - mean; sd_log: ((z - mean)^2 - var)/(2 sd);
        site_cv: ((x - mu)^2 - sd^2)/(2 sd mu) - sd (x - mu)/mu^2), i.e. the large-sample standard error times sqrt(N);
  N   = stations with flow > 0;  n(f) = f N ("simple") or the median station count of the usable windows ("counts");
  z_q = two-sided normal quantile for the share q of windows (0.8 -> 1.28, 0.9 -> 1.645, 0.95 -> 1.96).
Windows are the random windows of the main run (same seeds and usability rule), so f*_obs at q = 0.9 and 0.25 x IQR
reproduces the main-run value.
Writes out/e2/.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..common.cfg import load_config
from ..core import criteria, windows
from ..main.data import load_city

OUT = Path(__file__).resolve().parents[1] / "out"
PROBES = ("mean_log", "sd_log", "site_cv")
QS = (0.8, 0.9, 0.95)
MULTS = (0.1, 0.25, 0.5, 1.0)


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def probe_values(F: np.ndarray, M: np.ndarray) -> dict:
    """Probe values for each row of the station mask M (windows x stations)."""
    z = np.log(F)
    Mf = M.astype(float)
    n = Mf.sum(1)
    mz = (Mf @ z) / n
    vz = (Mf @ z ** 2 - n * mz ** 2) / (n - 1)
    mF = (Mf @ F) / n
    vF = (Mf @ F ** 2 - n * mF ** 2) / (n - 1)
    return dict(mean_log=mz, sd_log=np.sqrt(np.maximum(vz, 0)), site_cv=np.sqrt(np.maximum(vF, 0)) / mF)


def influence_sd(F: np.ndarray) -> dict:
    z = np.log(F)
    mu, sd = z.mean(), z.std(ddof=1)
    out = dict(mean_log=float(np.std(z - mu, ddof=1)),
               sd_log=float(np.std(((z - mu) ** 2 - sd ** 2) / (2 * sd), ddof=1)))
    m, s = F.mean(), F.std(ddof=1)
    out["site_cv"] = float(np.std(((F - m) ** 2 - s ** 2) / (2 * s * m) - s * (F - m) / m ** 2, ddof=1))
    return out


def windows_stage(main_tag: str, x0: Path):
    cfg = load_config()
    m = cfg["main"]
    od = OUT / "e2" / "windows"
    od.mkdir(parents=True, exist_ok=True)
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    fs = np.array([f for f in m["f_grid"] if f < 1.0])
    for c in cities:
        cd = load_city(c, x0, m)
        keep = cd.F > 0
        F, x, y = cd.F[keep], cd.x[keep], cd.y[keep]
        E = {k: float(v[0]) for k, v in probe_values(F, np.ones((1, F.size), bool)).items()}
        rec = dict(levels=fs, N=F.size, s=json.dumps(influence_sd(F)), theta_E=json.dumps(E))
        for i, f in enumerate(fs):
            wins = windows.random_windows(cd.extent, f, int(m["n_random"]), _rng(c, "main", "windows", f))
            M = np.array([windows.in_box(x, y, w) for w in wins])
            n = M.sum(1)
            use = n >= int(m["n_w"])
            rec[f"n_{i}"] = n
            if use.sum() >= int(m["min_usable_windows"]):
                V = probe_values(F, M[use])
                for p in PROBES:
                    rec[f"err_{p}_{i}"] = np.abs(V[p] - E[p])
        np.savez_compressed(od / f"{c}.npz", **rec)
        print(c, F.size, flush=True)


def _load(c):
    z = np.load(OUT / "e2" / "windows" / f"{c}.npz", allow_pickle=False)
    return z, json.loads(str(z["s"])), json.loads(str(z["theta_E"]))


def formula_stage(main_tag: str):
    cfg = load_config()
    m = cfg["main"]
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    od = OUT / "e2"
    thE = {}
    for c in cities:
        _, _, E = _load(c)
        thE[c] = E
    iqr = {p: float(np.subtract(*np.quantile([thE[c][p] for c in cities], [0.75, 0.25]))) for p in PROBES}
    rows, curves = [], []
    for c in cities:
        z, s, E = _load(c)
        fs, N = z["levels"], int(z["N"])
        for p in PROBES:
            for q in QS:
                zq = stats.norm.ppf((1 + q) / 2)
                stat = np.full(fs.size, np.nan)
                n_med = np.full(fs.size, np.nan)
                for i in range(fs.size):
                    n = z[f"n_{i}"]
                    if f"err_{p}_{i}" in z.files:
                        stat[i] = np.quantile(z[f"err_{p}_{i}"], q)
                        n_med[i] = np.median(n[n >= int(m["n_w"])])
                pred_counts = zq * s[p] * np.sqrt(np.maximum(1 / n_med - 1 / N, 0))
                pred_simple = zq * s[p] * np.sqrt((1 - fs) / (fs * N))
                if q == 0.9:
                    curves += [dict(city=c, probe=p, f=float(f), obs=float(o), pred_simple=float(ps), pred_counts=float(pc), n_med=float(nm), N=N)
                               for f, o, ps, pc, nm in zip(fs, stat, pred_simple, pred_counts, n_med)]
                for mult in MULTS:
                    eps = mult * iqr[p]
                    ob = criteria.critical_f(fs, stat, eps)
                    pc = criteria.critical_f(fs, np.where(np.isfinite(stat), pred_counts, np.nan), eps)
                    f_simple = 1 / (1 + N * (eps / (zq * s[p])) ** 2)
                    rows.append(dict(city=c, probe=p, q=q, mult=mult, eps=eps, N=N, s=s[p], f_obs=ob["f_star"], flag_obs=ob["flag"],
                                     f_pred_simple=f_simple, f_pred_counts=pc["f_star"], flag_pred_counts=pc["flag"]))
    res = pd.DataFrame(rows)
    res.to_csv(od / "fstar_obs_pred.csv", index=False)
    pd.DataFrame(curves).to_csv(od / "curves_obs_pred.csv", index=False)
    summ = []
    for (p, q, mult), g in res.groupby(["probe", "q", "mult"]):
        ok = g[g.flag_obs == "ok"]
        for col in ("f_pred_simple", "f_pred_counts"):
            d = ok[col] - ok.f_obs
            summ.append(dict(probe=p, q=q, mult=mult, predictor=col, n_ok=len(ok), n_cities=len(g),
                             median_obs=float(ok.f_obs.median()) if len(ok) else np.nan,
                             median_pred=float(ok[col].median()) if len(ok) else np.nan,
                             median_ratio=float((ok.f_obs / ok[col]).median()) if len(ok) else np.nan,
                             mae=float(d.abs().mean()) if len(ok) else np.nan,
                             share_within_0p1=float((d.abs() <= 0.1).mean()) if len(ok) else np.nan,
                             spearman=float(stats.spearmanr(ok.f_obs, ok[col])[0]) if len(ok) >= 5 else np.nan))
    summ = pd.DataFrame(summ)
    summ.to_csv(od / "formula_summary.csv", index=False)
    # censored cities: what the formula says
    cen = res[(res.q == 0.9) & (res.mult == 0.25) & (res.flag_obs != "ok")][["city", "probe", "N", "flag_obs", "f_pred_simple"]]
    cen.to_csv(od / "censored_cities.csv", index=False)
    json.dump(dict(iqr=iqr), open(od / "iqr.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(summ[(summ.q == 0.9)].round(3).to_string(index=False))
    print(cen.round(3).to_string(index=False))
    return res


def law_stage(main_tag: str, net_dir: str):
    """Per city and metric: log P90/eps on log((1 - f)/f) -> slope b, amplitude C at f = 0.5, R^2."""
    cw = pd.read_csv(OUT / main_tag / "critical_windows.csv")
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    rows = []

    def fit(c, metric, fs, st, eps, family):
        ok = np.isfinite(st) & (st > 0) & (fs < 1)
        if ok.sum() < 4:
            return
        u = np.log((1 - fs[ok]) / fs[ok])
        y = np.log(st[ok] / eps)
        b, a = np.polyfit(u, y, 1)
        r2 = 1 - np.sum((y - (a + b * u)) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append(dict(city=c, metric=metric, family=family, n_levels=int(ok.sum()), slope=b, C_over_eps=float(np.exp(a)), r2=r2))

    for c in cities:
        js = json.load(open(OUT / main_tag / c / "city.json", encoding="utf-8"))
        a = np.load(OUT / main_tag / c / "city.npz")
        fs = np.array(js["levels"], float)
        for p in ("sd_log", "mean_log", "site_cv", "hn500", "cv500"):
            eps = float(cw[(cw.probe == p) & (cw.city == c)].eps.iloc[0])
            fit(c, p, fs, a[f"{p}__p90_obs"], eps, "traffic")
    nd = OUT / net_dir
    if (nd / "summary.json").exists():
        eps = json.load(open(nd / "summary.json"))["eps"]
        for c in cities:
            if not (nd / f"{c}.csv").exists():
                continue
            d = pd.read_csv(nd / f"{c}.csv")
            E = d[d.f == 1.0].iloc[0]
            for k in ("R_random", "R_btw", "eff_loss"):
                lv = np.array(sorted(d.f[d.f < 1].unique()))
                st = []
                for f in lv:
                    w = d[np.isclose(d.f, f) & d[k].notna()][k]
                    st.append(float((w - E[k]).abs().quantile(0.9)) if len(w) >= 10 else np.nan)
                fit(c, k, lv, np.array(st), eps[k], "network")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "e2" / "law_fits.csv", index=False)
    s = res.groupby("metric").agg(cities=("city", "count"), slope_median=("slope", "median"), r2_median=("r2", "median"),
                                  share_r2_ge_0p9=("r2", lambda x: (x >= 0.9).mean()), C_median=("C_over_eps", "median"))
    s.to_csv(OUT / "e2" / "law_summary.csv")
    print(s.round(3).to_string())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["windows", "formula", "law"])
    ap.add_argument("--main", default="main34")
    ap.add_argument("--x0", default=str(OUT / "x0"))
    ap.add_argument("--net", default="resilience34/network")
    a = ap.parse_args(argv)
    (OUT / "e2").mkdir(parents=True, exist_ok=True)
    if a.what == "windows":
        windows_stage(a.main, Path(a.x0))
    elif a.what == "formula":
        formula_stage(a.main)
    else:
        law_stage(a.main, a.net)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
