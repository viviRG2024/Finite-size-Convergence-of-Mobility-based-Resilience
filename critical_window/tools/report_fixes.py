"""Bootstrap interval of f* and reporting of correlation lengths (run after run_main).

    python -m critical_window.tools.report_fixes [--dir critical_window/out/main] [--x0 critical_window/out/x0]

These steps do not change f*, a censoring flag or a test; they only change how uncertainty and
correlation lengths are reported.

D1  Bootstrap CI of f*. run_main computed the percentile CI from the replicates flagged 'ok' only, so for censored
    cities (e.g. 'above' in most replicates) the CI described the minority of replicates that met the criterion and
    could exclude the point estimate. Here every replicate enters: 'above' counts as larger than every level (> 0.9),
    'below' as smaller than every usable level. A CI end that falls on a censored replicate is reported as
    '> 0.9' or '<= f_min'. The shares of 'ok', 'above' and 'below' replicates are reported.
D2  Correlation length. When the first lag bin with >= 30 pairs is already below 1/e there is no crossing to interpolate;
    run_main then reported that bin's centre. Such values are upper bounds (xi_hat <= bin centre) and are flagged
    'upper_bound'; A* / xi_hat^2 and the H3 A/xi^2 comparison use only interpolated ('ok') values.

Writes <dir>/critical_windows_ci.csv, <dir>/xi_flags.csv and <dir>/h3_A_over_xi2.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.cfg import load_config, repo_path
from ..core import criteria, probes
from ..main.city_stage import PROBES
from ..main.data import load_city


def _ci(levels, usable, boot, eps):
    """Censoring-aware percentile CI of f* over bootstrap replicates (columns of boot)."""
    vals, flags = [], []
    for b in range(boot.shape[1]):
        r = criteria.critical_f(levels, np.where(usable, boot[:, b], np.nan), eps)
        flags.append(r["flag"])
        vals.append(np.inf if r["flag"] == "above" else (-np.inf if r["flag"] == "below" else r["f_star"]))
    v = np.array(vals, float)
    v = v[~np.isnan(v)]
    flags = np.array(flags)
    out = dict(boot_ok=float(np.mean(flags == "ok")), boot_above=float(np.mean(flags == "above")),
               boot_below=float(np.mean(flags == "below")))
    if v.size < 20:
        return dict(out, ci_lo=np.nan, ci_hi=np.nan, ci_lo_txt="", ci_hi_txt="")
    s = np.sort(v)
    lo, hi = s[int(np.floor(0.025 * (s.size - 1)))], s[int(np.ceil(0.975 * (s.size - 1)))]
    fmin = float(levels[np.flatnonzero(usable)[0]]) if usable.any() else np.nan
    txt = lambda x: ("> 0.9" if x == np.inf else (f"<= {fmin:g}" if x == -np.inf else f"{x:.3f}"))
    return dict(out, ci_lo=float(lo) if np.isfinite(lo) else np.nan, ci_hi=float(hi) if np.isfinite(hi) else np.nan,
                ci_lo_txt=txt(lo), ci_hi_txt=txt(hi))


def xi_flag(cd, tag="main") -> tuple:
    """Return (flag, first usable bin centre) for the X1 estimate: 'ok' (interpolated), 'upper_bound' or 'none'."""
    rng = np.random.default_rng(zlib.crc32("|".join((cd.city, tag, "xi")).encode()))   # same stream as city_stage._rng
    bins, rho, _ = probes.correlogram(cd.x, cd.y, np.log(cd.F), cd.short_side / 2, rng=rng)
    mid = 0.5 * (bins[:-1] + bins[1:])
    fin = np.flatnonzero(np.isfinite(rho))
    if fin.size == 0:
        return "none", np.nan
    below = fin[rho[fin] < np.exp(-1)]
    if below.size == 0:
        return "none", float(mid[fin[0]])
    return ("upper_bound" if below[0] == fin[0] else "ok"), float(mid[fin[0]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    cfg = load_config()
    out = repo_path(cfg["paths"]["out_dir"])
    ap.add_argument("--dir", default=str(out / "main"))
    ap.add_argument("--x0", default=str(out / "x0"))
    a = ap.parse_args(argv)
    d, x0 = Path(a.dir), Path(a.x0)
    m = cfg["main"]
    sc = json.load(open(d / "summary.json", encoding="utf-8"))
    eps, cities = sc["eps"], sc["cities"]
    cw = pd.read_csv(d / "critical_windows.csv")

    rows = []
    for c in cities:
        js = json.load(open(d / c / "city.json", encoding="utf-8"))
        arr = dict(np.load(d / c / "city.npz"))
        levels = np.array(js["levels"])
        usable = arr["lev__usable"] >= int(m["min_usable_windows"])
        for p in PROBES:
            rows.append(dict(probe=p, city=c, **_ci(levels, usable, arr[f"{p}__p90_boot"], eps[p])))
    ci = pd.DataFrame(rows)
    res = cw.drop(columns=["f_star_ci_lo", "f_star_ci_hi", "boot_ok_share"]).merge(ci, on=["probe", "city"])

    xrows = []
    for c in cities:
        js = json.load(open(d / c / "city.json", encoding="utf-8"))
        flag, first = xi_flag(load_city(c, x0, m))
        xrows.append(dict(city=c, xi_m=js["x1"]["xi"], xi_flag=flag, first_bin_centre_m=first,
                          xi_identifiable=js["x1"]["xi_identifiable"], short_km=js["x1"]["short_side_m"] / 1000))
    xf = pd.DataFrame(xrows)
    res = res.merge(xf[["city", "xi_flag"]], on="city")
    ok_xi = (res["xi_flag"] == "ok") & res["xi_identifiable"]
    res["A_over_xi2"] = np.where(ok_xi, res["A_over_xi2"], np.nan)

    h3 = []
    for p in PROBES:
        t = res[(res.probe == p) & (res.flag == "ok")]
        v = t["A_over_xi2"].to_numpy(float)
        v = v[np.isfinite(v) & (v > 0)]
        f = t.loc[np.isfinite(t["A_over_xi2"]), "f_star"].to_numpy(float)
        h3.append(dict(probe=p, n_cities=int(v.size), sd_log_A_over_xi2=float(np.std(np.log(v), ddof=1)) if v.size >= 3 else np.nan,
                       sd_log_f_same_cities=float(np.std(np.log(f), ddof=1)) if f.size >= 3 else np.nan))
    res.to_csv(d / "critical_windows_ci.csv", index=False)
    xf.to_csv(d / "xi_flags.csv", index=False)
    pd.DataFrame(h3).to_csv(d / "h3_A_over_xi2.csv", index=False)
    pd.set_option("display.width", 250)
    print(xf.round(3).to_string(index=False))
    print(pd.DataFrame(h3).round(3).to_string(index=False))
    show = res[res.probe.isin(["cv500", "hn500", "sd_log"])][["probe", "city", "flag", "f_star", "ci_lo_txt", "ci_hi_txt",
                                                               "boot_ok", "boot_above", "boot_below"]]
    print(show.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
