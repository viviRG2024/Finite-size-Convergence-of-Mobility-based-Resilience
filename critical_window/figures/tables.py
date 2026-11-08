"""Table 1 of the paper (six representative cities), Supplementary Table 5 (critical coverage of the six main metrics in
all 34 cities) and Supplementary Table 6 (further quantities per city and the four further metrics).

f* is the smallest coverage from which the 90th percentile of the window error stays within the tolerance (0.25 x the
interquartile range of whole-city values across the cities); '> 0.9' = not reached below f = 0.9; '–' = no usable window
level (too few stations) or, for the stations at f*, f* above 0.9. Footer rows: median over the cities with a usable
level, cities above 0.9 ranked above every found f* ('> x' when the median falls among them), and the number of cities
with f* <= 0.8.
Correlation length: 1/e crossing of the correlogram of log flows; '< 50' = within the first 50 m bin; 'n.i.' = not
identifiable (the correlogram does not fall below 1/e inside a sixth of the study area). 95% interval of f* (sigma_z):
2.5-97.5% over 1,000 bootstrap replicates that resample the weekdays and the windows (tools/report_fixes.py).
Writes out/figures/Table1.csv, SuppTable5.csv, SuppTable6.csv and Tables.xlsx.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C

REP = ("london", "losangeles", "madrid", "melbourne", "hamburg", "zurich")
MAIN_T = ("sd_log", "cv", "hn")
MAIN_N = ("R_random", "R_btw", "eff_loss")
SUPP_T = ("mean_log", "site_cv")
SUPP_N = ("dw_eff_loss", "acc_loss")


def fmt(f, flag):
    if flag == "ok":
        return f"{f:.2f}"
    if flag == "above":
        return "> 0.9"
    if flag == "below":
        return f"≤ {f:.2f}"
    return "–"


def col(k):
    """Column names as in the paper's tables."""
    if k == "sd_log":
        return "σz"
    return C.SHORT[k] if k in MAIN_T + MAIN_N else C.LAB[k]


def footer(keys, fst, first_col="City"):
    med, le = {first_col: "Median"}, {first_col: "Cities with f* ≤ 0.8"}
    for k in keys:
        st = C.fstar_stats(fst[k])
        med[col(k)] = f"{st['median']:.2f}" if np.isfinite(st["median"]) else (
            f"> {st['median_lower_bound']:.2f}" if st["median_lower_bound"] < 0.9 else "> 0.9")
        le[col(k)] = f"{int(round(st['share_le_0_8'] * st['n']))} / {st['n']}"
    return [med, le]


def build():
    tc, teps, tf = C.traffic_curves()
    nc, neps, nf, nlv = C.network_curves()
    fst = {**tf, **nf}
    t2 = pd.read_csv(C.OUT / C.MAIN / "table2_cities.csv").set_index("city")
    ci = pd.read_csv(C.OUT / C.MAIN / "critical_windows_ci.csv")
    ci = ci[ci.probe == "sd_log"].set_index("city")
    ns = pd.read_csv(C.OUT / "e6" / "e6_nstar.csv")
    ns = ns[(ns.variant == "main") & (ns.probe == "sd_log")].set_index("city")
    rows = []
    for c in sorted(C.cities(), key=C.name):
        xi = ("< 50" if t2.loc[c, "xi_identifiable"] and t2.loc[c, "xi_m"] <= 25 else
              f"{t2.loc[c, 'xi_m']:.0f}" if t2.loc[c, "xi_identifiable"] else "n.i.")
        r = {"City": C.name(c), "Stations": int(t2.loc[c, "n_stations"]), "Correlation length (m)": xi,
             "Stations at f* (σz)": f"{ns.loc[c, 'n_star']:,.0f}" if c in ns.index and np.isfinite(ns.loc[c, "n_star"]) else "–",
             "95% interval of f* (σz)": (f"{ci.loc[c, 'ci_lo_txt']}–{ci.loc[c, 'ci_hi_txt']}" if c in ci.index and ci.loc[c, "flag"] == "ok"
                                               and str(ci.loc[c, "ci_lo_txt"]) not in ("nan", "") else "–")}
        for k in MAIN_T + SUPP_T + MAIN_N + SUPP_N:
            q = fst[k]
            r[col(k)] = fmt(q.loc[c, "f_star"], q.loc[c, "flag"]) if c in q.index else "–"
        rows.append(r)
    allt = pd.DataFrame(rows)
    med_st = float(np.median(allt["Stations"]))
    nstar = pd.to_numeric(allt["Stations at f* (σz)"].str.replace(",", ""), errors="coerce")
    for cname in ("Stations",):
        allt[cname] = allt[cname].map(lambda v: f"{v:,}")
    t1cols = ["City", "Stations"] + [col(k) for k in MAIN_T + MAIN_N] + ["Stations at f* (σz)"]
    t1 = allt.set_index("City").loc[[C.name(c) for c in REP]].reset_index()[t1cols]
    med = footer(MAIN_T + MAIN_N, fst)[0]
    med.update({"City": "All cities, median", "Stations": f"{med_st:,.0f}",
                "Stations at f* (σz)": f"{np.nanmedian(nstar):,.0f}"})
    t1 = pd.concat([t1, pd.DataFrame([med])], ignore_index=True)[t1cols]
    t5cols = ["City"] + [col(k) for k in MAIN_T + MAIN_N]
    tab5 = pd.concat([allt[t5cols], pd.DataFrame(footer(MAIN_T + MAIN_N, fst))], ignore_index=True)
    s6cols = ["City", "Correlation length (m)", "Stations at f* (σz)", "95% interval of f* (σz)"] + [col(k) for k in SUPP_T + SUPP_N]
    sup6 = pd.concat([allt[s6cols], pd.DataFrame(footer(SUPP_T + SUPP_N, fst))], ignore_index=True)
    C.FIG.mkdir(parents=True, exist_ok=True)
    t1.to_csv(C.FIG / "Table1.csv", index=False)
    tab5.to_csv(C.FIG / "SuppTable5.csv", index=False)
    sup6.to_csv(C.FIG / "SuppTable6.csv", index=False)
    with pd.ExcelWriter(C.FIG / "Tables.xlsx", engine="xlsxwriter") as xw:
        for name_, df in (("Table 1", t1), ("Supplementary Table 5", tab5), ("Supplementary Table 6", sup6)):
            df.fillna("").to_excel(xw, sheet_name=name_, index=False)
            ws = xw.sheets[name_]
            for j, cname in enumerate(df.columns):
                ws.set_column(j, j, max(10, min(28, len(cname) + 2)))
    pd.set_option("display.width", 250)
    print(t1.to_string(index=False))
    print(tab5.tail(4).to_string(index=False))
    print(sup6.tail(3).to_string(index=False))
    return t1, tab5, sup6

if __name__ == "__main__":
    build()
