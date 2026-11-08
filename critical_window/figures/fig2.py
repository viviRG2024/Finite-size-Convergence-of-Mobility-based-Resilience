"""Fig. 2 | Small windows do not represent the city.

a  six representative cities: deviation of each random window's sigma_z from the whole-city value, in units of
   the tolerance, against coverage
b  window error relative to the tolerance against coverage, median over the cities (band: interquartile range),
   for sigma_z and for CV and H (500 m cells, averaged over 32 grid positions)
c  critical coverage f* per city; bar = median; cities whose error is not within tolerance below f = 0.9 listed as > 0.9
d  rank correlation (Spearman) between window values (one random window per city) and whole-city values across the
   cities, median over 1,000 draws; squares: whole-city values from a single weekday instead of all days
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common as C

REP = ("london", "losangeles", "madrid", "melbourne", "hamburg", "zurich")


def main():
    C.style()
    fig = plt.figure(figsize=(C.W2, 128 * C.MM))
    ga = fig.add_gridspec(2, 3, left=0.075, right=0.53, top=0.93, bottom=0.585, wspace=0.12, hspace=0.42)
    gb = fig.add_gridspec(1, 1, left=0.635, right=0.985, top=0.93, bottom=0.585)
    gc = fig.add_gridspec(1, 2, left=0.16, right=0.985, top=0.43, bottom=0.08, wspace=0.42, width_ratios=[1.0, 1])
    for letter, x, y in (("a", 0.005, 0.985), ("b", 0.57, 0.985), ("c", 0.005, 0.485), ("d", 0.57, 0.485)):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top", ha="left", color=C.INK)
    tc, teps, tf = C.traffic_curves()
    lv = C.levels()
    # ---- a
    rows = []
    for j, c in enumerate(REP):
        ax = fig.add_subplot(ga[j // 3, j % 3])
        r = json.load(open(C.OUT / "e8" / "cities" / f"{c}.json"))
        z = np.load(C.OUT / "e8" / "cities" / f"{c}.npz")
        thE = r["theta_E"]["sd_log"]
        ax.axhspan(-1, 1, color=C.BAND, alpha=0.6, lw=0, zorder=0)
        ax.axhline(0, color=C.INK, lw=0.5, zorder=1)
        rng = np.random.default_rng(j)
        for i, f in enumerate(r["levels"]):
            v = z[f"sd_log_{i}"]
            v = (v[np.isfinite(v)] - thE) / teps["sd_log"]
            if v.size < 25:
                continue
            ax.scatter(f * np.exp(rng.uniform(-0.07, 0.07, v.size)), v, s=0.6, color=C.COL["sd_log"], alpha=0.45, lw=0, zorder=2)
            rows += [dict(city=c, f=f, deviation_over_tolerance=float(x)) for x in v]
        ax.set_xscale("log"), ax.set_xlim(0.012, 1.0), ax.set_ylim(-9, 9)
        C.log_ticks(ax, "x", (0.03, 0.1, 0.3, 0.9))
        q = tf["sd_log"].loc[c]
        fz = "> 0.9" if q.flag == "above" else f"{q.f_star:.2f}"
        ax.set_title(f"{C.name(c)}  (f* {fz})", fontsize=5.5, loc="center", pad=2)
        if q.flag == "ok":
            ax.axvline(q.f_star, color=C.INK2, lw=0.5, ls=":")
        if j % 3:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Window − city\n(tolerance units)", fontsize=5.5)
        if j < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Coverage f", fontsize=5.5)
        ax.tick_params(labelsize=5)
    C.source("Fig2_a", pd.DataFrame(rows), "sigma_z of each usable random window minus the whole-city value, divided by the tolerance "
                                           f"({teps['sd_log']:.4f}).")
    # ---- b
    ax = fig.add_subplot(gb[0])
    rows = []
    for k in C.TRAFFIC:
        med, q1, q3, n = C.median_band(tc[k])
        ok = np.isfinite(med)
        ax.fill_between(lv[ok], q1[ok], q3[ok], color=C.COL[k], alpha=0.12, lw=0)
        ax.plot(lv[ok], med[ok], color=C.COL[k], lw=1.1, label=C.LAB[k] if k == "sd_log" else f"{C.LAB[k]} (32 grid positions)")
        rows += [dict(metric=k, f=f, median=m, q25=a, q75=b, n_cities=int(nn)) for f, m, a, b, nn in zip(lv, med, q1, q3, n)]
    ax.axhline(1, color=C.INK, lw=0.6, ls="--")
    ax.text(0.125, 1.08, "tolerance", fontsize=5, color=C.INK2, va="bottom")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(0.12, 1.0)
    C.log_ticks(ax, "x", (0.15, 0.2, 0.3, 0.5, 0.7, 0.9)), C.log_ticks(ax, "y", (0.3, 1, 3))
    ax.set_xlabel("Coverage f"), ax.set_ylabel("Window error / tolerance")
    ax.legend(loc="lower left")
    C.source("Fig2_b", pd.DataFrame(rows), "Median and quartiles over the cities with a usable level (shown where >= 17 cities).")
    # ---- c
    ax = fig.add_subplot(gc[0])
    rows, labs = [], []
    for i, k in enumerate(C.TRAFFIC):
        q = tf[k]
        st = C.fstar_stats(q)
        ok = q[q.flag.isin(["ok", "below"])].f_star.to_numpy()
        ab = st["n_above"]
        y = i + np.random.default_rng(i).uniform(-0.2, 0.2, ok.size)
        ax.scatter(ok, y, s=7, color=C.COL[k], alpha=0.85, lw=0.3, edgecolor="white", zorder=3)
        if ab:
            ax.scatter(np.full(ab, 0.965), i + np.linspace(-0.2, 0.2, ab), s=7, marker=">", color=C.COL[k], alpha=0.5, lw=0, zorder=3)
        if np.isfinite(st["median"]):
            ax.plot([st["median"]] * 2, [i - 0.32, i + 0.32], color=C.INK, lw=1.2, zorder=4)
        rows += [dict(metric=k, city=c, f_star=q.loc[c, "f_star"] if q.loc[c, "flag"] != "above" else np.nan, flag=q.loc[c, "flag"])
                 for c in q.index]
        labs.append(f"{C.LAB[k]}\n{C.fmt_median(st)}\n{ab} of {st['n']} above 0.9 (▸)")
    ax.set_yticks(range(3)), ax.set_yticklabels(labs, fontsize=5, linespacing=1.2)
    ax.set_ylim(2.6, -0.6), ax.set_xlim(0, 1.0)
    ax.set_xlabel("Critical coverage f*")
    C.source("Fig2_c", pd.DataFrame(rows), "flag: ok = f* found; above = not within the tolerance below f = 0.9; below = within the "
                                           "tolerance from the smallest usable level; none = no usable level (too few stations). "
                                           "Median over the cities with a usable level, cities 'above' ranked above every found f*.")
    # ---- d
    ax = fig.add_subplot(gc[1])
    rk = pd.read_csv(C.OUT / "e8" / "t_ranking.csv")
    rk["probe"] = rk.probe.replace({"cv_k32": "cv", "hn_k32": "hn"})
    rows = []
    for k in C.TRAFFIC:
        g = rk[(rk.probe == k) & rk["at"].str.startswith("f=")].copy()
        g["f"] = g["at"].str[2:].astype(float)
        ax.plot(g.f, g.rho_median, "-o", color=C.COL[k], lw=1.0, ms=2.5, mec="white", mew=0.3, label=C.LAB[k])
        d1 = rk[(rk.probe == k) & (rk["at"] == "single weekday")].rho_median.iloc[0]
        ax.scatter([0.965 + 0.012 * C.TRAFFIC.index(k)], [d1], marker="s", s=10, color=C.COL[k], lw=0.3, edgecolor="white", zorder=4)
        rows += [dict(metric=k, comparison="window location", f=f, rho_median=m, rho_p05=p) for f, m, p in zip(g.f, g.rho_median, g.rho_p05)]
        rows.append(dict(metric=k, comparison="single weekday, whole city", f=np.nan, rho_median=d1,
                         rho_p05=rk[(rk.probe == k) & (rk["at"] == "single weekday")].rho_p05.iloc[0]))
    ax.text(0.977, 0.94, "single\nweekday", fontsize=5, ha="center", va="top", color=C.INK2)
    ax.set_xlim(0.36, 1.0), ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("Coverage f"), ax.set_ylabel("Rank correlation with whole-city values")
    ax.legend(loc="lower right", bbox_to_anchor=(0.9, 0.0))
    C.source("Fig2_d", pd.DataFrame(rows), "Spearman rho between one random window per city and the whole-city values, median and 5th "
                                           "percentile over 1,000 draws (cities with a usable level).")
    C.save(fig, "Fig2")


if __name__ == "__main__":
    main()
