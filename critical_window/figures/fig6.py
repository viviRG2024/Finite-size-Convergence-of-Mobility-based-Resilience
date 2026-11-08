"""Fig. 6 | Grid-based indices: grid position and sensor geometry.

a  change of the whole-city CV and H when only the grid is shifted (90th percentile over 50 shifts), relative to the
   city-to-city interquartile range: one grid position (open) and the mean over 32 positions (filled)
b  window error relative to the tolerance (median over cities): one grid position (dashed) vs 32 positions (solid)
c  the exponent beta of H with grid size (200-1000 m) for the whole extent: observed flows (filled) and
   flows shuffled among stations (open) against beta with every station given the same flow (17 cities with extents
   >= 4 km)
d  beta against -D / <ln n>, D = box-counting dimension of the station layout, <ln n> = mean log number of active cells
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import common as C


def main():
    C.style()
    fig = plt.figure(figsize=(C.W2, 135 * C.MM))
    gt = fig.add_gridspec(1, 2, left=0.08, right=0.985, top=0.93, bottom=0.6, wspace=0.32)
    gb = fig.add_gridspec(1, 2, left=0.08, right=0.985, top=0.47, bottom=0.075, wspace=0.32)
    g = [gt[0], gt[1], gb[0], gb[1]]
    for letter, x, y in (("a", 0.005, 0.985), ("b", 0.52, 0.985), ("c", 0.005, 0.52), ("d", 0.52, 0.52)):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top", ha="left", color=C.INK)
    e3 = pd.read_csv(C.OUT / "e3" / "e3_cities.csv")
    e3s = pd.read_csv(C.OUT / "e3" / "e3_summary.csv").set_index(["variant", "probe"])
    # ---- a
    ax = fig.add_subplot(g[0])
    rows = []
    for i, p in enumerate(("cv", "hn")):
        for v, filled in (("sum", False), ("sum_k32", True)):
            q = e3[(e3.variant == v) & (e3.probe == p)]
            iqr = 4 * float(e3s.loc[(v, p), "eps"])
            val = q.floor.to_numpy() / iqr
            y = i + (0.17 if filled else -0.17) + np.random.default_rng(i).uniform(-0.1, 0.1, val.size)
            ax.scatter(val, y, s=7, facecolor=C.COL[p] if filled else "white", edgecolor=C.COL[p], lw=0.6, zorder=3,
                       label=("32 grid positions" if filled else "one grid position") if i == 0 else None)
            ax.plot([np.median(val)] * 2, [i + (0.17 if filled else -0.17) - 0.13, i + (0.17 if filled else -0.17) + 0.13], color=C.INK, lw=1.1, zorder=4)
            rows += [dict(index=p, grid=v, city=c, floor_over_iqr=float(x)) for c, x in zip(q.city, val)]
    ax.axvline(0.25, color=C.INK, lw=0.6, ls="--")
    ax.text(0.27, -0.42, "tolerance", fontsize=5, ha="left", va="center", color=C.INK2)
    ax.set_xscale("log"), ax.set_xlim(0.01, 3), C.log_ticks(ax, "x", (0.01, 0.03, 0.1, 0.3, 1, 3))
    ax.set_yticks([0, 1]), ax.set_yticklabels(["CV", "H"]), ax.set_ylim(1.5, -0.5)
    ax.set_xlabel("Change of the whole-city value from shifting the grid (fraction of the city-to-city range)")
    ax.legend(loc="upper left", fontsize=5, handletextpad=0.2)
    C.source("Fig6_a", pd.DataFrame(rows), "90th percentile of |whole-city value(shifted grid) - whole-city value| over 50 random grid shifts, "
                                           "divided by the interquartile range of whole-city values across cities; tolerance = 0.25.")
    # ---- b
    ax = fig.add_subplot(g[1])
    lv = C.levels()
    rows = []
    for p, k in (("cv", 0), ("hn", 1)):
        for v, ls in (("sum", "--"), ("sum_k32", "-")):
            eps = float(e3s.loc[(v, p), "eps"])
            cur = {c: np.array(json.load(open(C.OUT / "e3" / "cities" / f"{c}.json"))[f"{v}|p90"][k], float) / eps for c in C.cities()}
            med, q1, q3, n = C.median_band(cur)
            ok = np.isfinite(med)
            ax.plot(lv[ok], med[ok], color=C.COL[p], lw=1.0, ls=ls,
                    label=f"{C.LAB[p]}, " + ("one grid position" if v == "sum" else "32 grid positions"))
            rows += [dict(index=p, grid=v, f=f, median=m, q25=a, q75=b, n_cities=int(nn)) for f, m, a, b, nn in zip(lv, med, q1, q3, n)]
    ax.axhline(1, color=C.INK, lw=0.6, ls=":")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(0.12, 1.0)
    C.log_ticks(ax, "x", (0.15, 0.3, 0.5, 0.9)), C.log_ticks(ax, "y", (0.3, 1, 3))
    ax.set_xlabel("Coverage f"), ax.set_ylabel("Window error / tolerance")
    ax.legend(loc="lower left", fontsize=4.8)
    n1 = {p: int(e3s.loc[("sum", p), "n_ok"]) for p in ("cv", "hn")}
    n32 = {p: int(e3s.loc[("sum_k32", p), "n_ok"]) for p in ("cv", "hn")}
    ax.set_title(f"f* found: CV {n1['cv']} → {n32['cv']}, H {n1['hn']} → {n32['hn']} of 34", fontsize=5.5)
    C.source("Fig6_b", pd.DataFrame(rows), "Median and quartiles over cities (levels with >= 17 cities); each grid variant with its own "
                                           "tolerance (0.25 x IQR of its whole-city values).")
    # ---- c
    geo = pd.read_csv(C.OUT / "e5" / "geometry.csv")
    ge = geo[geo.estimable].copy()
    ax = fig.add_subplot(g[2])
    lim = (ge[["beta", "beta_equal", "beta_shuffled"]].min().min() - 0.03, ge[["beta", "beta_equal", "beta_shuffled"]].max().max() + 0.03)
    ax.plot(lim, lim, color=C.MUTED, lw=0.6)
    ax.scatter(ge.beta_equal, ge.beta_shuffled, s=12, facecolor="white", edgecolor=C.INK2, lw=0.6, label="shuffled flows", zorder=3)
    ax.scatter(ge.beta_equal, ge.beta, s=6, color=C.INK, lw=0, label="observed flows", zorder=4)
    ax.set_xlim(lim), ax.set_ylim(lim), ax.set_aspect("equal")
    ax.set_xlabel("β with equal flows at all stations"), ax.set_ylabel("Exponent β of H with grid size")
    rho = stats.spearmanr(ge.beta, ge.beta_equal)[0]
    ax.text(0.97, 0.05, f"median |difference| {np.median(np.abs(ge.beta - ge.beta_equal)):.3f}\nSpearman {rho:.2f}",
            transform=ax.transAxes, ha="right", fontsize=5, color=C.INK2)
    ax.legend(loc="upper left", fontsize=5, handletextpad=0.2)
    C.source("Fig6_c", ge[["city", "n_stations", "beta", "beta_shuffled", "beta_equal"]],
             "beta = slope of log H on log grid size over 200-1000 m (17 grid sizes) for the whole study area; shuffled = median over "
             "200 permutations of the station flows; equal = every station given the same flow.")
    # ---- d
    ax = fig.add_subplot(g[3])
    ax.plot(lim, lim, color=C.MUTED, lw=0.6)
    ax.scatter(ge.beta_geo, ge.beta, s=6, color=C.INK, lw=0, zorder=3)
    ax.set_xlim(lim), ax.set_ylim(lim), ax.set_aspect("equal")
    ax.set_xlabel("−D / ⟨ln n⟩ (sensor layout)"), ax.set_ylabel("Exponent β of H with grid size")
    r1 = stats.spearmanr(ge.beta, ge.beta_geo)[0]
    r2 = stats.spearmanr(ge.beta, ge.D_road)[0]
    r3 = stats.spearmanr(geo.D_station, geo.D_road)[0]
    ax.text(0.97, 0.05, f"Spearman with β: {r1:.2f}\nβ with road-network D: {r2:.2f}", transform=ax.transAxes, va="bottom", ha="right",
            fontsize=5, color=C.INK2)
    C.source("Fig6_d", ge[["city", "beta", "beta_geo", "D_station", "D_road"]],
             f"D = box-counting dimension over 200-1000 m (16 grid origins) of station positions (D_station) and of road lines (D_road, "
             f"points every 20 m); beta_geo = -D_station / mean ln(number of active cells). Spearman D_station vs D_road over the 34 cities: {r3:.2f}.")
    C.save(fig, "Fig6")


if __name__ == "__main__":
    main()
