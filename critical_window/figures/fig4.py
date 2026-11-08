"""Fig. 4 | Why the convergence is smooth.

a  intraclass correlation of log flows among the stations of the same square block, against block side (median over
   cities, band: interquartile range); dashed: 97.5th percentile after shuffling the flows among stations
b  side of the smallest usable window against the correlation length of log flows (cities where it is identifiable);
   lines: window = 10 and 100 correlation lengths
c  window error of sigma_z with the observed flows and with the flows shuffled among stations (median over cities of
   the median over 2,000 shuffles)
d  test for an abrupt transition (broken-stick against smooth decline of the window error, permutation p value) per
   city; dashed: p = 0.05
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common as C


def main():
    C.style()
    fig = plt.figure(figsize=(C.W2, 112 * C.MM))
    g = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.93, bottom=0.09, wspace=0.36, hspace=0.5)
    g2 = fig.add_gridspec(2, 2, left=0.085, right=0.935, top=0.93, bottom=0.09, wspace=0.55, hspace=0.5)
    for letter, x, y in (("a", 0.005, 0.985), ("b", 0.52, 0.985), ("c", 0.005, 0.49), ("d", 0.52, 0.49)):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top", ha="left", color=C.INK)
    cities = C.cities()
    # ---- a
    ax = fig.add_subplot(g[0, 0])
    bl = pd.read_csv(C.OUT / "e5" / "blocks.csv")
    s = bl.groupby("b").agg(m=("icc_obs", "median"), q1=("icc_obs", lambda x: x.quantile(0.25)), q3=("icc_obs", lambda x: x.quantile(0.75)),
                            sh=("icc_shuf_q975", "median"), n=("icc_obs", "count"))
    ax.fill_between(s.index, s.q1, s.q3, color=C.COL["sd_log"], alpha=0.12, lw=0)
    ax.plot(s.index, s.m, "-o", color=C.COL["sd_log"], lw=1.0, ms=2.5, mec="white", mew=0.3, label="observed flows")
    ax.plot(s.index, s.sh, color=C.MUTED, lw=0.9, ls="--", label="shuffled flows, 97.5th percentile")
    ax.set_xscale("log"), C.log_ticks(ax, "x", (250, 500, 1000, 2000, 4000))
    ax.set_ylim(0, 0.42), ax.set_xlabel("Block side (m)"), ax.set_ylabel("Correlation within a block")
    ax.legend(loc="upper right")
    C.source("Fig4_a", s.reset_index().rename(columns={"m": "icc_median", "q1": "icc_q25", "q3": "icc_q75", "sh": "shuffled_q975_median",
                                                     "n": "n_cities"}),
             "One-way ANOVA intraclass correlation of log station flows within square blocks (8 block origins averaged).")
    # ---- b
    ax = fig.add_subplot(g[0, 1])
    t = pd.read_csv(C.OUT / C.MAIN / "table2_cities.csv").set_index("city")
    rows = []
    for c in cities:
        js = json.load(open(C.OUT / C.MAIN / c / "city.json"))
        a = np.load(C.OUT / C.MAIN / c / "city.npz")
        lv = np.array(js["levels"])
        ok = np.flatnonzero(a["lev__usable"] >= 25)
        if not ok.size:
            continue
        e = js["extent"]
        side = np.sqrt(lv[ok.min()]) * min(e[2] - e[0], e[3] - e[1])
        rows.append(dict(city=c, xi_m=float(t.loc[c, "xi_m"]), xi_identifiable=bool(t.loc[c, "xi_identifiable"]),
                         smallest_usable_f=float(lv[ok.min()]), smallest_window_short_side_m=float(side)))
    d = pd.DataFrame(rows)
    di = d[d.xi_identifiable]
    xx = np.array([15, 2000])
    for k, lab, xl in ((10, "window = 10 ξ", 520), (100, "window = 100 ξ", 52)):
        ax.plot(xx, k * xx, color=C.MUTED, lw=0.6, ls=":")
        ax.text(xl, k * xl * 1.12, lab, fontsize=5, color=C.INK2, ha="right", va="bottom")
    ax.scatter(di.xi_m, di.smallest_window_short_side_m, s=10, color=C.COL["sd_log"], lw=0.3, edgecolor="white", zorder=3)
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(15, 2000), ax.set_ylim(300, 8000)
    C.log_ticks(ax, "x", (25, 50, 100, 250, 500, 1000)), C.log_ticks(ax, "y", (500, 1000, 2000, 5000))
    ax.set_xlabel("Correlation length ξ of log flows (m)"), ax.set_ylabel("Smallest usable window side (m)")
    ax.text(0.97, 0.05, f"{len(di)} cities; window / ξ: median {np.median(di.smallest_window_short_side_m / di.xi_m):.0f},"
                        f" min {np.min(di.smallest_window_short_side_m / di.xi_m):.1f}", transform=ax.transAxes, ha="right", fontsize=5, color=C.INK2)
    C.source("Fig4_b", d, "Smallest usable level: >= 25 usable windows (>= 20 stations each); side = sqrt(f) x short side of the study area. "
                          "Correlation length: 1/e crossing of the empirical correlogram (first bin 25 m means shorter than 50 m).")
    # ---- c
    ax = fig.add_subplot(g[1, 0])
    cw = pd.read_csv(C.OUT / C.MAIN / "critical_windows.csv")
    eps = float(cw[cw.probe == "sd_log"].eps.iloc[0])
    lv = C.levels()
    obs, perm = {}, {}
    for c in cities:
        a = np.load(C.OUT / C.MAIN / c / "city.npz")
        obs[c] = a["sd_log__p90_obs"] / eps
        with np.errstate(all="ignore"):
            perm[c] = np.nanmedian(a["sd_log__p90_perm"], 1) / eps
    rows = []
    for lab, cur, col, ls in (("observed flows", obs, C.COL["sd_log"], "-"), ("flows shuffled among stations", perm, C.MUTED, "--")):
        med, q1, q3, n = C.median_band(cur)
        ok = np.isfinite(med)
        if ls == "-":
            ax.fill_between(lv[ok], q1[ok], q3[ok], color=col, alpha=0.12, lw=0)
        ax.plot(lv[ok], med[ok], color=col, lw=1.1, ls=ls, label=lab)
        rows += [dict(flows=lab, f=f, median=m, q25=a_, q75=b_, n_cities=int(nn)) for f, m, a_, b_, nn in zip(lv, med, q1, q3, n)]
    ratio = np.nanmedian(np.array([obs[c] / perm[c] for c in cities]), 0)
    ax.axhline(1, color=C.INK, lw=0.6, ls=":")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(0.12, 1.0)
    C.log_ticks(ax, "x", (0.15, 0.2, 0.3, 0.5, 0.7, 0.9)), C.log_ticks(ax, "y", (0.3, 1, 3))
    ax.set_xlabel("Coverage f"), ax.set_ylabel(C.SIG + " window error / tolerance")
    ax.legend(loc="lower left")
    ok = np.isfinite(ratio) & (lv >= 0.15)
    ax.text(0.97, 0.95, f"observed / shuffled: median {np.median(ratio[ok]):.1f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=5, color=C.INK2)
    C.source("Fig4_c", pd.DataFrame(rows), "Shuffled: whole station series permuted among stations (2,000 permutations), same windows.")
    # ---- d
    ax = fig.add_subplot(g2[1, 1])
    h = pd.read_csv(C.OUT / C.MAIN / "h1_h2.csv")
    order = (("sd_log", C.SIG), ("mean_log", "Mean of log flows"), ("site_cv", "Site CV"), ("cv500", "CV (single grid)"), ("hn500", "H (single grid)"))
    rows = []
    for i, (p, lab) in enumerate(order):
        q = h[h.probe == p].dropna(subset=["h2_p"])
        y = i + np.random.default_rng(i).uniform(-0.22, 0.22, len(q))
        ax.scatter(q.h2_p, y, s=6, color=C.INK2, alpha=0.6, lw=0)
        nsig = int((q.h2_p < 0.05).sum())
        ax.text(1.02, i, f"{nsig}/{len(q)}", transform=ax.get_yaxis_transform(), fontsize=5, va="center", color=C.INK2)
        rows += [dict(probe=p, city=r.city, transition_p=r.h2_p, delta_bic=r.dbic) for r in q.itertuples()]
    ax.axvline(0.05, color=C.INK, lw=0.6, ls="--")
    ax.set_xscale("log"), ax.set_xlim(0.003, 1.05), C.log_ticks(ax, "x", (0.005, 0.05, 0.5))
    ax.set_yticks(range(len(order))), ax.set_yticklabels([lab for _, lab in order])
    ax.set_ylim(len(order) - 0.4, -0.6), ax.set_xlabel("p value for an abrupt transition")
    ax.text(1.02, -0.85, "p < 0.05", transform=ax.get_yaxis_transform(), fontsize=5, va="center", color=C.INK2, clip_on=False)
    C.source("Fig4_d", pd.DataFrame(rows), "Broken-stick vs smooth model of the window error (Delta BIC), p value against 200 simulated "
                                           "smooth fields (N2) on the same layout.")
    C.save(fig, "Fig4")


if __name__ == "__main__":
    main()
