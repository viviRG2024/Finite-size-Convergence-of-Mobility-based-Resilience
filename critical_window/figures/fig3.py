"""Fig. 3 | Road-network resilience also needs a critical window, and each metric its own.

a  how much a window overstates the resilience of the city's road network: median over cities of the relative difference
   window - city (robustness) or city - window (efficiency loss), band: interquartile range
b  window error relative to the tolerance (median over cities, band: interquartile range); dashed: sigma_z for reference
c  critical coverage per city; bar = median; triangles: not within tolerance below f = 0.9
d  share of cities whose network metric is within tolerance when the window is sized by a traffic diagnostic's own
   critical coverage
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common as C


def main():
    C.style()
    fig = plt.figure(figsize=(C.W2, 120 * C.MM))
    g = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.93, bottom=0.085, wspace=0.42, hspace=0.45)
    g2 = fig.add_gridspec(2, 2, left=0.175, right=0.985, top=0.93, bottom=0.085, wspace=0.52, hspace=0.45)
    for letter, x, y in (("a", 0.005, 0.985), ("b", 0.515, 0.985), ("c", 0.005, 0.49), ("d", 0.515, 0.49)):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top", ha="left", color=C.INK)
    lv = pd.read_csv(C.OUT / "e4b" / "levels.csv")
    nc, neps, nf, nlv = C.network_curves()
    tc, teps, tf = C.traffic_curves()
    tl = C.levels()
    # ---- a
    ax = fig.add_subplot(g[0, 0])
    rows = []
    for k in C.NETWORK:
        d = lv[lv.metric == k].copy()
        sgn = -1.0 if k == "eff_loss" else 1.0
        d["over"] = 100 * sgn * d.rel_bias
        s = d.groupby("f").over.agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), "count"])
        s.columns = ["median", "q25", "q75", "n"]
        s = s[s.n >= 17]
        ax.fill_between(s.index, s.q25, s.q75, color=C.COL[k], alpha=0.12, lw=0)
        ax.plot(s.index, s["median"], "-o", color=C.COL[k], ms=2.3, lw=1.0, mec="white", mew=0.3, label=C.LAB[k])
        rows += [dict(metric=k, f=f, overstatement_pct_median=r["median"], q25=r.q25, q75=r.q75, n_cities=int(r.n)) for f, r in s.iterrows()]
    ax.axhline(0, color=C.INK, lw=0.6)
    ax.set_xlim(0, 0.95), ax.set_xlabel("Coverage f"), ax.set_ylabel("Window overstates resilience (%)")
    ax.legend(loc="upper right")
    C.source("Fig3_a", pd.DataFrame(rows), "Relative difference (window - city)/city of robustness, and (city - window)/city of efficiency "
                                           "loss, in %; median and quartiles over cities (levels with >= 17 cities). "
                                           "Signed per-window values: out/e4b/levels.csv (rel_bias).")
    # ---- b
    ax = fig.add_subplot(g[0, 1])
    rows = []
    med, q1, q3, n = C.median_band(tc["sd_log"])
    ok = np.isfinite(med)
    ax.plot(tl[ok], med[ok], color=C.COL["sd_log"], lw=0.9, ls="--", label=C.LAB["sd_log"] + " (reference)")
    rows += [dict(metric="sd_log", f=f, median=m, q25=a, q75=b, n_cities=int(nn)) for f, m, a, b, nn in zip(tl, med, q1, q3, n)]
    for k in C.NETWORK:
        med, q1, q3, n = C.median_band(nc[k])
        ok = np.isfinite(med)
        ax.fill_between(nlv[ok], q1[ok], q3[ok], color=C.COL[k], alpha=0.12, lw=0)
        ax.plot(nlv[ok], med[ok], "-o", color=C.COL[k], ms=2.3, lw=1.0, mec="white", mew=0.3, label=C.LAB[k])
        rows += [dict(metric=k, f=f, median=m, q25=a, q75=b, n_cities=int(nn)) for f, m, a, b, nn in zip(nlv, med, q1, q3, n)]
    ax.axhline(1, color=C.INK, lw=0.6, ls=":")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(0.045, 1.0)
    C.log_ticks(ax, "x", (0.05, 0.1, 0.2, 0.5, 0.9)), C.log_ticks(ax, "y", (0.3, 1, 3, 10))
    ax.set_xlabel("Coverage f"), ax.set_ylabel("Window error / tolerance")
    ax.legend(loc="lower left", fontsize=5)
    C.source("Fig3_b", pd.DataFrame(rows), "Median and quartiles over cities of P90 |window - city| / tolerance.")
    # ---- c
    ax = fig.add_subplot(g2[1, 0])
    rows, labs = [], []
    order = ("sd_log",) + C.NETWORK
    for i, k in enumerate(order):
        q = tf[k] if k == "sd_log" else nf[k]
        st = C.fstar_stats(q)
        okv = q[q.flag.isin(["ok", "below"])].f_star.to_numpy()
        ab = st["n_above"]
        y = i + np.random.default_rng(i).uniform(-0.2, 0.2, okv.size)
        ax.scatter(okv, y, s=7, color=C.COL[k], alpha=0.5 if k == "sd_log" else 0.85, lw=0.3, edgecolor="white", zorder=3)
        if ab:
            ax.scatter(np.full(ab, 0.965), i + np.linspace(-0.2, 0.2, ab), s=7, marker=">", color=C.COL[k], alpha=0.5, lw=0, zorder=3)
        if np.isfinite(st["median"]):
            ax.plot([st["median"]] * 2, [i - 0.32, i + 0.32], color=C.INK, lw=1.2, zorder=4)
        labs.append(f"{C.SHORT[k]}\n{C.fmt_median(st)}\n{ab} of {st['n']} above 0.9")
        rows += [dict(metric=k, city=c, f_star=q.loc[c, "f_star"] if q.loc[c, "flag"] != "above" else np.nan, flag=q.loc[c, "flag"])
                 for c in q.index]
    ax.set_yticks(range(len(order))), ax.set_yticklabels(labs, fontsize=5, linespacing=1.2)
    ax.set_ylim(len(order) - 0.4, -0.6), ax.set_xlim(0, 1.0), ax.set_xlabel("Critical coverage f*")
    C.source("Fig3_c", pd.DataFrame(rows), "flag: ok = f* found; above = not within the tolerance below f = 0.9; none = no usable level. "
                                           "Median over the cities with a usable level, cities 'above' ranked above every found f*; "
                                           "'median > x' when the median falls among those cities.")
    # ---- d
    ax = fig.add_subplot(g[1, 1])
    tr = pd.read_csv(C.OUT / "e4b" / "transfer.csv")
    probes = (("sd_log", "sd_log"), ("cv_k32", "cv"), ("hn_k32", "hn"))
    rows = []
    wbar = 0.24
    for j, (pk, col) in enumerate(probes):
        for i, k in enumerate(C.NETWORK):
            r = tr[(tr.probe == pk) & (tr.metric == k)].iloc[0]
            x = i + (j - 1) * (wbar + 0.03)
            ax.bar(x, 100 * r.share_within, width=wbar, color=C.COL[col], lw=0, label=f"window sized by {C.LAB[col]}" if i == 0 else None)
            ax.text(x, 100 * r.share_within + 1.5, f"{100 * r.share_within:.0f}", ha="center", va="bottom", fontsize=5, color=C.INK2)
            rows.append(dict(window_from=col, network_metric=k, cities=int(r.cities), share_within_tolerance=r.share_within,
                             p90_over_tolerance_median=r.p90_over_eps_median))
    ax.set_xticks(range(3)), ax.set_xticklabels([C.SHORT[k] for k in C.NETWORK])
    ax.set_ylim(0, 100), ax.set_ylabel("Cities within tolerance (%)")
    ax.legend(loc="upper right", fontsize=5)
    C.source("Fig3_d", pd.DataFrame(rows), "Error of the network metric at the traffic diagnostic's critical coverage (log-log interpolation "
                                           "between levels), cities where that coverage is defined.")
    C.save(fig, "Fig3")


if __name__ == "__main__":
    main()
