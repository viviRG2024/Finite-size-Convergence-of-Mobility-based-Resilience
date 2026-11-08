"""Supplementary figures.

    python -m critical_window.figures.si [s1 s2 ...]

s1  window error per city, traffic diagnostics (sigma_z, CV, H)
s2  window error per city, road-network metrics (random failure, targeted attack, efficiency loss)
s3  the two flow-aware network metrics (flow-weighted efficiency loss, 2 km accessibility loss)
s4  robustness of f*: one change at a time; stations in the window vs coverage under other extent definitions
s5  criterion based on the stability of the exponent beta, on the IUTF flows: 24 h CV of beta and its spread across windows, observed vs shuffled flows
s6  window overlap and non-overlapping tiles
s7  windows with and without the main flow centre
s8  the three-step computation for mean log flow and site CV
s9  observation period per city
s10 rain events as a real disturbance (E10): flow, occupancy and speed index against normal days; the window needed to
    reproduce the city-wide slowdown; the slowdown before, during and after the rain (needs tools/rain_periods.py)
s11 city rankings of the road-network metrics from one window
s12 entropy against grid size over 39 sizes from 10 m to 10 km
s13 critical coverage against the number of stations, the observation period and the layout
"""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import common as C

EXTRA = {"dw_eff_loss": "#e87ba4", "acc_loss": "#008300"}


def _grid_cities(fig, gs, curves_by_metric, levels_by_metric, fst_by_metric, keys, tag):
    rows = []
    cities = sorted(C.cities(), key=C.name)
    for i, c in enumerate(cities):
        ax = fig.add_subplot(gs[i // 6, i % 6])
        for k in keys:
            y = curves_by_metric[k].get(c)
            if y is None:
                continue
            lv = levels_by_metric[k]
            ok = np.isfinite(y) & (y > 0)
            if ok.sum():
                ax.plot(lv[ok], y[ok], color=C.COL.get(k, EXTRA.get(k)), lw=0.8)
                rows += [dict(city=c, metric=k, f=f, p90_over_tolerance=v) for f, v in zip(lv[ok], y[ok])]
            q = fst_by_metric[k]
            if c in q.index and q.loc[c, "flag"] == "ok":
                ax.plot([q.loc[c, "f_star"]] * 2, [0.05, 0.09], color=C.COL.get(k, EXTRA.get(k)), lw=1.1, solid_capstyle="butt")
        ax.axhline(1, color=C.INK, lw=0.4, ls="--")
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(0.01, 1), ax.set_ylim(0.05, 30)
        ax.set_title(C.name(c), fontsize=5.5, loc="center", pad=1.5)
        ax.tick_params(labelsize=4.5, length=1.5)
        if i % 6:
            ax.set_yticklabels([])
        else:
            C.log_ticks(ax, "y", (0.1, 1, 10))
        if i < len(cities) - 6:
            ax.set_xticklabels([])
        else:
            C.log_ticks(ax, "x", (0.03, 0.3))
    return rows


def s1():
    tc, teps, tf = C.traffic_curves()
    lv = C.levels()
    fig = plt.figure(figsize=(C.W2, 205 * C.MM))
    gs = fig.add_gridspec(6, 6, left=0.07, right=0.985, top=0.95, bottom=0.06, wspace=0.12, hspace=0.38)
    rows = _grid_cities(fig, gs, tc, {k: lv for k in C.TRAFFIC}, tf, C.TRAFFIC, "traffic")
    fig.text(0.5, 0.02, "Coverage f", ha="center", fontsize=6)
    fig.text(0.015, 0.5, "Window error / tolerance", va="center", rotation=90, fontsize=6)
    for k in C.TRAFFIC:
        fig.lines.append(plt.Line2D([], [], color=C.COL[k]))
    hs = [plt.Line2D([], [], color=C.COL[k], lw=1) for k in C.TRAFFIC] + [plt.Line2D([], [], color=C.INK, lw=0.5, ls="--")]
    fig.legend(hs, [C.LAB["sd_log"], "CV (32 grid positions)", "H (32 grid positions)", "tolerance"], loc="upper center", ncol=4, fontsize=5.5,
               bbox_to_anchor=(0.5, 0.995))
    C.source("SuppFig1", pd.DataFrame(rows), "Per city: P90 window error / tolerance; ticks at the bottom mark f*.")
    C.save(fig, "SuppFig1")


def s2():
    nc, neps, nf, nlv = C.network_curves()
    fig = plt.figure(figsize=(C.W2, 205 * C.MM))
    gs = fig.add_gridspec(6, 6, left=0.07, right=0.985, top=0.95, bottom=0.06, wspace=0.12, hspace=0.38)
    rows = _grid_cities(fig, gs, nc, {k: nlv for k in C.NETWORK}, nf, C.NETWORK, "network")
    fig.text(0.5, 0.02, "Coverage f", ha="center", fontsize=6)
    fig.text(0.015, 0.5, "Window error / tolerance", va="center", rotation=90, fontsize=6)
    hs = [plt.Line2D([], [], color=C.COL[k], lw=1) for k in C.NETWORK] + [plt.Line2D([], [], color=C.INK, lw=0.5, ls="--")]
    fig.legend(hs, [C.LAB[k] for k in C.NETWORK] + ["tolerance"], loc="upper center", ncol=4, fontsize=5.5, bbox_to_anchor=(0.5, 0.995))
    C.source("SuppFig2", pd.DataFrame(rows), "Per city: P90 window error / tolerance of the road-network metrics; ticks mark f*.")
    C.save(fig, "SuppFig2")


def s3():
    lv = pd.read_csv(C.OUT / "e4b" / "levels.csv")
    nc, neps, nf, nlv = C.network_curves()
    fig, ax = plt.subplots(1, 3, figsize=(C.W2, 58 * C.MM), gridspec_kw=dict(wspace=0.42, left=0.07, right=0.985, top=0.88, bottom=0.2))
    rows_a, rows_b, rows_c = [], [], []
    for k, col in EXTRA.items():
        d = lv[lv.metric == k].copy()
        d["over"] = -100 * d.rel_bias
        s = d.groupby("f").over.agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), "count"])
        s.columns = ["median", "q25", "q75", "n"]
        s = s[s.n >= 17]
        ax[0].fill_between(s.index, s.q25, s.q75, color=col, alpha=0.12, lw=0)
        ax[0].plot(s.index, s["median"], "-o", color=col, ms=2.3, lw=1, mec="white", mew=0.3, label=C.LAB[k])
        rows_a += [dict(metric=k, f=f, overstatement_pct_median=r["median"], q25=r.q25, q75=r.q75, n_cities=int(r.n)) for f, r in s.iterrows()]
        med, q1, q3, n = C.median_band(nc[k])
        ok = np.isfinite(med)
        ax[1].fill_between(nlv[ok], q1[ok], q3[ok], color=col, alpha=0.12, lw=0)
        ax[1].plot(nlv[ok], med[ok], "-o", color=col, ms=2.3, lw=1, mec="white", mew=0.3, label=C.LAB[k])
        rows_b += [dict(metric=k, f=f, median=m, q25=a, q75=b, n_cities=int(nn)) for f, m, a, b, nn in zip(nlv, med, q1, q3, n)]
    ax[0].axhline(0, color=C.INK, lw=0.6)
    ax[0].set_xlabel("Coverage f"), ax[0].set_ylabel("Window overstates resilience (%)"), ax[0].legend(fontsize=5)
    ax[1].axhline(1, color=C.INK, lw=0.6, ls=":")
    ax[1].set_xscale("log"), ax[1].set_yscale("log"), C.log_ticks(ax[1], "x", (0.05, 0.1, 0.2, 0.5, 0.9)), C.log_ticks(ax[1], "y", (0.3, 1, 3, 10))
    ax[1].set_xlabel("Coverage f"), ax[1].set_ylabel("Window error / tolerance")
    labs_c = []
    for i, k in enumerate(EXTRA):
        q = nf[k]
        st = C.fstar_stats(q)
        okv = q[q.flag.isin(["ok", "below"])].f_star.to_numpy()
        ab = st["n_above"]
        ax[2].scatter(okv, i + np.random.default_rng(i).uniform(-0.2, 0.2, okv.size), s=7, color=EXTRA[k], lw=0.3, edgecolor="white")
        if ab:
            ax[2].scatter(np.full(ab, 0.965), i + np.linspace(-0.2, 0.2, ab), s=7, marker=">", color=EXTRA[k], alpha=0.5, lw=0)
        if np.isfinite(st["median"]):
            ax[2].plot([st["median"]] * 2, [i - 0.32, i + 0.32], color=C.INK, lw=1.2)
        rows_c += [dict(metric=k, city=c, f_star=q.loc[c, "f_star"] if q.loc[c, "flag"] != "above" else np.nan, flag=q.loc[c, "flag"])
                   for c in q.index]
        labs_c.append(f"{C.SHORT[k]}\n{C.fmt_median(st)}\n{ab} of {st['n']} above 0.9")
    ax[2].set_yticks([0, 1]), ax[2].set_yticklabels(labs_c, fontsize=5, linespacing=1.2)
    ax[2].set_ylim(1.6, -0.6), ax[2].set_xlim(0, 1), ax[2].set_xlabel("Critical coverage f*")
    for a, l in zip(ax, "abc"):
        C.panel(a, l, x=-0.2, y=1.04)
    C.source("SuppFig3_a", pd.DataFrame(rows_a), "(city - window)/city of the loss, %; positive = the window looks more resilient.")
    C.source("SuppFig3_b", pd.DataFrame(rows_b), "Median and quartiles over cities of P90 |window - city| / tolerance (as Fig. 3b).")
    C.source("SuppFig3_c", pd.DataFrame(rows_c), "flag: ok = f* found; above = not within the tolerance below f = 0.9. Median over all "
                                                 "cities, cities 'above' ranked above every found f*.")
    C.save(fig, "SuppFig3")


def s4():
    s = pd.read_csv(C.OUT / "e6" / "e6_cities.csv")
    ns = pd.read_csv(C.OUT / "e6" / "e6_nstar_summary.csv")
    lab = {"main": "Main analysis", "peak_hour": "08:00–08:59 only", "midday_hour": "12:00–12:59 only", "all_day": "Whole day (24 h)",
           "weekend": "Weekends", "dry_days": "Dry days only", "no_outlier_rule": "No outlier removal", "square": "Square windows",
           "circle": "Circular windows", "trim95": "Study area: central 95% of stations", "buffer500": "Study area: + 500 m",
           "hull250": "Study area: convex hull + 250 m", "road_bbox": "Study area: box of the road file", "grid250": "250 m cells",
           "grid1000": "1 km cells", "q90_eps0.1": "Tolerance × 0.4", "q90_eps0.5": "Tolerance × 2", "q90_eps1.0": "Tolerance × 4",
           "q80_eps0.25": "80% of windows", "q95_eps0.25": "95% of windows"}
    order = list(lab)
    fig, ax = plt.subplots(1, 2, figsize=(C.W2, 105 * C.MM), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.55, left=0.17, right=0.985,
                                                                                  top=0.92, bottom=0.1))
    a = ax[0]
    rows = []
    for p, off in (("sd_log", -0.22), ("cv", 0.0), ("hn", 0.22)):
        for i, v in enumerate(order):
            g = s[(s.variant == v) & (s.probe == p)].set_index("city")
            st = C.fstar_stats(g)
            if st["n"] < 3:
                continue
            q1, med, q3 = st["q25"], st["median"], st["q75"]
            x1 = q1 if np.isfinite(q1) else 0.965
            x3 = q3 if np.isfinite(q3) else 1.0
            a.plot([x1, x3], [i + off] * 2, color=C.COL[p], lw=1.4, alpha=0.6, solid_capstyle="butt")
            lbl = {"sd_log": C.SIG, "cv": "CV (32 grid positions)", "hn": "H (32 grid positions)"}[p] if i == 0 else None
            if np.isfinite(med):
                a.scatter([med], [i + off], color=C.COL[p], s=9, zorder=3, lw=0.3, edgecolor="white", label=lbl)
            else:
                a.scatter([0.965], [i + off], color=C.COL[p], s=9, marker=">", zorder=3, lw=0, label=lbl)
            rows.append(dict(variant=v, label=lab[v], metric=p, cities=st["n"], cities_found=st["n_ok"], cities_not_reached=st["n_above"],
                             cities_below=st["n_below"],
                             q25=q1, median=med, q75=q3))
    base = C.fstar_stats(s[(s.variant == "main") & (s.probe == "sd_log")].set_index("city"))["median"]
    a.axvline(base, color=C.COL["sd_log"], lw=0.5, ls=":")
    a.set_yticks(range(len(order))), a.set_yticklabels([lab[v] for v in order], fontsize=5)
    a.set_ylim(len(order) - 0.5, -0.5), a.set_xlim(0, 1)
    a.set_xlabel("Critical coverage f* (dot: median; line: interquartile range)")
    a.legend(loc="lower left", fontsize=5)
    C.source("SuppFig4_a", pd.DataFrame(rows), "One change at a time; tolerance recomputed for each variant from its own whole-city values "
                                               "(rows 'Tolerance x' rescale the main tolerance). Median and quartiles over the cities with a "
                                               "usable level (Kaplan-Meier): cities not within the tolerance at their largest usable window "
                                               "level are censored there (0.9, except for square and circular windows, whose largest "
                                               "feasible coverage depends on the shape of the study area); 'below' cities count at their smallest "
                                               "usable level; inf = the quantile is not reached.")
    a = ax[1]
    g = ns[(ns.probe == "sd_log") & (ns.variant != "main")].copy()
    y = np.arange(len(g))
    a.barh(y - 0.19, g.median_ratio_f_star_vs_main, height=0.34, color=C.MUTED, label="coverage f*")
    a.barh(y + 0.19, g.median_ratio_n_star_vs_main, height=0.34, color=C.COL["sd_log"], label="stations in the window at f*")
    a.axvline(1, color=C.INK, lw=0.6)
    a.set_yticks(y), a.set_yticklabels([lab[v] for v in g.variant], fontsize=5), a.set_ylim(len(g) - 0.5, -0.5)
    a.set_xlabel("Ratio to the main analysis (" + C.SIG + ", median over cities)")
    a.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=5)
    C.source("SuppFig4_b", g[["variant", "n_both", "median_ratio_f_star_vs_main", "median_ratio_n_star_vs_main", "f_median", "n_star_median"]],
             "Ratio of f* and of the median station count of the usable windows at f*, variant / main, median over cities found in both.")
    C.panel(ax[0], "a", x=-0.42, y=1.02), C.panel(ax[1], "b", x=-0.45, y=1.02)
    C.save(fig, "SuppFig4")


def s5(reference=False):
    """Criterion based on the stability of the scaling exponent beta, on the IUTF flows (observed and shuffled).
    reference=True adds the values of Table 1 of the reviewed manuscript and the SF = 0.4 line
    (saved as SuppFig5_reference, without Source Data)."""
    cur = pd.read_csv(C.OUT / "e7" / "old_design_curves.csv")
    pas = pd.read_csv(C.OUT / "e7" / "old_criterion_passing.csv")
    fit_name = {"old_road": "fit_200_1200m", "old_road_glob": "fit_39_sizes"}
    src_ = (lambda *a, **k: None) if reference else C.source
    fticks = [0.01, 0.04, 0.16, 0.64]

    def fax(a):
        a.set_xscale("log"), a.set_xticks(fticks), a.set_xticklabels([f"{t:g}" for t in fticks]), a.minorticks_off()
        a.set_xlabel("Coverage f (= SF²)")
        if reference:
            a.axvline(0.16, color=C.MUTED, lw=0.5, ls=":")

    fig, ax = plt.subplots(1, 4, figsize=(C.W2, 55 * C.MM), gridspec_kw=dict(wspace=0.45, left=0.06, right=0.985, top=0.85, bottom=0.2))
    rows = []
    for a, design, title in ((ax[0], "old_road", "β fitted over 200–1,000 m (17 sizes)"), (ax[1], "old_road_glob", "β fitted over 39 grid sizes")):
        for src, col, nm in (("observed", C.INK, "observed flows"), ("shuffled", C.MUTED, "shuffled flows")):
            g = cur[(cur.design == design) & (cur.source == src)]
            a.plot(g.sf ** 2, 100 * g.cvtemp_beta_nested, "-o", color=col, ms=2.3, lw=1, mec="white", mew=0.3, label=f"{nm}, centred window")
            a.plot(g.sf ** 2, 100 * g.cvtemp_beta_random_median, "--s", color=col, ms=2, lw=0.8, label=f"{nm}, random windows")
            rows += [dict(fit=fit_name[design], flows=src, sf=r.sf, f=r.sf ** 2, cvtemp_centred_pct=100 * r.cvtemp_beta_nested,
                          cvtemp_random_median_pct=100 * r.cvtemp_beta_random_median) for r in g.itertuples()]
        fax(a)
        a.set_ylabel("24 h CV of β (%)"), a.set_title(title, fontsize=5.5)
        a.set_ylim(0, 3)
    ax[1].legend(fontsize=4.5, loc="upper right")
    src_("SuppFig5_ab", pd.DataFrame(rows), "Median over the 34 cities; study area = bounding box of the road file; f = SF^2 (SF = window side / "
                                           "side of the study area); shuffled = median over 10 shuffles of the 24 h station series.")
    a = ax[2]
    rows = []
    for src, col in (("observed", C.INK), ("shuffled", C.MUTED)):
        g = cur[(cur.design == "old_road") & (cur.source == src) & (cur.sf < 1)]
        a.plot(g.sf ** 2, g.sd_beta_across_random, "-o", color=col, ms=2.3, lw=1, mec="white", mew=0.3, label=f"{src} flows")
        rows += [dict(flows=src, sf=r.sf, f=r.sf ** 2, sd_beta_across_random_windows=r.sd_beta_across_random) for r in g.itertuples()]
    if reference:
        a.plot([0.04, 0.16, 0.36, 0.64], [0.074, 0.039, 0.015, 0.005], "^:", color=C.COL["sd_log"], ms=3, lw=0.8,
               label="Table 1, reviewed manuscript")
    fax(a)
    a.set_yscale("log"), a.set_ylabel("SD of β across 20 random windows")
    a.set_title("Spread of β across window positions", fontsize=5.5), a.legend(fontsize=4.5)
    src_("SuppFig5_c", pd.DataFrame(rows), "Median over cities of the SD of the window beta (median over 24 h) across 20 random windows; "
                                          "f = SF^2.")
    a = ax[3]
    rows = []
    for src, col in (("observed", C.INK), ("shuffled", C.MUTED)):
        for thr, ls in ((0.015, "-"), (0.04, "--")):
            g = pas[(pas.design == "old_road_glob") & (pas.source == src) & np.isclose(pas.threshold, thr)]
            a.plot(g.sf ** 2, 100 * g.share_cities_passing, ls, color=col, lw=1, marker="o", ms=2, label=f"{src}, ≤ {100 * thr:g}%")
            rows += [dict(flows=src, threshold_pct=100 * thr, sf=r.sf, f=r.sf ** 2, share_cities_passing_pct=100 * r.share_cities_passing)
                     for r in g.itertuples()]
    fax(a)
    a.set_ylim(0, 100), a.set_ylabel("Cities meeting the stability criterion (%)")
    a.set_title("Stability of β (39 grid sizes)", fontsize=5.5), a.legend(fontsize=4.5, loc="lower right")
    src_("SuppFig5_d", pd.DataFrame(rows), "Share of cities whose centred-window CV_temp(beta) stays at or below the threshold from that "
                                          "coverage on; f = SF^2.")
    for a, l in zip(ax, "abcd"):
        C.panel(a, l, x=-0.3, y=1.06)
    C.save(fig, "SuppFig5_reference" if reference else "SuppFig5")


def s6():
    ov = pd.read_csv(C.OUT / "e8" / "t_overlap.csv")
    fig, ax = plt.subplots(1, 2, figsize=(C.W1 * 1.6, 55 * C.MM), gridspec_kw=dict(wspace=0.45, left=0.11, right=0.98, top=0.88, bottom=0.2))
    ax[0].plot(ov.f, 100 * ov.overlap_median, "-o", color=C.INK, ms=2.2, lw=1, label="random windows (median over cities)")
    ax[0].plot(ov.f, 100 * ov.overlap_analytic, "--", color=C.MUTED, lw=0.9, label="expected for uniform placement")
    ax[0].set_xlabel("Coverage f"), ax[0].set_ylabel("Mean pairwise overlap (%)"), ax[0].legend(fontsize=5)
    t = ov[ov.cities_with_tiles >= 5]
    for p, col, nm in (("sd_log", C.COL["sd_log"], C.SIG), ("cv500", C.COL["cv"], "CV (single grid)"), ("hn500", C.COL["hn"], "H (single grid)")):
        ax[1].plot(t.f, t[f"tiles_over_random_{p}"], "-o", color=col, ms=2.2, lw=1, label=nm)
    ax[1].axhline(1, color=C.INK, lw=0.6)
    ax[1].set_ylim(0.4, 1.6), ax[1].set_xlabel("Coverage f"), ax[1].set_ylabel("Error of non-overlapping tiles /\nerror of random windows")
    ax[1].legend(fontsize=5)
    C.panel(ax[0], "a", x=-0.25, y=1.04), C.panel(ax[1], "b", x=-0.3, y=1.04)
    C.source("SuppFig6", ov, "Overlap: mean over window pairs of the shared area / window area. Tiles: m x m non-overlapping windows "
                              "(m = floor(1/sqrt f)) with >= 20 stations; ratio of P90 errors, median over cities with >= 2 tiles.")
    C.save(fig, "SuppFig6")


def s7():
    ce = pd.read_csv(C.OUT / "e8" / "t_centres_levels.csv")
    ce["probe"] = ce.probe.replace({"cv_k32": "cv", "hn_k32": "hn"})
    fig, ax = plt.subplots(1, 2, figsize=(C.W1 * 1.6, 55 * C.MM), gridspec_kw=dict(wspace=0.4, left=0.1, right=0.98, top=0.85, bottom=0.2))
    rows = []
    for k in C.TRAFFIC:
        g = ce[ce.probe == k]
        s = g.assign(r=g.p90_main / g.p90_none).groupby("f").r.agg(["median", "count"])
        s = s[s["count"] >= 10]
        ax[0].plot(s.index, s["median"], "-o", color=C.COL[k], ms=2.2, lw=1, label=C.LAB[k])
        poly = g[g.n_centres >= 2]
        s2 = poly.assign(r=poly.p90_secondary_only / poly.p90_main).groupby("f").r.agg(["median", "count"])
        s2 = s2[s2["count"] >= 8]
        ax[1].plot(s2.index, s2["median"], "-o", color=C.COL[k], ms=2.2, lw=1, label=C.LAB[k])
        rows += [dict(metric=k, comparison="main centre / no centre", f=f, ratio_median=r["median"], cities=int(r["count"])) for f, r in s.iterrows()]
        rows += [dict(metric=k, comparison="secondary only / main centre (>= 2 centres)", f=f, ratio_median=r["median"], cities=int(r["count"]))
                 for f, r in s2.iterrows()]
    for a in ax:
        a.axhline(1, color=C.INK, lw=0.6)
        a.set_xscale("log"), C.log_ticks(a, "x", (0.03, 0.1, 0.3)), a.set_xlabel("Coverage f")
    ax[0].set_ylabel("Error ratio"), ax[0].set_title("Windows with the main centre / without any centre", fontsize=5.5)
    ax[1].set_title("Secondary centre only / main centre", fontsize=5.5)
    ax[0].legend(fontsize=5)
    C.panel(ax[0], "a", x=-0.22, y=1.06), C.panel(ax[1], "b", x=-0.22, y=1.06)
    C.source("SuppFig7", pd.DataFrame(rows), "Centres: peaks of the flow-weighted station density (>= 50% of the highest), merged within 2 km. "
                                             "P90 errors per group of windows (>= 5 windows per group); median of the ratio over cities.")
    C.save(fig, "SuppFig7")


def s8():
    from .fig5 import STEPS, step_frame, step_panel
    de = pd.read_csv(C.OUT / "e5" / "deff.csv")
    tc, teps, tf = C.traffic_curves()
    i5 = int(np.argmin(np.abs(C.levels() - 0.5)))
    fig, ax = plt.subplots(2, 3, figsize=(C.W1 * 1.6, 100 * C.MM), gridspec_kw=dict(wspace=0.15, hspace=0.55, left=0.1, right=0.98, top=0.92,
                                                                                         bottom=0.1))
    rows = []
    for r_, k in enumerate(("mean_log", "site_cv")):
        s = step_frame(de, k, tc, i5)
        for j, (col, fcol, title) in enumerate(STEPS):
            a = ax[r_, j]
            e, rr = step_panel(a, s, col, fcol, C.COL[k], ms=6)
            a.set_title(f"{title}\nmean error {e.mean():.2f}; {100 * (e <= 0.1).mean():.0f}% within ±0.1", fontsize=5)
            if j:
                a.set_yticklabels([])
            else:
                a.set_ylabel(f"Observed f* ({C.LAB[k]})", fontsize=5.5)
            a.set_xlabel("Computed f*", fontsize=5.5)
            rows += [dict(metric=k, step=title, **r) for r in rr]
    C.source("SuppFig8", pd.DataFrame(rows), "As Fig. 5d for the mean of log flows and the site CV. flag 'above' = above 0.9 (drawn at 0.95, "
                                             "open symbols); errors over the cities whose observed f* is found.")
    C.save(fig, "SuppFig8")


def s9():
    d = pd.read_csv(C.OUT / "e8" / "t_dates.csv")
    d["first"] = pd.to_datetime(d.first_date)
    d["last"] = pd.to_datetime(d.last_date)
    d = d.sort_values("first")
    fig, ax = plt.subplots(1, 2, figsize=(C.W2, 95 * C.MM), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.08, left=0.1, right=0.985,
                                                                                 top=0.95, bottom=0.1))
    y = np.arange(len(d))
    ax[0].hlines(y, d["first"], d["last"], color=C.COL["sd_log"], lw=2)
    ax[0].scatter(d["first"], y, s=6, color=C.COL["sd_log"], zorder=3)
    ax[0].set_yticks(y), ax[0].set_yticklabels([C.name(c) for c in d.city], fontsize=5), ax[0].set_ylim(len(d) - 0.5, -0.5)
    ax[0].set_xlabel("Observation period (first to last date in the data)")
    ax[1].barh(y - 0.18, d.n_city_weekdays, height=0.34, color=C.COL["sd_log"], label="weekdays used")
    ax[1].barh(y + 0.18, d.n_weekend_days, height=0.34, color=C.MUTED, label="weekend days")
    ax[1].set_xscale("log"), ax[1].set_xlim(0.8, 100), C.log_ticks(ax[1], "x", (1, 2, 5, 10, 20, 50, 100))
    ax[1].set_yticks(y), ax[1].set_yticklabels([]), ax[1].set_ylim(len(d) - 0.5, -0.5)
    ax[1].set_xlabel("Days"), ax[1].legend(fontsize=5, loc="lower right")
    C.panel(ax[0], "a", x=-0.12, y=1.0), C.panel(ax[1], "b", x=-0.05, y=1.0)
    C.source("SuppFig9", d.drop(columns=["first", "last"]), "Weekdays used: non-holiday weekdays with >= 50% of stations valid, hours "
                                                            "07:00-19:59.")
    C.save(fig, "SuppFig9")



RAIN = "#e87ba4"


def s10(thr="thr0p5"):
    """Rain events (E10): whole-city change against comparison days, the window needed for the city-wide slowdown (a-d),
    and the slowdown hour by hour before, during and after the rain (e, f; tools/rain_periods.py)."""
    od = C.OUT / "e10" / thr
    ev = pd.read_csv(od / "events.csv")
    ps = pd.read_csv(od / "pseudo.csv")
    lev = pd.read_csv(od / "levels.csv")
    cr = pd.read_csv(od / "critical.csv")
    sm = json.load(open(od / "summary.json"))
    per = pd.read_csv(C.OUT / "e10" / "periods" / "summary.csv", dtype={"period": str})
    fig, ax = plt.subplots(3, 2, figsize=(C.W2, 176 * C.MM), gridspec_kw=dict(wspace=0.32, hspace=0.62, left=0.12, right=0.985,
                                                                                  top=0.965, bottom=0.07))
    ax = ax.ravel()
    # ---- a: z of rain events against the comparison days themselves (placebo)
    a = ax[0]
    names = {"Lq": "Flow", "Lo": "Occupancy", "Lv": "Speed index"}
    rows, shares = [], []
    for i, mn in enumerate(("Lq", "Lo", "Lv")):
        pz = []
        for _, g in ps.groupby(["city", "event"]):
            x = g[mn].dropna().to_numpy()
            if x.size < 6:
                continue
            for k in range(x.size):
                o = np.delete(x, k)
                if o.std(ddof=1) > 0:
                    pz.append((x[k] - o.mean()) / o.std(ddof=1))
        pz = np.clip(np.array(pz), -4, 6)
        ez = np.clip(ev[f"{mn}_z"].dropna().to_numpy(), -4, 6)
        sign = -1 if mn == "Lq" else 1
        rng = np.random.default_rng(i)
        a.scatter(pz, i + 0.17 + rng.uniform(-0.1, 0.1, pz.size), s=3, color=C.MUTED, alpha=0.35, lw=0, zorder=2,
                  label="comparison days" if i == 0 else None)
        a.scatter(ez, i - 0.17 + rng.uniform(-0.1, 0.1, ez.size), s=8, color=RAIN, lw=0.3, edgecolor="white", zorder=3,
                  label="rain events" if i == 0 else None)
        for y0, v in ((i + 0.17, np.median(pz)), (i - 0.17, np.median(ez))):
            a.plot([v, v], [y0 - 0.13, y0 + 0.13], color=C.INK, lw=1.0, zorder=4)
        shares.append(100 * sm[mn]["share_above_pseudo_median"])
        rows += [dict(measure=names[mn], group="rain event", z=float(v)) for v in ev[f"{mn}_z"].dropna()]
        rows += [dict(measure=names[mn], group="comparison day", z=float(v)) for v in pz]
    a.axvline(0, color=C.INK, lw=0.5)
    labs = [f"{nm}\n{s:.0f}% of events above\nthe median day" for nm, s in zip(("Flow drop", "Occupancy rise", "Speed-index drop"), shares)]
    a.set_yticks(range(3)), a.set_yticklabels(labs, fontsize=5, linespacing=1.15)
    a.set_ylim(2.6, -0.6), a.set_xlim(-4.2, 6.0)
    a.set_xlabel("Change against comparison days (z)")
    a.legend(loc="lower right", fontsize=5, handletextpad=0.1, borderaxespad=0.2)
    C.source("SuppFig10_a", pd.DataFrame(rows), "z = (event value - mean of the comparison days) / SD of the comparison days; comparison days "
             "= dry weekdays of the same data block within 14 days; for comparison days, each day against the others (placebo). "
             "Flow: drop of the whole-city flow; occupancy: rise; speed index (flow / occupancy): drop. Clipped to [-4, 6] in the plot.")
    # ---- b: city-wide slowdown against the amount of rain
    a = ax[1]
    det = ev.Lv_detect.astype(str) == "True"
    a.axhline(0, color=C.INK, lw=0.5)
    a.scatter(ev.total_mm[det], 100 * ev.Lv[det], s=10, color=RAIN, lw=0.3, edgecolor="white", zorder=3,
              label="beyond the 95th percentile of comparison days")
    a.scatter(ev.total_mm[~det], 100 * ev.Lv[~det], s=10, facecolor="white", edgecolor=RAIN, lw=0.6, zorder=3, label="other events")
    a.set_xscale("log"), C.log_ticks(a, "x", (0.5, 1, 2, 5, 10, 20))
    a.set_xlabel("Rain during the event (mm)"), a.set_ylabel("City-wide slowdown (%)")
    rho = stats.spearmanr(ev.total_mm, ev.Lv)[0]
    a.text(0.03, 0.97, f"{100 * np.mean(ev.Lv > 0):.0f}% of {len(ev)} events slower; Spearman {rho:.2f}", transform=a.transAxes,
           va="top", fontsize=5, color=C.INK2)
    a.legend(loc="lower right", fontsize=4.8, handletextpad=0.1)
    C.source("SuppFig10_b", ev[["city", "day", "hour_start", "hour_end", "max_mmh", "total_mm", "n_stations", "n_comparison",
                                "Lq", "Lo", "Lv", "Lv_z", "Lv_detect", "recovery_h"]],
             "One row per rain event (ERA5 >= 0.5 mm/h over the study area, 07:00-19:59 of a weekday). Lv = 1 - (sum flow / sum occupancy) / "
             "(the same on the comparison days); recovery_h = hours after the rain until the hourly Lv is back below the 95th percentile "
             "of the comparison days (detectable events; empty = not reached by 20:00).")
    # ---- c: window error of the city-wide slowdown (median over cities) with sigma_z for reference
    a = ax[2]
    eps = sm["eps"]["Lv"]
    cities = sorted(cr[cr.measure == "Lv"].city)
    lv_levels = np.array(sorted(lev.f.unique()))
    cur = {c: lev[lev.city == c].set_index("f").reindex(lv_levels).Lv_p90.to_numpy() / eps for c in cities}
    med, q1, q3, n = C.median_band(cur, min_n=6)
    ok = np.isfinite(med)
    a.fill_between(lv_levels[ok], q1[ok], q3[ok], color=RAIN, alpha=0.18, lw=0)
    a.plot(lv_levels[ok], med[ok], "-o", color=RAIN, ms=2.3, lw=1.0, mec="white", mew=0.3, label="city-wide slowdown in rain")
    tc, teps, tf = C.traffic_curves()
    tl = C.levels()
    m2, _, _, n2 = C.median_band({c: tc["sd_log"][c] for c in cities if c in tc["sd_log"]}, min_n=6)
    ok2 = np.isfinite(m2)
    a.plot(tl[ok2], m2[ok2], "--", color=C.COL["sd_log"], lw=1.0, label=C.SIG + " (same cities)")
    a.axhline(1, color=C.INK, lw=0.6, ls=":")
    a.set_xscale("log"), a.set_yscale("log"), a.set_xlim(0.1, 1.0)
    C.log_ticks(a, "x", (0.1, 0.2, 0.3, 0.5, 0.9)), C.log_ticks(a, "y", (0.3, 1, 3, 10))
    a.set_xlabel("Coverage f"), a.set_ylabel("Window error / tolerance")
    a.legend(loc="lower left", fontsize=5)
    rows = [dict(metric="rain_slowdown", f=f, median=m_, q25=x1, q75=x3, n_cities=int(nn)) for f, m_, x1, x3, nn in zip(lv_levels, med, q1, q3, n)]
    rows += [dict(metric="sd_log", f=f, median=m_, n_cities=int(nn)) for f, m_, nn in zip(tl, m2, n2)]
    C.source("SuppFig10_c", pd.DataFrame(rows), f"P90 over windows and events of |window slowdown - city-wide slowdown| / tolerance; tolerance = "
             f"0.25 x IQR of the city-wide slowdown over the events of the {len(cities)} cities with >= 3 events ({eps:.3f}); "
             "median and quartiles over these cities (levels with >= 6 cities); sigma_z for the same cities.")
    # ---- d: critical coverage of the slowdown against that of sigma_z
    a = ax[3]
    g = cr[cr.measure == "Lv"].copy()
    g = g[g.flag_sd_log == "ok"]
    y = np.where(g.flag == "ok", g.f_star, 0.95)
    a.plot([0, 1], [0, 1], color=C.MUTED, lw=0.6)
    a.axhline(0.9, color=C.MUTED, lw=0.4, ls=":")
    okk = (g.flag == "ok").to_numpy()
    a.scatter(g.f_star_sd_log[okk], y[okk], s=12, color=RAIN, lw=0.3, edgecolor="white", zorder=3)
    a.scatter(g.f_star_sd_log[~okk], y[~okk], s=12, facecolor="white", edgecolor=RAIN, lw=0.6, zorder=3)
    off = {"hamburg": (4, 1), "toronto": (-4, -7), "london": (4, 1), "madrid": (-3, -8), "taipeh": (4, 1), "bordeaux": (4, -6),
           "torino": (-3, 4), "strasbourg": (-5, 4), "essen": (4, -2), "manchester": (4, -2), "augsburg": (4, -2)}
    for c, xx, yy in zip(g.city, g.f_star_sd_log, y):
        dx, dy = off.get(c, (3, -1))
        a.annotate(C.name(c), (xx, yy), xytext=(dx, dy), textcoords="offset points", fontsize=4.5, color=C.INK2,
                   ha="right" if dx < 0 else "left")
    w = sm["windows"]["Lv"]
    nab = int(((g.flag == "above") | ((g.flag == "ok") & (g.f_star > g.f_star_sd_log))).sum())
    a.text(0.97, 0.03, f"{nab} of {len(g)} above the diagonal\none-point formula: mean error {w['one_point_mae']:.2f}",
           transform=a.transAxes, ha="right", va="bottom", fontsize=5, color=C.INK2)
    a.set_xlim(0, 1), a.set_ylim(0, 1), a.set_aspect("equal")
    a.set_xlabel("Critical coverage f* of " + C.SIG), a.set_ylabel("f* of the city-wide slowdown")
    C.source("SuppFig10_d", cr[cr.measure == "Lv"][["city", "n_events", "f_star", "flag", "C_at_half", "f_star_one_point", "b", "r2",
                                                     "f_star_sd_log", "flag_sd_log", "err_at_sd_log_fstar"]],
             "Per city (>= 3 events): f* of the city-wide slowdown (flag above = not within tolerance below 0.9, drawn at 0.95), "
             "error at f = 0.5 in tolerance units, f* computed from it (b = 0.7), fitted exponent b and R^2 of the coverage law, "
             "f* of sigma_z, and the slowdown error at the sigma_z f* in tolerance units.")
    # ---- e, f: city-wide slowdown hour by hour around the rain (events with >= 5 comparison days)
    order = ["-2", "-1", "R", "+1", "+2", "+3"]
    labs = ["2 h\nbefore", "1 h\nbefore", "During\nrain", "1 h\nafter", "2 h\nafter", "3 h\nafter"]
    st = per.set_index("period").loc[order]
    x = np.arange(len(order))
    a = ax[4]
    a.axvspan(1.5, 2.5, color=RAIN, alpha=0.08, lw=0)
    a.fill_between(x, st.comparison_q25, st.comparison_q75, color=C.MUTED, alpha=0.25, lw=0,
                   label="comparison days, interquartile range")
    a.plot(x, st.comparison_median, "--o", color=C.MUTED, lw=0.9, ms=2.3, label="comparison days, median")
    em = st.events_median.to_numpy(float)
    a.errorbar(x + 0.07, em, yerr=[em - st.events_q25.to_numpy(float), st.events_q75.to_numpy(float) - em], fmt="-o",
               color=RAIN, ms=3, lw=1.0, elinewidth=0.8, capsize=2, mec="white", mew=0.3,
               label="rain events, median and interquartile range")
    a.axhline(0, color=C.INK, lw=0.5)
    a.set_xticks(x), a.set_xticklabels(labs, fontsize=5)
    a.set_ylim(-4, 22), a.set_ylabel("City-wide slowdown (%)")
    for i, n_ in enumerate(st.n_events):
        a.text(i, -3.3, f"n = {int(n_)}", ha="center", fontsize=4.5, color=C.INK2)
    a.legend(loc="upper left", fontsize=4.8, handletextpad=0.3, borderaxespad=0.2)
    a = ax[5]
    a.axvspan(1.5, 2.5, color=RAIN, alpha=0.08, lw=0)
    a.bar(x, st.events_beyond_p95, width=0.55, color=RAIN, label="rain events")
    a.plot(x, st.comparison_beyond_p95, "--o", color=C.MUTED, lw=0.9, ms=2.3, label="comparison days (normal variation)")
    for i, v in enumerate(st.events_beyond_p95):
        a.text(i, v + 1.0, f"{v:.0f}", ha="center", fontsize=5, color=C.INK2)
    a.set_xticks(x), a.set_xticklabels(labs, fontsize=5)
    a.set_ylim(0, 50), a.set_ylabel("Beyond the 95th percentile\nof comparison days (%)")
    a.legend(loc="upper right", fontsize=4.8, handletextpad=0.3, borderaxespad=0.2)
    C.source("SuppFig10_ef", per, "City-wide slowdown (%) per period around the rain: two and one hours before, the rain hours, and one "
             "to three hours after; events with >= 5 comparison days. Events: the event against the median of its comparison days; "
             "comparison days: each day against the median of the other comparison days. Median and quartiles over events and over "
             "comparison days (e); share of events and of comparison days beyond the 95th percentile of the comparison days in the "
             "same period, each comparison day judged against the others (f). From tools/rain_periods.py.")
    for a_, l in zip(ax, "abcdef"):
        C.panel(a_, l, x=-0.2, y=1.03)
    C.save(fig, "SuppFig10")


def _rank_x(lab):
    return 1.02 if lab == "f*" else float(lab.split("=")[1])


def s11():
    """Spearman between window values (one random window per city) and whole-city values across cities."""
    rn = pd.read_csv(C.OUT / "checks" / "ranking_network.csv")
    rt = pd.read_csv(C.OUT / "e8" / "t_ranking.csv")
    rt = rt[(rt.comparison == "window location") & (rt.probe == "sd_log")]
    fig, ax = plt.subplots(1, 2, figsize=(C.W2, 62 * C.MM), gridspec_kw=dict(wspace=0.25, left=0.08, right=0.985, top=0.9, bottom=0.2))
    rows = []
    for a, keys in ((ax[0], C.NETWORK), (ax[1], ("dw_eff_loss", "acc_loss"))):
        g = rt.copy()
        g["x"] = g["at"].map(_rank_x)
        lv = g[g["at"] != "f*"]
        a.plot(lv.x, lv.rho_median, "--", color=C.COL["sd_log"], lw=0.9, label=C.SIG + " (Fig. 2f)")
        a.scatter([1.02], g[g["at"] == "f*"].rho_median, s=10, color=C.COL["sd_log"], zorder=3)
        for k in keys:
            col = EXTRA.get(k, C.COL.get(k))
            d = rn[rn.metric == k].copy()
            d["x"] = d["at"].map(_rank_x)
            lv = d[d["at"] != "f*"].sort_values("x")
            a.fill_between(lv.x, lv.rho_p05, lv.rho_median, color=col, alpha=0.12, lw=0)
            a.plot(lv.x, lv.rho_median, "-o", color=col, ms=2.5, lw=1.0, mec="white", mew=0.3, label=C.LAB[k])
            fs = d[d["at"] == "f*"]
            a.errorbar([1.02], fs.rho_median, yerr=[fs.rho_median - fs.rho_p05, [0] * len(fs)], fmt="o", color=col, ms=3, lw=0.8,
                       mec="white", mew=0.3, zorder=4)
            rows += [dict(metric=k, at=r["at"], cities=int(r.cities), rho_median=r.rho_median, rho_p05=r.rho_p05) for _, r in d.iterrows()]
        a.axvline(0.96, color=C.GRID, lw=0.6)
        a.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.02]), a.set_xticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "own f*"])
        a.set_xlim(0.37, 1.06), a.set_ylim(0.4, 1.0)
        a.set_xlabel("Coverage f"), a.set_ylabel("Rank correlation with whole-city values")
        a.legend(loc="lower right", fontsize=5)
    rows += [dict(metric="sd_log", at=r["at"], cities=int(r.cities), rho_median=r.rho_median, rho_p05=r.rho_p05) for _, r in rt.iterrows()]
    C.panel(ax[0], "a", x=-0.13, y=1.03), C.panel(ax[1], "b", x=-0.13, y=1.03)
    C.source("SuppFig11", pd.DataFrame(rows), "Spearman rho across the 34 cities between the metric in one random window per city (one of the "
             "20 windows per level) and its whole-city value, 1000 draws: median and 5th percentile; 'f*' = each city's own critical "
             "coverage (first level at or above it; 0.9 when above 0.9). sigma_z from E8 (31 cities).")
    C.save(fig, "SuppFig11")


def s12():
    """H at the 39 grid sizes from 10 m to 10 km (whole extent, one grid origin)."""
    cu = pd.read_csv(C.OUT / "checks" / "segments_curves.csv")
    sg = pd.read_csv(C.OUT / "checks" / "segments.csv")
    fig, ax = plt.subplots(1, 3, figsize=(C.W2, 62 * C.MM), gridspec_kw=dict(wspace=0.42, left=0.07, right=0.985, top=0.9, bottom=0.2))
    rows = []
    # a: H against L, all cities grey, London highlighted with one power law over the 39 sizes
    a = ax[0]
    for c, g in cu.groupby("city"):
        g = g[g.H > 0]
        a.plot(g.L, g.H, color=C.GRID if c != "london" else C.INK, lw=0.5 if c != "london" else 1.1, zorder=2 if c != "london" else 4)
    g = cu[(cu.city == "london") & (cu.H > 0)]
    b, a0 = np.polyfit(np.log(g.L), np.log(g.H), 1)
    LL = np.geomspace(10, 10000, 50)
    a.plot(LL, np.exp(a0) * LL ** b, "--", color=C.COL["sd_log"], lw=0.9, zorder=5, label=f"one power law, London (exponent {b:.2f})")
    a.axvspan(200, 1000, color=C.BAND, alpha=0.5, lw=0, zorder=1)
    a.text(450, 10.5, "200–1000 m", ha="center", va="top", fontsize=5, color=C.INK2)
    a.set_xscale("log"), a.set_yscale("log"), C.log_ticks(a, "x", (10, 100, 1000, 10000)), C.log_ticks(a, "y", (0.3, 1, 3, 10))
    a.set_ylim(0.18, 12)
    a.set_xlabel("Grid size L (m)"), a.set_ylabel("Entropy H of the whole study area")
    a.plot([], [], color=C.INK, lw=1.1, label="London"), a.plot([], [], color=C.GRID, lw=0.8, label="other cities")
    a.legend(loc="lower left", fontsize=4.8)
    # b: H against ln(number of cells with flow)
    a = ax[1]
    g = cu[(cu.H > 0) & (cu.n_nodes > 1)]
    a.scatter(np.log(g.n_nodes), g.H, s=2, color=C.INK2, alpha=0.35, lw=0)
    lim = (0, max(np.log(g.n_nodes).max(), g.H.max()) + 0.3)
    a.plot(lim, lim, color=C.MUTED, lw=0.6)
    r = np.corrcoef(np.log(g.n_nodes), g.H)[0, 1]
    a.text(0.04, 0.96, f"r = {r:.3f}", transform=a.transAxes, va="top", fontsize=5.5, color=C.INK2)
    a.set_xlim(lim), a.set_ylim(lim), a.set_aspect("equal")
    a.set_xlabel("ln(number of cells in the adjacency graph)"), a.set_ylabel("Entropy H")
    # c: local log-log slope
    a = ax[2]
    sl = []
    for c, g in cu.groupby("city"):
        g = g[g.H > 0].sort_values("L")
        x, y = np.log(g.L.to_numpy()), np.log(g.H.to_numpy())
        s_ = np.diff(y) / np.diff(x)
        Lm = np.exp((x[1:] + x[:-1]) / 2)
        sl += [dict(city=c, L_mid=float(l_), slope=float(v)) for l_, v in zip(Lm, s_)]
    sl = pd.DataFrame(sl)
    sl["L_bin"] = sl.L_mid.round(-1)
    q = sl.groupby("L_bin").slope.agg(["median", lambda v: v.quantile(0.25), lambda v: v.quantile(0.75), "count"])
    q.columns = ["median", "q25", "q75", "n"]
    q = q[q.n >= 17]
    a.axvspan(200, 1000, color=C.BAND, alpha=0.5, lw=0)
    a.fill_between(q.index, q.q25, q.q75, color=C.INK2, alpha=0.18, lw=0)
    a.plot(q.index, q["median"], color=C.INK, lw=1.0)
    a.axhline(0, color=C.INK, lw=0.4)
    a.set_xscale("log"), C.log_ticks(a, "x", (10, 100, 1000, 10000))
    a.set_xlabel("Grid size L (m)"), a.set_ylabel("Local slope d ln H / d ln L")
    h = sg[(sg.measure == "H") & sg.dBIC.notna()]
    n17 = h[h.range == "200-1000 m"]
    nall = h[h.range == "all sizes"]
    a.text(0.03, 0.04, f"one power law, 39 sizes: median R² {nall.r2_single.median():.2f}\n"
           f"two segments better (200–1000 m): {int((n17.dBIC > 10).sum())} of {len(n17)} cities",
           transform=a.transAxes, fontsize=4.8, color=C.INK2, va="bottom")
    for a_, l in zip(ax, "abc"):
        C.panel(a_, l, x=-0.22, y=1.03)
    C.source("SuppFig12_ab", cu[["city", "L", "H", "CV", "n_cells", "n_nodes"]], "Panels a and b. Whole study area, station flows (weekday daytime "
             "means), one grid origin at the corner of the study area, 39 grid sizes from 10 m to 10 km; n_cells = cells with flow; n_nodes = cells with flow "
             "that have at least one neighbouring cell with flow, i.e. the nodes of the adjacency graph on which H is computed (panel b: ln n_nodes).")
    C.source("SuppFig12_c", q.reset_index().rename(columns={"L_bin": "L_m"}), "Local slope of ln H on ln L between consecutive grid sizes, "
             "median and quartiles over cities (sizes with >= 17 cities).")
    C.source("SuppFig12_fits", sg, "Per city: one power law against two joined at a break (least squares in log-log, BIC); dBIC > 10 = "
             "two segments clearly better.")
    C.save(fig, "SuppFig12")


def s13():
    """Critical coverage against stations, observation period and layout."""
    cr = pd.read_csv(C.OUT / "checks" / "correlates.csv")
    att = pd.read_csv(C.OUT / "checks" / "city_attributes.csv").set_index("city")
    tc, teps, tf = C.traffic_curves()
    q = tf["sd_log"]
    fig = plt.figure(figsize=(C.W2, 70 * C.MM))
    g = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.75, left=0.13, right=0.985, top=0.9, bottom=0.2)
    # a: heatmap of Spearman rho
    a = fig.add_subplot(g[0])
    attrs = ["stations", "area_km2", "density_per_km2", "empty_cells_500m", "D_station", "xi_m", "weekdays", "span_days"]
    alab = ["Stations", "Study area", "Station density", "Empty 500 m cells", "Layout dimension D", "Correlation length",
            "Weekdays used", "Calendar span"]
    mets = ["sd_log", "cv", "hn", "R_random", "R_btw", "eff_loss"]
    mlab = [C.SIG, "CV", "H", "Random\nfailure", "Targeted\nattack", "Efficiency\nloss"]
    M = np.array([[cr[(cr.attribute == at_) & (cr.metric == mt)].spearman.iloc[0] for mt in mets] for at_ in attrs])
    P = np.array([[cr[(cr.attribute == at_) & (cr.metric == mt)].p.iloc[0] for mt in mets] for at_ in attrs])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#a32c2c"])
    im = a.imshow(M, cmap=cmap, vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i, j]:.2f}" + ("*" if P[i, j] < 0.05 else ""), ha="center", va="center", fontsize=4.6,
                   color=C.INK if abs(M[i, j]) < 0.55 else "white")
    a.set_xticks(range(len(mets))), a.set_xticklabels(mlab, fontsize=4.3)
    a.set_yticks(range(len(attrs))), a.set_yticklabels(alab, fontsize=5)
    a.tick_params(length=0)
    for s_ in a.spines.values():
        s_.set_visible(False)
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=4.8), cb.set_label("Spearman ρ with f*", fontsize=5)
    # b, c: sigma_z f* against stations and weekdays
    st = C.fstar_stats(q)
    rows = []
    for a_i, (col, lab, logx) in enumerate((("stations", "Stations", True), ("weekdays", "Weekdays used", True))):
        a = fig.add_subplot(g[1 + a_i])
        okc = [c for c in q.index if q.loc[c, "flag"] == "ok"]
        abc = [c for c in q.index if q.loc[c, "flag"] == "above"]
        a.scatter(att.loc[okc, col], q.loc[okc, "f_star"], s=9, color=C.COL["sd_log"], lw=0.3, edgecolor="white", zorder=3)
        a.scatter(att.loc[abc, col], np.full(len(abc), 0.95), s=9, facecolor="white", edgecolor=C.COL["sd_log"], lw=0.6, zorder=3)
        a.axhline(0.9, color=C.MUTED, lw=0.4, ls=":")
        r_ = cr[(cr.attribute == col) & (cr.metric == "sd_log")].iloc[0]
        a.text(0.97, 0.04, f"Spearman {r_.spearman:.2f} (p = {r_.p:.2g})", transform=a.transAxes, ha="right", fontsize=5, color=C.INK2)
        if logx:
            a.set_xscale("log")
            ticks = (30, 100, 300, 1000, 3000) if col == "stations" else (1, 3, 10, 30, 100)
            C.log_ticks(a, "x", ticks)
        a.set_ylim(0, 1), a.set_xlabel(lab), a.set_ylabel("Critical coverage f* (" + C.SIG + ")")
        rows += [dict(city=c, attribute=col, value=float(att.loc[c, col]), f_star=float(q.loc[c, "f_star"]) if q.loc[c, "flag"] == "ok" else np.nan,
                      flag=q.loc[c, "flag"]) for c in okc + abc]
    for a_, l, x in zip(fig.axes[:1] + fig.axes[2:], "abc", (-0.42, -0.3, -0.3)):
        C.panel(a_, l, x=x, y=1.03)
    C.source("SuppFig13_a", cr, "Spearman correlation across cities between f* and each attribute; cities above 0.9 enter with 0.95 "
             "(ranked last), cities without a usable level are left out; * p < 0.05.")
    C.source("SuppFig13_bc", pd.DataFrame(rows), "sigma_z f* (open symbols: above 0.9, drawn at 0.95) against the number of stations and "
             "the weekdays used.")
    C.save(fig, "SuppFig13")


ALL = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13")

if __name__ == "__main__":
    C.style()
    for w in (sys.argv[1:] or ALL):
        C.style()
        globals()[w]()
