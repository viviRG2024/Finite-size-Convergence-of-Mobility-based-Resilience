"""Fig. 5 | One law for every metric, and the required coverage in three steps.

a  traffic diagnostics: each city's window error divided by its error at half coverage, against (1 - f)/f; lines:
   power laws with exponent b = 0.5 and 0.7
b  the same for the road-network metrics
c  critical coverage against the error at half coverage (in tolerance units), all cities and metrics; curves:
   f* = 1 / (1 + (tolerance / C)^(1/b)) for b = 0.5 and 0.7
d  sigma_z, required coverage computed in three steps against the observed critical coverage: (1) number of stations and
   their spread only; (2) times the design effect of the spatial correlation from block averaging; (3) from the error
   measured once at half coverage (b = 0.7). Band: +-0.1
e  step 3 for the six metrics
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common as C

B = 0.7
TOP = 0.95          # where cities whose f* lies above 0.9 are drawn


def fpred(Cv, b=B):
    return 1 / (1 + (1 / Cv) ** (1 / b))


STEPS = (("f_mix", "flag_mix", "1  Stations and spread"), ("f_block", "flag_block", "2  + Block averaging"),
         ("f_one", "flag_one", "3  + One measurement"))


def unit_ticks(ax):
    for f in (ax.set_xticks, ax.set_yticks):
        f([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "", "0.5", "", "1"]), ax.set_yticklabels(["0", "", "0.5", "", "1"])


def step_frame(de, k, tc, i5):
    """Cities with a usable level for probe k: observed f* (ok / above) and the f* computed in the three steps."""
    s = de[(de.probe == k) & de.flag_obs.isin(["ok", "above"])].set_index("city").copy()
    c5 = np.array([tc[k][c][i5] for c in s.index], float)
    with np.errstate(all="ignore"):
        s["f_one"] = np.where(np.isfinite(c5) & (c5 > 0), fpred(c5), np.nan)
    s["flag_one"] = np.where(s.f_one.notna(), "ok", "none")
    return s


def step_panel(ax, s, col, fcol, color, ms=7):
    """Computed against observed f*; values above 0.9 drawn at TOP with open symbols. Returns the errors over the cities
    whose observed f* is found (a computed value above 0.9 counts at 0.9, a lower bound of its error) and Source Data rows."""
    ax.fill_between([0, 1], [-0.1, 0.9], [0.1, 1.1], color=C.BAND, alpha=0.6, lw=0)
    ax.plot([0, 1], [0, 1], color=C.INK, lw=0.6)
    ax.axhline(0.9, color=C.MUTED, lw=0.4, ls=":")
    obs_ok = (s.flag_obs == "ok").to_numpy()
    comp_ab = (s[fcol] == "above").to_numpy()
    if comp_ab.any():
        ax.axvline(0.9, color=C.MUTED, lw=0.4, ls=":")
    valid = s[fcol].isin(["ok", "above"]).to_numpy()
    x = np.where(comp_ab, TOP, s[col].to_numpy(float))
    y = np.where(obs_ok, s.f_obs.to_numpy(float), TOP)
    full = valid & obs_ok & ~comp_ab
    part = valid & ~full
    ax.scatter(x[full], y[full], s=ms, color=color, lw=0.3, edgecolor="white", zorder=3)
    ax.scatter(x[part], y[part], s=ms, facecolor="white", edgecolor=color, lw=0.5, zorder=3)
    sel = valid & obs_ok
    e = np.abs(np.where(comp_ab, 0.9, s[col].to_numpy(float)) - s.f_obs.to_numpy(float))[sel]
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect("equal")
    unit_ticks(ax)
    rows = [dict(city=c, f_computed=float(v) if fc == "ok" else np.nan, computed_flag=fc,
                 f_observed=float(o) if fo == "ok" else np.nan, observed_flag=fo)
            for c, v, fc, o, fo in zip(s.index, s[col], s[fcol], s.f_obs, s.flag_obs)]
    return e, rows


def main():
    C.style()
    fig = plt.figure(figsize=(C.W2, 125 * C.MM))
    gt = fig.add_gridspec(1, 3, left=0.065, right=0.985, top=0.94, bottom=0.6, wspace=0.38)
    gd = fig.add_gridspec(1, 3, left=0.065, right=0.585, top=0.43, bottom=0.085, wspace=0.12)
    ge = fig.add_gridspec(1, 2, left=0.655, right=0.985, top=0.43, bottom=0.085, wspace=0.12)
    for letter, x, y in (("a", 0.005, 0.985), ("b", 0.335, 0.985), ("c", 0.665, 0.985), ("d", 0.005, 0.5), ("e", 0.615, 0.5)):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top", ha="left", color=C.INK)
    tc, teps, tf = C.traffic_curves()
    nc, neps, nf, nlv = C.network_curves()
    tl = C.levels()
    i5t = int(np.argmin(np.abs(tl - 0.5)))
    i5n = int(np.argmin(np.abs(nlv - 0.5)))
    # ---- a, b
    rows = []
    for ax_i, (fam, keys, lv, i5, cur) in enumerate((("traffic", C.TRAFFIC, tl, i5t, tc), ("network", C.NETWORK, nlv, i5n, nc))):
        ax = fig.add_subplot(gt[ax_i])
        u = (1 - lv) / lv
        for k in keys:
            for c, y in cur[k].items():
                if not np.isfinite(y[i5]) or y[i5] <= 0:
                    continue
                v = y / y[i5]
                ok = np.isfinite(v) & (v > 0)
                ax.plot(u[ok], v[ok], color=C.COL[k], lw=0.35, alpha=0.45)
                rows += [dict(family=fam, metric=k, city=c, f=f, u=uu, error_over_error_at_half=vv) for f, uu, vv in zip(lv[ok], u[ok], v[ok])]
        uu = np.geomspace(0.1, 25, 50)
        ax.plot(uu, uu ** 0.5, color=C.INK, lw=0.8, ls="--", label="exponent b = 0.5")
        ax.plot(uu, uu ** 0.7, color=C.INK, lw=0.8, ls=":", label="exponent b = 0.7")
        for k in keys:
            ax.plot([], [], color=C.COL[k], lw=1.2, label=C.LAB[k] if fam == "traffic" else C.SHORT[k])
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(0.1, 25), ax.set_ylim(0.03, 30)
        C.log_ticks(ax, "x", (0.1, 0.3, 1, 3, 10)), C.log_ticks(ax, "y", (0.1, 1, 10))
        ax.set_xlabel("(1 − f) / f"), ax.set_ylabel("Error / error at f = 0.5")
        ax.set_title("Traffic diagnostics" if fam == "traffic" else "Road-network resilience", fontsize=6)
        ax.legend(loc="upper left", fontsize=5)
    C.source("Fig5_ab", pd.DataFrame(rows), "Window error (P90 / tolerance) divided by its value at f = 0.5, per city and metric.")
    # ---- c
    ax = fig.add_subplot(gt[2])
    rows = []
    for fam, keys, i5, cur, fst, mk in (("traffic", C.TRAFFIC, i5t, tc, tf, "o"), ("network", C.NETWORK, i5n, nc, nf, "^")):
        for k in keys:
            for c, y in cur[k].items():
                if c not in fst[k].index or fst[k].loc[c, "flag"] not in ("ok", "above") or not np.isfinite(y[i5]):
                    continue
                fl = fst[k].loc[c, "flag"]
                rows.append(dict(family=fam, metric=k, city=c, C_over_tolerance=float(y[i5]),
                                 f_star=float(fst[k].loc[c, "f_star"]) if fl == "ok" else np.nan, flag=fl))
    d = pd.DataFrame(rows)
    for fam, mk, lab in (("traffic", "o", "traffic diagnostics"), ("network", "^", "network resilience")):
        g = d[(d.family == fam) & (d.flag == "ok")]
        ax.scatter(g.C_over_tolerance, g.f_star, s=7, marker=mk, color=C.INK2, alpha=0.55, lw=0, label=lab)
    for fam, mk in (("traffic", "o"), ("network", "^")):
        g = d[(d.family == fam) & (d.flag == "above")]
        ax.scatter(g.C_over_tolerance, np.full(len(g), TOP), s=7, marker=mk, facecolor="white", edgecolor=C.INK2, lw=0.5, alpha=0.8,
                   label="open: f* above 0.9" if fam == "traffic" else None)
    ax.axhline(0.9, color=C.MUTED, lw=0.4, ls=":")
    cc = np.geomspace(0.2, 30, 200)
    for b, ls in ((0.5, "--"), (0.7, ":")):
        ax.plot(cc, fpred(cc, b), color=C.INK, lw=0.8, ls=ls, label=f"b = {b:g}")
    ax.set_xscale("log"), ax.set_xlim(0.2, 30), ax.set_ylim(0, 1)
    C.log_ticks(ax, "x", (0.3, 1, 3, 10, 30))
    ax.set_xlabel("Error at f = 0.5, C (tolerance units)"), ax.set_ylabel("Critical coverage f*")
    ax.legend(loc="lower right", fontsize=5, title=r"$f^* = 1\,/\,(1 + C^{-1/b})$", title_fontsize=5.5, alignment="left")
    C.source("Fig5_c", d, "C = P90 window error / tolerance at f = 0.5. flag 'above': error not within the tolerance below f = 0.9 "
                          f"(plotted at f* = {TOP}).")
    # ---- d: three steps for sigma_z
    de = pd.read_csv(C.OUT / "e5" / "deff.csv")
    s = step_frame(de, "sd_log", tc, i5t)
    rows = []
    for j, (col, fcol, title) in enumerate(STEPS):
        ax = fig.add_subplot(gd[j])
        e, rr = step_panel(ax, s, col, fcol, C.COL["sd_log"])
        ax.set_title(f"{title}\nmean error {e.mean():.2f}; {100 * (e <= 0.1).mean():.0f}% within ±0.1", fontsize=5.5)
        ax.set_xlabel("Computed f*", fontsize=5.5)
        if j == 0:
            ax.set_ylabel("Observed f*  (" + C.SIG + ")")
        else:
            ax.set_yticklabels([])
        rows += [dict(step=title, **r) for r in rr]
    C.source("Fig5_d", pd.DataFrame(rows), "Step 1: mixture of normal errors with each window's station count (independent stations); "
                                           "step 2: step 1 times sqrt(design effect) from block averaging of the stations' influence values; "
                                           "step 3: f* = 1/(1 + (1/C)^(1/0.7)) with C measured at f = 0.5. flag 'above' = above 0.9 "
                                           f"(drawn at {TOP}, open symbols). Mean error and share within 0.1 over the cities whose observed "
                                           "f* is found; a computed value above 0.9 counts at 0.9.")
    # ---- e: one-point for six metrics
    rows = []
    for j, (fam, keys, i5, cur, fst) in enumerate((("Traffic diagnostics", C.TRAFFIC, i5t, tc, tf), ("Road-network resilience", C.NETWORK, i5n, nc, nf))):
        ax = fig.add_subplot(ge[j])
        ax.fill_between([0, 1], [-0.1, 0.9], [0.1, 1.1], color=C.BAND, alpha=0.6, lw=0)
        ax.plot([0, 1], [0, 1], color=C.INK, lw=0.6)
        ax.axhline(0.9, color=C.MUTED, lw=0.4, ls=":")
        allerr = []
        for k in keys:
            q = fst[k]
            has = [c for c in q.index if c in cur[k] and np.isfinite(cur[k][c][i5]) and cur[k][c][i5] > 0]
            okc = [c for c in has if q.loc[c, "flag"] == "ok"]
            abc = [c for c in has if q.loc[c, "flag"] == "above"]
            fp = np.array([fpred(cur[k][c][i5]) for c in okc])
            fo = q.loc[okc, "f_star"].to_numpy()
            ax.scatter(fp, fo, s=6, color=C.COL[k], lw=0.3, edgecolor="white", zorder=3, label=C.SHORT[k])
            fa = np.array([fpred(cur[k][c][i5]) for c in abc])
            ax.scatter(fa, np.full(fa.size, TOP), s=6, facecolor="white", edgecolor=C.COL[k], lw=0.5, zorder=3)
            allerr += list(np.abs(fp - fo))
            rows += [dict(family=fam, metric=k, city=c, f_computed=float(a), f_observed=float(b), observed_flag="ok")
                     for c, a, b in zip(okc, fp, fo)]
            rows += [dict(family=fam, metric=k, city=c, f_computed=float(a), f_observed=np.nan, observed_flag="above")
                     for c, a in zip(abc, fa)]
        allerr = np.array(allerr)
        ax.set_title(f"{fam}\nmean error {allerr.mean():.2f}; {100 * (allerr <= 0.1).mean():.0f}% within ±0.1", fontsize=5.5)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect("equal"), ax.set_xlabel("Computed f* (step 3)", fontsize=5.5)
        unit_ticks(ax)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Observed f*")
        ax.legend(loc="lower right", fontsize=4.5, handletextpad=0.2, borderaxespad=0.1)
    C.source("Fig5_e", pd.DataFrame(rows), "Step 3 for the six main metrics. observed_flag 'above' = error not within the tolerance below "
                                           f"f = 0.9 (drawn at {TOP}, open symbols). Mean error and share within 0.1 over the cities whose "
                                           "observed f* is found.")
    C.save(fig, "Fig5")


if __name__ == "__main__":
    main()
