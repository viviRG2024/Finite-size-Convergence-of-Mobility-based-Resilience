"""Shared style, colours and data access for the paper figures (Nature Communications layout).

Every panel writes its plotted values to out/figures/source_data/<Fig>_<panel>.csv through `source()`; make_all.py
compiles them into Source_Data.xlsx (one sheet per panel).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from ..core.criteria import fstar_stats  # noqa: F401  (median f* with cities above the last level)

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "out"
FIG = OUT / "figures"
SRC = FIG / "source_data"
X0 = OUT / "x0"
IUTFD = PKG.parent / "data" / "input" / "IUTFD"
MAIN = "main34"

MM = 1 / 25.4
W2, W1 = 183 * MM, 89 * MM                     # double / single column width (inches)

# ink and surface
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
BAND = "#d9d8d2"                                # tolerance band / reference wash
# entity colours, fixed across all figures (validated: traffic trio and network trio all-pairs; network trio with blue)
COL = {"sd_log": "#2a78d6", "cv": "#eb6834", "hn": "#1baf7a",
       "R_random": "#4a3aa7", "R_btw": "#e34948", "eff_loss": "#eda100",
       "mean_log": "#e87ba4", "site_cv": "#008300", "dw_eff_loss": "#898781", "acc_loss": "#52514e"}
SIG = r"$\sigma_z$"
LAB = {"sd_log": SIG, "cv": "CV", "hn": "H", "mean_log": "Mean of log flows", "site_cv": "Site CV",
       "R_random": "Random-failure robustness", "R_btw": "Targeted-attack robustness", "eff_loss": "Efficiency loss",
       "dw_eff_loss": "Flow-weighted efficiency loss", "acc_loss": "Accessibility loss (2 km)"}
SHORT = {"sd_log": SIG, "cv": "CV", "hn": "H", "R_random": "Random failure", "R_btw": "Targeted attack",
         "eff_loss": "Efficiency loss", "mean_log": "Mean of log flows", "site_cv": "Site CV",
         "dw_eff_loss": "Flow-weighted eff. loss", "acc_loss": "Accessibility loss"}
TRAFFIC = ("sd_log", "cv", "hn")
NETWORK = ("R_random", "R_btw", "eff_loss")
CONTINENT = {"losangeles": "North America", "toronto": "North America", "taipeh": "Asia", "melbourne": "Oceania"}
DISPLAY = {"losangeles": "Los Angeles", "taipeh": "Taipei", "constance": "Constance", "zurich": "Zurich",
           "torino": "Turin", "bolton": "Bolton", "groningen": "Groningen", "innsbruck": "Innsbruck"}


def name(c: str) -> str:
    return DISPLAY.get(c, c.capitalize())


def style():
    for f in ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 6, "axes.titlesize": 6.5,
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold", "mathtext.sf": "Liberation Sans",
        "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
        "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.4,
        "lines.linewidth": 1.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "lines.markersize": 3,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
        "axes.titlelocation": "left", "axes.titlepad": 3})


def panel(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, name_):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name_}.pdf", dpi=600)
    fig.savefig(FIG / f"{name_}.png", dpi=450)
    plt.close(fig)
    print(FIG / f"{name_}.pdf")


def source(fig_panel: str, df: pd.DataFrame, note: str = ""):
    """Write the plotted values of one panel (Source Data)."""
    SRC.mkdir(parents=True, exist_ok=True)
    df.to_csv(SRC / f"{fig_panel}.csv", index=False)
    if note:
        (SRC / f"{fig_panel}.txt").write_text(note, encoding="utf-8")


def log_ticks(ax, axis="x", ticks=(0.01, 0.03, 0.1, 0.3, 1), fmt=None):
    labs = [fmt(t) if fmt else (f"{t:g}") for t in ticks]
    if axis == "x":
        ax.set_xticks(ticks), ax.set_xticklabels(labs)
    else:
        ax.set_yticks(ticks), ax.set_yticklabels(labs)
    ax.minorticks_off()


# ------------------------------------------------------------------------------------------------ data access
def cities():
    return json.load(open(OUT / MAIN / "summary.json", encoding="utf-8"))["cities"]


def levels():
    return np.array(json.load(open(OUT / MAIN / cities()[0] / "city.json"))["levels"], float)


def traffic_curves():
    """{metric: {city: P90/eps over the 18 levels}}, {metric: eps}, {metric: DataFrame(city, f_star, flag)}."""
    cw = pd.read_csv(OUT / MAIN / "critical_windows.csv")
    e3 = pd.read_csv(OUT / "e3" / "e3_cities.csv")
    e3s = pd.read_csv(OUT / "e3" / "e3_summary.csv")
    cur, eps, fst = {}, {}, {}
    for p in ("sd_log", "mean_log", "site_cv"):
        q = cw[cw.probe == p].set_index("city")
        eps[p] = float(q.eps.iloc[0])
        cur[p] = {c: np.load(OUT / MAIN / c / "city.npz")[f"{p}__p90_obs"] / eps[p] for c in q.index}
        fst[p] = q[["f_star", "flag"]]
    for p in ("cv", "hn"):
        q = e3[(e3.variant == "sum_k32") & (e3.probe == p)].set_index("city")
        eps[p] = float(e3s[(e3s.variant == "sum_k32") & (e3s.probe == p)].eps.iloc[0])
        k = 0 if p == "cv" else 1
        cur[p] = {c: np.array(json.load(open(OUT / "e3" / "cities" / f"{c}.json"))["sum_k32|p90"][k], float) / eps[p] for c in q.index}
        fst[p] = q[["f_star", "flag"]]
    return cur, eps, fst


def network_curves(kind="e4b"):
    lv = pd.read_csv(OUT / kind / "levels.csv")
    cr = pd.read_csv(OUT / kind / "critical.csv")
    eps = json.load(open(OUT / kind / "summary.json"))["eps"]
    cur, fst = {}, {}
    for k in ("R_random", "R_btw", "eff_loss", "dw_eff_loss", "acc_loss"):
        g = lv[lv.metric == k]
        cur[k] = {c: gg.sort_values("f").p90_over_eps.to_numpy() for c, gg in g.groupby("city")}
        fst[k] = cr[cr.metric == k].set_index("city")[["f_star", "flag"]]
    return cur, eps, fst, np.array(sorted(lv.f.unique()))


def fmt_median(st: dict) -> str:
    if np.isfinite(st["median"]):
        return f"median {st['median']:.2f}"
    return f"median > {st['median_lower_bound']:.2f}" if st["median_lower_bound"] < 0.9 else "median > 0.9"


def median_band(curves: dict, min_n: int = 17):
    M = np.array(list(curves.values()), float)
    n = np.isfinite(M).sum(0)
    ok = n >= min_n
    with np.errstate(all="ignore"):
        med = np.where(ok, np.nanmedian(M, 0), np.nan)
        q1 = np.where(ok, np.nanquantile(M, 0.25, 0), np.nan)
        q3 = np.where(ok, np.nanquantile(M, 0.75, 0), np.nan)
    return med, q1, q3, n
