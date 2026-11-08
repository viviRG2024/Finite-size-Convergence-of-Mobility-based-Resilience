"""Fig. 1a, right panel: window estimates of sigma_z against coverage in Zurich, with the whole-city value, the tolerance
band and the distribution of the window estimates at 10%, 30% and 60% coverage (insets).

    python -m critical_window.figures.fig1a_panel      # after figures.make_all (reads Source Data Fig1_c, Fig1_d)

The schematic parts of Fig. 1 are drawn by hand; this script draws the data panel that is placed in it.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from scipy.stats import gaussian_kde

from . import common as C

SD = C.OUT / "figures" / "source_data"
DST = C.OUT / "figures"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.9,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
c = pd.read_csv(SD / "Fig1_c.csv"); d = pd.read_csv(SD / "Fig1_d.csv")
c = c[c.f >= 0.1]
whole, tol = c.whole_city.iloc[0], c.tolerance.iloc[0]
fstar = float(d[(d.metric == "sd_log") & d.f_star.notna()].f_star.iloc[0])
err = d[(d.metric == "sd_log") & d.f.notna()].set_index("f").p90_over_tolerance

NAVY, RED, BAND, DOT = "#1d3557", "#e5383b", "#e9edf2", "#6c8fb3"
KDE_FILL, KDE_LINE = "#b7dfc0", "#3c9a5f"
PICK = [(0.1, "①", "10%"), (0.3, "②", "30%"), (0.6, "③", "60%")]

fig = plt.figure(figsize=(4.0, 3.4))
# rounded frame, matching the schematic of Fig. 1
fig.patches.append(FancyBboxPatch((0.015, 0.015), 0.97, 0.90, boxstyle="round,pad=0,rounding_size=0.035",
                                  transform=fig.transFigure, fill=False, lw=1.6, ec="#16242f"))
fig.text(0.5, 0.955, "Defining sufficient coverage", ha="center", va="center", fontsize=11.5, fontweight="bold")

ax = fig.add_axes([0.12, 0.135, 0.83, 0.44])
ins = [fig.add_axes([0.075 + i * 0.315, 0.66, 0.255, 0.14]) for i in range(3)]

# main panel: window estimates against coverage
ax.axhspan(whole - tol, whole + tol, color=BAND, lw=0, zorder=0)
ax.axhline(whole, color=NAVY, lw=1.1, zorder=1)
ax.scatter(c.f, c.window_value, s=5, color=DOT, alpha=0.55, lw=0, zorder=2)
ax.axvline(fstar, color=NAVY, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.text(fstar + 0.012, 0.585, "f*", color=NAVY, fontsize=9, fontstyle="italic", va="bottom")
ax.text(0.95, whole + tol + 0.012, "whole city ± tolerance", ha="right", va="bottom", fontsize=6.8, color=NAVY)
for f, num, _ in PICK:
    v = c[c.f == f].window_value
    ax.scatter(np.full(v.size, f), v, s=7, color=RED, alpha=0.8, lw=0, zorder=3)
ax.set_xlim(0.05, 0.95); ax.set_ylim(0.58, 1.08)
ax.set_xticks([0.1, 0.3, 0.6, 0.9]); ax.set_xticklabels(["\u2460 10%", "\u2461 30%", "\u2462 60%", "90%"], fontsize=7)
ax.set_yticks([])
ax.set_xlabel("Observed share of the city (coverage)", fontsize=8, color=NAVY, labelpad=2)
ax.set_ylabel("Estimate in the window", fontsize=8, color=NAVY, labelpad=4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(NAVY)
ax.tick_params(colors=NAVY, length=2.5, width=0.8)
for t in ax.get_xticklabels()[:3]:
    t.set_color(RED)
ax.plot(1, 0, ">", transform=ax.transAxes, color=NAVY, ms=4, clip_on=False)
ax.plot(0, 1, "^", transform=ax.transAxes, color=NAVY, ms=4, clip_on=False)

# insets: distribution of the window estimates at the three coverages
grid = np.linspace(0.6, 1.08, 300)
for a, (f, num, lab) in zip(ins, PICK):
    v = c[c.f == f].window_value.to_numpy()
    kd = gaussian_kde(v); kd.set_bandwidth(0.014 / v.std(ddof=1)); k = kd(grid); k = k / k.max()
    a.axvspan(whole - tol, whole + tol, color=BAND, lw=0)
    a.fill_between(grid, k, color=KDE_FILL, lw=0)
    a.plot(grid, k, color=KDE_LINE, lw=1.0)
    a.axvline(whole, color=NAVY, lw=0.9)
    a.set_xlim(0.6, 1.08); a.set_ylim(0, 1.15)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color("#16242f"); s.set_linewidth(0.9)
    e = err.loc[f]
    a.text(0.0, 1.27, f"{num}  {lab} coverage", transform=a.transAxes, ha="left", va="bottom", fontsize=7.3, color="#16242f")
    a.text(0.0, 1.04, f"error = {e:.1f} \u00d7 tolerance", transform=a.transAxes, ha="left", va="bottom", fontsize=6.3,
           color=RED if e > 1 else KDE_LINE)
    hi = v.max() + 0.012
    con = ConnectionPatch(xyA=(f, hi), coordsA=ax.transData, xyB=(0.5, 0.0), coordsB=a.transAxes,
                          color=RED, lw=0.9, ls=(0, (2.5, 1.5)))
    fig.add_artist(con)

fig.savefig(DST / "Fig1a_right.png", dpi=400)
fig.savefig(DST / "Fig1a_right.pdf")
fig.savefig(DST / "Fig1a_right.svg")
print("f*", round(fstar, 3), "errors", {f: round(err.loc[f], 2) for f, _, _ in PICK})
