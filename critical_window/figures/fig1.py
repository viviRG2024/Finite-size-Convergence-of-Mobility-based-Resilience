"""Fig. 1 | Study design: 34 sensor networks, the window design and how the critical coverage is read.

a  sensor stations (dots) and road network (lines) of the 34 cities, each inside its observation extent
b  Zurich: the observation extent with three random windows covering 10%, 30% and 60% of its area
c  Zurich: sigma_z of every usable random window against coverage; whole-city value and tolerance band
d  Zurich: the window error (90% of windows within) relative to the tolerance; f* where it falls below 1 for good
"""
from __future__ import annotations

import json
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from ..core import windows
from ..main.data import load_city
from ..common.cfg import load_config
from ..resil import network_e4
from . import common as C

EXAMPLE = "zurich"


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def city_geometry(c):
    m = load_config()["main"]
    cd = load_city(c, C.X0, m)
    inv = pd.read_csv(C.X0 / "network_inventory.csv").set_index("city")
    qc = json.load(open(C.X0 / "per_city" / c / "qc.json", encoding="utf-8"))
    road_file = C.IUTFD / c / inv.loc[c, "road_file"].replace("\\", "/")
    if not road_file.exists():          # the maps need the raw IUTF road files; without them they show the stations only
        print(f"{c}: road file not found ({road_file}); map drawn without roads")
        return cd, np.zeros((0, 2, 2))
    seg = network_e4.load_segments_xy(road_file, inv.loc[c, "road_source"], int(qc["epsg"]))
    a, b, ln, mx, my, vx, vy = seg
    e = cd.extent
    k = (mx >= e[0]) & (mx <= e[2]) & (my >= e[1]) & (my <= e[3])
    lines = np.stack([np.column_stack([vx[a[k]], vy[a[k]]]), np.column_stack([vx[b[k]], vy[b[k]]])], 1)
    return cd, lines


def order_cities():
    cs = C.cities()
    eu = sorted([c for c in cs if c not in C.CONTINENT], key=C.name)
    other = [c for c in ("losangeles", "toronto", "taipeh", "melbourne") if c in cs]
    return eu + other


def main():
    C.style()
    fig = plt.figure(figsize=(C.W2, 150 * C.MM))
    gl = fig.add_gridspec(6, 6, left=0.01, right=0.565, top=0.955, bottom=0.045, wspace=0.10, hspace=0.55)
    gr = fig.add_gridspec(3, 1, left=0.66, right=0.985, top=0.965, bottom=0.06, height_ratios=[1.45, 1, 1], hspace=0.42)
    # ---- a: 34 layouts
    rows = []
    for i, c in enumerate(order_cities()):
        ax = fig.add_subplot(gl[i // 6, i % 6])
        cd, lines = city_geometry(c)
        e = cd.extent
        ax.add_collection(LineCollection(lines, colors="#c9c8c1", linewidths=0.12, zorder=1, rasterized=True))
        ax.scatter(cd.x, cd.y, s=0.35, color=C.COL["sd_log"], lw=0, zorder=2, rasterized=True)
        ax.add_patch(Rectangle((e[0], e[1]), e[2] - e[0], e[3] - e[1], fill=False, lw=0.35, ec=C.INK2, zorder=3))
        pad = 0.02 * max(e[2] - e[0], e[3] - e[1])
        ax.set_xlim(e[0] - pad, e[2] + pad), ax.set_ylim(e[1] - pad, e[3] + pad)
        ax.set_aspect("equal"), ax.axis("off")
        cont = C.CONTINENT.get(c, "Europe")
        cell = gl[i // 6, i % 6].get_position(fig)
        xc = (cell.x0 + cell.x1) / 2
        fig.text(xc, cell.y1 + 0.004, C.name(c), fontsize=5, ha="center", va="bottom", color=C.INK)
        sub = (f"{cd.F.size:,} stations\n{(e[2] - e[0]) / 1000:.1f} × {(e[3] - e[1]) / 1000:.1f} km" +
               ("" if cont == "Europe" else f"\n{cont}"))
        fig.text(xc, cell.y0 - 0.004, sub, fontsize=4, ha="center", va="top", color=C.INK2, linespacing=1.15)
        if i == 0:
            fig.text(0.005, 0.985, "a", fontsize=8, fontweight="bold", va="top", ha="left", color=C.INK)
        rows.append(dict(city=c, continent=cont, n_stations=int(cd.F.size), study_area_w_km=(e[2] - e[0]) / 1000, study_area_h_km=(e[3] - e[1]) / 1000,
                         area_km2=cd.area_km2, road_segments_in_study_area=int(lines.shape[0])))
    C.source("Fig1_a", pd.DataFrame(rows), "Station positions and road segments per city are in the IUTF data; this table lists the counts shown.")
    # ---- b: example city with windows
    cd, lines = city_geometry(EXAMPLE)
    e = cd.extent
    ax = fig.add_subplot(gr[0])
    ax.add_collection(LineCollection(lines, colors="#d4d3cc", linewidths=0.18, zorder=1, rasterized=True))
    ax.scatter(cd.x, cd.y, s=1.2, color=C.COL["sd_log"], lw=0, zorder=2)
    ax.add_patch(Rectangle((e[0], e[1]), e[2] - e[0], e[3] - e[1], fill=False, lw=0.6, ec=C.INK, zorder=3))
    wrows = []
    for f, ls in ((0.1, "-"), (0.3, "--"), (0.6, ":")):
        w = windows.random_windows(e, f, 50, _rng(EXAMPLE, "main", "windows", f))[0]
        ax.add_patch(Rectangle((w[0], w[1]), w[2] - w[0], w[3] - w[1], fill=False, lw=0.9, ec=C.INK, ls=ls, zorder=4))
        ax.text(w[0] + 150, w[3] - 150, f"f = {f:g}", fontsize=5.5, va="top", ha="left", color=C.INK, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
        wrows.append(dict(f=f, x0=w[0], y0=w[1], x1=w[2], y1=w[3], n_stations=int(windows.in_box(cd.x, cd.y, w).sum())))
    ax.set_xlim(e[0] - 300, e[2] + 300), ax.set_ylim(e[1] - 300, e[3] + 300), ax.set_aspect("equal"), ax.axis("off")
    ax.plot([e[0], e[0] + 2000], [e[1] - 200, e[1] - 200], color=C.INK, lw=1)
    ax.text(e[0] + 1000, e[1] - 320, "2 km", fontsize=5, ha="center", va="top")
    ax.set_title(f"Zurich ({cd.F.size} stations)", fontsize=6, loc="center")
    C.panel(ax, "b", x=-0.02, y=1.02)
    C.source("Fig1_b", pd.DataFrame(wrows + [dict(f=1.0, x0=e[0], y0=e[1], x1=e[2], y1=e[3], n_stations=int(cd.F.size))]),
             "Window corners in UTM metres (EPSG 32632); f = window area / study area.")
    # ---- c: window values of sigma_z
    r = json.load(open(C.OUT / "e8" / "cities" / f"{EXAMPLE}.json"))
    z = np.load(C.OUT / "e8" / "cities" / f"{EXAMPLE}.npz")
    lv = np.array(r["levels"])
    thE = r["theta_E"]["sd_log"]
    cw = pd.read_csv(C.OUT / C.MAIN / "critical_windows.csv")
    q = cw[(cw.probe == "sd_log") & (cw.city == EXAMPLE)].iloc[0]
    eps = float(q.eps)
    ax = fig.add_subplot(gr[1])
    ax.axhspan(thE - eps, thE + eps, color=C.BAND, alpha=0.6, lw=0, zorder=0)
    ax.axhline(thE, color=C.INK, lw=0.7, zorder=1)
    vrows = []
    jit = _rng("fig1c")
    for i, f in enumerate(lv):
        v = z[f"sd_log_{i}"]
        v = v[np.isfinite(v)]
        if v.size < 25:
            continue
        xs = f * np.exp(jit.uniform(-0.06, 0.06, v.size))
        ax.scatter(xs, v, s=1.0, color=C.COL["sd_log"], alpha=0.45, lw=0, zorder=2)
        vrows += [dict(f=float(f), window_value=float(x)) for x in v]
    ax.set_xscale("log"), C.log_ticks(ax, "x", (0.03, 0.1, 0.3, 0.9))
    ax.set_xlim(0.025, 1.0)
    ax.set_xlabel("Coverage f"), ax.set_ylabel(C.SIG + " in the window")
    ax.text(0.98, 0.04, "whole city ± tolerance", transform=ax.transAxes, ha="right", va="bottom", fontsize=5, color=C.INK2)
    C.panel(ax, "c", x=-0.22, y=1.04)
    C.source("Fig1_c", pd.DataFrame(vrows).assign(whole_city=thE, tolerance=eps), "Usable random windows (>= 20 stations) of the main run.")
    # ---- d: P90 / eps and f*
    ax = fig.add_subplot(gr[2])
    tc, teps, tf = C.traffic_curves()
    nc, neps, nf, nlv = C.network_curves()
    drows = []
    for k, fs_, cur, fst in (("sd_log", lv, tc["sd_log"], tf["sd_log"]), ("R_random", nlv, nc["R_random"], nf["R_random"]),
                             ("R_btw", nlv, nc["R_btw"], nf["R_btw"])):
        y = cur[EXAMPLE]
        ok = np.isfinite(y)
        ax.plot(fs_[ok], y[ok], "-o", color=C.COL[k], ms=2.2, lw=0.9, label=C.SHORT[k], mec="white", mew=0.3)
        fz = float(fst.loc[EXAMPLE, "f_star"])
        ax.axvline(fz, ymax=0.12, color=C.COL[k], lw=1.2)
        drows += [dict(metric=k, f=float(f), p90_over_tolerance=float(v)) for f, v in zip(fs_[ok], y[ok])]
        drows.append(dict(metric=k, f=np.nan, p90_over_tolerance=np.nan, f_star=fz, flag=fst.loc[EXAMPLE, "flag"]))
    ax.axhline(1, color=C.INK, lw=0.6, ls="--")
    ax.set_xscale("log"), ax.set_yscale("log")
    C.log_ticks(ax, "x", (0.03, 0.1, 0.3, 0.9)), C.log_ticks(ax, "y", (0.3, 1, 3, 10))
    ax.set_xlim(0.025, 1.0)
    ax.set_xlabel("Coverage f"), ax.set_ylabel("Window error / tolerance")
    ax.legend(loc="upper right", fontsize=5)
    ax.text(0.03, 0.06, "ticks: critical coverage f*", transform=ax.transAxes, fontsize=5, color=C.INK2)
    C.panel(ax, "d", x=-0.22, y=1.04)
    C.source("Fig1_d", pd.DataFrame(drows), "Window error = 90th percentile of |window value - whole-city value| over the usable windows; "
                                             "tolerance = 0.25 x IQR of the whole-city values across the 34 cities.")
    C.save(fig, "Fig1")


if __name__ == "__main__":
    main()
