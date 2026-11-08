"""Fig. 2: a, London windows of increasing coverage drawn from their coordinates; b, two windows of the same coverage at
different positions (cell flows at three cell sizes, adjacency weights, station flows); c-f, results from the Source Data
written by figures/fig2.py.

    python -m critical_window.figures.fig2_panels      # after figures.make_all

Panels a and b draw the road network of London from its IUTF road file (data/input/IUTFD/london/roads/); without the
file they show the stations only.
"""
import zlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.colors import LogNorm
from . import fig1 as F1
from . import common as C
from ..core import windows
from ..tools import e3_grid as G
OUT = C.OUT / "figures"
SD = C.OUT / "figures" / "source_data"
_FS = pd.read_csv(SD / "Fig2_c.csv")
LONDON_FSTAR = float(_FS[(_FS.metric == "sd_log") & (_FS.city == "london")].f_star.iloc[0])
plt.rcParams.update({"font.family": "sans-serif", "font.size": 6, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"sd_log": "#2a78d6", "cv": "#eb6834", "hn": "#1baf7a"}
LAB = {"sd_log": "σz", "cv": "CV", "hn": "H"}
_CW = pd.read_csv(C.OUT / C.MAIN / "critical_windows.csv")
_E3 = pd.read_csv(C.OUT / "e3" / "e3_summary.csv")
EPS = {"sd_log": float(_CW[_CW.probe == "sd_log"].eps.iloc[0]),            # tolerances (0.25 x IQR across cities)
       "cv": float(_E3[(_E3.variant == "sum_k32") & (_E3.probe == "cv")].eps.iloc[0]),
       "hn": float(_E3[(_E3.variant == "sum_k32") & (_E3.probe == "hn")].eps.iloc[0])}
RED, INK, MUT = "#e5383b", "#1b1b1b", "#8a8a8a"
NAME = {"london": "London", "losangeles": "Los Angeles", "madrid": "Madrid", "melbourne": "Melbourne", "hamburg": "Hamburg", "zurich": "Zurich"}

cd, lines = F1.city_geometry("london")
e = np.array(cd.extent, float); x, y, F = cd.x, cd.y, cd.F
mx = (lines[:, 0, 0] + lines[:, 1, 0]) / 2; my = (lines[:, 0, 1] + lines[:, 1, 1]) / 2
fig = plt.figure(figsize=(7.2, 9.4))
def letter(xf, yf, s):
    fig.text(xf, yf, s, fontsize=9, fontweight="bold", va="top", ha="left")

# ---------------- a: windows of increasing coverage
p = np.array([np.median(x), np.median(y)])
FS = [0.02, 0.05, 0.1, 0.3, 0.6, 1.0]
def nest(f):
    s = np.sqrt(f); return np.array([p[0] + s * (e[0] - p[0]), p[1] + s * (e[1] - p[1]), p[0] + s * (e[2] - p[0]), p[1] + s * (e[3] - p[1])])
W = [nest(f) for f in FS]
rows_a = [dict(panel=i + 1, coverage=f, x0=w[0], y0=w[1], x1=w[2], y1=w[3], width_km=(w[2] - w[0]) / 1000,
               n_stations=int(((x >= w[0]) & (x <= w[2]) & (y >= w[1]) & (y <= w[3])).sum())) for i, (f, w) in enumerate(zip(FS, W))]
axa = [fig.add_axes([0.035 + i * 0.161, 0.855, 0.147, 0.095 * 7.2 / 9.4 * 1.0 / 0.62 * 0.62 * 1.62 / 1.0 * 0.62]) for i in range(6)]
asp = (e[3] - e[1]) / (e[2] - e[0])
for i, ax in enumerate(axa):
    ax.set_position([0.035 + i * 0.161, 0.86, 0.147, 0.147 * asp * 7.2 / 9.4])
for i, (ax, w, f) in enumerate(zip(axa, W, FS)):
    k = (mx >= w[0] - 200) & (mx <= w[2] + 200) & (my >= w[1] - 200) & (my <= w[3] + 200)
    lw = 0.45 if f <= 0.05 else 0.28 if f <= 0.1 else 0.16 if f <= 0.3 else 0.1
    ax.add_collection(LineCollection(lines[k], colors=INK, linewidths=lw, rasterized=True))
    m = (x >= w[0]) & (x <= w[2]) & (y >= w[1]) & (y <= w[3])
    ax.scatter(x[m], y[m], s=1.0 if f <= 0.05 else 0.5 if f <= 0.1 else 0.15, color=COL["sd_log"], alpha=0.85, lw=0, zorder=3, rasterized=True)
    ax.set_xlim(w[0], w[2]); ax.set_ylim(w[1], w[3]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(INK); s.set_linewidth(0.7)
    if i:
        q = W[i - 1]
        ax.add_patch(Rectangle((q[0], q[1]), q[2] - q[0], q[3] - q[1], fill=False, ec=RED, lw=1.1, zorder=4))
        for xa, ya, yb in ((q[0], q[3], 1), (q[0], q[1], 0)):
            fig.add_artist(ConnectionPatch(xyA=(1, yb), coordsA=axa[i - 1].transAxes, xyB=(xa, ya), coordsB=ax.transData, color=RED, lw=0.6, ls=(0, (2, 1.5))))
    ax.text(0, 1.05, "❶❷❸❹❺❻"[i] + f" f = {f:g}", transform=ax.transAxes, fontsize=6.5, fontweight="bold", va="bottom")
    ax.text(0, -0.05, f"{int(m.sum()):,} stations, {(w[2] - w[0]) / 1000:.1f} km", transform=ax.transAxes, fontsize=5, va="top", color="#444")
fig.text(0.035 + 2.5 * 0.161 + 0.147 / 2 + 0.0055, 0.988, f"London f* = {LONDON_FSTAR:.2f}", fontsize=5.5, color=RED, ha="center", va="top")
fig.add_artist(plt.Line2D([0.035 + 3 * 0.161 - 0.007] * 2, [0.85, 0.975], transform=fig.transFigure, color=RED, lw=0.7, ls=(0, (1, 1))))
letter(0.005, 0.99, "a")

# ---------------- b: same coverage, different positions
def vals(m, bbox):
    cv, hn = G.variant_values(x[m], y[m], F[m][:, None], bbox, "sum_k32")
    return np.std(np.log(F[m]), ddof=1), float(cv[0]), float(hn[0])
whole = vals(np.ones(F.size, bool), e)
rng = np.random.default_rng(zlib.crc32("|".join(map(str, ("london", "main", "windows", 0.1))).encode()))
WR = windows.random_windows(e, 0.1, 50, rng)
INS = [(x >= w[0]) & (x <= w[2]) & (y >= w[1]) & (y <= w[3]) for w in WR]
dev = np.array([(vals(m, w)[0] - whole[0]) / EPS["sd_log"] if m.sum() >= 20 else np.nan for m, w in zip(INS, WR)])
o = np.argsort(dev); n = np.isfinite(dev).sum(); pick = [o[int(0.1 * n)], o[int(0.9 * n)]]
def grid(m, w, L):
    nx = max(int(np.ceil((w[2] - w[0]) / L)), 1); ny = max(int(np.ceil((w[3] - w[1]) / L)), 1)
    gx = np.clip(((x[m] - w[0]) / L).astype(int), 0, nx - 1); gy = np.clip(((y[m] - w[1]) / L).astype(int), 0, ny - 1)
    g = np.zeros(nx * ny); np.add.at(g, gy * nx + gx, F[m]); return g.reshape(ny, nx)
def adjacency(g):
    ny, nx = g.shape; act = [(r, c) for r in range(ny) for c in range(nx) if g[r, c] > 0]; idx = {a: i for i, a in enumerate(act)}
    A = np.full((len(act), len(act)), np.nan)
    for (r, c), i in idx.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                j = idx.get((r + dr, c + dc))
                if (dr or dc) and j is not None:
                    A[i, j] = np.sqrt(g[r, c] * g[r + dr, c + dc])
    return A
yb0, yb1 = 0.555, 0.815
axm = fig.add_axes([0.035, 0.60, 0.2, 0.2 * asp * 7.2 / 9.4])
axm.add_collection(LineCollection(lines, colors="#c4c3bc", linewidths=0.08, rasterized=True))
axm.scatter(x, y, s=0.1, color=COL["sd_log"], lw=0, rasterized=True)
for k, i in enumerate(pick):
    w = WR[i]; axm.add_patch(Rectangle((w[0], w[1]), w[2] - w[0], w[3] - w[1], fill=False, ec=RED, lw=1.0))
    axm.text(w[0] + 200, w[3] - 200, "①②"[k], color=RED, fontsize=7, fontweight="bold", va="top")
axm.set_xlim(e[0], e[2]); axm.set_ylim(e[1], e[3]); axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([])
for s in axm.spines.values():
    s.set_visible(True); s.set_linewidth(0.6)
axm.set_title("London: two windows, each 10%", fontsize=6, pad=3)
axm.text(0, -0.06, "whole city: σz {:.3f}, CV {:.3f}, H {:.3f}".format(*whole), transform=axm.transAxes, fontsize=5, va="top", color="#444")
SIZES = [250, 500, 1000]
gs = {L: [grid(INS[i], WR[i], L) for i in pick] for L in SIZES}
norms = {L: LogNorm(vmin=max(min(np.min(g[g > 0]) for g in gs[L]), 1), vmax=max(np.max(g) for g in gs[L])) for L in SIZES}
bins = np.linspace(np.log10(F.min()), np.log10(F.max()), 28)
rows_b = []
for k, i in enumerate(pick):
    m, w = INS[i], WR[i]; v = vals(m, w); yr = 0.695 - k * 0.125
    for j, L in enumerate(SIZES):
        a = fig.add_axes([0.27 + j * 0.105, yr, 0.095, 0.095 * asp * 7.2 / 9.4 * 1.0])
        a.imshow(np.where(gs[L][k] > 0, gs[L][k], np.nan), origin="lower", cmap="viridis", norm=norms[L], interpolation="nearest", aspect="auto")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True); s.set_linewidth(0.5)
        if k == 0:
            a.set_title(f"{L if L < 1000 else '1 km'}{' m' if L < 1000 else ''} cells", fontsize=5.6, pad=2)
        if j == 0:
            a.text(-0.16, 0.5, "①②"[k], transform=a.transAxes, color=RED, fontsize=8, fontweight="bold", va="center")
    a = fig.add_axes([0.595, yr - 0.004, 0.075, 0.075 * 7.2 / 9.4 * 1.25])
    A = adjacency(gs[500][k]); a.imshow(A, cmap="Blues", norm=LogNorm(), interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(True); s.set_linewidth(0.5)
    if k == 0:
        a.set_title("adjacency (500 m)", fontsize=5.6, pad=2)
    a = fig.add_axes([0.70, yr, 0.11, 0.075])
    a.hist(np.log10(F), bins=bins, density=True, histtype="step", color=MUT, lw=0.7)
    a.hist(np.log10(F[m]), bins=bins, density=True, color="#a9d0f5", ec=COL["sd_log"], lw=0.25)
    a.set_yticks([]); a.spines["left"].set_visible(False); a.tick_params(labelsize=5)
    if k == 0:
        a.set_title("station flows", fontsize=5.6, pad=2); a.set_xticklabels([])
    else:
        a.set_xlabel("log₁₀ mean daily flow", fontsize=5, labelpad=1)
    rows = [(LAB[q], (vv - wv) / EPS[q], q) for q, vv, wv in zip(("sd_log", "cv", "hn"), v, whole)]
    rows_b += [dict(window="①②"[k], x0=w[0], y0=w[1], x1=w[2], y1=w[3], n_stations=int(m.sum()), metric=q, window_value=vv,
                    whole_city=wv, tolerance=EPS[q], deviation_over_tolerance=(vv - wv) / EPS[q])
               for q, vv, wv in zip(("sd_log", "cv", "hn"), v, whole)]
    fig.text(0.83, yr + 0.072, f"{int(m.sum())} stations", fontsize=5.5, va="top")
    for r, (lab, d, q) in enumerate(rows):
        fig.text(0.83, yr + 0.055 - r * 0.016, f"{lab}", fontsize=5.8, color=COL[q], va="top")
        fig.text(0.865, yr + 0.055 - r * 0.016, f"{d:+.1f} × tol.", fontsize=5.8, va="top", color=RED if abs(d) > 1 else "#2e8b57")
letter(0.005, 0.83, "b")

# ---------------- c: six cities
a2 = pd.read_csv(SD / "Fig2_a.csv"); fs = pd.read_csv(SD / "Fig2_c.csv")
cities = list(dict.fromkeys(a2.city))
for i, c in enumerate(cities):
    ax = fig.add_axes([0.075 + (i % 3) * 0.17, 0.43 - (i // 3) * 0.125, 0.145, 0.09])
    d = a2[a2.city == c]
    ax.axhspan(-1, 1, color="#e8e8e4", lw=0); ax.axhline(0, color=INK, lw=0.6)
    ax.scatter(d.f, d.deviation_over_tolerance, s=0.8, color=COL["sd_log"], alpha=0.6, lw=0, rasterized=True)
    r = fs[(fs.metric == "sd_log") & (fs.city == c)].iloc[0]
    if r.flag == "ok":
        ax.axvline(r.f_star, color=INK, lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_xlim(0.009, 1); ax.set_ylim(-8, 8)
    ax.set_xticks([0.01, 0.1, 1]); ax.set_xticklabels(["0.01", "0.1", "1"] if i // 3 else [])
    ax.set_yticks([-5, 0, 5]); ax.tick_params(labelsize=5)
    if i % 3: ax.set_yticklabels([])
    ax.set_title(f"{NAME[c]} (f* {'> 0.9' if r.flag != 'ok' else f'{r.f_star:.2f}'})", fontsize=5.6, pad=2)
fig.text(0.03, 0.42, "Window − city (tolerance units)", rotation=90, fontsize=5.8, va="center")
fig.text(0.075 + 1.5 * 0.17 - 0.012, 0.272, "Coverage f", fontsize=5.8, ha="center")
letter(0.005, 0.545, "c")

# ---------------- d: window error, median over cities
b2 = pd.read_csv(SD / "Fig2_b.csv")
ax = fig.add_axes([0.66, 0.31, 0.31, 0.215])
for q in ("sd_log", "cv", "hn"):
    d = b2[(b2.metric == q) & (b2.n_cities >= 17)].dropna()
    ax.fill_between(d.f, d.q25, d.q75, color=COL[q], alpha=0.13, lw=0); ax.plot(d.f, d["median"], color=COL[q], lw=1.1, label=LAB[q])
ax.axhline(1, color=INK, lw=0.6, ls="--"); ax.text(0.155, 1.06, "tolerance", fontsize=5, color="#444")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([0.15, 0.3, 0.5, 0.9]); ax.set_xticklabels(["0.15", "0.3", "0.5", "0.9"])
ax.set_yticks([0.3, 1, 3]); ax.set_yticklabels(["0.3", "1", "3"]); ax.minorticks_off()
ax.set_xlabel("Coverage f", fontsize=5.8); ax.set_ylabel("Window error / tolerance", fontsize=5.8); ax.tick_params(labelsize=5)
ax.legend(frameon=False, fontsize=5.5, loc="lower left", handlelength=1.5)
ax.set_title("median and interquartile range over cities", fontsize=5.6, pad=3)
letter(0.61, 0.545, "d")

# ---------------- e: f* per city
ax = fig.add_axes([0.11, 0.05, 0.42, 0.17])
rng2 = np.random.default_rng(3)
for r, q in enumerate(("sd_log", "cv", "hn")):
    d = fs[(fs.metric == q) & (fs.flag != "none")]; yv = 2 - r
    ok = d[d.flag == "ok"]; ab = d[d.flag == "above"]
    ax.scatter(ok.f_star, yv + rng2.uniform(-0.18, 0.18, len(ok)), s=5, color=COL[q], lw=0)
    ax.scatter(np.full(len(ab), 0.965), yv + np.linspace(-0.2, 0.2, max(len(ab), 1))[:len(ab)], s=6, marker=">", color=COL[q], alpha=0.6, lw=0)
    vals_ = np.r_[ok.f_star.to_numpy(), np.full(len(ab), np.inf)]; med = np.median(vals_)
    ax.plot([med, med], [yv - 0.3, yv + 0.3], color=INK, lw=1.4)
    ax.text(-0.02, yv, f"{LAB[q]}\nmedian {med:.2f}\n{len(ab)} of {len(d)} > 0.9", fontsize=5, ha="right", va="center", transform=ax.get_yaxis_transform())
ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlim(0, 1); ax.set_ylim(-0.6, 2.6)
ax.set_xlabel("Critical coverage f*", fontsize=5.8); ax.tick_params(labelsize=5)
letter(0.005, 0.245, "e")

# ---------------- f: rank correlation
dd = pd.read_csv(SD / "Fig2_d.csv")
ax = fig.add_axes([0.66, 0.05, 0.31, 0.17])
for q in ("sd_log", "cv", "hn"):
    d = dd[(dd.metric == q) & dd.f.notna()]; s = dd[(dd.metric == q) & dd.f.isna()]
    ax.plot(d.f, d.rho_median, "-o", color=COL[q], lw=1.0, ms=2.2, label=LAB[q])
    ax.plot(0.97, s.rho_median.iloc[0], "s", color=COL[q], ms=3)
ax.text(0.97, 0.69, "single\nweekday", fontsize=5, ha="center", color="#444")
ax.set_xlim(0.35, 1.0); ax.set_ylim(0.6, 1.0); ax.set_xticks([0.4, 0.6, 0.8]); ax.tick_params(labelsize=5)
ax.set_xlabel("Coverage f", fontsize=5.8); ax.set_ylabel("Rank correlation with city", fontsize=5.8)
ax.legend(frameon=False, fontsize=5.5, loc="lower right", handlelength=1.5, bbox_to_anchor=(0.88, 0.0))
letter(0.61, 0.245, "f")
fig.savefig(OUT / "Fig2_panels.png", dpi=350); fig.savefig(OUT / "Fig2_panels.pdf")
C.source("Fig2_London_windows", pd.DataFrame(rows_a), "Panel a: windows of London covering 2%-100% of the study area, sharing one "
         "anchor point (the median station coordinates) for illustration; UTM coordinates (m) and number of stations in each window.")
C.source("Fig2_London_two_windows", pd.DataFrame(rows_b), "Panel b: two random windows of the main analysis covering 10% of London "
         "(those at the 10th and 90th percentile of the deviation of sigma_z); window and whole-city values of sigma_z, and of CV and H "
         "on 500 m cells averaged over 32 grid origins; deviation in units of the tolerance (0.25 x IQR of the whole-city values).")
from .make_all import compile_source_data
compile_source_data()
print("done", np.round(whole, 3), dev[pick].round(2))
