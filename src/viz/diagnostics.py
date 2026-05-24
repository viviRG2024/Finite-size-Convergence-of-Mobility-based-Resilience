"""Process-side diagnostic plots for the pipeline.

These are *off by default* and run only when ``main.py --viz`` is passed.
They are written so a reviewer can see, per (city, date, window):

* whether the window bbox sits where it should within the city's segments;
* what the active-cell grid looks like at two representative resolutions;
* whether the local power-law fits for ``CV(L)`` and ``H(L)`` are clean
  inside the ``[fit_min_l, fit_max_l]`` window.

And, per city, a single 24-hour x window-index heatmap of the fitted
exponents -- the converged regime (SF >= 0.4) is the bottom four rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ..analysis.multi_scale import (
    _assign_segments,
    _cell_flow_matrix,
    _make_grid,
)
from ..analysis.scaling_fit import fit_local_power_law

REF_HOUR = 10
REF_L_FINE = 200
REF_L_COARSE = 1000
CV_COLOR = "#2E6FB5"
H_COLOR = "#D7352B"


def _log_norm(values: np.ndarray) -> mcolors.Normalize | None:
    v = values[values > 0]
    if v.size == 0:
        return None
    return mcolors.LogNorm(vmin=max(float(v.min()), 1e-6), vmax=float(v.max()))


def _segment_scatter(ax, segments: list, bbox: tuple) -> None:
    xs = np.array([s["centroid_x"] for s in segments])
    ys = np.array([s["centroid_y"] for s in segments])
    daily = np.array([np.nansum(s["flows"]) for s in segments])
    sc = ax.scatter(xs, ys, s=2, c=daily, cmap="viridis",
                    norm=_log_norm(daily))
    ax.add_patch(Rectangle(
        (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
        fill=False, ec="red", lw=1.5,
    ))
    ax.set_aspect("equal")
    ax.set_title(f"segments (n = {len(segments)}) + window bbox", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, shrink=0.7, label="daily flow")


def _cell_flow_heatmap(ax, segments: list, bbox: tuple, L_m: int,
                      hour: int) -> int:
    grid = _make_grid(bbox, L_m)
    cells = _assign_segments(segments, grid)
    flows = _cell_flow_matrix(segments, cells, grid)
    nx, ny = grid["n_cells_x"], grid["n_cells_y"]
    img = flows[:, hour].reshape(ny, nx)
    masked = np.ma.masked_where(img <= 0, img)
    im = ax.imshow(
        masked, origin="lower",
        extent=(bbox[0], bbox[2], bbox[1], bbox[3]),
        cmap="viridis", aspect="equal", interpolation="nearest",
    )
    n_active = int((flows[:, hour] > 0).sum())
    ax.set_title(f"active cells at L = {L_m} m  (n = {n_active})", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.7, label=f"hour-{hour} cell flow")
    return n_active


def _loglog_fit_panel(ax, grid_sizes: Sequence[float], values_by_scale,
                     hour: int, fit_min_l: float, fit_max_l: float,
                     max_valid_l: float, label: str, color: str) -> dict:
    L = np.asarray(grid_sizes, dtype=float)
    y = np.asarray(values_by_scale)[:, hour]
    upper = min(fit_max_l, max_valid_l)
    valid = (L > 0) & (y > 0) & np.isfinite(y)
    ax.loglog(L[valid], y[valid], "o", ms=5, color=color,
              mfc="white", mec=color, lw=0)
    ax.axvspan(fit_min_l, upper, color=color, alpha=0.10)
    fit = fit_local_power_law(
        grid_sizes, y, fit_min_l=fit_min_l, fit_max_l=fit_max_l,
        max_valid_l=max_valid_l, min_points=4,
    )
    if fit["ok"]:
        L_line = np.array([fit_min_l, upper])
        y_line = np.exp(fit["intercept"]) * L_line ** fit["exponent"]
        ax.loglog(L_line, y_line, "-", color=color, lw=2)
        ax.set_title(
            f"{label}(L) ~ L^{{{fit['exponent']:+.3f}}}   "
            f"R^2 = {fit['r_squared']:.2f}",
            fontsize=9,
        )
    else:
        ax.set_title(f"{label}(L): fit failed (n = {fit['n_points']})",
                     fontsize=9)
    ax.set_xlabel("L (m)", fontsize=8)
    ax.set_ylabel(label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    return fit


def plot_window_diagnostics(
    *,
    segments: list,
    bbox: tuple,
    ms: dict,
    city: str,
    date_str: str,
    window_idx: int,
    fit_min_l: float,
    fit_max_l: float,
    max_valid_l: float,
    out_path: Path,
) -> None:
    """Six-panel diagnostic PNG for one (city, date, window)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"{city} {date_str}  window {window_idx}  hour {REF_HOUR}",
        fontsize=11, fontweight="bold",
    )

    _segment_scatter(axes[0, 0], segments, bbox)
    _cell_flow_heatmap(axes[0, 1], segments, bbox, REF_L_FINE, REF_HOUR)
    _cell_flow_heatmap(axes[0, 2], segments, bbox, REF_L_COARSE, REF_HOUR)

    fit_cv = _loglog_fit_panel(
        axes[1, 0], ms["grid_sizes"], ms["cv_by_scale"], REF_HOUR,
        fit_min_l, fit_max_l, max_valid_l, "CV", CV_COLOR,
    )
    fit_h = _loglog_fit_panel(
        axes[1, 1], ms["grid_sizes"], ms["entropy_by_scale"], REF_HOUR,
        fit_min_l, fit_max_l, max_valid_l, "H", H_COLOR,
    )

    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        f"city          : {city}",
        f"date          : {date_str}",
        f"window_idx    : {window_idx}",
        f"bbox dim (m)  : {bbox[2] - bbox[0]:.0f} x {bbox[3] - bbox[1]:.0f}",
        f"n_segments    : {len(segments)}",
        f"fit window (m): [{fit_min_l:.0f}, {min(fit_max_l, max_valid_l):.0f}]",
        "",
        f"hour {REF_HOUR} alpha = {fit_cv['exponent']:+.3f}"
        f"  (R^2 = {fit_cv['r_squared']:.2f})",
        f"hour {REF_HOUR}  beta = {fit_h['exponent']:+.3f}"
        f"  (R^2 = {fit_h['r_squared']:.2f})",
    ]
    ax.text(0.0, 0.95, "\n".join(lines), family="monospace", fontsize=9,
            va="top", ha="left", transform=ax.transAxes)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_city_convergence_heatmap(
    *,
    city: str,
    n_windows: int,
    dates: Sequence[str],
    hours: Sequence[int],
    fits_by_window_date: dict,
    out_path: Path,
) -> None:
    """24h x window heatmap of the median alpha and beta for one city.

    ``fits_by_window_date`` is keyed by ``(window_idx, date_str)`` and holds
    the per-hour fit dict written by ``stage_fit``.
    """
    alpha = np.full((n_windows, len(hours)), np.nan)
    beta = np.full((n_windows, len(hours)), np.nan)
    for w_idx in range(n_windows):
        for h_pos, h in enumerate(hours):
            a_vals: list = []
            b_vals: list = []
            for date_str in dates:
                fits = fits_by_window_date.get((w_idx, date_str))
                if not fits:
                    continue
                ab = fits["alpha_by_hour"]
                bb = fits["beta_by_hour"]
                a = ab.get(str(h)) or ab.get(h)
                b = bb.get(str(h)) or bb.get(h)
                if a and a.get("ok"):
                    a_vals.append(a["exponent"])
                if b and b.get("ok"):
                    b_vals.append(b["exponent"])
            if a_vals:
                alpha[w_idx, h_pos] = float(np.median(a_vals))
            if b_vals:
                beta[w_idx, h_pos] = float(np.median(b_vals))

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.7 * n_windows + 2.5))
    fig.suptitle(
        f"{city}: median exponents per (window_idx, hour)",
        fontsize=11, fontweight="bold",
    )

    a_vmax = float(np.nanmax(np.abs(alpha))) if np.isfinite(alpha).any() else 1.0
    im_a = axes[0].imshow(alpha, aspect="auto", origin="lower",
                          cmap="coolwarm", vmin=-a_vmax, vmax=a_vmax)
    axes[0].set_title(r"$\alpha$ (CV slope)", fontsize=10)
    plt.colorbar(im_a, ax=axes[0], shrink=0.85)

    im_b = axes[1].imshow(beta, aspect="auto", origin="lower",
                          cmap="viridis")
    axes[1].set_title(r"$\beta$ (H slope)", fontsize=10)
    plt.colorbar(im_b, ax=axes[1], shrink=0.85)

    for ax in axes:
        ax.set_xlabel("hour")
        ax.set_ylabel("window_idx")
        ax.set_xticks(range(len(hours)))
        ax.set_xticklabels([str(h) for h in hours], fontsize=8)
        ax.set_yticks(range(n_windows))
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
