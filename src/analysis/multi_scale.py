"""Multi-scale CV(L) and structural entropy H(L).

Pipeline per spatial scale ``L``:

1. Tile the bounding box with a regular grid of cell side ``L``.
2. Aggregate per-segment hourly flows into per-cell hourly flows.
3. ``CV(L, h)`` = std/mean across active cells (flow > 0) at hour ``h``.
4. Build the cell-cell adjacency on active cells (4- or 8-connectivity)
   with edge weight ``min``, ``mean`` or ``geometric mean`` of endpoint flows.
5. ``H_struct(L, h)`` = degree-based Shannon entropy of the cell network at
   hour ``h``.

These match the implementation used in the published 34-city analysis.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Gridding
# ---------------------------------------------------------------------------

def _make_grid(bbox: tuple, grid_size_m: int) -> dict:
    """Build a regular grid covering ``bbox`` with cells of side ``grid_size_m``."""
    min_x, min_y, max_x, max_y = bbox
    n_cells_x = int(np.ceil((max_x - min_x) / grid_size_m))
    n_cells_y = int(np.ceil((max_y - min_y) / grid_size_m))
    return {
        "grid_size_m": grid_size_m,
        "n_cells_x": max(n_cells_x, 1),
        "n_cells_y": max(n_cells_y, 1),
        "min_x": min_x,
        "min_y": min_y,
    }


def _assign_segments(segments: list, grid: dict) -> dict:
    """Map each cell_id to a list of segment indices."""
    nx, ny = grid["n_cells_x"], grid["n_cells_y"]
    s = grid["grid_size_m"]
    cells: dict = defaultdict(list)
    for i, seg in enumerate(segments):
        gx = int((seg["centroid_x"] - grid["min_x"]) / s)
        gy = int((seg["centroid_y"] - grid["min_y"]) / s)
        gx = max(0, min(gx, nx - 1))
        gy = max(0, min(gy, ny - 1))
        cells[gy * nx + gx].append(i)
    return cells


def _cell_flow_matrix(segments: list, cells: dict, grid: dict) -> np.ndarray:
    """Return a (n_total_cells, 24) matrix of per-cell hourly flows."""
    n_total = grid["n_cells_x"] * grid["n_cells_y"]
    mat = np.zeros((n_total, 24))
    for cid, idxs in cells.items():
        for i in idxs:
            mat[cid] += np.nan_to_num(segments[i]["flows"], nan=0.0)
    return mat


# ---------------------------------------------------------------------------
# CV and entropy
# ---------------------------------------------------------------------------

def compute_cv_per_hour(flow_matrix: np.ndarray) -> np.ndarray:
    """``CV(h) = std/mean`` over active cells (flow > 0) for each hour."""
    out = np.zeros(flow_matrix.shape[1])
    for h in range(flow_matrix.shape[1]):
        active = flow_matrix[:, h]
        active = active[active > 0]
        if active.size > 1:
            mu = active.mean()
            out[h] = active.std(ddof=1) / mu if mu > 0 else 0.0
    return out


def _build_adjacency(flow_matrix: np.ndarray, grid: dict, connectivity: str,
                     edge_method: str) -> dict:
    """For each hour, build {(cell_i, cell_j): weight} on active cells."""
    nx, ny = grid["n_cells_x"], grid["n_cells_y"]
    if connectivity == "4":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]

    n_hours = flow_matrix.shape[1]
    adjacency = {}
    for h in range(n_hours):
        flows = flow_matrix[:, h]
        active = np.where(flows > 0)[0]
        edges: dict = {}
        for cid in active:
            gx, gy = cid % nx, cid // nx
            f_i = flows[cid]
            for dx, dy in offsets:
                nx2, ny2 = gx + dx, gy + dy
                if not (0 <= nx2 < nx and 0 <= ny2 < ny):
                    continue
                nid = ny2 * nx + nx2
                f_j = flows[nid]
                if f_j <= 0:
                    continue
                if edge_method == "min":
                    w = min(f_i, f_j)
                elif edge_method == "mean":
                    w = 0.5 * (f_i + f_j)
                else:
                    w = float(np.sqrt(f_i * f_j))
                key = (cid, nid) if cid < nid else (nid, cid)
                edges.setdefault(key, w)
        adjacency[h] = edges
    return adjacency


def compute_structural_entropy_per_hour(adjacency: dict) -> np.ndarray:
    """Degree-based Shannon entropy ``-sum p_i log p_i`` of the cell network."""
    out = np.zeros(len(adjacency))
    for h, edges in adjacency.items():
        if not edges:
            continue
        deg: dict = defaultdict(float)
        for (i, j), w in edges.items():
            deg[i] += w
            deg[j] += w
        vol = sum(deg.values())
        if vol <= 0:
            continue
        out[h] = float(-sum((d / vol) * np.log(d / vol)
                            for d in deg.values() if d > 0))
    return out


def _flow_distribution_entropy(flow_matrix: np.ndarray) -> np.ndarray:
    """Fallback flow-distribution entropy (used when ``entropy_type='flow'``)."""
    out = np.zeros(flow_matrix.shape[1])
    for h in range(flow_matrix.shape[1]):
        flows = flow_matrix[:, h]
        flows = flows[flows > 0]
        s = flows.sum()
        if s <= 0:
            continue
        p = flows / s
        out[h] = float(-np.sum(p * np.log(p)))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_single_scale(
    segments: list,
    bbox: tuple,
    grid_size_m: int,
    connectivity: str = "8",
    edge_weight_method: str = "geometric",
    entropy_type: str = "structural",
) -> dict:
    """Run the CV/H computation at one spatial scale."""
    grid = _make_grid(bbox, grid_size_m)
    cells = _assign_segments(segments, grid)
    flow_matrix = _cell_flow_matrix(segments, cells, grid)

    cv = compute_cv_per_hour(flow_matrix)
    if entropy_type == "structural":
        adj = _build_adjacency(flow_matrix, grid, connectivity, edge_weight_method)
        H = compute_structural_entropy_per_hour(adj)
    else:
        H = _flow_distribution_entropy(flow_matrix)

    valid = np.isfinite(cv) & np.isfinite(H)
    if valid.sum() >= 3:
        rho, p = pearsonr(cv[valid], H[valid])
    else:
        rho, p = 0.0, 1.0

    return {
        "grid_size_m": grid_size_m,
        "cv_per_hour": cv.tolist(),
        "entropy_per_hour": H.tolist(),
        "correlation": float(rho),
        "p_value": float(p),
        "n_total_cells": grid["n_cells_x"] * grid["n_cells_y"],
        "n_active_cells_per_hour": [
            int((flow_matrix[:, h] > 0).sum()) for h in range(24)
        ],
    }


def analyse_multi_scale(
    segments: list,
    bbox: tuple,
    grid_sizes: Sequence[int],
    connectivity: str = "8",
    edge_weight_method: str = "geometric",
    entropy_type: str = "structural",
    progress: bool = True,
) -> dict:
    """Run :func:`analyse_single_scale` over a sequence of scales.

    Returns a dict with arrays indexed by scale, ready to be JSON-dumped.
    """
    cv_by_scale: list = []
    h_by_scale: list = []
    correlations: list = []
    p_values: list = []
    per_scale: list = []

    for L in grid_sizes:
        if progress:
            print(f"  L = {L:>5} m", flush=True)
        res = analyse_single_scale(
            segments, bbox, L,
            connectivity=connectivity,
            edge_weight_method=edge_weight_method,
            entropy_type=entropy_type,
        )
        cv_by_scale.append(res["cv_per_hour"])
        h_by_scale.append(res["entropy_per_hour"])
        correlations.append(res["correlation"])
        p_values.append(res["p_value"])
        per_scale.append(res)

    rho_abs = np.abs(correlations)
    opt_idx = int(np.argmax(rho_abs)) if rho_abs.size else 0

    return {
        "grid_sizes": list(grid_sizes),
        "cv_by_scale": cv_by_scale,
        "entropy_by_scale": h_by_scale,
        "correlations": correlations,
        "p_values": p_values,
        "scale_results": per_scale,
        "optimal_scale_idx": opt_idx,
        "optimal_scale_m": int(grid_sizes[opt_idx]),
        "optimal_correlation": float(correlations[opt_idx]) if correlations else 0.0,
    }
