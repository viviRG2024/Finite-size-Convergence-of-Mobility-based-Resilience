"""E4b: the five road-network metrics of E4 with the Monte-Carlo noise of the sampled computations made small.

E4 (and X11) estimated betweenness from 500 sampled sources, global efficiency from 100 sampled sources, the random-
failure curve from 20 removal orders and accessibility from 100 sampled origins. Repeating these with other seeds moves
the values by 0.2-0.6 x eps (out/e4/noise_summary.csv), which is of the order of the window errors themselves. Here:
  betweenness   exact for graphs with <= 15000 nodes, else 2000 sampled sources (hop distance, as before)
  R_random      mean over 200 random removal orders (union-find, compiled with numba when available)
  R_btw         removal in decreasing betweenness order (as before)
  eff_loss      1 - E(after) / E(before), global efficiency over the remaining nodes, length-weighted; exact for
                graphs with <= 2000 nodes, else 2000 sampled sources (the same sources before and after)
  dw_eff_loss   demand-weighted efficiency loss between sensor nodes (stations snapped to the nearest node within
                200 m, weight = station flow), all sensor nodes as sources (exact)
  acc_loss      1 - A(after) / A(before), A = mean number of nodes within 2 km network distance, all nodes (exact)
Removal set for the last three: the 10% of nodes with the highest betweenness. Graph, windows and levels as in E4.
"""
from __future__ import annotations

import zlib

import igraph as ig
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from ..core import windows
from . import network, network_e4
from .network_e4 import ACC_RADIUS, ALL_LEVELS, SNAP_MAX, wmatrix

BTW_EXACT_MAX, N_SRC_BTW = 15000, 2000
EFF_EXACT_MAX, N_SRC_EFF = 2000, 2000
N_ORDERS = 200
BATCH = 500

try:
    from numba import njit
except Exception:                                   # numba missing: same algorithm in pure Python
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


@njit(cache=True)
def _lcc_mean(n, indptr, indices, order):
    """Mean over q of S(q), the largest-component share after removing order[:q+1] (as network._lcc_curve)."""
    parent = np.arange(n)
    size = np.ones(n, np.int64)
    present = np.zeros(n, np.bool_)
    best = 0
    total = 0.0
    for q in range(n - 1, -1, -1):
        total += best
        u = order[q]
        present[u] = True
        ru = u
        while parent[ru] != ru:
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            if present[v]:
                rv = v
                while parent[rv] != rv:
                    parent[rv] = parent[parent[rv]]
                    rv = parent[rv]
                if rv != ru:
                    if size[ru] < size[rv]:
                        ru, rv = rv, ru
                    parent[rv] = ru
                    size[ru] += size[rv]
        if size[ru] > best:
            best = size[ru]
    return total / n / n


def _csr(n, E):
    A = coo_matrix((np.ones(2 * E.shape[0]), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    return A.indptr.astype(np.int64), A.indices.astype(np.int64)


def _inv_dist_sum(W, srcs, tgt_mask):
    """Sum over sources and alive targets of 1/d (d finite, > 0), in batches."""
    tot = 0.0
    for i in range(0, srcs.size, BATCH):
        D = dijkstra(W, directed=False, indices=srcs[i:i + BATCH])[:, tgt_mask]
        with np.errstate(divide="ignore"):
            tot += float(np.where(np.isfinite(D) & (D > 0), 1.0 / D, 0.0).sum())
    return tot


def _reach_counts(W, srcs, tgt_mask, radius):
    out = np.zeros(srcs.size)
    for i in range(0, srcs.size, BATCH):
        D = dijkstra(W, directed=False, indices=srcs[i:i + BATCH], limit=radius)
        out[i:i + BATCH] = (np.isfinite(D) & tgt_mask[None, :]).sum(1)
    return out


def metrics(n, E, L, xy, sx, sy, sF, rng):
    out = dict(n_nodes=n, n_edges=int(E.shape[0]))
    g = ig.Graph(n=n, edges=E.tolist())
    if n <= BTW_EXACT_MAX:
        btw = np.array(g.betweenness(directed=False))
    else:
        btw = np.array(g.betweenness(sources=rng.choice(n, N_SRC_BTW, replace=False).tolist(), directed=False))
    order_b = np.argsort(-btw, kind="stable").astype(np.int64)
    indptr, indices = _csr(n, E)
    out["R_btw"] = float(_lcc_mean(n, indptr, indices, order_b))
    out["R_random"] = float(np.mean([_lcc_mean(n, indptr, indices, rng.permutation(n).astype(np.int64)) for _ in range(N_ORDERS)]))
    rem = order_b[: int(round(0.1 * n))]
    alive = np.ones(n, bool)
    alive[rem] = False
    W = wmatrix(n, E, L)
    W2 = W.tolil()
    W2[rem, :] = 0
    W2[:, rem] = 0
    W2 = W2.tocsr()
    # global efficiency over the remaining nodes (X11 definition), exact or with common sampled sources
    srcs = np.arange(n) if n <= EFF_EXACT_MAX else np.sort(rng.choice(n, N_SRC_EFF, replace=False))
    e0 = _inv_dist_sum(W, srcs, np.ones(n, bool)) / (srcs.size * (n - 1))
    s1 = srcs[alive[srcs]]
    e1 = _inv_dist_sum(W2, s1, alive) / (s1.size * (alive.sum() - 1)) if s1.size else np.nan
    out["eff_loss"] = float(1 - e1 / e0) if e0 > 0 else np.nan
    # demand-weighted efficiency between sensor nodes, all sensor nodes as sources
    out["dw_eff_loss"], out["n_sensor_nodes"] = np.nan, 0
    if sx.size:
        d, j = cKDTree(xy).query(np.column_stack([sx, sy]))
        ok = d <= SNAP_MAX
        w = np.bincount(j[ok], weights=sF[ok], minlength=n)
        sn = np.flatnonzero(w > 0)
        out["n_sensor_nodes"] = int(sn.size)
        if sn.size >= 5:
            def ew(Wm, al):
                s_ok = sn[al[sn]]
                if s_ok.size < 2:
                    return 0.0
                tot = 0.0
                for i in range(0, s_ok.size, BATCH):
                    D = dijkstra(Wm, directed=False, indices=s_ok[i:i + BATCH])[:, s_ok]
                    ww = w[s_ok[i:i + BATCH]][:, None] * w[s_ok][None, :]
                    with np.errstate(divide="ignore"):
                        tot += float((ww * np.where(np.isfinite(D) & (D > 0), 1.0 / D, 0.0)).sum())
                return tot
            num0 = ew(W, np.ones(n, bool))
            if num0 > 0:
                out["dw_eff_loss"] = float(1 - ew(W2, alive) / num0)
    # local accessibility, all nodes
    a0 = _reach_counts(W, np.arange(n), np.ones(n, bool), ACC_RADIUS)
    a1 = np.zeros(n)
    if alive.any():
        a1[alive] = _reach_counts(W2, np.flatnonzero(alive), alive, ACC_RADIUS)
    out["acc_loss"] = float(1 - a1.sum() / a0.sum()) if a0.sum() > 0 else np.nan
    return out


def city_e4b(city: str, seg, cd, n_pos=network.N_POS):
    rows = []
    n, E, L, xy = network_e4._window_graph(seg, cd.extent)
    r = dict(f=1.0, pos=0, n_stations=int(cd.F.size))
    r.update(metrics(n, E, L, xy, cd.x, cd.y, cd.F, _rng(city, "e4b", "E")))
    rows.append(r)
    for f in ALL_LEVELS:
        wins = windows.random_windows(cd.extent, f, 50, _rng(city, "main", "windows", f))[:n_pos]
        for k, w in enumerate(wins):
            n, E, L, xy = network_e4._window_graph(seg, w)
            r = dict(f=f, pos=k, n_nodes=n)
            if n < network.MIN_NODES:
                rows.append(r)
                continue
            inb = windows.in_box(cd.x, cd.y, w)
            r["n_stations"] = int(inb.sum())
            r.update(metrics(n, E, L, xy, cd.x[inb], cd.y[inb], cd.F[inb], _rng(city, "e4b", f, k)))
            rows.append(r)
    return rows
