"""E4: road-network metrics on finer coverage levels, plus two flow-aware metrics.

Graph as in network.py (vertex graph of the road lines, degree-2 vertices contracted, edges weighted by length), with
node coordinates kept. Per graph, after removing the 10% of nodes with the highest initial betweenness (the same
removal set as eff_loss):
  dw_eff_loss : 1 - E_w(after) / E_w(before), E_w = sum_ij w_i w_j / d_ij / sum_ij w_i w_j over sensor nodes
                (stations snapped to their nearest node within 200 m; w = station flow, summed per node); up to 150
                sensor nodes as sources; demand-weighted efficiency, i.e. the network efficiency that matters for the
                observed traffic.
  acc_loss    : 1 - A(after) / A(before), A = mean number of nodes within 2 km network distance of 100 sampled
                nodes (local accessibility).
Coverage levels: the 8 X11 levels plus 0.15, 0.25, 0.6, 0.8 (positions = the first 20 random windows of the main run).
R_random, R_btw and eff_loss are computed here only on the 4 new levels; the 8 X11 levels are reused.
"""
from __future__ import annotations

import zlib

import igraph as ig
import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from ..common.geo import utm_forward, wkb_parts
from ..core import windows
from ..x0.x05_inventory import DRIVE_CLASSES
from . import network

NEW_LEVELS = (0.15, 0.25, 0.60, 0.80)
ALL_LEVELS = tuple(sorted(set(network.LEVELS) | set(NEW_LEVELS)))
SNAP_MAX = 200.0
ACC_RADIUS = 2000.0


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def load_segments_xy(path, source: str, epsg: int, snap: float = 1.0):
    """Vertex-graph edges (a, b, length, mid x, mid y) and vertex coordinates (vx, vy) in UTM metres."""
    names = pq.ParquetFile(path).schema_arrow.names
    tab = pq.read_table(path, columns=["geometry"] + (["highway"] if "highway" in names else [])).to_pandas()
    if source == "raw_osm" and "highway" in tab:
        tab = tab[tab["highway"].astype(str).isin(DRIVE_CLASSES)]
    lines = []
    for g in tab["geometry"]:
        if g is None:
            continue
        try:
            _, parts = wkb_parts(g)
        except Exception:
            continue
        lines.extend(pt for pt in parts if len(pt) >= 2)
    lon = np.concatenate([ln[:, 0] for ln in lines])
    lat = np.concatenate([ln[:, 1] for ln in lines])
    zone, south = int(epsg) % 100, int(epsg) // 100 == 327
    x, y = utm_forward(lon, lat, zone, south)
    key = np.round(x / snap).astype(np.int64) * 4_000_000_000 + np.round(y / snap).astype(np.int64)
    _, first_idx, node = np.unique(key, return_index=True, return_inverse=True)
    vx, vy = x[first_idx], y[first_idx]
    sizes = np.array([len(ln) for ln in lines])
    first = np.r_[0, np.cumsum(sizes)[:-1]]
    ok = np.ones(x.size, bool)
    ok[first + sizes - 1] = False
    ia = np.flatnonzero(ok)
    a, b = node[ia], node[ia + 1]
    ln = np.hypot(x[ia + 1] - x[ia], y[ia + 1] - y[ia])
    mx, my = (x[ia] + x[ia + 1]) / 2, (y[ia] + y[ia + 1]) / 2
    keep = a != b
    return a[keep], b[keep], ln[keep], mx[keep], my[keep], vx, vy


def contracted_ids(a, b, ln):
    """network.contracted, step for step, plus the original vertex id of every kept node (row i of the result = node i)."""
    nodes, inv = np.unique(np.r_[a, b], return_inverse=True)
    a, b = inv[:a.size], inv[a.size:]
    n = nodes.size
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    two = deg == 2
    inner = two[a] & two[b]
    lab = np.arange(n)
    if inner.any():
        g = coo_matrix((np.ones(inner.sum()), (a[inner], b[inner])), shape=(n, n))
        _, cl = connected_components(g, directed=False)
        lab = np.where(two, n + cl, np.arange(n))
    la, lb = lab[a], lab[b]
    junction_edges = ~two[a] & ~two[b]
    E = [np.column_stack([a[junction_edges], b[junction_edges]])]
    L = [ln[junction_edges]]
    touch = two[a] | two[b]
    cid = np.where(two[a], la, lb)[touch]
    other = np.where(two[a] & ~two[b], b, np.where(two[b] & ~two[a], a, -1))[touch]
    clen = ln[touch]
    order = np.argsort(cid, kind="stable")
    cid, other, clen = cid[order], other[order], clen[order]
    bounds = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        ends = other[s:e]
        ends = ends[ends >= 0]
        if ends.size == 2 and ends[0] != ends[1]:
            E.append(ends[None, :])
            L.append(np.array([clen[s:e].sum()]))
    E = np.vstack(E) if E else np.zeros((0, 2), int)
    L = np.concatenate(L) if L else np.zeros(0)
    keep_nodes = np.unique(E.ravel())
    remap = -np.ones(n, int)
    remap[keep_nodes] = np.arange(keep_nodes.size)
    return keep_nodes.size, remap[E], L, nodes[keep_nodes]


def contracted_xy(a, b, ln, vx, vy):
    n, E, L, orig = contracted_ids(a, b, ln)
    return n, E, L, np.column_stack([vx[orig], vy[orig]])


def wmatrix(n, E, L):
    """Symmetric length matrix; parallel edges keep the shortest length (a sparse sum would add them up)."""
    if E.shape[0] == 0:
        return coo_matrix((n, n)).tocsr()
    u, v = np.minimum(E[:, 0], E[:, 1]), np.maximum(E[:, 0], E[:, 1])
    key = u.astype(np.int64) * n + v
    order = np.lexsort((L, key))
    key, u, v, Ls = key[order], u[order], v[order], L[order]
    first = np.r_[True, key[1:] != key[:-1]]
    u, v, Ls = u[first], v[first], Ls[first]
    return coo_matrix((np.r_[Ls, Ls], (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()


def _window_graph(seg, box):
    a, b, ln, mx, my, vx, vy = seg
    m = (mx >= box[0]) & (mx <= box[2]) & (my >= box[1]) & (my <= box[3])
    if m.sum() == 0:
        return 0, np.zeros((0, 2), int), np.zeros(0), np.zeros((0, 2))
    return contracted_xy(a[m], b[m], ln[m], vx, vy)


def extra_metrics(n, E, L, xy, sx, sy, sF, rng, n_src_btw=500, n_src_dw=150, n_src_acc=100):
    """dw_eff_loss, acc_loss after removing the top 10% betweenness nodes."""
    out = dict(dw_eff_loss=np.nan, acc_loss=np.nan, n_sensor_nodes=0)
    if n < 2 or E.shape[0] == 0:
        return out
    g = ig.Graph(n=n, edges=E.tolist())
    src = rng.choice(n, min(n, n_src_btw), replace=False).tolist()
    btw = np.array(g.betweenness(sources=src, directed=False))
    rem = np.argsort(-btw, kind="stable")[: int(round(0.1 * n))]
    W = wmatrix(n, E, L)
    W2 = W.tolil()
    W2[rem, :] = 0
    W2[:, rem] = 0
    W2 = W2.tocsr()
    alive = np.ones(n, bool)
    alive[rem] = False
    # demand-weighted efficiency over sensor nodes
    d, j = cKDTree(xy).query(np.column_stack([sx, sy]))
    ok = d <= SNAP_MAX
    if ok.sum() >= 2:
        w = np.bincount(j[ok], weights=sF[ok], minlength=n)
        sn = np.flatnonzero(w > 0)
        out["n_sensor_nodes"] = int(sn.size)
        if sn.size >= 5:
            srcs = sn if sn.size <= n_src_dw else rng.choice(sn, n_src_dw, replace=False)

            def ew(Wm, al):
                s_ok = srcs[al[srcs]]
                t_ok = sn[al[sn]]
                if s_ok.size == 0 or t_ok.size < 2:
                    return 0.0, 1.0
                D = dijkstra(Wm, directed=False, indices=s_ok)[:, t_ok]
                ww = w[s_ok][:, None] * w[t_ok][None, :]
                with np.errstate(divide="ignore"):
                    inv = np.where(np.isfinite(D) & (D > 0), 1.0 / D, 0.0)
                return float((ww * inv).sum()), float(ww[D > 0].sum() if np.any(D > 0) else 1.0)

            num0, den0 = ew(W, np.ones(n, bool))
            num1, _ = ew(W2, alive)                      # removed sensor nodes count as lost demand
            if num0 > 0:
                out["dw_eff_loss"] = float(1 - num1 / num0)
    # local accessibility: nodes within ACC_RADIUS of sampled nodes
    s_acc = rng.choice(n, min(n, n_src_acc), replace=False)
    D0 = dijkstra(W, directed=False, indices=s_acc, limit=ACC_RADIUS)
    a0 = np.isfinite(D0).sum(1).astype(float)
    s_alive = alive[s_acc]
    D1 = dijkstra(W2, directed=False, indices=s_acc, limit=ACC_RADIUS)
    a1 = np.where(s_alive, (np.isfinite(D1) & alive[None, :]).sum(1), 0).astype(float)
    if a0.sum() > 0:
        out["acc_loss"] = float(1 - a1.sum() / a0.sum())
    return out


def city_e4(city: str, seg, cd, n_pos=network.N_POS, n_random=20):
    rows = []
    n, E, L, xy = _window_graph(seg, cd.extent)
    r = dict(f=1.0, pos=0, n_nodes=n, n_stations=int(cd.F.size))
    r.update(extra_metrics(n, E, L, xy, cd.x, cd.y, cd.F, _rng(city, "e4", "E")))
    rows.append(r)
    for f in ALL_LEVELS:
        wins = windows.random_windows(cd.extent, f, 50, _rng(city, "main", "windows", f))[:n_pos]
        for k, w in enumerate(wins):
            n, E, L, xy = _window_graph(seg, w)
            r = dict(f=f, pos=k, n_nodes=n)
            if n < network.MIN_NODES:
                rows.append(r)
                continue
            inb = windows.in_box(cd.x, cd.y, w)
            r["n_stations"] = int(inb.sum())
            r.update(extra_metrics(n, E, L, xy, cd.x[inb], cd.y[inb], cd.F[inb], _rng(city, "e4", f, k)))
            if f in NEW_LEVELS:
                rr = network.robustness(n, E, L, _rng(city, "x11", f, k), n_random=n_random)
                r.update({kk: rr[kk] for kk in ("R_random", "R_btw", "eff_loss")})
            rows.append(r)
    return rows
