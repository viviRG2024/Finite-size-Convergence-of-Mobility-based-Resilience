"""X11  Road-network robustness and its critical observation window.

Graph: the X0 vertex graph of the road lines (vertices within 1 m merged; same road file as X0), restricted to the
segments whose midpoint lies in the window, with degree-2 vertices contracted (a node is a junction or a dead end;
an edge is the road between two such nodes, weighted by its length). Removing a shape point of a curved road is not
a node failure, hence the contraction.
Per graph:
  R_random : area under the largest-component curve S(q) (fraction of nodes in the largest component after removing
             q nodes), mean over q = 1..N, averaged over 20 random removal orders (Schneider et al. 2011);
  R_btw    : the same for removal in decreasing order of initial betweenness (500 sampled sources, hop distance);
  eff_loss : 1 - E_after / E_before, global efficiency (length-weighted, 100 sampled sources) after removing the 10%
             of nodes with the highest initial betweenness.
Windows: 8 coverage levels x the first 20 random windows of the main run (same seeds); a window is usable with
>= 100 nodes.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import igraph as ig
import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from ..common.geo import utm_forward, wkb_parts
from ..core import windows
from ..x0.x05_inventory import DRIVE_CLASSES

LEVELS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90)
N_POS = 20
MIN_NODES = 100


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def load_segments(path: Path, source: str, epsg: int, snap: float = 1.0):
    """Vertex-graph edges (node a, node b, length, midpoint x, y) of a road file in UTM metres."""
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
    _, node = np.unique(key, return_inverse=True)
    sizes = np.array([len(ln) for ln in lines])
    first = np.r_[0, np.cumsum(sizes)[:-1]]
    ok = np.ones(x.size, bool)
    ok[first + sizes - 1] = False
    ia = np.flatnonzero(ok)
    a, b = node[ia], node[ia + 1]
    ln = np.hypot(x[ia + 1] - x[ia], y[ia + 1] - y[ia])
    mx, my = (x[ia] + x[ia + 1]) / 2, (y[ia] + y[ia + 1]) / 2
    keep = a != b
    return a[keep], b[keep], ln[keep], mx[keep], my[keep]


def contracted(a, b, ln):
    """Contract degree-2 vertices. Returns (n_nodes, edge array (k x 2), edge lengths) with nodes relabelled 0..n-1."""
    nodes, inv = np.unique(np.r_[a, b], return_inverse=True)
    a, b = inv[:a.size], inv[a.size:]
    n = nodes.size
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    two = deg == 2
    # chains: components of the subgraph made of edges between two degree-2 vertices
    inner = two[a] & two[b]
    lab = np.arange(n)
    if inner.any():
        g = coo_matrix((np.ones(inner.sum()), (a[inner], b[inner])), shape=(n, n))
        _, cl = connected_components(g, directed=False)
        lab = np.where(two, n + cl, np.arange(n))              # chain id offset by n
    la, lb = lab[a], lab[b]
    junction_edges = ~two[a] & ~two[b]
    E = [np.column_stack([a[junction_edges], b[junction_edges]])]
    L = [ln[junction_edges]]
    # edges touching a chain: collect per chain its end junctions and total length
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
    return keep_nodes.size, remap[E], L


def _lcc_curve(n, edges, order):
    """S(q) for q = 1..n when nodes are removed in `order` (reverse percolation with union-find)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    parent = np.arange(n)
    size = np.ones(n, int)
    present = np.zeros(n, bool)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    best = 0
    S = np.zeros(n)
    for q in range(n - 1, -1, -1):                 # add nodes back from the last removed
        S[q] = best                                 # after removing order[:q+1]... see below
        u = order[q]
        present[u] = True
        ru = find(u)
        for v in adj[u]:
            if present[v]:
                rv = find(v)
                if rv != ru:
                    if size[ru] < size[rv]:
                        ru, rv = rv, ru
                    parent[rv] = ru
                    size[ru] += size[rv]
        best = max(best, size[ru])
    return S / n                                    # S[q] = largest component after removing order[:q+1]


def robustness(n, edges, lengths, rng, n_random=20, n_src_btw=500, n_src_eff=100):
    g = ig.Graph(n=n, edges=edges.tolist())
    src = rng.choice(n, min(n, n_src_btw), replace=False).tolist()
    btw = np.array(g.betweenness(sources=src, directed=False))
    order_b = np.argsort(-btw, kind="stable")
    R_btw = float(_lcc_curve(n, edges, order_b).mean())
    R_rand = float(np.mean([_lcc_curve(n, edges, rng.permutation(n)).mean() for _ in range(n_random)]))
    W = coo_matrix((np.r_[lengths, lengths], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    s2 = rng.choice(n, min(n, n_src_eff), replace=False)

    def eff(Wm, alive):
        ss = s2[alive[s2]]
        if ss.size == 0:
            return np.nan
        d = dijkstra(Wm, directed=False, indices=ss)
        d = d[:, alive]
        with np.errstate(divide="ignore"):
            inv = np.where(np.isfinite(d) & (d > 0), 1.0 / d, 0.0)
        return inv.sum() / (ss.size * (alive.sum() - 1))

    alive = np.ones(n, bool)
    e0 = eff(W, alive)
    rem = order_b[: int(round(0.1 * n))]
    alive[rem] = False
    keep = np.flatnonzero(alive)
    W2 = W.tolil()
    W2[rem, :] = 0
    W2[:, rem] = 0
    e1 = eff(W2.tocsr(), alive)
    _ = keep
    return dict(R_random=R_rand, R_btw=R_btw, eff_loss=float(1 - e1 / e0) if e0 and np.isfinite(e0) else np.nan, n_nodes=n,
                n_edges=int(edges.shape[0]))


def window_graph(seg, box):
    a, b, ln, mx, my = seg
    m = (mx >= box[0]) & (mx <= box[2]) & (my >= box[1]) & (my <= box[3])
    if m.sum() == 0:
        return 0, np.zeros((0, 2), int), np.zeros(0)
    return contracted(a[m], b[m], ln[m])


def city_network(city: str, seg, extent, n_pos=N_POS, n_random=20):
    rows = []
    n, E, L = window_graph(seg, extent)
    full = robustness(n, E, L, _rng(city, "x11", "E"), n_random=n_random)
    rows.append(dict(f=1.0, pos=0, **full))
    for f in LEVELS:
        wins = windows.random_windows(extent, f, 50, _rng(city, "main", "windows", f))[:n_pos]   # main-run windows
        for k, w in enumerate(wins):
            n, E, L = window_graph(seg, w)
            if n < MIN_NODES:
                rows.append(dict(f=f, pos=k, n_nodes=n))
                continue
            rows.append(dict(f=f, pos=k, **robustness(n, E, L, _rng(city, "x11", f, k), n_random=n_random)))
    return rows
