"""Geometry helpers with numpy only: UTM projection, WKB parsing, distance-based clustering.

The UTM forward projection uses the Krueger series (WGS84), accurate to well below 1 mm inside a zone;
tests/test_geo.py checks it against pyproj when pyproj is installed.
"""
from __future__ import annotations

import struct

import numpy as np

_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996


def utm_zone(lon: float, lat: float) -> tuple[int, bool]:
    """UTM zone number and hemisphere (south=True) for a point; the city centre is used for a whole city."""
    zone = int(np.floor((lon + 180.0) / 6.0)) + 1
    return min(max(zone, 1), 60), lat < 0


def utm_epsg(zone: int, south: bool) -> int:
    return (32700 if south else 32600) + zone


def utm_forward(lon, lat, zone: int, south: bool):
    """Project WGS84 lon/lat (degrees) to UTM easting/northing (metres) in a fixed zone."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = _F / (2 - _F)
    A = _A / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alpha = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16,
             13 * n ** 2 / 48 - 3 * n ** 3 / 5,
             61 * n ** 3 / 240)
    lam0 = np.radians((zone - 1) * 6 - 180 + 3)
    phi = np.radians(lat)
    lam = np.radians(lon) - lam0
    c = 2 * np.sqrt(n) / (1 + n)
    t = np.sinh(np.arctanh(np.sin(phi)) - c * np.arctanh(c * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t ** 2))
    E = xi * 0 + eta
    N = xi.copy()
    for j, a in enumerate(alpha, start=1):
        E = E + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        N = N + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    x = 500000.0 + _K0 * A * E
    y = (10000000.0 if south else 0.0) + _K0 * A * N
    return x, y


# ------------------------------------------------------------------------------------------------
# WKB (points, lines, polygons and their multi versions; ISO and EWKB flags)
# ------------------------------------------------------------------------------------------------
def _parse(b: bytes, off: int):
    end = "<" if b[off] == 1 else ">"
    off += 1
    (t,) = struct.unpack_from(end + "I", b, off)
    off += 4
    has_z = bool(t & 0x80000000)
    has_m = bool(t & 0x40000000)
    if t & 0x20000000:          # EWKB SRID
        off += 4
    base = t & 0x0FFFFFFF
    if base >= 1000:            # ISO WKB: 1000 = Z, 2000 = M, 3000 = ZM
        has_z = has_z or (base // 1000) in (1, 3)
        has_m = has_m or (base // 1000) in (2, 3)
        base %= 1000
    dim = 2 + int(has_z) + int(has_m)
    dt = np.dtype(end + "f8")
    if base == 1:
        p = np.frombuffer(b, dtype=dt, count=dim, offset=off).reshape(1, dim)[:, :2]
        return base, [p], off + 8 * dim
    if base == 2:
        (n,) = struct.unpack_from(end + "I", b, off)
        off += 4
        arr = np.frombuffer(b, dtype=dt, count=n * dim, offset=off).reshape(n, dim)[:, :2]
        return base, [arr], off + 8 * dim * n
    if base == 3:
        (nr,) = struct.unpack_from(end + "I", b, off)
        off += 4
        rings = []
        for _ in range(nr):
            (n,) = struct.unpack_from(end + "I", b, off)
            off += 4
            rings.append(np.frombuffer(b, dtype=dt, count=n * dim, offset=off).reshape(n, dim)[:, :2])
            off += 8 * dim * n
        return base, rings, off
    if base in (4, 5, 6, 7):
        (ng,) = struct.unpack_from(end + "I", b, off)
        off += 4
        parts = []
        for _ in range(ng):
            _, sub, off = _parse(b, off)
            parts.extend(sub)
        return base, parts, off
    raise ValueError(f"unsupported WKB geometry type {t}")


def wkb_parts(b) -> tuple[int, list]:
    """Return (base geometry type, list of (n, 2) coordinate arrays) for a WKB blob."""
    if isinstance(b, memoryview):
        b = b.tobytes()
    base, parts, _ = _parse(bytes(b), 0)
    return base, parts


# ------------------------------------------------------------------------------------------------
# Clustering points closer than a distance (single linkage) with a grid hash + union-find
# ------------------------------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, i: int) -> int:
        p = self.p
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            p[i], i = root, p[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            if ri < rj:
                self.p[rj] = ri
            else:
                self.p[ri] = rj

    def labels(self) -> np.ndarray:
        roots = np.array([self.find(i) for i in range(len(self.p))])
        _, lab = np.unique(roots, return_inverse=True)
        return lab


def cluster_points(x, y, dist: float) -> np.ndarray:
    """Label points so that any two points closer than `dist` share a label (single linkage)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    uf = UnionFind(n)
    if n == 0:
        return np.zeros(0, int)
    gx = np.floor((x - x.min()) / dist).astype(np.int64)
    gy = np.floor((y - y.min()) / dist).astype(np.int64)
    cells: dict = {}
    for i, key in enumerate(zip(gx.tolist(), gy.tolist())):
        cells.setdefault(key, []).append(i)
    d2 = dist * dist
    for (cx, cy), members in cells.items():
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(cells.get((cx + dx, cy + dy), ()))
        cand = np.array(cand)
        for i in members:
            close = cand[(x[cand] - x[i]) ** 2 + (y[cand] - y[i]) ** 2 <= d2]
            for j in close:
                if j != i:
                    uf.union(i, int(j))
    return uf.labels()
