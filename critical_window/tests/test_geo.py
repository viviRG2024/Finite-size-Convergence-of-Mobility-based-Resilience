"""X0.4 checks of the numpy geometry helpers (UTM, WKB, clustering).

Run from the repository root:  python -m critical_window.tests.test_geo
pyproj / shapely are optional; comparisons that need them are skipped when they are missing.
"""
from __future__ import annotations

import struct
import sys

import numpy as np

from ..common.geo import cluster_points, utm_forward, utm_zone, wkb_parts

CITY_POINTS = {"london": (-0.09, 51.51), "melbourne": (144.96, -37.81), "losangeles": (-118.26, 34.03),
               "taipeh": (121.54, 25.05), "vilnius": (25.27, 54.69), "toronto": (-79.38, 43.67)}


def test_utm() -> str:
    try:
        from pyproj import Transformer
    except ImportError:
        return "utm: pyproj not installed, comparison skipped"
    rng = np.random.default_rng(0)
    worst = 0.0
    for name, (lon0, lat0) in CITY_POINTS.items():
        zone, south = utm_zone(lon0, lat0)
        lon = lon0 + rng.uniform(-0.3, 0.3, 500)
        lat = lat0 + rng.uniform(-0.2, 0.2, 500)
        x, y = utm_forward(lon, lat, zone, south)
        tr = Transformer.from_crs(4326, (32700 if south else 32600) + zone, always_xy=True)
        px, py = tr.transform(lon, lat)
        worst = max(worst, float(np.max(np.hypot(x - px, y - py))))
    assert worst < 1e-3, worst
    return f"utm: max difference to pyproj {worst * 1000:.4f} mm over 6 cities"


def test_wkb() -> str:
    line = struct.pack("<BII", 1, 2, 3) + struct.pack("<6d", 0, 0, 1, 1, 2, 0)
    base, parts = wkb_parts(line)
    assert base == 2 and parts[0].shape == (3, 2) and parts[0][2, 0] == 2
    multi = struct.pack("<BII", 1, 5, 2) + line + line
    base, parts = wkb_parts(multi)
    assert base == 5 and len(parts) == 2
    big_endian_point = struct.pack(">BI2d", 0, 1, 7.5, 47.5)
    base, parts = wkb_parts(big_endian_point)
    assert base == 1 and np.allclose(parts[0], [[7.5, 47.5]])
    try:
        from shapely.geometry import LineString, Polygon
        import shapely.wkb as swkb
    except ImportError:
        return "wkb: hand-made cases pass (shapely not installed)"
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    base, parts = wkb_parts(swkb.dumps(poly))
    assert base == 3 and np.allclose(parts[0], np.array(poly.exterior.coords))
    ls = LineString([(7.1, 47.2), (7.2, 47.3), (7.25, 47.31)])
    base, parts = wkb_parts(swkb.dumps(ls))
    assert np.allclose(parts[0], np.array(ls.coords))
    return "wkb: hand-made and shapely-encoded geometries parse correctly"


def test_cluster() -> str:
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 500, 400)
    y = rng.uniform(0, 500, 400)
    lab = cluster_points(x, y, 10.0)
    # brute force single linkage
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) <= 10.0
    comp = -np.ones(x.size, int)
    c = 0
    for i in range(x.size):
        if comp[i] >= 0:
            continue
        stack = [i]
        comp[i] = c
        while stack:
            j = stack.pop()
            for k in np.flatnonzero(d[j] & (comp < 0)):
                comp[k] = c
                stack.append(k)
        c += 1
    same = (lab[:, None] == lab[None, :]) == (comp[:, None] == comp[None, :])
    assert same.all()
    return f"cluster: identical to brute-force single linkage ({c} clusters from 400 points)"


def main() -> int:
    print("\n".join([test_utm(), test_wkb(), test_cluster()]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
