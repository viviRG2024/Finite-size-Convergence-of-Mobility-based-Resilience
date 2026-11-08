"""Active cells (cells containing at least one station) and stations per active cell at each of the 39 cell sizes,
for the whole observation extent with the grid origin at the corner of the extent (Supplementary Table 3).

    python -m critical_window.tools.active_cells [--x0 out/x0] [--main main34]

Outputs in out/checks/: active_cells_by_size.csv (one row per city and cell size) and active_cells_summary.csv
(medians over cities at selected sizes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.cfg import load_config, repo_path
from ..core.probes import GRID_SIZES
from ..main.data import load_city

SHOW = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def city_rows(c, x0, m):
    cd = load_city(c, x0, m)
    x, y = cd.x, cd.y
    bx0, by0, bx1, by1 = cd.extent
    rows = []
    for L in GRID_SIZES:
        nx = max(int(np.ceil((bx1 - bx0) / L)), 1)
        ny = max(int(np.ceil((by1 - by0) / L)), 1)
        gx = np.clip(((x - bx0) / L).astype(int), 0, nx - 1)
        gy = np.clip(((y - by0) / L).astype(int), 0, ny - 1)
        cells, cnt = np.unique(gy * nx + gx, return_counts=True)
        occupied = set(cells.tolist())
        cgx, cgy = cells % nx, cells // nx
        with_nb = 0
        for a, b in zip(cgx, cgy):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (dx or dy) and 0 <= a + dx < nx and 0 <= b + dy < ny and (b + dy) * nx + a + dx in occupied:
                        with_nb += 1
                        break
                else:
                    continue
                break
        rows.append(dict(city=c, L=float(L), n_stations=int(x.size), cells_in_extent=nx * ny, active=int(cells.size),
                         active_with_neighbour=with_nb, stations_per_active=x.size / cells.size,
                         share_single=float((cnt == 1).mean()), active_ge10=int((cnt >= 10).sum()),
                         share_ge10=float((cnt >= 10).mean())))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x0", default="")
    ap.add_argument("--main", default="main34")
    a = ap.parse_args(argv)
    cfg = load_config()
    out_root = repo_path(cfg["paths"]["out_dir"])
    x0 = Path(a.x0) if a.x0 else out_root / "x0"
    cities = json.load(open(out_root / a.main / "summary.json", encoding="utf-8"))["cities"]
    rows = []
    for c in cities:
        rows += city_rows(c, x0, cfg["main"])
    d = pd.DataFrame(rows)
    od = out_root / "checks"
    od.mkdir(parents=True, exist_ok=True)
    d.to_csv(od / "active_cells_by_size.csv", index=False)
    s = d[d.L.isin(SHOW)].groupby("L").agg(
        active_median=("active", "median"), active_min=("active", "min"), active_max=("active", "max"),
        with_neighbour_median=("active_with_neighbour", "median"), stations_per_active_median=("stations_per_active", "median"),
        share_single_median=("share_single", "median"), cities_with_cell_ge10=("active_ge10", lambda v: int((v > 0).sum())))
    s.to_csv(od / "active_cells_summary.csv")
    print(s.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
