"""X0.5  Rainfall and road-network inventory (data availability only; no traffic response is computed).

Rainfall: ERA5 hourly total precipitation on a 0.25 degree grid (weather/grid_info.parquet and
weather/datetime/local_hourly_rainfall_YYYY-MM-DD.parquet). `total_precipitation` is in metres accumulated
over the hour ENDING at `local_time`, so the value at local_time h+1 describes the traffic hour h:00-h:59.
The city series is the overlap-area-weighted mean of the cells that intersect the station extent.

Network: IUTFD centre-line road segments (roads/<year>/centerline_roads.parquet). Segment end points closer
than 1 m are one node; we report how well the segments connect and the size of the largest component.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..common.geo import UnionFind, utm_forward, wkb_parts


# ------------------------------------------------------------------------------------------------ rain
def _cell_boxes(grid: pd.DataFrame) -> np.ndarray:
    boxes = []
    for _, r in grid.iterrows():
        box = None
        g = r.get("geometry")
        if g is not None:
            try:
                _, parts = wkb_parts(g)
                pts = np.vstack(parts)
                box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
            except Exception:
                box = None
        if box is None:
            box = (r["longitude"] - 0.125, r["latitude"] - 0.125, r["longitude"] + 0.125, r["latitude"] + 0.125)
        boxes.append(box)
    return np.array(boxes, float)


def rain_inventory(city_dir: Path, days: pd.DataFrame, ext: dict, cfg: dict) -> dict:
    out = dict(city=city_dir.name)
    wdir = city_dir / "weather"
    gpath = wdir / "grid_info.parquet"
    if not gpath.exists() or days is None or days.empty or "extent_lon_min" not in ext:
        out["status"] = "no weather grid" if not gpath.exists() else "no station extent"
        return out
    grid = pq.read_table(gpath).to_pandas()
    boxes = _cell_boxes(grid)
    e = (ext["extent_lon_min"], ext["extent_lat_min"], ext["extent_lon_max"], ext["extent_lat_max"])
    ow = np.clip(np.minimum(boxes[:, 2], e[2]) - np.maximum(boxes[:, 0], e[0]), 0, None)
    oh = np.clip(np.minimum(boxes[:, 3], e[3]) - np.maximum(boxes[:, 1], e[1]), 0, None)
    w = ow * oh
    if w.sum() <= 0:                                   # degenerate extent: nearest cell
        cx, cy = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
        w = np.zeros(len(grid))
        w[np.argmin((grid["longitude"] - cx) ** 2 + (grid["latitude"] - cy) ** 2)] = 1.0
    wmap = dict(zip(grid["grid_id"].tolist(), w.tolist()))
    out["cells_used"] = int((w > 0).sum())

    dates = pd.to_datetime(days["date"])
    want = set(dates.dt.strftime("%Y-%m-%d")) | set((dates + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"))
    frames = []
    for d in sorted(want):
        p = wdir / "datetime" / f"local_hourly_rainfall_{d}.parquet"
        if p.exists():
            frames.append(pq.read_table(p, columns=["grid_id", "total_precipitation", "local_time"]).to_pandas())
    out["n_rain_files"] = len(frames)
    if not frames:
        out["status"] = "no rain files for traffic dates"
        return out
    r = pd.concat(frames, ignore_index=True)
    r["w"] = r["grid_id"].map(wmap).fillna(0.0)
    r = r[r["w"] > 0]
    r["mm_w"] = r["total_precipitation"].astype(float) * 1000.0 * r["w"]
    s = r.groupby("local_time").agg(mm_w=("mm_w", "sum"), w=("w", "sum"))
    mmh = (s["mm_w"] / s["w"]).sort_index()
    hour_start = pd.DatetimeIndex(mmh.index) - pd.Timedelta(hours=1)
    series = pd.Series(mmh.values, index=hour_start)
    series = series[~series.index.duplicated()]
    out["n_hours_series"] = int(series.size)
    out["max_mmh"] = round(float(series.max()), 3)

    cw = set(days.loc[days["city_weekday"], "date"])
    h0, h1 = int(cfg["daytime_start_hour"]), int(cfg["daytime_end_hour"])
    idx = series.index
    daytime = np.array([(d in cw) for d in idx.strftime("%Y-%m-%d")]) & (idx.hour >= h0) & (idx.hour < h1)
    out["n_daytime_weekday_hours"] = int(daytime.sum())
    out["n_city_weekdays_with_rain_data"] = int(len(set(idx[daytime].strftime("%Y-%m-%d"))))
    gap = int(cfg["rain_event_gap_h"])
    for thr in cfg["rain_thresholds_mmh"]:
        wet = series.values >= float(thr)
        tag = str(thr).replace(".", "p")
        out[f"hours_ge_{tag}"] = int((wet & daytime).sum())
        # events over the full series: wet hours merged when separated by <= gap dry hours
        ev = 0
        hours = np.flatnonzero(wet)
        if hours.size:
            t = idx[hours].values.astype("datetime64[h]").astype(np.int64)
            breaks = np.flatnonzero(np.diff(t) > gap + 1)
            starts = np.r_[0, breaks + 1]
            ends = np.r_[breaks, hours.size - 1]
            for a, b in zip(starts, ends):
                if daytime[hours[a:b + 1]].any():
                    ev += 1
        out[f"events_ge_{tag}"] = ev
    out["status"] = "ok"
    return out


# ------------------------------------------------------------------------------------------------ network
DRIVE_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
                 "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
                 "service", "road"}


def _road_file(city_dir: Path, years_hint: list):
    """Centre-line segments if present, else raw OSM ways; the year of the traffic data is preferred."""
    rdir = city_dir / "roads"
    if not rdir.exists():
        return None, None
    years = sorted(p.name for p in rdir.iterdir() if p.is_dir())
    order = [str(y) for y in years_hint if str(y) in years] + [y for y in reversed(years) if y not in map(str, years_hint)]
    for y in order:
        for name, source in (("centerline_roads.parquet", "centerline"), ("raw_roads.parquet", "raw_osm")):
            if (rdir / y / name).exists():
                return rdir / y / name, source
    return None, None


def _components(n_nodes: int, ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    uf = UnionFind(n_nodes)
    for i, j in zip(ea.tolist(), eb.tolist()):
        uf.union(i, j)
    return uf.labels()


def network_inventory(city_dir: Path, ext: dict, epsg: int | None, years_hint: list, cfg: dict) -> dict:
    """Vertex graph of the road lines: vertices within `network_snap_m` are one node, consecutive vertices are edges."""
    out = dict(city=city_dir.name)
    p, source = _road_file(city_dir, years_hint)
    if p is None:
        out["status"] = "no road network"
        return out
    out["road_file"] = str(p.relative_to(city_dir))
    out["road_source"] = source
    names = pq.ParquetFile(p).schema_arrow.names
    cols = ["geometry"] + (["highway"] if "highway" in names else [])
    tab = pq.read_table(p, columns=cols).to_pandas()
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
    if not lines:
        out["status"] = "no geometries"
        return out
    lon_all = np.concatenate([ln[:, 0] for ln in lines])
    lat_all = np.concatenate([ln[:, 1] for ln in lines])
    if epsg:
        zone, south = int(epsg) % 100, int(epsg) // 100 == 327
    else:
        from ..common.geo import utm_zone
        zone, south = utm_zone(float(np.median(lon_all)), float(np.median(lat_all)))
    x, y = utm_forward(lon_all, lat_all, zone, south)
    snap = float(cfg["network_snap_m"])
    key = np.round(x / snap).astype(np.int64) * 4_000_000_000 + np.round(y / snap).astype(np.int64)
    _, node = np.unique(key, return_inverse=True)
    n_nodes = int(node.max()) + 1
    sizes = np.array([len(ln) for ln in lines])
    first = np.r_[0, np.cumsum(sizes)[:-1]]
    last = first + sizes - 1
    ok = np.ones(x.size, bool)
    ok[last] = False                                  # edge i -> i+1 except across line boundaries
    ia = np.flatnonzero(ok)
    ea, eb = node[ia], node[ia + 1]
    elen = np.hypot(x[ia + 1] - x[ia], y[ia + 1] - y[ia])
    keep = ea != eb
    ea, eb, elen, emx, emy = ea[keep], eb[keep], elen[keep], (x[ia] + x[ia + 1])[keep] / 2, (y[ia] + y[ia + 1])[keep] / 2
    deg = np.bincount(ea, minlength=n_nodes) + np.bincount(eb, minlength=n_nodes)
    endpoint_nodes = np.r_[node[first], node[last]]
    comp = _components(n_nodes, ea, eb)[ea]
    comp_len = np.bincount(comp, weights=elen)
    out.update(status="ok", n_lines=int(len(lines)), length_km=round(float(elen.sum()) / 1000, 2), n_nodes=n_nodes,
               n_junctions=int((deg >= 3).sum()), share_endpoints_shared=round(float((deg[endpoint_nodes] >= 2).mean()), 3),
               n_components=int((comp_len > 0).sum()), lcc_length_share=round(float(comp_len.max() / comp_len.sum()), 3))
    if "extent_xmin" in ext:
        inside = ((emx >= ext["extent_xmin"]) & (emx <= ext["extent_xmax"]) &
                  (emy >= ext["extent_ymin"]) & (emy <= ext["extent_ymax"]))
        if inside.any():
            c2 = _components(n_nodes, ea[inside], eb[inside])[ea[inside]]
            cl = np.bincount(c2, weights=elen[inside])
            deg2 = np.bincount(ea[inside], minlength=n_nodes) + np.bincount(eb[inside], minlength=n_nodes)
            out.update(ext_length_km=round(float(elen[inside].sum()) / 1000, 2), ext_junctions=int((deg2 >= 3).sum()),
                       ext_lcc_length_share=round(float(cl.max() / cl.sum()), 3))
    return out
