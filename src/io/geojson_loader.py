"""GeoJSON loading with unified CRS handling.

The IUTFD GeoJSONs use one of two coordinate conventions:

* WGS84 (EPSG:4326) -- the standard for GeoJSON.
* A pre-projected metric CRS (e.g. British National Grid for London).

For any spatial-grid analysis we need metric coordinates.  This module
auto-detects the input CRS, reprojects to a target metric CRS (auto-picked
UTM zone if unspecified) and exposes three helpers used elsewhere:

* :func:`load_geojson_to_metric` -- internal: returns a GeoDataFrame in metric CRS.
* :func:`load_road_segments`     -- segments with centroid + 24-hour flow vector.
* :func:`network_bbox_and_centre` -- bbox and centre point for window construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape


# ---------------------------------------------------------------------------
# CRS detection
# ---------------------------------------------------------------------------

def _read_geojson_crs(data: dict) -> Optional[str]:
    """Pull a CRS string out of a GeoJSON's optional ``crs`` field."""
    crs = data.get("crs")
    if not crs:
        return None
    name = crs.get("properties", {}).get("name", "")
    if "EPSG::" in name:
        return f"EPSG:{name.split('EPSG::')[-1]}"
    if name.startswith("EPSG:"):
        return name
    return None


def _looks_projected(gdf: gpd.GeoDataFrame) -> bool:
    """Heuristic: large coordinate magnitudes => projected (UTM-like)."""
    if len(gdf) == 0:
        return False
    g = gdf.geometry.iloc[0]
    if g.geom_type == "LineString":
        x, y = g.coords[0][0], g.coords[0][1]
    elif g.geom_type == "MultiLineString":
        x, y = g.geoms[0].coords[0][0], g.geoms[0].coords[0][1]
    else:
        return False
    return abs(x) > 100_000 or abs(y) > 100_000


def _utm_zone_for(gdf: gpd.GeoDataFrame) -> str:
    """Pick a UTM EPSG code from the geographic centroid of the network."""
    minx, miny, maxx, maxy = gdf.total_bounds
    lon = (minx + maxx) / 2.0
    lat = (miny + maxy) / 2.0
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:326{zone:02d}" if lat >= 0 else f"EPSG:327{zone:02d}"


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_geojson_to_metric(
    geojson_path: str | Path,
    target_crs: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, str]:
    """Load a GeoJSON, reproject into a metric CRS, return (gdf, target_crs).

    Parameters
    ----------
    geojson_path : str or Path
        Path to the IUTFD GeoJSON file.
    target_crs : str, optional
        Force a target CRS (e.g. ``"EPSG:27700"``).  If ``None``, the function
        keeps an existing projected CRS or auto-picks a UTM zone.
    """
    path = Path(geojson_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    geojson_crs = _read_geojson_crs(data)

    geoms, props = [], []
    for feat in data["features"]:
        geoms.append(shape(feat["geometry"]))
        props.append(feat["properties"])

    gdf = gpd.GeoDataFrame(props, geometry=geoms)
    if geojson_crs and gdf.crs is None:
        gdf.set_crs(geojson_crs, inplace=True)

    is_proj = _looks_projected(gdf)

    # Decide input CRS if still unset.
    if gdf.crs is None:
        gdf.set_crs(target_crs if (target_crs and is_proj) else "EPSG:4326",
                    inplace=True)

    # Decide target CRS.
    if target_crs is None:
        target_crs = str(gdf.crs) if is_proj else _utm_zone_for(gdf)

    if str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)

    return gdf, target_crs


def load_road_segments(
    geojson_path: str | Path,
    modal_weights: dict,
    target_crs: Optional[str] = None,
) -> Tuple[list, Tuple[float, float, float, float], str]:
    """Return (segments, bbox, target_crs).

    Each segment is a dict ``{id, centroid_x, centroid_y, flows}`` with a
    24-element ``flows`` array.  IUTFD format ``flow_0..flow_23`` is preferred
    when present, otherwise per-mode columns are weighted by ``modal_weights``.
    """
    gdf, target_crs = load_geojson_to_metric(geojson_path, target_crs)

    segments, xs, ys = [], [], []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            if len(geom.geoms) == 0:
                continue
            coords = list(geom.geoms[0].coords)
        else:
            continue
        if not coords:
            continue
        cx = float(np.mean([c[0] for c in coords]))
        cy = float(np.mean([c[1] for c in coords]))
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue

        props = row.drop("geometry").to_dict()
        flows = np.zeros(24)
        for h in range(24):
            key = f"flow_{h}"
            if key in props and props[key] is not None:
                flows[h] = float(props[key])
            else:
                acc = 0.0
                for mode, w in modal_weights.items():
                    mk = f"{mode}_{h}"
                    if mk in props and props[mk] is not None:
                        acc += w * float(props[mk])
                flows[h] = acc

        seg_id = props.get("id", idx)
        try:
            seg_id = int(seg_id) if pd.notna(seg_id) else int(idx)
        except (TypeError, ValueError):
            seg_id = int(idx)

        segments.append({"id": seg_id, "centroid_x": cx,
                         "centroid_y": cy, "flows": flows})
        xs.append(cx)
        ys.append(cy)

    if not segments:
        raise ValueError(f"No usable segments in {geojson_path}")

    bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
    return segments, bbox, target_crs


def network_bbox_and_centre(
    geojson_path: str | Path,
    target_crs: Optional[str] = None,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float], str]:
    """Return ``(bbox, (cx, cy), target_crs)`` in the chosen metric CRS."""
    gdf, target_crs = load_geojson_to_metric(geojson_path, target_crs)
    bounds = gdf.total_bounds
    bbox = (float(bounds[0]), float(bounds[1]),
            float(bounds[2]), float(bounds[3]))
    centre = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    return bbox, centre, target_crs
