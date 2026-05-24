"""Build a single Point-GeoJSON with the 34 IUTFD cities and their data volume.

For each city we read one daily GeoJSON, extract the network centroid in
WGS84, count the road segments, and combine with the number of dates on disk
to estimate the total record count (segments * dates * 24 hours).

Output: ``clean_code/results/cities_34.geojson`` (one Point per city).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CITIES = [
    "augsburg", "basel", "bern", "essen", "groningen", "hamburg",
    "innsbruck", "london", "madrid", "manchester", "rotterdam",
    "santander", "vilnius",
    "birmingham", "bolton", "bordeaux", "bremen", "cagliari", "constance",
    "darmstadt", "frankfurt", "graz", "kassel", "losangeles", "marseille",
    "melbourne", "speyer", "strasbourg", "taipeh", "torino", "toronto",
    "utrecht", "wolfsburg", "zurich",
]

DATA_ROOT = Path(r"G:\005_code\044\data\output\IUTFD_bk")


def _read_crs(data: dict) -> str | None:
    crs = data.get("crs")
    if not crs:
        return None
    name = crs.get("properties", {}).get("name", "")
    if "EPSG::" in name:
        return f"EPSG:{name.split('EPSG::')[-1]}"
    if name.startswith("EPSG:"):
        return name
    return None


def city_record(city: str) -> dict | None:
    city_dir = DATA_ROOT / city
    files = sorted(city_dir.glob("*.geojson"))
    if not files:
        return None
    n_dates = len(files)

    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    geoms = [shape(feat["geometry"]) for feat in data["features"]
             if feat.get("geometry")]
    n_segments = len(geoms)
    if n_segments == 0:
        return None

    src_crs = _read_crs(data) or "EPSG:4326"
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=src_crs).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    return {
        "city": city,
        "lon": float(cx),
        "lat": float(cy),
        "bbox_min_lon": float(minx),
        "bbox_min_lat": float(miny),
        "bbox_max_lon": float(maxx),
        "bbox_max_lat": float(maxy),
        "n_dates": n_dates,
        "n_segments": n_segments,
        "n_records": n_dates * n_segments * 24,
    }


def main() -> int:
    features = []
    for city in CITIES:
        rec = city_record(city)
        if rec is None:
            print(f"  [skip] {city}: no usable GeoJSON")
            continue
        print(f"  {city}: {rec['n_dates']} dates, {rec['n_segments']} segments "
              f"-> {rec['n_records']:,} records  ({rec['lon']:.3f}, {rec['lat']:.3f})")
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [rec["lon"], rec["lat"]]},
            "properties": {k: v for k, v in rec.items() if k not in ("lon", "lat")},
        })

    out = ROOT / "results" / "cities_34.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "type": "FeatureCollection",
            "crs": {"type": "name",
                    "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
            "features": features,
        }, f, indent=2)
    print(f"\nWrote {out} ({len(features)} cities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
