"""I/O helpers: GeoJSON loading + CRS handling."""

from .geojson_loader import (
    load_geojson_to_metric,
    load_road_segments,
    network_bbox_and_centre,
)

__all__ = [
    "load_geojson_to_metric",
    "load_road_segments",
    "network_bbox_and_centre",
]
