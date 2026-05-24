"""Nested window construction and segment clipping.

A *nested window* is a square sub-bbox centred on the network centroid whose
side length is ``scale_factor * full_extent``.  The same set of windows is
re-used across all dates of a city, so they are computed once from the
reference (full-network) GeoJSON.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from ..io import network_bbox_and_centre


Bbox = Tuple[float, float, float, float]


def build_nested_windows(
    reference_geojson: str,
    scale_factors: Sequence[float] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    target_crs: str | None = None,
) -> Tuple[List[Bbox], Tuple[float, float], str]:
    """Build nested square windows for one city.

    Returns
    -------
    windows : list of (min_x, min_y, max_x, max_y)
    centre  : (cx, cy) in the metric CRS
    crs     : target CRS used for the bbox coordinates
    """
    bbox, centre, target_crs = network_bbox_and_centre(reference_geojson, target_crs)
    cx, cy = centre
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    windows: List[Bbox] = []
    for sf in scale_factors:
        w, h = width * sf, height * sf
        windows.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return windows, centre, target_crs


def clip_segments_to_bbox(segments: list, bbox: Bbox) -> list:
    """Return only segments whose centroid lies inside ``bbox``."""
    min_x, min_y, max_x, max_y = bbox
    return [
        s for s in segments
        if min_x <= s["centroid_x"] <= max_x and min_y <= s["centroid_y"] <= max_y
    ]
