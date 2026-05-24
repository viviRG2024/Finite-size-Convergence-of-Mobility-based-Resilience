"""Throwaway: invoke the viz functions against cached Melbourne results."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from main import (
    diag_city_path, diag_window_path, load_config, scaling_path,
)
from src.analysis import clip_segments_to_bbox
from src.io import load_road_segments
from src.utils import load_json
from src.viz import plot_city_convergence_heatmap, plot_window_diagnostics

cfg = load_config(ROOT / "config.yaml")
out_root = ROOT / "results"
city = "melbourne"
city_dir = ROOT / "data" / "input" / city

win_cfg = load_json(out_root / city / "windows_config.json")
windows = [tuple(b) for b in win_cfg["windows"]]
target_crs = win_cfg["target_crs"]

w_idx = 4  # SF = 0.8 (a converged window)
bbox = windows[w_idx]

ms_dir = out_root / city / "multi_scale" / f"window{w_idx}"
ms_files = sorted(ms_dir.glob(f"{city}_*.json"))
assert ms_files, f"no cached multi_scale outputs under {ms_dir}"
ms_path = ms_files[0]
date_str = ms_path.stem.replace(f"{city}_", "")
print(f"using {ms_path}")

ms = load_json(ms_path)
geo = city_dir / f"result_traffic_flow_hourly_{date_str}.geojson"
segments_full, _, _ = load_road_segments(str(geo), cfg["modal_weights"],
                                         target_crs=target_crs)
segs = clip_segments_to_bbox(segments_full, bbox)
print(f"  segments in window {w_idx}: {len(segs)}")

win_min_dim = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
max_valid_l = cfg["max_valid_l_factor"] * win_min_dim

out_window = diag_window_path(out_root, city, w_idx, date_str)
plot_window_diagnostics(
    segments=segs, bbox=bbox, ms=ms,
    city=city, date_str=date_str, window_idx=w_idx,
    fit_min_l=cfg["fit_min_l"], fit_max_l=cfg["fit_max_l"],
    max_valid_l=max_valid_l, out_path=out_window,
)
print(f"  wrote {out_window}")

dates: list = []
fits_by_window_date: dict = {}
for wi in range(len(windows)):
    for p in sorted((out_root / city / "scaling" / f"window{wi}").glob(
            f"{city}_*.json")):
        d = p.stem.replace(f"{city}_", "")
        dates.append(d)
        fits_by_window_date[(wi, d)] = load_json(p)
dates = sorted(set(dates))

out_heat = diag_city_path(out_root, city)
plot_city_convergence_heatmap(
    city=city, n_windows=len(windows),
    dates=dates, hours=cfg["workday_hours"],
    fits_by_window_date=fits_by_window_date, out_path=out_heat,
)
print(f"  wrote {out_heat}")
