"""End-to-end pipeline for the cleaned NC-submission codebase.

Stages
------
* ``load``      -- discover GeoJSONs and build nested windows per city.
* ``compute``   -- multi-scale CV(L) and H(L) per (city, date, window).
* ``fit``       -- alpha and beta per (city, date, window, hour).
* ``aggregate`` -- one-row-per-city converged-exponent table (Table 2).

The default ``all`` runs them all in order.  Each stage is idempotent: it
skips work whose output JSON already exists unless ``--force`` is supplied.

Usage
-----
    python main.py
    python main.py --city melbourne --stage compute
    python main.py --windows 2 3 4 5 --hours 8 9 10 11 12 13 14 15
    python main.py --no-tests
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analysis import (
    analyse_multi_scale,
    bootstrap_correlation_ci,
    bootstrap_scaling_exponents,
    build_nested_windows,
    clip_segments_to_bbox,
    compute_alpha_beta_per_hour,
    ks_test_correlation_curves,
    permutation_test_optimal_scale,
)
from src.io import load_road_segments
from src.utils import dump_json, load_json


# ---------------------------------------------------------------------------
# Config + path helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_dates(city_dir: Path) -> list:
    """Return the YYYY_MM_DD strings found as GeoJSONs in ``city_dir``."""
    dates = []
    for f in sorted(city_dir.glob("result_traffic_flow_hourly_*.geojson")):
        dates.append(f.stem.replace("result_traffic_flow_hourly_", ""))
    return dates


def geojson_path(city_dir: Path, date_str: str) -> Path:
    return city_dir / f"result_traffic_flow_hourly_{date_str}.geojson"


def windows_config_path(out_root: Path, city: str) -> Path:
    return out_root / city / "windows_config.json"


def multi_scale_path(out_root: Path, city: str, window_idx: int, date_str: str) -> Path:
    return out_root / city / "multi_scale" / f"window{window_idx}" / f"{city}_{date_str}.json"


def scaling_path(out_root: Path, city: str, window_idx: int, date_str: str) -> Path:
    return out_root / city / "scaling" / f"window{window_idx}" / f"{city}_{date_str}.json"


def stats_path(out_root: Path, city: str, window_idx: int, date_str: str) -> Path:
    return out_root / city / "stats" / f"window{window_idx}" / f"{city}_{date_str}.json"


def diag_window_path(out_root: Path, city: str, window_idx: int,
                     date_str: str) -> Path:
    return (out_root / city / "diagnostics" / f"window{window_idx}"
            / f"{city}_{date_str}.png")


def diag_city_path(out_root: Path, city: str) -> Path:
    return out_root / city / "diagnostics" / f"{city}_convergence_heatmap.png"


# ---------------------------------------------------------------------------
# Stage 1: discover dates and build per-city windows
# ---------------------------------------------------------------------------

def stage_load(cfg: dict, cities: list, in_root: Path, out_root: Path,
               force: bool = False) -> dict:
    """Resolve dates and build (or read) windows for each city."""
    resolved: dict = {}
    sf = cfg["window_scale_factors"]

    for city in cities:
        city_dir = in_root / city
        if not city_dir.exists():
            raise FileNotFoundError(f"Missing city directory: {city_dir}")

        cfg_dates = cfg["cities"][city].get("dates")
        target_crs = cfg["cities"][city].get("target_crs")

        all_dates = discover_dates(city_dir)
        dates = cfg_dates if cfg_dates else all_dates
        # Keep only dates that actually exist on disk.
        dates = [d for d in dates if geojson_path(city_dir, d).exists()]
        if not dates:
            raise FileNotFoundError(f"No GeoJSONs found for {city} in {city_dir}")

        win_cfg = windows_config_path(out_root, city)
        if win_cfg.exists() and not force:
            data = load_json(win_cfg)
            windows = [tuple(b) for b in data["windows"]]
            centre = tuple(data["centre"])
            target_crs = data["target_crs"]
        else:
            ref = geojson_path(city_dir, dates[0])
            windows, centre, target_crs = build_nested_windows(
                str(ref), scale_factors=sf, target_crs=target_crs,
            )
            dump_json({
                "city": city,
                "reference_geojson": ref.name,
                "scale_factors": list(sf),
                "windows": [list(w) for w in windows],
                "centre": list(centre),
                "target_crs": target_crs,
            }, win_cfg)

        resolved[city] = {
            "city_dir": city_dir,
            "dates": dates,
            "windows": windows,
            "centre": centre,
            "target_crs": target_crs,
        }
        print(f"[load] {city}: {len(dates)} dates, {len(windows)} windows, "
              f"CRS={target_crs}", flush=True)
    return resolved


# ---------------------------------------------------------------------------
# Stage 2: multi-scale CV and H per (city, date, window)
# ---------------------------------------------------------------------------

def stage_compute(cfg: dict, ctx: dict, out_root: Path, force: bool = False,
                  viz: bool = False) -> None:
    """Compute and persist CV(L), H(L) for every (date, window)."""
    grid_sizes = cfg["grid_sizes"]
    connectivity = cfg["connectivity"]
    edge_method = cfg["edge_weight_method"]
    entropy_type = cfg["entropy_type"]
    modal_weights = cfg["modal_weights"]
    fit_min = cfg["fit_min_l"]
    fit_max = cfg["fit_max_l"]
    factor = cfg["max_valid_l_factor"]

    if viz:
        from src.viz import plot_window_diagnostics

    for city, info in ctx.items():
        for date_str in info["dates"]:
            t0 = time.time()
            geo = geojson_path(info["city_dir"], date_str)
            print(f"[compute] {city} {date_str}", flush=True)
            segments_full, _bbox_full, _crs = load_road_segments(
                str(geo), modal_weights, target_crs=info["target_crs"],
            )
            print(f"  loaded {len(segments_full)} segments", flush=True)

            for w_idx, bbox in enumerate(info["windows"]):
                out = multi_scale_path(out_root, city, w_idx, date_str)
                ms = None
                if out.exists() and not force:
                    if viz:
                        ms = load_json(out)
                        if ms.get("skipped"):
                            continue
                    else:
                        continue
                if ms is None:
                    segs = clip_segments_to_bbox(segments_full, bbox)
                    if len(segs) < 10:
                        dump_json({"skipped": True,
                                   "reason": "too few segments",
                                   "n_segments": len(segs),
                                   "window_idx": w_idx,
                                   "bbox": list(bbox)}, out)
                        continue
                    ms = analyse_multi_scale(
                        segs, bbox, grid_sizes,
                        connectivity=connectivity,
                        edge_weight_method=edge_method,
                        entropy_type=entropy_type,
                        progress=False,
                    )
                    ms["city"] = city
                    ms["date"] = date_str
                    ms["window_idx"] = w_idx
                    ms["bbox"] = list(bbox)
                    ms["n_segments"] = len(segs)
                    # Drop bulky scale_results to keep JSON small.
                    ms.pop("scale_results", None)
                    dump_json(ms, out)
                else:
                    segs = clip_segments_to_bbox(segments_full, bbox)
                if viz and len(segs) >= 10:
                    win_min_dim = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
                    plot_window_diagnostics(
                        segments=segs, bbox=bbox, ms=ms,
                        city=city, date_str=date_str, window_idx=w_idx,
                        fit_min_l=fit_min, fit_max_l=fit_max,
                        max_valid_l=factor * win_min_dim,
                        out_path=diag_window_path(out_root, city, w_idx,
                                                  date_str),
                    )
            print(f"  done in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Stage 3: alpha, beta fits per (city, date, window, hour)
# ---------------------------------------------------------------------------

def stage_fit(cfg: dict, ctx: dict, out_root: Path, force: bool = False,
              viz: bool = False) -> None:
    """Fit local power laws to CV(L) and H(L) for each (date, window)."""
    fit_min = cfg["fit_min_l"]
    fit_max = cfg["fit_max_l"]
    factor = cfg["max_valid_l_factor"]
    min_pts = cfg["min_points_local"]
    hours = cfg["workday_hours"]

    if viz:
        from src.viz import plot_city_convergence_heatmap

    for city, info in ctx.items():
        for date_str in info["dates"]:
            for w_idx, bbox in enumerate(info["windows"]):
                ms_path = multi_scale_path(out_root, city, w_idx, date_str)
                if not ms_path.exists():
                    continue
                ms = load_json(ms_path)
                if ms.get("skipped"):
                    continue
                out = scaling_path(out_root, city, w_idx, date_str)
                if out.exists() and not force:
                    continue
                win_min_dim = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
                max_valid_l = factor * win_min_dim
                fits = compute_alpha_beta_per_hour(
                    ms,
                    fit_min_l=fit_min, fit_max_l=fit_max,
                    max_valid_l=max_valid_l, min_points=min_pts,
                    hours=hours,
                )
                fits.update({
                    "city": city, "date": date_str, "window_idx": w_idx,
                    "fit_min_l": fit_min, "fit_max_l": fit_max,
                    "max_valid_l": max_valid_l, "hours": list(hours),
                })
                dump_json(fits, out)

        if viz:
            fits_by_window_date: dict = {}
            for date_str in info["dates"]:
                for w_idx in range(len(info["windows"])):
                    p = scaling_path(out_root, city, w_idx, date_str)
                    if p.exists():
                        fits_by_window_date[(w_idx, date_str)] = load_json(p)
            if fits_by_window_date:
                plot_city_convergence_heatmap(
                    city=city,
                    n_windows=len(info["windows"]),
                    dates=list(info["dates"]),
                    hours=list(hours),
                    fits_by_window_date=fits_by_window_date,
                    out_path=diag_city_path(out_root, city),
                )


# ---------------------------------------------------------------------------
# Stage 3b (optional): bootstrap CIs and KS tests per window
# ---------------------------------------------------------------------------

def stage_tests(cfg: dict, ctx: dict, out_root: Path, force: bool = False) -> None:
    """Bootstrap CIs and pairwise KS tests on rho(L) curves."""
    n_boot = int(cfg["n_bootstrap"])
    seed = int(cfg["random_seed"])
    fit_min, fit_max = cfg["fit_min_l"], cfg["fit_max_l"]
    factor = cfg["max_valid_l_factor"]
    hours = cfg["workday_hours"]
    rng = np.random.default_rng(seed)

    for city, info in ctx.items():
        for date_str in info["dates"]:
            for w_idx, bbox in enumerate(info["windows"]):
                ms_path = multi_scale_path(out_root, city, w_idx, date_str)
                if not ms_path.exists():
                    continue
                ms = load_json(ms_path)
                if ms.get("skipped"):
                    continue
                out = stats_path(out_root, city, w_idx, date_str)
                if out.exists() and not force:
                    continue

                grid_sizes = ms["grid_sizes"]
                cv = np.asarray(ms["cv_by_scale"])  # (n_scales, 24)
                H = np.asarray(ms["entropy_by_scale"])
                win_min_dim = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
                max_valid_l = factor * win_min_dim

                rho_ci_per_scale = []
                exp_ci_per_hour = {}
                for s_idx in range(len(grid_sizes)):
                    lo, hi, _ = bootstrap_correlation_ci(
                        cv[s_idx], H[s_idx],
                        n_bootstrap=n_boot, rng=rng,
                    )
                    rho_ci_per_scale.append({"grid_size_m": grid_sizes[s_idx],
                                             "ci_low": lo, "ci_high": hi})
                for h in hours:
                    cv_h = cv[:, h]
                    H_h = H[:, h]
                    exp_ci_per_hour[h] = bootstrap_scaling_exponents(
                        grid_sizes, cv_h, H_h,
                        fit_min_l=fit_min, fit_max_l=fit_max,
                        max_valid_l=max_valid_l,
                        n_bootstrap=n_boot, rng=rng,
                    )

                dump_json({
                    "city": city, "date": date_str, "window_idx": w_idx,
                    "rho_bootstrap_ci_per_scale": rho_ci_per_scale,
                    "exponent_bootstrap_ci_per_hour": exp_ci_per_hour,
                    "n_bootstrap": n_boot, "random_seed": seed,
                }, out)


# ---------------------------------------------------------------------------
# Stage 4: cross-city aggregation
# ---------------------------------------------------------------------------

def stage_aggregate(cfg: dict, ctx: dict, out_root: Path) -> None:
    """Build the converged-exponent table (Table 2) over all cities."""
    converged = set(cfg["converged_window_indices"])
    hours = cfg["workday_hours"]
    rows = []
    per_city_summary = {}

    for city, info in ctx.items():
        alphas, betas = [], []
        for date_str in info["dates"]:
            for w_idx in range(len(info["windows"])):
                if w_idx not in converged:
                    continue
                p = scaling_path(out_root, city, w_idx, date_str)
                if not p.exists():
                    continue
                fits = load_json(p)
                for h in hours:
                    a = fits["alpha_by_hour"].get(str(h)) or fits["alpha_by_hour"].get(h)
                    b = fits["beta_by_hour"].get(str(h)) or fits["beta_by_hour"].get(h)
                    if a and a.get("ok"):
                        alphas.append(a["exponent"])
                    if b and b.get("ok"):
                        betas.append(b["exponent"])
        a_arr = np.asarray(alphas) if alphas else np.array([np.nan])
        b_arr = np.asarray(betas) if betas else np.array([np.nan])
        row = {
            "city": city,
            "n_alpha": int(np.isfinite(a_arr).sum()),
            "n_beta": int(np.isfinite(b_arr).sum()),
            "alpha_median": float(np.nanmedian(a_arr)),
            "alpha_iqr": float(np.nanpercentile(a_arr, 75) - np.nanpercentile(a_arr, 25)),
            "alpha_min": float(np.nanmin(a_arr)),
            "alpha_max": float(np.nanmax(a_arr)),
            "beta_median": float(np.nanmedian(b_arr)),
            "beta_iqr": float(np.nanpercentile(b_arr, 75) - np.nanpercentile(b_arr, 25)),
            "beta_min": float(np.nanmin(b_arr)),
            "beta_max": float(np.nanmax(b_arr)),
        }
        rows.append(row)
        per_city_summary[city] = row

    csv_path = out_root / "cross_city_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    dump_json({"converged_window_indices": list(converged),
               "workday_hours": list(hours),
               "cities": per_city_summary}, out_root / "cross_city_summary.json")
    print(f"[aggregate] wrote {csv_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STAGES = ["load", "compute", "fit", "tests", "aggregate", "all"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--input-dir", type=Path, default=ROOT / "data" / "input")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--city", action="append",
                   help="restrict to one city (repeatable)")
    p.add_argument("--windows", type=int, nargs="+",
                   help="override window indices for aggregate stage")
    p.add_argument("--hours", type=int, nargs="+",
                   help="override workday hours")
    p.add_argument("--no-tests", action="store_true",
                   help="skip the bootstrap / KS test stage in 'all'")
    p.add_argument("--force", action="store_true",
                   help="recompute outputs even if they already exist")
    p.add_argument("--viz", action="store_true",
                   help="also write per-window diagnostic PNGs and a per-city "
                        "convergence heatmap into results/<city>/diagnostics/")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.windows is not None:
        cfg["converged_window_indices"] = args.windows
    if args.hours is not None:
        cfg["workday_hours"] = args.hours

    cities = args.city or list(cfg["cities"].keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cities  : {cities}")
    print(f"Stage   : {args.stage}")
    print(f"Output  : {args.output_dir}")
    print()

    ctx = stage_load(cfg, cities, args.input_dir, args.output_dir,
                     force=args.force)
    if args.stage == "load":
        return 0

    if args.stage in ("compute", "all"):
        stage_compute(cfg, ctx, args.output_dir, force=args.force,
                      viz=args.viz)
    if args.stage in ("fit", "all"):
        stage_fit(cfg, ctx, args.output_dir, force=args.force,
                  viz=args.viz)
    if args.stage == "all" and not args.no_tests:
        stage_tests(cfg, ctx, args.output_dir, force=args.force)
    if args.stage == "tests":
        stage_tests(cfg, ctx, args.output_dir, force=args.force)
    if args.stage in ("aggregate", "all"):
        stage_aggregate(cfg, ctx, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
