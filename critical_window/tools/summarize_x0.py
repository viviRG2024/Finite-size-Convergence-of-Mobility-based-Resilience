"""Summarise the X0 outputs into one table per city (needs pandas only; reads CSV/JSON).

    python -m critical_window.tools.summarize_x0
Writes critical_window/out/x0/x0_summary.csv and prints it.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from ..common.cfg import PKG_DIR, city_table


def main() -> int:
    out = PKG_DIR / "out" / "x0"
    qc = pd.read_csv(out / "qc_city.csv")
    inc = pd.read_csv(out / "inclusion.csv")
    rain = pd.read_csv(out / "rain_inventory.csv")
    net = pd.read_csv(out / "network_inventory.csv")
    cities = pd.DataFrame(city_table().values())[["city", "continent"]]
    t = (inc.merge(cities, on="city", how="left")
         .merge(qc[[c for c in ("city", "interval_s", "labelling", "n_days", "n_weekdays_nonholiday", "n_stations",
                                "n_detectors_readings", "extent_area_km2", "extent_short_km", "unit_classes",
                                "unit_median_ratio_top2", "n_records", "n_valid", "n_error_flag",
                                "station_days_any_member", "station_days_lost_incomplete_lanes",
                                "station_days_valid", "station_days_outlier") if c in qc]], on="city", how="left")
         .merge(rain[[c for c in ("city", "events_ge_0p5", "hours_ge_0p5", "events_ge_1p0") if c in rain]], on="city", how="left")
         .merge(net[[c for c in ("city", "road_source", "lcc_length_share") if c in net]], on="city", how="left"))
    t["valid_record_share"] = (t["n_valid"] / t["n_records"]).round(3)
    t["error_share"] = (t["n_error_flag"] / t["n_records"]).round(3)
    t["lost_lane_share"] = (t["station_days_lost_incomplete_lanes"] / t["station_days_any_member"]).round(3)
    t["outlier_share"] = (t["station_days_outlier"] / t["station_days_valid"]).round(3)
    keep = ["city", "continent", "include_main", "include_sensitivity", "reason_excluded", "n_valid_stations",
            "n_city_weekdays", "n_weekdays_nonholiday", "n_detectors_readings", "n_stations", "interval_s", "labelling",
            "unit_classes", "unit_median_ratio_top2", "valid_record_share", "error_share", "lost_lane_share",
            "outlier_share", "extent_area_km2", "extent_short_km", "events_ge_0p5", "hours_ge_0p5", "events_ge_1p0",
            "road_source", "lcc_length_share"]
    t = t[[c for c in keep if c in t]].sort_values(["include_main", "n_valid_stations"], ascending=[False, False])
    t.to_csv(out / "x0_summary.csv", index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.max_colwidth", 60)
    print(t.to_string(index=False))
    main_ = t[t["include_main"]]
    print(f"\nmain: {len(main_)} cities; by continent: {main_['continent'].value_counts().to_dict()}")
    print(f"sensitivity (N>=100): {int(t['include_sensitivity'].sum())} cities")
    print(f"main cities with >= 3 rain events (0.5 mm/h): {int((main_['events_ge_0p5'] >= 3).sum())}")
    with open(out / "x0_summary.json", "w", encoding="utf-8") as fh:
        json.dump(dict(main=main_["city"].tolist(), sensitivity=t.loc[t["include_sensitivity"], "city"].tolist(),
                       rain_ge3=main_.loc[main_["events_ge_0p5"] >= 3, "city"].tolist()), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
