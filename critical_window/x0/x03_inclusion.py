"""X0.3  City inclusion: valid stations >= N_min and city weekdays >= min_weekdays."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _int(v) -> int:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0
    return int(v) if np.isfinite(v) else 0


def inclusion(qc_city: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    n_main, n_sens, min_wd = int(cfg["n_min_main"]), int(cfg["n_min_sensitivity"]), int(cfg["min_weekdays"])
    for _, r in qc_city.iterrows():
        ns = _int(r.get("n_valid_stations", 0))
        nw = _int(r.get("n_city_weekdays", 0))
        status = str(r.get("status", "ok")) if pd.notna(r.get("status", "ok")) else "unknown"
        reasons = []
        if status != "ok":
            reasons.append(status)
        if nw < min_wd:
            reasons.append(f"{nw} weekdays < {min_wd}")
        if ns < n_main:
            reasons.append(f"{ns} stations < {n_main}")
        rows.append(dict(city=r["city"], n_valid_stations=ns, n_city_weekdays=nw,
                         include_main=(status == "ok" and ns >= n_main and nw >= min_wd),
                         include_sensitivity=(status == "ok" and ns >= n_sens and nw >= min_wd),
                         reason_excluded="; ".join(reasons)))
    return pd.DataFrame(rows).sort_values("city").reset_index(drop=True)
