"""Smoke test: run the pipeline on Melbourne, one window, one hour.

Designed to finish in well under a minute on a laptop.  It checks that:

* The pipeline imports and runs end-to-end.
* alpha and beta are finite for at least one (date, window, hour).
* The cross-city CSV is written.
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from main import (  # noqa: E402
    load_config, stage_aggregate, stage_compute, stage_fit, stage_load,
)


def test_smoke_melbourne(tmp_path: Path | None = None) -> None:
    """End-to-end smoke test on a single Melbourne date and window."""
    cfg = load_config(ROOT / "config.yaml")
    # Pin to a small subset so the test stays fast.
    cfg["grid_sizes"] = [100, 200, 400, 800, 1000]
    cfg["window_scale_factors"] = [1.0]
    cfg["converged_window_indices"] = [0]
    cfg["workday_hours"] = [10]
    cfg["cities"] = {"melbourne": cfg["cities"]["melbourne"]}
    cfg["cities"]["melbourne"]["dates"] = None

    out = Path(tempfile.mkdtemp(prefix="clean_code_smoke_"))
    try:
        ctx = stage_load(cfg, ["melbourne"], ROOT / "data" / "input", out)
        assert "melbourne" in ctx
        # Use only the first date to keep it fast.
        ctx["melbourne"]["dates"] = ctx["melbourne"]["dates"][:1]

        stage_compute(cfg, ctx, out)
        stage_fit(cfg, ctx, out)
        stage_aggregate(cfg, ctx, out)

        csv_path = out / "cross_city_summary.csv"
        assert csv_path.exists(), "aggregate CSV not produced"
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, "aggregate CSV is empty"
        row = rows[0]
        assert row["city"] == "melbourne"
        assert np.isfinite(float(row["beta_median"])), "beta_median not finite"
        print(f"OK: alpha_median={row['alpha_median']}, "
              f"beta_median={row['beta_median']}")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    test_smoke_melbourne()
    print("Smoke test passed.")
