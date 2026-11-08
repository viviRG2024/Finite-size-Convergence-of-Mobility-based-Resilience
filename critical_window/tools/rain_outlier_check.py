"""Rainfall analysis (threshold 0.5 mm/h) with the per-station outlier rule switched off: the station-days flagged by the
rule are kept. Used to check that the rainfall results do not depend on the outlier rule.

    python -m critical_window.tools.rain_outlier_check [--x0 out/x0] [--raw data/input/IUTFD]

Runs tools/e10_rain.py (run and summary) unchanged except for the station-days used; writes
out/e10/thr0p5_no_outlier_rule/ with the same files as out/e10/thr0p5/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..common.cfg import load_config, repo_path
from ..main import data as D
from . import e10_rain as E


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x0", default="")
    ap.add_argument("--raw", default="data/input/IUTFD")
    ap.add_argument("--main", default="main34")
    a = ap.parse_args(argv)
    out_root = repo_path(load_config()["paths"]["out_dir"])
    od = out_root / "e10" / "thr0p5_no_outlier_rule"
    x0 = a.x0 or str(out_root / "x0")
    load_city_default = D.load_city

    def load_city_keep(city, x0_dir, cfg, daily_all=None):
        d = pd.read_parquet(Path(x0_dir) / "per_city" / city / "station_daily.parquet")
        d["outlier"] = False
        d["city"] = city
        return load_city_default(city, x0_dir, cfg, daily_all=d)

    E.load_city = load_city_keep
    E.outdir = lambda _a: (out_root, od)
    common = ["--x0", x0, "--weather", a.raw, "--raw", a.raw, "--main", a.main]
    E.main(["run"] + common)
    E.main(["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
