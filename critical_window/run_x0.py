"""Run the X0 stage (data preparation and quality control) for all IUTFD cities.

    cd <repo root>            (the folder that contains data/ and critical_window/)
    python -m critical_window.run_x0                 # all cities
    python -m critical_window.run_x0 --cities basel,bremen --no-hash

Needs Python >= 3.8 with numpy, pandas and pyarrow. Results go to critical_window/out/x0/.
Cities already processed are skipped unless --force is given, so an interrupted run can be restarted.
X0 computes no diagnostic, window or spatial statistic.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .common.cfg import city_table, holidays_by_city, load_config, read_metadata, repo_path
from .x0.x01_manifest import manifest
from .x0.x02_stations import process_city
from .x0.x03_inclusion import inclusion
from .x0.x05_inventory import network_inventory, rain_inventory


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", default="", help="comma-separated subset (default: all cities in data/cities.csv)")
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 of the raw files")
    ap.add_argument("--force", action="store_true", help="reprocess cities that already have results")
    ap.add_argument("--no-manifest", action="store_true", help="keep the existing manifest.csv (skip X0.1)")
    ap.add_argument("--config", default="", help="alternative config.yaml")
    args = ap.parse_args(argv)

    cfg_all = load_config(Path(args.config) if args.config else None)
    cfg = cfg_all["x0"]
    iutfd = repo_path(cfg_all["paths"]["iutfd_dir"])
    out = repo_path(cfg_all["paths"]["out_dir"]) / "x0"
    (out / "per_city").mkdir(parents=True, exist_ok=True)
    log_path = out / "run_log.txt"
    log = open(log_path, "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    import pyarrow
    say(f"X0 start | python {platform.python_version()} numpy {np.__version__} pandas {pd.__version__} "
        f"pyarrow {pyarrow.__version__} | {platform.platform()}")
    say(f"IUTFD folder: {iutfd}")
    if not iutfd.exists():
        say("ERROR: IUTFD folder not found; run from the repository root or fix paths.iutfd_dir in config.yaml")
        return 2

    table = city_table()
    hol = holidays_by_city()
    cities = [c.strip() for c in args.cities.split(",") if c.strip()] or sorted(table)
    cities = [c for c in cities if (iutfd / c).exists()]
    say(f"{len(cities)} cities: {', '.join(cities)}")

    # ---- X0.1 manifest
    t = time.time()
    if args.no_manifest and (out / "manifest.csv").exists():
        say("X0.1 manifest: kept existing manifest.csv")
    else:
        _write_manifest(iutfd, cities, out, not args.no_hash, say, t)

    # ---- X0.2 + X0.5 per city
    return _cities(args, cfg, iutfd, out, cities, table, hol, say, log)


def _write_manifest(iutfd, cities, out, do_hash, say, t):
    man = manifest(iutfd, cities, do_hash=do_hash)
    man.to_csv(out / "manifest.csv", index=False)
    with open(out / "checksums.sha256", "w", encoding="utf-8") as fh:
        for _, r in man[man["sha256"] != ""].iterrows():
            fh.write(f"{r['sha256']}  {r['file']}\n")
    say(f"X0.1 manifest: {len(man)} files, {man['bytes'].sum() / 1e9:.2f} GB, hashed {int((man['sha256'] != '').sum())} "
        f"({time.time() - t:.0f} s)")


def _cities(args, cfg, iutfd, out, cities, table, hol, say, log) -> int:
    for city in cities:
        cdir = iutfd / city
        pdir = out / "per_city" / city
        if (pdir / "qc.json").exists() and not args.force:
            say(f"{city}: already done, skipped")
            continue
        pdir.mkdir(parents=True, exist_ok=True)
        t = time.time()
        meta = read_metadata(cdir)
        qc = dict(city=city)
        try:
            if not (cdir / "sensors" / "5min_readings.parquet").exists():
                qc["status"] = "no traffic readings"
                res = None
            else:
                res = process_city(cdir, meta, cfg, hol.get(city, set()))
                qc.update(res["qc"])
                qc["status"] = "ok" if len(res["stations"]) else "no valid stations"
                res["stations"].to_csv(pdir / "stations.csv", index=False)
                res["detectors"].to_csv(pdir / "detectors.csv", index=False)
                res["days"].to_csv(pdir / "days.csv", index=False)
                res["daily"].to_parquet(pdir / "station_daily.parquet", index=False)
                res["hourly"].to_parquet(pdir / "station_hourly.parquet", index=False)
        except Exception as exc:
            qc["status"] = f"error: {type(exc).__name__}: {exc}"
            res = None
            say(f"{city}: ERROR\n{traceback.format_exc()}")
        # X0.5 inventories
        ext = {k: v for k, v in qc.items() if k.startswith("extent_")}
        days = res["days"] if res is not None else pd.DataFrame()
        years = sorted({d[:4] for d in days["date"]}) if len(days) else []
        try:
            rain = rain_inventory(cdir, days, ext, cfg)
        except Exception as exc:
            rain = dict(city=city, status=f"error: {exc}")
            say(f"{city}: rain inventory error\n{traceback.format_exc()}")
        try:
            net = network_inventory(cdir, ext, qc.get("epsg"), years, cfg)
        except Exception as exc:
            net = dict(city=city, status=f"error: {exc}")
            say(f"{city}: network inventory error\n{traceback.format_exc()}")
        qc["seconds"] = round(time.time() - t, 1)
        for name, obj in (("rain.json", rain), ("network.json", net), ("qc.json", qc)):
            with open(pdir / name, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=1, default=_json_default)
        say(f"{city}: {qc.get('status')} | interval {qc.get('interval_s')} s ({qc.get('labelling')}) | "
            f"stations {qc.get('n_stations', 0)} valid {qc.get('n_valid_stations', 0)} | "
            f"weekdays {qc.get('n_city_weekdays', 0)} | {qc['seconds']} s")

    # ---- combine
    qcs, rains, nets, dailies = [], [], [], []
    for city in cities:
        pdir = out / "per_city" / city
        if not (pdir / "qc.json").exists():
            continue
        with open(pdir / "qc.json", encoding="utf-8") as fh:
            qcs.append(json.load(fh))
        for name, bucket in (("rain.json", rains), ("network.json", nets)):
            if (pdir / name).exists():
                with open(pdir / name, encoding="utf-8") as fh:
                    bucket.append(json.load(fh))
        if (pdir / "station_daily.parquet").exists():
            dailies.append(pd.read_parquet(pdir / "station_daily.parquet"))
    qc_city = pd.DataFrame(qcs)
    qc_city.to_csv(out / "qc_city.csv", index=False)
    pd.DataFrame(rains).to_csv(out / "rain_inventory.csv", index=False)
    pd.DataFrame(nets).to_csv(out / "network_inventory.csv", index=False)
    if dailies:
        pd.concat(dailies, ignore_index=True).to_parquet(out / "station_daily_all.parquet", index=False)
    inc = inclusion(qc_city, cfg)
    inc.to_csv(out / "inclusion.csv", index=False)
    say(f"X0.3 inclusion: main {int(inc['include_main'].sum())} cities, "
        f"sensitivity (N>={cfg['n_min_sensitivity']}) {int(inc['include_sensitivity'].sum())} cities")
    say(f"done -> {out}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
