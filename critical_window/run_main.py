"""Run the main analysis X1-X9 and the cross-city summary.

    python -m critical_window.run_main                     # all included cities, full replicate counts
    python -m critical_window.run_main --dry-run --quick   # permuted flows, reduced counts (code and timing check)

Reads critical_window/out/x0/ (or --x0 DIR); writes critical_window/out/main/ (dry run: out/dryrun/).
The dry run permutes whole station series among stations before anything is computed, so no real result is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .common.cfg import PKG_DIR, load_config, repo_path
from .main.city_stage import city_stage
from .main.cross_city import cross_city
from .main.data import load_city, permuted

FULL = dict(n_perm=2000, n_perm_exponents=200, n_n2=200, n_n3=200, n_boot=1000, n_boot_exponents=200, n_boot_xi=1000)
QUICK = dict(n_perm=200, n_perm_exponents=20, n_n2=50, n_n3=50, n_boot=100, n_boot_exponents=20, n_boot_xi=50)


def _code_fingerprint() -> dict:
    out = {}
    for p in sorted(PKG_DIR.rglob("*.py")) + [PKG_DIR / "config.yaml"]:
        if "out" in p.parts or "__pycache__" in p.parts:
            continue
        out[p.relative_to(PKG_DIR).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def _job(args):
    city, x0_dir, m, out_dir, tag, counts, dry = args
    t = time.time()
    try:
        cd = load_city(city, Path(x0_dir), m)
        if dry:
            cd = permuted(cd, np.random.default_rng(zlib.crc32(f"dryrun|{city}".encode())))
        r = city_stage(cd, m, Path(out_dir), tag, counts)
        return city, "ok", round(time.time() - t, 1), r["n_stations"]
    except Exception as exc:  # keep other cities running
        import traceback
        return city, f"error: {exc}\n{traceback.format_exc()}", round(time.time() - t, 1), 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", default="", help="subset (default: include_main cities from X0)")
    ap.add_argument("--x0", default="", help="X0 output folder (default critical_window/out/x0)")
    ap.add_argument("--dry-run", action="store_true", help="permute station series first (no real results)")
    ap.add_argument("--quick", action="store_true", help="reduced replicate counts")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-cities", action="store_true", help="only rerun the cross-city stage")
    ap.add_argument("--out-tag", default="", help="output folder name under out/ (default: main, or dryrun)")
    args = ap.parse_args(argv)

    cfg_all = load_config()
    m = cfg_all["main"]
    x0_dir = Path(args.x0) if args.x0 else repo_path(cfg_all["paths"]["out_dir"]) / "x0"
    tag = "dryrun" if args.dry_run else "main"
    out_dir = repo_path(cfg_all["paths"]["out_dir"]) / (args.out_tag or (tag + ("_quick" if args.quick else "")))
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = QUICK if args.quick else FULL
    inc = pd.read_csv(x0_dir / "inclusion.csv")
    cities = [c.strip() for c in args.cities.split(",") if c.strip()] or inc.loc[inc["include_main"], "city"].tolist()
    log = open(out_dir / "run_log.txt", "a", encoding="utf-8")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"run_main {tag} | counts {counts} | {len(cities)} cities | python {platform.python_version()} numpy {np.__version__}")
    with open(out_dir / "code_fingerprint.json", "w", encoding="utf-8") as fh:
        json.dump(_code_fingerprint(), fh, indent=1)
    if not args.skip_cities:
        jobs = [(c, str(x0_dir), m, str(out_dir), tag, counts, args.dry_run) for c in cities]
        big_first = sorted(jobs, key=lambda j: -len(pd.read_csv(x0_dir / "per_city" / j[0] / "stations.csv")))
        with Pool(max(1, args.workers)) as pool:
            for city, status, secs, n in pool.imap_unordered(_job, big_first):
                say(f"{city}: {status.splitlines()[0]} ({n} stations, {secs} s)")
                if status != "ok":
                    log.write(status + "\n")
    done = [c for c in cities if (out_dir / c / "city.json").exists()]
    city_data = {c: load_city(c, x0_dir, m) for c in done}
    if args.dry_run:
        city_data = {c: permuted(cd, np.random.default_rng(zlib.crc32(f"dryrun|{c}".encode()))) for c, cd in city_data.items()}
    m2 = dict(m)
    if args.quick:
        m2["rank_reps"] = 100
    summary = cross_city(out_dir, done, m2, city_data=city_data, tag=tag)
    say("H1 / H2 shares: " + "; ".join(f"{p}: H1 {v['h1_share']}, H2 {v['h2_share']}"
                                        for p, v in summary["tests"]["per_probe"].items()))
    say(f"done -> {out_dir}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
