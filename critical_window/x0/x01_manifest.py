"""X0.1  Data manifest: every raw file the pipeline reads, with size and SHA-256."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

USED_PATTERNS = ("sensors/5min_readings.parquet", "sensors/detectors_info.parquet", "_metadata.json",
                 "weather/grid_info.parquet", "weather/datetime/", "roads/")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def manifest(iutfd_dir: Path, cities: list, do_hash: bool = True) -> pd.DataFrame:
    rows = []
    for city in cities:
        cdir = iutfd_dir / city
        if not cdir.exists():
            continue
        for p in sorted(cdir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(iutfd_dir).as_posix()
            used = any(pat in rel for pat in USED_PATTERNS)
            rows.append(dict(city=city, file=rel, bytes=p.stat().st_size, used=used,
                             sha256=sha256(p) if (do_hash and used) else ""))
    return pd.DataFrame(rows)
