"""X0.4 checks of the X0 reader against IUTFD's own hourly aggregates and a synthetic file.

Run from the repository root:  python -m critical_window.tests.test_x0_io
Cities without raw files are skipped.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.cfg import load_config, read_metadata, repo_path
from ..x0.io_iutfd import load_detectors, read_readings


def check_against_iutfd_hourly(city: str) -> str:
    cfg_all = load_config()
    cdir = repo_path(cfg_all["paths"]["iutfd_dir"]) / city
    hp = cdir / "sensors" / "hourly_readings.parquet"
    if not (cdir / "sensors" / "5min_readings.parquet").exists() or not hp.exists():
        return f"{city}: skipped (files missing)"
    det = load_detectors(cdir)
    R = read_readings(cdir, det, read_metadata(cdir), 1e12, 2_000_000)
    nH = R.hours.size
    ours = pd.DataFrame(dict(detid=np.array(R.detids, dtype=object)[R.key // nH],
                             t=R.hours[R.key % nH].astype("datetime64[ns]"),
                             n=R.n_valid, mean=R.sum_flow / R.n_valid))
    h = pd.read_parquet(hp)
    h["t"] = pd.to_datetime(h["datetime"], format="%d/%m/%Y %H:%M:%S")
    h["detid"] = h["detid"].astype(str)
    h["n_iutfd"] = (h["flow_sum"] / h["flow_mean_5min"]).round()
    full = 3600 // R.dt_s
    h = h[(h["error_mean"].fillna(0) == 0) & (h["n_iutfd"] == full)]
    m = h.merge(ours, on=["detid", "t"], how="inner")
    m = m[m["n"] == full]
    if R.labelling == "end":
        # IUTFD bins end-labelled stamps by floor(t): its hour h holds intervals starting h:00-5min ... h:55-5min,
        # so exact agreement is not expected; report the share that agrees.
        agree = np.isclose(m["flow_mean_5min"], m["mean"]).mean() if len(m) else np.nan
        return f"{city}: end-labelled; {len(m)} complete hours, share identical to IUTFD {agree:.3f} (misaligned by design)"
    ok = np.allclose(m["flow_mean_5min"], m["mean"])
    assert ok, f"{city}: hourly means differ from IUTFD"
    return f"{city}: {len(m)} complete error-free detector-hours identical to IUTFD flow_mean_5min"


def check_synthetic() -> str:
    """Synthetic end-labelled 5-min file with known errors, gaps and a spike; checks every QC branch."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp) / "testcity"
        (cdir / "sensors").mkdir(parents=True)
        ts = pd.date_range("2016-10-24 00:05", "2016-10-26 00:00", freq="5min")
        rows = []
        for d, lanes in (("A_1", 1), ("A_2", 1), ("B_1", 2)):
            for i, t in enumerate(ts):
                s = t - pd.Timedelta(minutes=5)                       # interval start (stamps are end-labelled)
                flow = 120.0 * (1 + (d == "B_1"))                       # constant 120 or 240 veh/h
                err = 0.0
                if d == "A_2" and s.hour == 8:
                    err = 1.0                                         # whole hour flagged -> hour invalid
                if d == "B_1" and s.hour == 9 and s.minute < 20:
                    continue                                          # 4 missing of 12 -> 8 < 9 -> invalid
                if d == "B_1" and s.hour == 10 and s.minute == 30:
                    flow = 99999.0                                    # spike above cap -> dropped, hour valid
                rows.append((t.strftime("%d/%m/%Y %H:%M:%S"), d, flow, err))
        df = pd.DataFrame(rows, columns=["datetime", "detid", "flow", "error"])
        pq.write_table(pa.Table.from_pandas(df), cdir / "sensors" / "5min_readings.parquet")
        det = pd.DataFrame(dict(detid=["A_1", "A_2", "B_1"], long=[7.0, 7.00005, 7.01], lat=[47.0, 47.0, 47.0],
                                lanes=[1.0, 1.0, 2.0], fclass=["primary"] * 3))
        pq.write_table(pa.Table.from_pandas(det), cdir / "sensors" / "detectors_info.parquet")
        R = read_readings(cdir, load_detectors(cdir), {}, 3600.0, 1000)
        assert R.dt_s == 300 and R.labelling == "end", (R.dt_s, R.labelling)
        nH = R.hours.size
        tab = pd.DataFrame(dict(det=np.array(R.detids)[R.key // nH], hour=pd.DatetimeIndex(R.hours[R.key % nH]).hour,
                                n=R.n_valid, mean=R.sum_flow / R.n_valid))
        a2_8 = tab[(tab.det == "A_2") & (tab.hour == 8)]
        assert a2_8.empty, "flagged hour should have no valid records"
        b1_9 = tab[(tab.det == "B_1") & (tab.hour == 9)]
        assert (b1_9.n == 8).all(), b1_9
        b1_10 = tab[(tab.det == "B_1") & (tab.hour == 10)]
        assert (b1_10.n == 11).all() and np.allclose(b1_10["mean"], 240.0), b1_10
        assert R.qc["n_above_cap"] == 2 and R.qc["n_error_flag"] == 24
        full = tab[(tab.det == "A_1")]
        assert (full.n == 12).all() and np.allclose(full["mean"], 120.0)
    return "synthetic: interval, labelling, error flag, gap and cap rules behave as specified"


def main() -> int:
    msgs = [check_synthetic()]
    for c in ("bremen", "basel", "london"):
        msgs.append(check_against_iutfd_hourly(c))
    print("\n".join(msgs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
