"""Streaming reader for IUTFD detector readings (sensors/5min_readings.parquet).

Despite the file name, the native interval differs by city (3, 5 or 15 min, or 1 h). Flow values are
rates in vehicles per hour (UTD19 convention: count x 3600 / interval); this is checked per city.
The reader never keeps all records in memory: it streams record batches and aggregates them to
detector-hour sums and counts of valid records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TS_FORMATS = ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S",
              "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S")
STD_INTERVALS = np.array([60, 120, 180, 300, 600, 900, 1800, 3600])


def _index_in(arr, value_set):
    try:
        out = pc.index_in(arr, value_set=value_set)
    except TypeError:                                   # older pyarrow
        out = pc.index_in(arr, options=pc.SetLookupOptions(value_set=value_set))
    out = pc.fill_null(out, -1)
    if isinstance(out, pa.ChunkedArray):
        return np.concatenate([c.to_numpy(zero_copy_only=False) for c in out.chunks]) if out.num_chunks else np.zeros(0, np.int64)
    return out.to_numpy(zero_copy_only=False)


def _to_float(arr) -> np.ndarray:
    arr = pc.cast(arr, pa.float64())
    if isinstance(arr, pa.ChunkedArray):
        if arr.num_chunks == 0:
            return np.zeros(0)
        return np.concatenate([c.to_numpy(zero_copy_only=False) for c in arr.chunks])
    return arr.to_numpy(zero_copy_only=False)


def parse_timestamps(values: list, t_start=None, t_end=None):
    """Parse unique timestamp strings; choose the first format that parses all values and fits the metadata range."""
    s = pd.Series(values, dtype="object")
    fallback = None
    for fmt in TS_FORMATS:
        t = pd.to_datetime(s, format=fmt, errors="coerce")
        if t.isna().any():
            continue
        if t_start is not None and t_end is not None:
            if t.min() < t_start - pd.Timedelta(days=2) or t.max() > t_end + pd.Timedelta(days=3):
                fallback = fallback or (t, fmt)
                continue
        return t.values.astype("datetime64[ns]"), fmt
    if fallback is not None:
        return fallback[0].values.astype("datetime64[ns]"), fallback[1] + " (outside metadata range)"
    t = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return t.values.astype("datetime64[ns]"), "inferred(dayfirst)"


def detect_interval(ts_sorted: np.ndarray) -> int:
    d = np.diff(ts_sorted).astype("timedelta64[s]").astype(np.int64)
    d = d[d > 0]
    if d.size == 0:
        return 3600
    mode = np.bincount(np.clip(d, 0, 7200)).argmax()
    return int(STD_INTERVALS[np.argmin(np.abs(STD_INTERVALS - mode))])


def detect_labelling(ts_sorted: np.ndarray, dt_s: int) -> str:
    """Whether stamps mark the start or the end of each interval, voted over contiguous blocks of data.

    A block that begins at 00:00 or ends at 24:00 - dt votes 'start'; a block that begins at 00:00 + dt or
    ends at 00:00 (or 23:59:59) votes 'end'. Blocks that begin and end mid-day carry no information; if no
    block votes, or the votes tie, the result is 'start?' (start assumed; the choice shifts hours by one interval).
    """
    if ts_sorted.size == 0:
        return "start?"
    sec = ts_sorted.astype("datetime64[s]").astype(np.int64)
    gaps = np.flatnonzero(np.diff(sec) > dt_s)
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps, sec.size - 1]
    tod = np.mod(sec, 86400)
    tod = np.where(tod >= 86400 - 1, 0, tod)          # 23:59:59 written for midnight
    v_start = v_end = 0
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        s, e = tod[a], tod[b]
        v_start += int(s == 0) + int(e == 86400 - dt_s)
        v_end += int(s == dt_s) + int(e == 0)
    if v_start > v_end:
        return "start"
    if v_end > v_start:
        return "end"
    return "start?"


def load_detectors(city_dir: Path) -> pd.DataFrame:
    p = city_dir / "sensors" / "detectors_info.parquet"
    names = pq.ParquetFile(p).schema_arrow.names
    cols = [c for c in ("detid", "long", "lon", "longitude", "lat", "latitude", "fclass", "lanes", "road",
                        "limit", "geometry") if c in names]
    det = pq.read_table(p, columns=cols).to_pandas()
    det["detid"] = det["detid"].astype(str)
    if "long" not in det:
        for alt in ("lon", "longitude"):
            if alt in det:
                det["long"] = det[alt]
                break
    if "lat" not in det and "latitude" in det:
        det["lat"] = det["latitude"]
    if ("long" not in det or "lat" not in det) and "geometry" in det:
        from ..common.geo import wkb_parts
        xy = np.array([wkb_parts(g)[1][0][0] if g is not None else (np.nan, np.nan) for g in det["geometry"]], float)
        det["long"], det["lat"] = xy[:, 0], xy[:, 1]
    for c in ("long", "lat"):
        det[c] = pd.to_numeric(det.get(c, np.nan), errors="coerce")
    det = det.drop(columns=[c for c in ("geometry",) if c in det]).drop_duplicates("detid")
    if "lanes" not in det:
        det["lanes"] = np.nan
    det["lanes"] = pd.to_numeric(det["lanes"], errors="coerce")
    return det.reset_index(drop=True)


@dataclass
class Readings:
    city: str
    detids: list                      # detector id per detector code
    hours: np.ndarray                 # datetime64 start of each hour code
    dt_s: int                         # native interval (s)
    labelling: str                    # 'start' or 'end' (of interval)
    ts_format: str
    key: np.ndarray                   # detector code * n_hours + hour code (sorted, unique)
    n_valid: np.ndarray               # valid records in that detector-hour
    sum_flow: np.ndarray              # sum of valid flow rates (veh/h)
    qc: dict = field(default_factory=dict)
    det_units: dict = field(default_factory=dict)   # per detector: counts of positive / integer / multiple-of-k values


def read_readings(city_dir: Path, det: pd.DataFrame, meta: dict, cap_per_lane: float, batch_rows: int) -> Readings:
    path = city_dir / "sensors" / "5min_readings.parquet"
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    tr = meta.get("time_range", {}) if meta else {}
    t0 = pd.Timestamp(tr["start"]) if tr.get("start") else None
    t1 = pd.Timestamp(tr["end"]) + pd.Timedelta(days=1) if tr.get("end") else None

    # ---- pass 1: all distinct timestamps -> format, interval, labelling, hour codes
    ts_type = pf.schema_arrow.field("datetime").type
    uniq: dict = {}
    for b in pf.iter_batches(batch_size=batch_rows, columns=["datetime"]):
        col = b.column(0)
        for v in pc.unique(col).to_pylist():
            if v is not None and v not in uniq:
                uniq[v] = len(uniq)
    values = list(uniq.keys())
    if pa.types.is_timestamp(ts_type):
        ts = np.array(values, dtype="datetime64[ns]")
        fmt = "timestamp"
    else:
        ts, fmt = parse_timestamps([str(v) for v in values], t0, t1)
    ok_ts = ~np.isnat(ts)
    order = np.argsort(ts[ok_ts])
    ts_sorted = ts[ok_ts][order]
    dt_s = detect_interval(ts_sorted)
    lab = detect_labelling(ts_sorted, dt_s)
    start = ts - np.timedelta64(dt_s, "s") if lab == "end" else ts
    hour_start = start.astype("datetime64[h]")
    hours, hour_of_ts = np.unique(hour_start[ok_ts], return_inverse=True)
    hcode = np.full(ts.size, -1, np.int64)
    hcode[np.flatnonzero(ok_ts)] = hour_of_ts
    value_set = pa.array(values, type=ts_type if pa.types.is_timestamp(ts_type) else pa.string())
    nH = hours.size

    # ---- detector attributes for the plausibility cap
    lanes = det.set_index("detid")["lanes"].to_dict()
    det_code: dict = {}
    det_cap: list = []
    det_in_info: list = []

    keys, sums, cnts = [], [], []
    UNIT_K = (4, 12, 20, 60)
    unit_acc = {name: np.zeros(0, np.int64) for name in ["n_pos", "n_int"] + [f"n_m{k}" for k in UNIT_K]}
    qc = dict(n_records=0, n_ts_unparsed=0, n_flow_missing=0, n_negative=0, n_error_flag=0,
              n_above_cap=0, n_unknown_detector=0, n_valid=0, n_pos=0, n_pos_multiple=0)
    step = 3600 // dt_s if dt_s <= 3600 else 1
    cols = [c for c in ("datetime", "detid", "flow", "error") if c in names]
    for b in pf.iter_batches(batch_size=batch_rows, columns=cols):
        n = b.num_rows
        qc["n_records"] += n
        tcode = _index_in(b.column(cols.index("datetime")), value_set)
        h = np.where(tcode >= 0, hcode[np.maximum(tcode, 0)], -1)
        dcol = b.column(cols.index("detid"))
        if not pa.types.is_string(dcol.type):
            dcol = pc.cast(dcol, pa.string())
        loc_u = pc.unique(dcol)
        loc_codes = _index_in(dcol, loc_u)
        g = np.empty(len(loc_u), np.int64)
        for i, name in enumerate(loc_u.to_pylist()):
            name = str(name)
            if name not in det_code:
                det_code[name] = len(det_code)
                ln = lanes.get(name, np.nan)
                try:
                    ln = float(ln)
                except (TypeError, ValueError):
                    ln = np.nan
                det_cap.append(cap_per_lane * (ln if np.isfinite(ln) and ln >= 1 else 1.0))
                det_in_info.append(name in lanes)
            g[i] = det_code[name]
        d = g[loc_codes]
        flow = _to_float(b.column(cols.index("flow")))
        err = _to_float(b.column(cols.index("error"))) if "error" in cols else np.zeros(n)
        cap = np.asarray(det_cap)[d]
        in_info = np.asarray(det_in_info, dtype=bool)[d]
        m_ts = h >= 0
        m_nan = ~np.isfinite(flow)
        m_neg = flow < 0
        m_err = np.nan_to_num(err, nan=0.0) != 0
        m_cap = flow > cap
        valid = m_ts & ~m_nan & ~m_neg & ~m_err & ~m_cap & in_info
        qc["n_ts_unparsed"] += int((~m_ts).sum())
        qc["n_flow_missing"] += int(m_nan.sum())
        qc["n_negative"] += int((m_neg & ~m_nan).sum())
        qc["n_error_flag"] += int((m_err & ~m_nan).sum())
        qc["n_above_cap"] += int((m_cap & ~m_err & ~m_nan).sum())
        qc["n_unknown_detector"] += int((~in_info).sum())
        qc["n_valid"] += int(valid.sum())
        pos = flow[valid & (flow > 0)]
        qc["n_pos"] += int(pos.size)
        qc["n_pos_multiple"] += int(np.sum(np.isclose(np.mod(pos, step), 0) | np.isclose(np.mod(pos, step), step)))
        # per-detector unit fingerprints: integer values and multiples of 4, 12, 20, 60 among positive values
        vp = valid & (flow > 0)
        dp = d[vp]
        fv = flow[vp]
        ri = np.round(fv)
        is_int = np.isclose(fv, ri)
        nd = len(det_code)
        for name in unit_acc:
            if unit_acc[name].size < nd:
                unit_acc[name] = np.r_[unit_acc[name], np.zeros(nd - unit_acc[name].size, np.int64)]
        unit_acc["n_pos"] += np.bincount(dp, minlength=nd)
        unit_acc["n_int"] += np.bincount(dp[is_int], minlength=nd)
        for kk in UNIT_K:
            m = is_int & (np.mod(ri, kk) == 0)
            unit_acc[f"n_m{kk}"] += np.bincount(dp[m], minlength=nd)
        k = d[valid].astype(np.int64) * nH + h[valid]
        u, inv = np.unique(k, return_inverse=True)
        keys.append(u)
        sums.append(np.bincount(inv, weights=flow[valid], minlength=u.size))
        cnts.append(np.bincount(inv, minlength=u.size))

    if keys:
        k = np.concatenate(keys)
        u, inv = np.unique(k, return_inverse=True)
        s = np.bincount(inv, weights=np.concatenate(sums), minlength=u.size)
        c = np.bincount(inv, weights=np.concatenate(cnts), minlength=u.size).astype(np.int64)
    else:
        u, s, c = np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
    qc.update(interval_s=dt_s, labelling=lab, ts_format=fmt, n_timestamps=int(ts.size),
              first_ts=str(ts_sorted[0]) if ts_sorted.size else "", last_ts=str(ts_sorted[-1]) if ts_sorted.size else "",
              n_detectors_readings=len(det_code), n_detectors_info=len(lanes),
              n_detectors_without_info=int(sum(1 for k2 in det_code if k2 not in lanes)))
    detids = [None] * len(det_code)
    for name, code in det_code.items():
        detids[code] = name
    nd = len(det_code)
    for name in unit_acc:
        if unit_acc[name].size < nd:
            unit_acc[name] = np.r_[unit_acc[name], np.zeros(nd - unit_acc[name].size, np.int64)]
    return Readings(city=city_dir.name, detids=detids, hours=hours, dt_s=dt_s, labelling=lab, ts_format=fmt,
                    key=u, n_valid=c, sum_flow=s, qc=qc, det_units=unit_acc)
