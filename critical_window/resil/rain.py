"""X10  Rain performance loss and its critical observation window.

Implementation choices:
  - data block: runs of city weekdays separated by more than 14 days;
  - dry city weekday: every hour 07:00-19:59 below 0.2 mm/h (the lowest X0 threshold);
  - an event spans its first to last rain hour (rain hours merged across gaps <= 1 h), clipped to 07:00-19:59 of a
    city weekday; a station enters an event only if all event hours and its baseline for them are available;
  - loss of a station set S: L = 1 - sum(event flow) / sum(baseline flow) over S and the event hours;
  - disturbance threshold: 95th percentile of the whole-extent L of pseudo-events (same hours on every dry day of the
    block); recovery: hours after the event until the whole-extent hourly flow is within +-5% of baseline, censored at 20:00.
Windows are the random windows of the main run (same seeds), usable with >= N_w stations.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..core import criteria, windows
from ..main.data import CityData
from ..x0.x05_inventory import _cell_boxes

H0, H1 = 7, 20
NH = H1 - H0


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def city_rain_series(weather_dir: Path, qc: dict, dates) -> pd.Series:
    """Overlap-weighted ERA5 precipitation (mm/h) indexed by the START of the traffic hour (local time)."""
    grid = pq.read_table(weather_dir / "grid_info.parquet").to_pandas()
    boxes = _cell_boxes(grid)
    e = (qc["extent_lon_min"], qc["extent_lat_min"], qc["extent_lon_max"], qc["extent_lat_max"])
    ow = np.clip(np.minimum(boxes[:, 2], e[2]) - np.maximum(boxes[:, 0], e[0]), 0, None)
    oh = np.clip(np.minimum(boxes[:, 3], e[3]) - np.maximum(boxes[:, 1], e[1]), 0, None)
    w = ow * oh
    if w.sum() <= 0:
        cx, cy = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
        w = np.zeros(len(grid))
        w[np.argmin((grid["longitude"] - cx) ** 2 + (grid["latitude"] - cy) ** 2)] = 1.0
    wmap = dict(zip(grid["grid_id"].tolist(), w.tolist()))
    dates = pd.to_datetime(pd.Series(list(dates)))
    want = set(dates.dt.strftime("%Y-%m-%d")) | set((dates + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"))
    frames = [pq.read_table(weather_dir / "datetime" / f"local_hourly_rainfall_{d}.parquet",
                            columns=["grid_id", "total_precipitation", "local_time"]).to_pandas()
              for d in sorted(want) if (weather_dir / "datetime" / f"local_hourly_rainfall_{d}.parquet").exists()]
    if not frames:
        return pd.Series(dtype=float)
    r = pd.concat(frames, ignore_index=True)
    r["w"] = r["grid_id"].map(wmap).fillna(0.0)
    r = r[r["w"] > 0]
    r["mm_w"] = r["total_precipitation"].astype(float) * 1000.0 * r["w"]
    s = r.groupby("local_time").agg(mm_w=("mm_w", "sum"), w=("w", "sum"))
    mmh = (s["mm_w"] / s["w"]).sort_index()
    idx = pd.DatetimeIndex(mmh.index) - pd.Timedelta(hours=1)
    out = pd.Series(mmh.values, index=idx)
    return out[~out.index.duplicated()]


def hourly_cube(cd: CityData, hourly: pd.DataFrame) -> np.ndarray:
    """Flows (veh/h) as stations x city weekdays x 13 daytime hours; NaN where missing or the station-day is not clean."""
    ts = pd.to_datetime(hourly["hour_start"])
    h = hourly.assign(date=ts.dt.strftime("%Y-%m-%d").to_numpy(), hour=ts.dt.hour.to_numpy())
    h = h[(h.hour >= H0) & (h.hour < H1)]
    sidx = {s: i for i, s in enumerate(cd.station_ids)}
    didx = {d: j for j, d in enumerate(cd.days)}
    si = h.station_id.map(sidx)
    dj = h.date.map(didx)
    ok = si.notna().to_numpy() & dj.notna().to_numpy()
    si, dj = si[ok].astype(int).to_numpy(), dj[ok].astype(int).to_numpy()
    hh = h.hour.to_numpy()[ok] - H0
    v = h.flow_vehh.to_numpy(float)[ok]
    clean = np.isfinite(cd.X[si, dj])
    cube = np.full((cd.F.size, cd.days.size, NH), np.nan)
    cube[si[clean], dj[clean], hh[clean]] = v[clean]
    return cube


def blocks(days, gap_days: int = 14) -> np.ndarray:
    d = pd.to_datetime(pd.Series(list(days)))
    return np.r_[0, np.cumsum(d.diff().dt.days.to_numpy()[1:] > gap_days)].astype(int)


def rain_matrix(series: pd.Series, days) -> np.ndarray:
    """Rain (mm/h) as city weekdays x 13 daytime hours (NaN where ERA5 is missing)."""
    out = np.full((len(days), NH), np.nan)
    for j, d in enumerate(days):
        for k in range(NH):
            t = pd.Timestamp(d) + pd.Timedelta(hours=H0 + k)
            if t in series.index:
                out[j, k] = series[t]
    return out


def find_events(R: np.ndarray, thr: float, gap: int = 1) -> list:
    """(day index, first hour index, last hour index, max mm/h) of each event within 07:00-19:59."""
    ev = []
    for j in range(R.shape[0]):
        wet = np.flatnonzero(R[j] >= thr)
        if wet.size == 0:
            continue
        start = prev = wet[0]
        for k in wet[1:]:
            if k - prev > gap + 1:
                ev.append((j, start, prev, float(np.nanmax(R[j, start:prev + 1]))))
                start = k
            prev = k
        ev.append((j, start, prev, float(np.nanmax(R[j, start:prev + 1]))))
    return ev


def station_sums(cube, base, day, h0, h1):
    """Per-station event flow and baseline flow over hours h0..h1 of a day; valid if all present."""
    e = cube[:, day, h0:h1 + 1]
    b = base[:, h0:h1 + 1]
    v = np.isfinite(e).all(1) & np.isfinite(b).all(1)
    return np.where(v, e.sum(1), 0.0), np.where(v, b.sum(1), 0.0), v


def loss(M: np.ndarray, e: np.ndarray, b: np.ndarray) -> np.ndarray:
    """L for each row of the station mask M (windows x stations); e, b may carry extra columns (stations x K)."""
    num = M.astype(float) @ e
    den = M.astype(float) @ b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 1.0 - num / den, np.nan)


def city_rain(cd: CityData, hourly: pd.DataFrame, series: pd.Series, m: dict, thr: float = 0.5, n_perm: int = 200,
              dry_thr: float = 0.2) -> dict:
    """Events, whole-extent losses, pseudo-event threshold, recovery and per-level window errors for one city."""
    cube = hourly_cube(cd, hourly)
    R = rain_matrix(series, cd.days)
    blk = blocks(cd.days)
    dry = np.isfinite(R).all(1) & (np.nanmax(np.where(np.isfinite(R), R, 0), 1) < dry_thr)
    base = {}
    for b_ in np.unique(blk):
        dd = np.flatnonzero((blk == b_) & dry)
        base[b_] = np.nanmedian(cube[:, dd, :], axis=1) if dd.size else np.full((cd.F.size, NH), np.nan)
    evs = find_events(R, thr)
    rows = []
    E_all = np.ones((1, cd.F.size), bool)
    for (j, h0, h1, mx) in evs:
        b_ = blk[j]
        e, bb, v = station_sums(cube, base[b_], j, h0, h1)
        if v.sum() < int(m["n_w"]):
            continue
        L_E = float(loss(E_all, e[:, None], bb[:, None])[0, 0])
        # pseudo-events: same hours on the dry days of the block (other days)
        pseudo = []
        for jj in np.flatnonzero((blk == b_) & dry):
            if jj == j:
                continue
            pe, pb, pv = station_sums(cube, base[b_], jj, h0, h1)
            if pv.sum() >= int(m["n_w"]):
                pseudo.append(float(loss(E_all, pe[:, None], pb[:, None])[0, 0]))
        # recovery after the event (whole extent, hour by hour)
        rec, censored = np.nan, True
        for k in range(h1 + 1, NH):
            fk, bk = cube[:, j, k], base[b_][:, k]
            ok = np.isfinite(fk) & np.isfinite(bk)
            if ok.sum() >= int(m["n_w"]) and abs(1 - fk[ok].sum() / bk[ok].sum()) <= 0.05:
                rec, censored = float(k - h1), False
                break
        rows.append(dict(day=str(cd.days[j]), block=int(b_), hour_start=H0 + h0, hour_end=H0 + h1 + 1, n_hours=int(h1 - h0 + 1),
                         max_mmh=mx, n_stations=int(v.sum()), L_E=L_E, n_pseudo=len(pseudo),
                         pseudo_p95=float(np.quantile(pseudo, 0.95)) if len(pseudo) >= 5 else np.nan,
                         pseudo_p05=float(np.quantile(pseudo, 0.05)) if len(pseudo) >= 5 else np.nan,
                         recovery_h=rec, recovery_censored=censored, _j=j, _h0=h0, _h1=h1))
    ev = pd.DataFrame(rows)
    out = dict(city=cd.city, n_events=len(ev), n_dry_days=int(dry.sum()), n_days=int(cd.days.size), events=ev)
    if len(ev) < 1:
        return out
    # station sums per event (stations x events), plus N1-style permutations of station identity (exploratory)
    e_mat = np.zeros((cd.F.size, len(ev)))
    b_mat = np.zeros((cd.F.size, len(ev)))
    for q, r in ev.iterrows():
        e_mat[:, q], b_mat[:, q], _ = station_sums(cube, base[r.block], r._j, r._h0, r._h1)
    fs = np.array(m["f_grid"], float)
    levels = fs[fs < 1.0]
    n_w, n_rand = int(m["n_w"]), int(m["n_random"])
    L_E = ev.L_E.to_numpy()
    prng = _rng(cd.city, "x10", "perm")
    perms = [prng.permutation(cd.F.size) for _ in range(n_perm)]
    lev, errs = [], {}
    for f in levels:
        wins = windows.random_windows(cd.extent, f, n_rand, _rng(cd.city, "main", "windows", f))   # same windows as run_main
        M = np.array([windows.in_box(cd.x, cd.y, w_) for w_ in wins])
        use = M.sum(1) >= n_w
        r = dict(f=float(f), usable=int(use.sum()))
        if use.sum() >= int(m["min_usable_windows"]):
            Lw = loss(M[use], e_mat, b_mat)                         # windows x events
            err = np.abs(Lw - L_E[None, :])
            errs[float(f)] = err
            r.update(p90=float(np.nanquantile(err, 0.9)), med=float(np.nanmedian(err)),
                     p90_per_event_median=float(np.nanmedian(np.nanquantile(err, 0.9, axis=0))))
            pp = []
            for p in perms:
                Lp = loss(M[use], e_mat[p], b_mat[p])
                pp.append(np.nanquantile(np.abs(Lp - L_E[None, :]), 0.9))
            r.update(p90_perm_median=float(np.median(pp)), p90_perm_p05=float(np.quantile(pp, 0.05)))
        lev.append(r)
    out["levels"] = pd.DataFrame(lev)
    out["errs"] = errs
    return out


def critical(levels_df: pd.DataFrame, eps: float, col: str = "p90") -> dict:
    lv = levels_df.f.to_numpy()
    st = levels_df[col].to_numpy() if col in levels_df else np.full(lv.size, np.nan)
    return criteria.critical_f(lv, st, eps)
