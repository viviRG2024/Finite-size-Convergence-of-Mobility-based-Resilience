"""X0.2  Station-level hourly and daily flows with quality control.

Steps (all counts go to the QC table):
  1. record validity: timestamp parsed, flow present, flow >= 0, IUTFD error flag not set,
     flow <= 3600 veh/h per lane, detector listed in detectors_info
  2. detector-hour: mean of valid interval rates (veh/h) if >= 75% of the expected intervals are valid
  3. detector-day: 07:00-19:59 total = mean valid hour x 13 if >= 12 of the 13 hours are valid
  4. stations: detectors with >= 1 valid day, merged when closer than 10 m (single linkage)
  5. station-hour / station-day valid only if every member detector is valid; value = sum over members
  6. per-station outlier rule on non-holiday weekdays: |x - median| > 5 x 1.4826 x MAD
  7. city weekdays: non-holiday weekdays on which >= 50% of stations are valid;
     valid stations: valid on >= 50% of those weekdays
No probe, window or spatial statistic of the flows is computed here.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.geo import cluster_points, utm_epsg, utm_forward, utm_zone
from .io_iutfd import load_detectors, read_readings


def _mode(values):
    vals = [v for v in values if isinstance(v, str) and v]
    return max(sorted(set(vals)), key=vals.count) if vals else ""  # ties broken alphabetically (deterministic)


def process_city(city_dir: Path, meta: dict, cfg: dict, holidays: set) -> dict:
    city = city_dir.name
    det = load_detectors(city_dir)
    R = read_readings(city_dir, det, meta, float(cfg["max_rate_per_lane_vehh"]), int(cfg["batch_rows"]))
    qc = dict(R.qc)
    nD, nH = len(R.detids), R.hours.size
    expected = max(1, 3600 // R.dt_s) if R.dt_s <= 3600 else 1
    need_h = int(math.ceil(float(cfg["hour_min_valid_frac"]) * expected - 1e-9))
    h0, h1 = int(cfg["daytime_start_hour"]), int(cfg["daytime_end_hour"])
    n_day_hours = h1 - h0
    need_d = int(math.ceil(float(cfg["day_min_valid_frac"]) * n_day_hours - 1e-9))

    # ---- detector-hour matrix (veh/h)
    dcode = R.key // max(nH, 1)
    hcode = R.key % max(nH, 1)
    ok_h = R.n_valid >= need_h
    HF = np.full((nD, nH), np.nan, dtype=np.float64)
    HF[dcode[ok_h], hcode[ok_h]] = R.sum_flow[ok_h] / R.n_valid[ok_h]
    qc.update(expected_per_hour=expected, need_per_hour=need_h,
              det_hours_with_data=int(R.key.size), det_hours_valid=int(ok_h.sum()))

    # ---- detector-day matrix (vehicles, 07:00-19:59)
    hrs = pd.DatetimeIndex(R.hours)
    hdate = hrs.normalize()
    hod = hrs.hour
    days = pd.DatetimeIndex(np.unique(hdate))
    day_of_h = days.get_indexer(hdate)
    DF = np.full((nD, days.size), np.nan)
    NOK = np.zeros((nD, days.size), dtype=np.int16)
    for j in range(days.size):
        cols = np.flatnonzero((day_of_h == j) & (hod >= h0) & (hod < h1))
        if cols.size == 0:
            continue
        block = HF[:, cols]
        nok = np.isfinite(block).sum(1)
        NOK[:, j] = nok
        good = nok >= need_d
        if good.any():
            DF[good, j] = np.nanmean(block[good], axis=1) * n_day_hours
    qc.update(n_days=int(days.size), need_hours_per_day=need_d,
              det_days_valid=int(np.isfinite(DF).sum()))

    # ---- stations
    info = det.set_index("detid")
    lon = np.array([info["long"].get(d, np.nan) for d in R.detids], float)
    lat = np.array([info["lat"].get(d, np.nan) for d in R.detids], float)
    has_day = np.isfinite(DF).any(1)
    use = np.flatnonzero(has_day & np.isfinite(lon) & np.isfinite(lat))
    qc.update(detectors_with_valid_day=int(has_day.sum()), detectors_used=int(use.size))
    empty = dict(city=city, qc=qc, stations=pd.DataFrame(), daily=pd.DataFrame(), hourly=pd.DataFrame(),
                 detectors=pd.DataFrame(), days=pd.DataFrame())
    if use.size == 0:
        return empty
    zone, south = utm_zone(float(np.median(lon[use])), float(np.median(lat[use])))
    x, y = utm_forward(lon[use], lat[use], zone, south)
    lab = cluster_points(x, y, float(cfg["station_merge_m"]))
    order = np.lexsort((use, lab))
    use_o, lab_o = use[order], lab[order]
    starts = np.flatnonzero(np.r_[True, np.diff(lab_o) != 0])
    n_mem = np.diff(np.r_[starts, lab_o.size])
    nS = starts.size
    sid = np.array([f"{city}_s{i:05d}" for i in range(nS)])

    # station-day and station-hour: all members valid
    Dm = DF[use_o]
    s_day_ok = np.add.reduceat(np.isfinite(Dm).astype(np.int32), starts, axis=0) == n_mem[:, None]
    s_day = np.add.reduceat(np.nan_to_num(Dm), starts, axis=0)
    s_day[~s_day_ok] = np.nan
    Hm = HF[use_o]
    s_hr_ok = np.add.reduceat(np.isfinite(Hm).astype(np.int32), starts, axis=0) == n_mem[:, None]
    s_hr = np.add.reduceat(np.nan_to_num(Hm), starts, axis=0)
    s_hr[~s_hr_ok] = np.nan
    any_member_day = np.add.reduceat(np.isfinite(Dm).astype(np.int32), starts, axis=0) > 0
    qc.update(n_stations=int(nS), members_mean=float(n_mem.mean()), members_max=int(n_mem.max()),
              station_days_any_member=int(any_member_day.sum()), station_days_valid=int(s_day_ok.sum()),
              station_days_lost_incomplete_lanes=int((any_member_day & ~s_day_ok).sum()),
              station_hours_valid=int(s_hr_ok.sum()))

    # ---- calendar
    dstr = days.strftime("%Y-%m-%d")
    weekday = days.dayofweek.values
    is_hol = np.array([s in holidays for s in dstr])
    workday = (weekday < 5) & ~is_hol

    # ---- per-station outlier rule on weekday totals
    k = float(cfg["outlier_mad_k"])
    min_n = int(cfg["outlier_min_days"])
    W = s_day[:, workday]
    outl = np.zeros_like(s_day, dtype=bool)
    if W.size:
        n_ok = np.isfinite(W).sum(1)
        Wz = np.where(n_ok[:, None] > 0, W, 0.0)
        med = np.nanmedian(Wz, axis=1)
        mad = np.nanmedian(np.abs(Wz - med[:, None]), axis=1) * 1.4826
        test = (n_ok >= min_n) & (mad > 0)
        with np.errstate(invalid="ignore"):
            flag_w = np.abs(W - med[:, None]) > k * mad[:, None]
        flag_w &= test[:, None] & np.isfinite(W)
        outl[:, workday] = flag_w
    qc.update(station_days_outlier=int(outl.sum()))
    clean = np.isfinite(s_day) & ~outl

    # ---- city weekdays and valid stations
    st_any = clean.any(1)
    share = clean[st_any][:, workday].mean(0) if st_any.any() and workday.any() else np.zeros(int(workday.sum()))
    city_wd_mask = np.zeros(days.size, bool)
    city_wd_mask[np.flatnonzero(workday)] = share >= float(cfg["city_day_min_station_frac"])
    n_city_wd = int(city_wd_mask.sum())
    if n_city_wd:
        st_share = clean[:, city_wd_mask].mean(1)
        valid_station = st_share >= float(cfg["station_min_day_frac"])
    else:
        st_share = np.zeros(nS)
        valid_station = np.zeros(nS, bool)
    qc.update(n_weekdays_nonholiday=int(workday.sum()), n_holidays_on_weekdays=int(((weekday < 5) & is_hol).sum()),
              n_city_weekdays=n_city_wd, n_valid_stations=int(valid_station.sum()))

    # ---- station table
    xs = np.add.reduceat(x[order], starts) / n_mem
    ys = np.add.reduceat(y[order], starts) / n_mem
    lons = np.add.reduceat(lon[use_o], starts) / n_mem
    lats = np.add.reduceat(lat[use_o], starts) / n_mem
    det_ids = np.array(R.detids, dtype=object)
    fcl = info["fclass"] if "fclass" in info else pd.Series(dtype=object)
    lan = pd.to_numeric(info["lanes"], errors="coerce")
    st_rows = []
    for i in range(nS):
        mem = det_ids[use_o[starts[i]:starts[i] + n_mem[i]]]
        st_rows.append(dict(
            station_id=sid[i], x_m=round(float(xs[i]), 2), y_m=round(float(ys[i]), 2),
            lon=round(float(lons[i]), 7), lat=round(float(lats[i]), 7), epsg=utm_epsg(zone, south),
            n_detectors=int(n_mem[i]), detids=";".join(map(str, mem)),
            fclass=_mode([fcl.get(m) for m in mem]) if len(fcl) else "",
            lanes=float(np.nansum([lan.get(m, np.nan) for m in mem])) if len(lan) else np.nan,
            n_valid_weekdays=int(clean[i, city_wd_mask].sum()), valid_share=round(float(st_share[i]), 3),
            valid_station=bool(valid_station[i])))
    stations = pd.DataFrame(st_rows)

    # ---- long tables
    si, dj = np.nonzero(np.isfinite(s_day))
    daily = pd.DataFrame(dict(
        city=city, station_id=sid[si], date=dstr[dj], weekday=weekday[dj].astype(np.int8),
        holiday=is_hol[dj], city_weekday=city_wd_mask[dj], flow_veh=s_day[si, dj].astype(np.float32),
        outlier=outl[si, dj]))
    si, hj = np.nonzero(np.isfinite(s_hr))
    hourly = pd.DataFrame(dict(city=city, station_id=sid[si], hour_start=R.hours[hj].astype("datetime64[ns]"),
                               flow_vehh=s_hr[si, hj].astype(np.float32)))
    member_of = np.full(nD, "", dtype=object)
    member_of[use_o] = np.repeat(sid, n_mem)
    detectors = pd.DataFrame(dict(
        detid=R.detids, station_id=member_of, n_valid_hours=np.isfinite(HF).sum(1),
        n_valid_days=np.isfinite(DF).sum(1), lon=lon, lat=lat))
    # unit fingerprints: share of positive values that are integers / multiples of k, and the daytime level
    U = R.det_units
    npos = np.maximum(U.get("n_pos", np.zeros(nD)), 1)
    for name in ("n_int", "n_m4", "n_m12", "n_m20", "n_m60"):
        if name in U:
            detectors["share_" + name[2:]] = np.round(U[name] / npos, 3)
    day_cols = (hod >= h0) & (hod < h1)
    import warnings
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        detectors["mean_daytime_vehh"] = np.round(np.nanmean(np.where(np.isfinite(HF[:, day_cols]), HF[:, day_cols], np.nan), axis=1), 1) \
            if day_cols.any() else np.nan
    cls = np.full(nD, "float", dtype=object)
    for name, lab_ in (("share_int", "int"), ("share_m4", "m4"), ("share_m12", "m12"), ("share_m20", "m20"), ("share_m60", "m60")):
        if name in detectors:
            cls = np.where(detectors[name].to_numpy() >= 0.95, lab_, cls)
    cls = np.where(U.get("n_pos", np.zeros(nD)) > 0, cls, "none")
    detectors["unit_class"] = cls
    used = detectors[detectors["station_id"] != ""]
    summ = used.groupby("unit_class")["mean_daytime_vehh"].agg(["size", "median"]).sort_values("size", ascending=False)
    qc["unit_classes"] = "; ".join(f"{k}:{int(r['size'])} (median {r['median']:.0f} veh/h)" for k, r in summ.iterrows())
    if len(summ) >= 2 and summ["median"].iloc[1] > 0:
        qc["unit_median_ratio_top2"] = round(float(summ["median"].iloc[0] / summ["median"].iloc[1]), 2)
    day_tab = pd.DataFrame(dict(date=dstr, weekday=weekday, holiday=is_hol, workday=workday,
                                city_weekday=city_wd_mask,
                                share_stations_valid=np.round(np.r_[clean[st_any].mean(0)] if st_any.any() else 0, 3)))
    ext = {}
    if valid_station.any():
        vx, vy = xs[valid_station], ys[valid_station]
        w, h = vx.max() - vx.min(), vy.max() - vy.min()
        ext = dict(extent_xmin=vx.min(), extent_ymin=vy.min(), extent_xmax=vx.max(), extent_ymax=vy.max(),
                   extent_w_km=w / 1000, extent_h_km=h / 1000, extent_area_km2=w * h / 1e6,
                   extent_short_km=min(w, h) / 1000,
                   extent_lon_min=float(lons[valid_station].min()), extent_lon_max=float(lons[valid_station].max()),
                   extent_lat_min=float(lats[valid_station].min()), extent_lat_max=float(lats[valid_station].max()))
    qc.update(ext)
    qc.update(epsg=utm_epsg(zone, south))
    return dict(city=city, qc=qc, stations=stations, daily=daily, hourly=hourly, detectors=detectors, days=day_tab)
