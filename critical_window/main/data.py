"""Analysis-ready station data for one city."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CityData:
    city: str
    station_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    extent: tuple            # (x0, y0, x1, y1) in UTM metres
    X: np.ndarray            # stations x city weekdays, clean daily totals (NaN where not clean)
    days: np.ndarray         # city weekday dates
    F: np.ndarray            # station analysis flow = mean over clean days
    fclass: np.ndarray
    epsg: int
    n_stuck_days: int

    @property
    def short_side(self) -> float:
        return min(self.extent[2] - self.extent[0], self.extent[3] - self.extent[1])

    @property
    def area_km2(self) -> float:
        return (self.extent[2] - self.extent[0]) * (self.extent[3] - self.extent[1]) / 1e6

    def bootstrap(self, n: int, rng) -> np.ndarray:
        """n day-bootstrap replicates of the station flows (stations x n)."""
        nd = self.days.size
        ok = np.isfinite(self.X)
        Xz = np.where(ok, self.X, 0.0)
        out = np.empty((self.F.size, n))
        for b in range(n):
            w = np.bincount(rng.integers(0, nd, nd), minlength=nd).astype(float)
            den = ok @ w
            num = Xz @ w
            out[:, b] = np.where(den > 0, num / np.maximum(den, 1e-12), self.F)
        return out


def _stuck_days(hourly: pd.DataFrame, run_hours: int, h0: int = 7, h1: int = 20) -> set:
    """(station_id, date) pairs with >= run_hours identical consecutive daytime values or all-zero daytime hours."""
    if hourly.empty:
        return set()
    h = hourly.copy()
    ts = pd.to_datetime(h["hour_start"])
    h["date"] = ts.dt.strftime("%Y-%m-%d")
    h["hour"] = ts.dt.hour
    h = h[(h["hour"] >= h0) & (h["hour"] < h1)].sort_values(["station_id", "date", "hour"])
    key = h["station_id"] + "|" + h["date"]
    same = (h["flow_vehh"].diff() == 0) & (key == key.shift()) & (h["hour"].diff() == 1)
    run = (~same).cumsum()
    run_len = same.groupby(run).transform("sum") + 1
    stuck = set(key[run_len >= run_hours])
    zero = h.groupby(key)["flow_vehh"].max()
    stuck |= set(zero[zero <= 0].index)
    return {tuple(k.split("|", 1)) for k in stuck}


def load_city(city: str, x0_dir: Path, cfg: dict, daily_all: pd.DataFrame | None = None) -> CityData:
    pdir = x0_dir / "per_city" / city
    st = pd.read_csv(pdir / "stations.csv")
    st = st[st["valid_station"]].reset_index(drop=True)
    days = pd.read_csv(pdir / "days.csv")
    cw = days.loc[days["city_weekday"], "date"].astype(str).to_numpy()
    if daily_all is None and (pdir / "station_daily.parquet").exists():
        daily = pd.read_parquet(pdir / "station_daily.parquet")
    else:
        if daily_all is None:
            daily_all = pd.read_parquet(x0_dir / "station_daily_all.parquet", filters=[("city", "==", city)])
        daily = daily_all[daily_all["city"] == city]
    daily = daily[daily["city_weekday"] & ~daily["outlier"] & daily["station_id"].isin(st["station_id"])]
    hp = pdir / "station_hourly.parquet"
    stuck = _stuck_days(pd.read_parquet(hp), int(cfg["stuck_run_hours"])) if hp.exists() else set()
    if stuck:
        k = list(zip(daily["station_id"], daily["date"].astype(str)))
        daily = daily[[kk not in stuck for kk in k]]
    sidx = {s: i for i, s in enumerate(st["station_id"])}
    didx = {d: j for j, d in enumerate(cw)}
    X = np.full((len(st), cw.size), np.nan)
    rows = daily["station_id"].map(sidx).to_numpy()
    cols = daily["date"].astype(str).map(didx)
    ok = cols.notna().to_numpy()
    X[rows[ok], cols[ok].astype(int).to_numpy()] = daily["flow_veh"].to_numpy(float)[ok]
    has = np.isfinite(X).any(1)
    st, X = st[has].reset_index(drop=True), X[has]
    F = np.nanmean(X, axis=1)
    x, y = st["x_m"].to_numpy(float), st["y_m"].to_numpy(float)
    with open(pdir / "qc.json", encoding="utf-8") as fh:
        qc = json.load(fh)
    return CityData(city=city, station_ids=st["station_id"].to_numpy(), x=x, y=y,
                    extent=(x.min(), y.min(), x.max(), y.max()), X=X, days=cw, F=F,
                    fclass=st["fclass"].fillna("").astype(str).to_numpy() if "fclass" in st else np.array([""] * len(st)),
                    epsg=int(qc.get("epsg", 0) or 0), n_stuck_days=len(stuck))


def permuted(cd: CityData, rng) -> CityData:
    """Dry-run copy: whole station series (rows) permuted among stations; layout kept."""
    p = rng.permutation(cd.F.size)
    return CityData(city=cd.city, station_ids=cd.station_ids, x=cd.x, y=cd.y, extent=cd.extent, X=cd.X[p], days=cd.days,
                    F=cd.F[p], fclass=cd.fclass[p], epsg=cd.epsg, n_stuck_days=cd.n_stuck_days)
