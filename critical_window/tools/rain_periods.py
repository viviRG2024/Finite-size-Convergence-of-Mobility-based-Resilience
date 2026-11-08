"""City-wide slowdown hour by hour around rainfall events: two hours before, during, and up to three hours after the rain,
for the events and, in the same way, for each comparison day against the other comparison days (Supplementary Fig. 10e,f).

    python -m critical_window.tools.rain_periods run       [--x0 out/x0] [--raw data/input/IUTFD] [--main main34]
    python -m critical_window.tools.rain_periods summary

Uses the events, comparison days, baselines and measures of tools/e10_rain.py (threshold 0.5 mm/h).
Outputs in out/e10/periods/: events.csv, pseudo.csv (one row per event or comparison day and period) and summary.csv
(per period: median and quartiles over events and over comparison days, and the share beyond the 95th percentile of the
comparison days, for the events with at least five comparison days).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.cfg import load_config, repo_path
from ..main.data import load_city
from ..resil import rain
from . import e10_rain as E

PERIODS = ["-2", "-1", "R", "+1", "+2", "+3"]


def out_dir() -> Path:
    return repo_path(load_config()["paths"]["out_dir"]) / "e10" / "periods"


def run(a) -> int:
    cfg = load_config()
    m = cfg["main"]
    n_w = int(m["n_w"])
    out_root = repo_path(cfg["paths"]["out_dir"])
    x0 = Path(a.x0) if a.x0 else out_root / "x0"
    raw = Path(a.raw)
    cities = json.load(open(out_root / a.main / "summary.json", encoding="utf-8"))["cities"]
    ev_rows, ps_rows = [], []
    for c in cities:
        t0 = time.time()
        cd = load_city(c, x0, m)
        qc = json.load(open(x0 / "per_city" / c / "qc.json", encoding="utf-8"))
        series = rain.city_rain_series(raw / c / "weather", qc, cd.days)
        if series.empty:
            continue
        hourly = pd.read_parquet(x0 / "per_city" / c / "station_hourly.parquet")
        Q = rain.hourly_cube(cd, hourly)
        O = E.occupancy_cube(cd, raw, x0 / "per_city" / c / "detectors.csv")
        if O is None:
            continue
        R = rain.rain_matrix(series, cd.days)
        blk = rain.blocks(cd.days)
        dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(cd.days))).to_numpy().astype("datetime64[D]"))
        whole = np.ones((1, cd.F.size), bool)

        def slowdown(day, bq, bo, a0, b0):
            if a0 < 0 or b0 > E.NH - 1:
                return np.nan
            pq, pbq, po, pbo, pv = E.sums(Q, O, day, bq, bo, a0, b0, True)
            if pv.sum() < n_w:
                return np.nan
            return float(E.measures(whole, pq[:, None], pbq[:, None], po[:, None], pbo[:, None], True)[2][0, 0])

        n_ev = 0
        for k, (j, h0, h1, _mx) in enumerate(rain.find_events(R, 0.5)):
            dd = E.comparison_days(R, blk, dates, j, h0, h1)
            if dd.size == 0:
                continue
            bq = np.nanmedian(Q[:, dd, :], 1)
            bo = np.nanmedian(O[:, dd, :], 1)
            if E.sums(Q, O, j, bq, bo, h0, h1, True)[4].sum() < n_w:
                continue
            seg = {"-2": (h0 - 2, h0 - 2), "-1": (h0 - 1, h0 - 1), "R": (h0, h1),
                   "+1": (h1 + 1, h1 + 1), "+2": (h1 + 2, h1 + 2), "+3": (h1 + 3, h1 + 3)}
            for r in PERIODS:
                ev_rows.append(dict(city=c, event=k, day=str(cd.days[j]), hour_start=h0 + E.H0, hour_end=h1 + E.H0,
                                    n_comp=int(dd.size), rel=r, Lv=slowdown(j, bq, bo, *seg[r])))
            for jj in dd:
                oth = dd[dd != jj]
                if oth.size == 0:
                    continue
                bq_ = np.nanmedian(Q[:, oth, :], 1)
                bo_ = np.nanmedian(O[:, oth, :], 1)
                for r in PERIODS:
                    ps_rows.append(dict(city=c, event=k, comp_day=str(cd.days[jj]), rel=r, Lv=slowdown(jj, bq_, bo_, *seg[r])))
            n_ev += 1
        print(f"{c}: {n_ev} events ({time.time() - t0:.0f} s)", flush=True)
    od = out_dir()
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ev_rows).to_csv(od / "events.csv", index=False)
    pd.DataFrame(ps_rows).to_csv(od / "pseudo.csv", index=False)
    return 0


def summary(a) -> int:
    od = out_dir()
    ev = pd.read_csv(od / "events.csv")
    ps = pd.read_csv(od / "pseudo.csv")
    ok = ev[ev.rel == "R"].query("n_comp >= 5")[["city", "event"]]
    E_ = ev.merge(ok, on=["city", "event"])
    P_ = ps.merge(ok, on=["city", "event"])
    rows = []
    for rel in PERIODS:
        e = E_[E_.rel == rel].Lv.dropna() * 100
        p = P_[P_.rel == rel].Lv.dropna() * 100
        ea, pa = [], []
        for (c, k), g in P_[P_.rel == rel].groupby(["city", "event"]):
            x = g.Lv.to_numpy(float)
            x = x[np.isfinite(x)]
            v = E_[(E_.city == c) & (E_.event == k) & (E_.rel == rel)].Lv.to_numpy(float)
            if len(x) >= 5 and v.size and np.isfinite(v[0]):
                ea.append(v[0] > np.quantile(x, 0.95))
            for i in range(len(x)):
                o = np.delete(x, i)
                if len(o) >= 5:
                    pa.append(x[i] > np.quantile(o, 0.95))
        rows.append(dict(period=rel, n_events=len(e), events_median=e.median(), events_q25=e.quantile(0.25),
                         events_q75=e.quantile(0.75), comparison_median=p.median(), comparison_q25=p.quantile(0.25),
                         comparison_q75=p.quantile(0.75), events_beyond_p95=100 * np.mean(ea), n_events_tested=len(ea),
                         comparison_beyond_p95=100 * np.mean(pa), n_comparison_tested=len(pa)))
    pd.DataFrame(rows).round(3).to_csv(od / "summary.csv", index=False)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--main", default="main34")
    ap.add_argument("--x0", default="")
    ap.add_argument("--raw", default="data/input/IUTFD")
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    raise SystemExit(main())
