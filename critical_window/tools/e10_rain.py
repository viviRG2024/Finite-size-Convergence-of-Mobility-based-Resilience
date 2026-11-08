"""E10 (exploratory): rain events as a city-wide disturbance, measured with flow, occupancy and a speed index.

    python -m critical_window.tools.e10_rain run      [--thr 0.5] [--main main34] [--x0 out/x0]
                                                        [--weather data/input/IUTFD] [--raw data/input/IUTFD]
    python -m critical_window.tools.e10_rain summary  [--thr 0.5]

Events: ERA5 precipitation over the observation extent >= thr mm/h, 07:00-19:59 of a city weekday, rain hours merged
across gaps <= 1 h (resil/rain.py). For every event and every comparison day:
  comparison days = the other city weekdays of the same data block (runs of days separated by <= 14 days) with less
  than 0.2 mm/h in every hour from one hour before to one hour after the event hours;
  baseline = per station and hour, the median over the comparison days (for a comparison day itself: over the others);
  stations enter when flow (and occupancy, where the city has it) is present for all event hours on the day and in the
  baseline.
Whole-extent changes against the baseline:
  Lq = 1 - sum q / sum q_base                       (flow loss)
  Lo = sum o / sum o_base - 1                       (occupancy rise; o = mean lane occupancy of the station)
  Lv = 1 - (sum q / sum o) / (sum q_base / sum o_base)   (loss of the speed index flow / occupancy)
Normal fluctuation: the same quantities for every comparison day (pseudo-events). z = (L - mean) / SD of the pseudo-events;
an event is a detectable disturbance when L exceeds the 95th percentile of its pseudo-events (>= 5 needed).
Recovery (speed index, detectable events): hours after the event until the hourly Lv is back at or below the 95th
percentile of the comparison days in that hour (censored at 20:00).
Windows: the random windows of the main run (same seeds); P90 over windows and events of |L_window - L_city| per level.

Outputs in out/e10/thr<thr>/: events.csv, pseudo.csv, levels.csv, cities.csv, summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..common.cfg import load_config, repo_path
from ..core import criteria, windows
from ..main.data import load_city
from ..resil import rain

H0, H1 = rain.H0, rain.H1
NH = H1 - H0
DRY = 0.2
MEASURES = ("Lq", "Lo", "Lv")


def occupancy_cube(cd, raw_dir: Path, det_csv: Path):
    """Mean lane occupancy per station as stations x city weekdays x 13 hours (NaN where missing or not clean);
    None when the city has no occupancy."""
    p = raw_dir / cd.city / "sensors" / "hourly_readings.parquet"
    if not p.exists():
        return None
    h = pd.read_parquet(p, columns=["datetime", "detid", "occ_mean", "error_mean"])
    if h.occ_mean.notna().mean() < 0.5:
        return None
    det = pd.read_csv(det_csv, usecols=["detid", "station_id"])
    det["detid"] = det.detid.astype(str)
    h["detid"] = h.detid.astype(str)
    h = h[(h.error_mean.fillna(0) == 0) & h.occ_mean.notna() & (h.occ_mean >= 0)]
    top = 1.0 if h.occ_mean.quantile(0.99) <= 1.0 else 100.0          # fraction or per cent
    h = h[h.occ_mean < 0.95 * top].merge(det, on="detid")             # values at the top of the scale = stuck loops
    qc = json.load(open(det_csv.parent / "qc.json", encoding="utf-8"))
    t = pd.to_datetime(h.datetime, format=qc.get("ts_format") or None, errors="coerce").dt.floor("h")
    h = h.assign(date=t.dt.strftime("%Y-%m-%d"), hour=t.dt.hour)
    h = h[(h.hour >= H0) & (h.hour < H1)]
    g = h.groupby(["station_id", "date", "hour"]).occ_mean.mean().reset_index()
    sidx = {s: i for i, s in enumerate(cd.station_ids)}
    didx = {d: j for j, d in enumerate(cd.days)}
    si, dj = g.station_id.map(sidx), g.date.map(didx)
    ok = si.notna().to_numpy() & dj.notna().to_numpy()
    si, dj = si[ok].astype(int).to_numpy(), dj[ok].astype(int).to_numpy()
    hh = g.hour.to_numpy()[ok] - H0
    clean = np.isfinite(cd.X[si, dj])
    cube = np.full((cd.F.size, cd.days.size, NH), np.nan)
    cube[si[clean], dj[clean], hh[clean]] = g.occ_mean.to_numpy(float)[ok][clean]
    return cube


WIN_DAYS = 14


def comparison_days(R, blk, dates, j, h0, h1):
    """Other city weekdays of the same data block, within +-WIN_DAYS calendar days, dry from one hour before to one
    hour after the event hours."""
    a, b = max(0, h0 - 1), min(NH - 1, h1 + 1)
    seg = R[:, a:b + 1]
    dry = np.isfinite(seg).all(1) & (np.nanmax(np.where(np.isfinite(seg), seg, 0), 1) < DRY)
    near = np.abs((dates - dates[j]).days) <= WIN_DAYS
    dd = np.flatnonzero((blk == blk[j]) & dry & near)
    return dd[dd != j]


def station_median_lv(sq, sbq, so, sbo, v):
    """Median over stations of the station's own speed-index loss (robustness check against a few detectors)."""
    ok = v & (so > 0) & (sbo > 0) & (sbq > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.median(1 - (sq[ok] / so[ok]) / (sbq[ok] / sbo[ok])))


def sums(Q, O, day, base_q, base_o, h0, h1, need_occ):
    """Per-station sums over the event hours on `day` and in the baseline; validity mask."""
    q, bq = Q[:, day, h0:h1 + 1], base_q[:, h0:h1 + 1]
    v = np.isfinite(q).all(1) & np.isfinite(bq).all(1)
    if need_occ:
        o, bo = O[:, day, h0:h1 + 1], base_o[:, h0:h1 + 1]
        v &= np.isfinite(o).all(1) & np.isfinite(bo).all(1)
        so, sbo = np.where(v, o.sum(1), 0.0), np.where(v, bo.sum(1), 0.0)
    else:
        so = sbo = np.zeros(Q.shape[0])
    return np.where(v, q.sum(1), 0.0), np.where(v, bq.sum(1), 0.0), so, sbo, v


def measures(M, sq, sbq, so, sbo, has_occ):
    """Lq, Lo, Lv for each row of the station mask M (k x stations); sums may carry extra columns (stations x K)."""
    Mf = M.astype(float)
    Q, BQ = Mf @ sq, Mf @ sbq
    with np.errstate(invalid="ignore", divide="ignore"):
        Lq = np.where(BQ > 0, 1 - Q / BQ, np.nan)
        if not has_occ:
            nan = np.full_like(Lq, np.nan)
            return Lq, nan, nan
        Oc, BO = Mf @ so, Mf @ sbo
        Lo = np.where(BO > 0, Oc / BO - 1, np.nan)
        Lv = np.where((BO > 0) & (Oc > 0) & (BQ > 0), 1 - (Q / Oc) / (BQ / BO), np.nan)
    return Lq, Lo, Lv


def city_events(cd, Q, O, R, blk, thr, n_w):
    has_occ = O is not None
    dates = pd.to_datetime(pd.Series(list(cd.days))).to_numpy().astype("datetime64[D]")
    dates = pd.DatetimeIndex(dates)
    Oz = O if has_occ else np.full_like(Q, np.nan)
    E_all = np.ones((1, cd.F.size), bool)
    evs, pseudo, keep = [], [], []
    for k, (j, h0, h1, mx) in enumerate(rain.find_events(R, thr)):
        dd = comparison_days(R, blk, dates, j, h0, h1)
        if dd.size == 0:
            continue
        base_q = np.nanmedian(Q[:, dd, :], axis=1)
        base_o = np.nanmedian(Oz[:, dd, :], axis=1) if has_occ else None
        sq, sbq, so, sbo, v = sums(Q, Oz, j, base_q, base_o, h0, h1, has_occ)
        if v.sum() < n_w:
            continue
        Lq, Lo, Lv = (float(x[0, 0]) for x in measures(E_all, sq[:, None], sbq[:, None], so[:, None], sbo[:, None], has_occ))
        ps = []
        for jj in dd:
            others = dd[dd != jj]
            if others.size == 0:
                continue
            bq_ = np.nanmedian(Q[:, others, :], axis=1)
            bo_ = np.nanmedian(Oz[:, others, :], axis=1) if has_occ else None
            pq, pbq, po, pbo, pv = sums(Q, Oz, jj, bq_, bo_, h0, h1, has_occ)
            if pv.sum() < n_w:
                continue
            m = measures(E_all, pq[:, None], pbq[:, None], po[:, None], pbo[:, None], has_occ)
            ps.append(dict(event=k, day=str(cd.days[jj]), Lq=float(m[0][0, 0]), Lo=float(m[1][0, 0]), Lv=float(m[2][0, 0]),
                           Lv_med=station_median_lv(pq, pbq, po, pbo, pv) if has_occ else np.nan))
        row = dict(event=k, day=str(cd.days[j]), block=int(blk[j]), hour_start=H0 + h0, hour_end=H0 + h1 + 1, n_hours=int(h1 - h0 + 1),
                   max_mmh=mx, total_mm=float(np.nansum(R[j, h0:h1 + 1])), n_stations=int(v.sum()), n_comparison=int(dd.size),
                   n_pseudo=len(ps), Lq=Lq, Lo=Lo, Lv=Lv, Lv_med=station_median_lv(sq, sbq, so, sbo, v) if has_occ else np.nan,
                   has_occ=has_occ)
        pdf = pd.DataFrame(ps)
        for mname in MEASURES + ("Lv_med",):
            x = pdf[mname].dropna().to_numpy() if len(pdf) else np.array([])
            ok = x.size >= 5 and np.isfinite(row[mname])
            row[f"{mname}_p95"] = float(np.quantile(x, 0.95)) if ok else np.nan
            row[f"{mname}_z"] = float((row[mname] - x.mean()) / x.std(ddof=1)) if ok and x.std(ddof=1) > 0 else np.nan
            row[f"{mname}_detect"] = bool(row[mname] > row[f"{mname}_p95"]) if ok else None
            row[f"{mname}_rank"] = float(np.mean(x < row[mname])) if ok else np.nan     # share of pseudo-events below
        # recovery of the speed index (hour by hour) for detectable events
        rec, cens = np.nan, None
        if has_occ and row["Lv_detect"]:
            cens = True
            for hh in range(h1 + 1, NH):
                hv = []
                for jj in dd:
                    others = dd[dd != jj]
                    if others.size == 0:
                        continue
                    bq_ = np.nanmedian(Q[:, others, :], axis=1)
                    bo_ = np.nanmedian(Oz[:, others, :], axis=1)
                    pq, pbq, po, pbo, pv = sums(Q, Oz, jj, bq_, bo_, hh, hh, True)
                    if pv.sum() >= n_w:
                        hv.append(measures(E_all, pq[:, None], pbq[:, None], po[:, None], pbo[:, None], True)[2][0, 0])
                eq, ebq, eo, ebo, ev_ = sums(Q, Oz, j, base_q, base_o, hh, hh, True)
                if ev_.sum() < n_w or len(hv) < 5:
                    continue
                lv_h = measures(E_all, eq[:, None], ebq[:, None], eo[:, None], ebo[:, None], True)[2][0, 0]
                if lv_h <= np.quantile(hv, 0.95):
                    rec, cens = float(hh - h1), False
                    break
        row.update(recovery_h=rec, recovery_censored=cens)
        evs.append(row)
        pseudo += ps
        keep.append((k, j, h0, h1, sq, sbq, so, sbo))
    return pd.DataFrame(evs), pd.DataFrame(pseudo), keep


def run(a):
    cfg = load_config()
    m = cfg["main"]
    out_root, od = outdir(a)
    x0 = Path(a.x0) if a.x0 else out_root / "x0"
    wroot = Path(a.weather)
    raw = Path(a.raw)
    od.mkdir(parents=True, exist_ok=True)
    cities = json.load(open(out_root / a.main / "summary.json", encoding="utf-8"))["cities"]
    n_w, n_rand, min_use = int(m["n_w"]), int(m["n_random"]), int(m["min_usable_windows"])
    fs = np.array(m["f_grid"], float)
    levels = fs[fs < 1.0]
    ev_all, ps_all, lev_all, crow = [], [], [], []
    for c in cities:
        t0 = time.time()
        cd = load_city(c, x0, m)
        qc = json.load(open(x0 / "per_city" / c / "qc.json", encoding="utf-8"))
        series = rain.city_rain_series(wroot / c / "weather", qc, cd.days)
        hourly = pd.read_parquet(x0 / "per_city" / c / "station_hourly.parquet")
        Q = rain.hourly_cube(cd, hourly)
        O = occupancy_cube(cd, raw, x0 / "per_city" / c / "detectors.csv")
        R = rain.rain_matrix(series, cd.days)
        blk = rain.blocks(cd.days)
        ev, ps, keep = city_events(cd, Q, O, R, blk, a.thr, n_w)
        crow.append(dict(city=c, n_days=int(cd.days.size), n_rain_days=int((np.nanmax(np.where(np.isfinite(R), R, 0), 1) >= a.thr).sum()),
                         has_occupancy=O is not None, n_events=len(ev)))
        if len(ev):
            ev_all.append(ev.assign(city=c))
            if len(ps):
                ps_all.append(ps.assign(city=c))
            # windows: errors of the window estimate against the whole-extent value, per level and measure
            sq = np.stack([kk[4] for kk in keep], 1)
            sbq = np.stack([kk[5] for kk in keep], 1)
            so = np.stack([kk[6] for kk in keep], 1)
            sbo = np.stack([kk[7] for kk in keep], 1)
            Lc = {mn: ev[mn].to_numpy(float) for mn in MEASURES}
            for f in levels:
                wins = windows.random_windows(cd.extent, f, n_rand, rain._rng(cd.city, "main", "windows", f))
                M = np.array([windows.in_box(cd.x, cd.y, w_) for w_ in wins])
                use = M.sum(1) >= n_w
                r = dict(city=c, f=float(f), usable=int(use.sum()))
                if use.sum() >= min_use:
                    Lw = measures(M[use], sq, sbq, so, sbo, O is not None)
                    for mn, L in zip(MEASURES, Lw):
                        err = np.abs(L - Lc[mn][None, :])
                        r[f"{mn}_p90"] = float(np.nanquantile(err, 0.9)) if np.isfinite(err).any() else np.nan
                lev_all.append(r)
        print(f"[{time.strftime('%H:%M:%S')}] {c}: {len(ev)} events, occupancy {O is not None} ({time.time() - t0:.0f} s)", flush=True)
    ev = pd.concat(ev_all, ignore_index=True)
    ev.to_csv(od / "events.csv", index=False)
    pd.concat(ps_all, ignore_index=True).to_csv(od / "pseudo.csv", index=False)
    pd.DataFrame(lev_all).to_csv(od / "levels.csv", index=False)
    pd.DataFrame(crow).to_csv(od / "cities.csv", index=False)
    print(f"events: {len(ev)} in {ev.city.nunique()} cities -> {od}")
    return 0


def outdir(a):
    cfg = load_config()
    out_root = repo_path(cfg["paths"]["out_dir"])
    tag = f"thr{a.thr:g}".replace(".", "p") + ("" if a.win_days == 14 else f"_win{a.win_days}")
    return out_root, out_root / "e10" / tag


def at_f(lv, y, f0):
    """Log-log interpolation of a P90 curve at coverage f0 (NaN outside the usable levels)."""
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2 or f0 < lv[ok].min() or f0 > lv[ok].max():
        return np.nan
    return float(np.exp(np.interp(np.log(f0), np.log(lv[ok]), np.log(y[ok]))))


def summary(a):
    out_root, od = outdir(a)
    ev = pd.read_csv(od / "events.csv")
    lev = pd.read_csv(od / "levels.csv")
    res = dict(threshold_mmh=a.thr, comparison_window_days=a.win_days, n_events=int(len(ev)), n_cities=int(ev.city.nunique()),
               all_events={mn: dict(median=float(ev[mn].median()), share_positive=float((ev[mn] > 0).mean()))
                           for mn in MEASURES + ("Lv_med",)})
    for mn in MEASURES + ("Lv_med",):
        e = ev[ev[f"{mn}_z"].notna()]
        if not len(e):
            continue
        z = e[f"{mn}_z"].to_numpy()
        rk = e[f"{mn}_rank"].to_numpy()
        res[mn] = dict(
            n_events_tested=int(len(e)), n_cities=int(e.city.nunique()),
            median_L=float(e[mn].median()), median_z=float(np.median(z)), mean_z=float(z.mean()),
            share_detectable=float((e[f"{mn}_detect"].astype(str) == "True").mean()),
            share_above_pseudo_median=float(np.mean(rk > 0.5)),
            p_sign=float(stats.binomtest(int(np.sum(rk > 0.5)), int(len(rk)), 0.5, alternative="greater").pvalue),
            p_wilcoxon_z=float(stats.wilcoxon(z, alternative="greater").pvalue) if len(z) >= 6 else np.nan,
            spearman_z_maxmmh=float(stats.spearmanr(e.max_mmh, z)[0]) if len(e) >= 6 else np.nan,
            spearman_z_totalmm=float(stats.spearmanr(e.total_mm, z)[0]) if len(e) >= 6 else np.nan)
    # placebo: each comparison day treated as an event against the other comparison days of the same event
    ps = pd.read_csv(od / "pseudo.csv")
    for mn in MEASURES + ("Lv_med",):
        hits, zs = [], []
        for _, g in ps.groupby(["city", "event"]):
            x = g[mn].dropna().to_numpy()
            if x.size < 6:
                continue
            for i in range(x.size):
                o = np.delete(x, i)
                hits.append(x[i] > np.quantile(o, 0.95))
                if o.std(ddof=1) > 0:
                    zs.append((x[i] - o.mean()) / o.std(ddof=1))
        if hits:
            ez = ev[f"{mn}_z"].dropna().to_numpy()
            res.setdefault("placebo", {})[mn] = dict(
                n=len(hits), share_detectable=float(np.mean(hits)), median_z=float(np.median(zs)) if zs else None,
                p_events_vs_placebo_z=float(stats.mannwhitneyu(ez, zs, alternative="greater").pvalue) if len(ez) and zs else None)
    rec = ev[ev.Lv_detect.astype(str) == "True"]
    res["recovery_h"] = dict(n_detectable=int(len(rec)), n_measured=int(rec.recovery_h.notna().sum()),
                             median=float(rec.recovery_h.median()) if rec.recovery_h.notna().any() else None,
                             censored_at_20h=int((rec.recovery_censored.astype(str) == "True").sum()))
    # ---- windows: tolerance = 0.25 x IQR of the whole-extent values over the events of cities with >= 3 events
    big = ev.groupby("city").filter(lambda g: len(g) >= 3)
    cw = pd.read_csv(out_root / a.main / "critical_windows.csv")
    sz = cw[cw.probe == "sd_log"].set_index("city")
    crit, law = [], []
    for mn in MEASURES:
        x = big[mn].dropna()
        if len(x) < 8:
            continue
        eps = 0.25 * float(np.subtract(*np.quantile(x, [0.75, 0.25])))
        res.setdefault("eps", {})[mn] = eps
        for c, g in lev[lev.city.isin(big.city.unique())].groupby("city"):
            g = g.sort_values("f")
            if f"{mn}_p90" not in g or g[f"{mn}_p90"].isna().all():
                continue
            fl, y = g.f.to_numpy(), g[f"{mn}_p90"].to_numpy() / eps
            cr = criteria.critical_f(fl, y * eps, eps)
            c5 = at_f(fl, y, 0.5)
            f_one = 1 / (1 + (1 / c5) ** (1 / 0.7)) if np.isfinite(c5) and c5 > 0 else np.nan
            ok = np.isfinite(y) & (y > 0) & (fl < 1)
            b = r2 = np.nan
            if ok.sum() >= 4:
                u = np.log((1 - fl[ok]) / fl[ok])
                b, a0 = np.polyfit(u, np.log(y[ok]), 1)
                r2 = 1 - np.var(np.log(y[ok]) - (a0 + b * u)) / np.var(np.log(y[ok]))
            fsz = sz.f_star.get(c, np.nan) if sz.flag.get(c, "") == "ok" else np.nan
            crit.append(dict(city=c, measure=mn, n_events=int((big.city == c).sum()), f_star=cr["f_star"], flag=cr["flag"],
                             C_at_half=c5, f_star_one_point=f_one, b=b, r2=r2,
                             f_star_sd_log=sz.f_star.get(c, np.nan), flag_sd_log=sz.flag.get(c, ""),
                             err_at_sd_log_fstar=at_f(fl, y, fsz) if np.isfinite(fsz) else np.nan))
    crit = pd.DataFrame(crit)
    if len(crit):
        crit.to_csv(od / "critical.csv", index=False)
        for mn, g in crit.groupby("measure"):
            okk = g[(g.flag == "ok") & (g.flag_sd_log == "ok")]
            one = g[(g.flag == "ok") & g.f_star_one_point.notna()]
            tr = g[g.err_at_sd_log_fstar.notna()]
            res.setdefault("windows", {})[mn] = dict(
                n_cities=int(len(g)), n_found=int((g.flag == "ok").sum()),
                f_star_median_found=float(g[g.flag == "ok"].f_star.median()) if (g.flag == "ok").any() else None,
                f_star_median_all=criteria.fstar_stats(g)["median"],
                f_star_sd_log_median_same_cities=criteria.fstar_stats(
                    g[["f_star_sd_log", "flag_sd_log"]].rename(columns={"f_star_sd_log": "f_star", "flag_sd_log": "flag"}))["median"],
                n_above_sd_log_diagonal=int(((g.flag == "above") | ((g.flag == "ok") & (g.f_star > g.f_star_sd_log))).sum()),
                share_cities_f_star_above_sd_log=float(np.mean(okk.f_star > okk.f_star_sd_log)) if len(okk) else None,
                spearman_f_star_vs_sd_log=float(stats.spearmanr(okk.f_star, okk.f_star_sd_log)[0]) if len(okk) >= 5 else None,
                b_median=float(g.b.median()), r2_median=float(g.r2.median()),
                one_point_mae=float((one.f_star_one_point - one.f_star).abs().mean()) if len(one) else None,
                one_point_within_0p1=float(((one.f_star_one_point - one.f_star).abs() <= 0.1).mean()) if len(one) else None,
                share_within_tolerance_at_sd_log_fstar=float(np.mean(tr.err_at_sd_log_fstar <= 1)) if len(tr) else None,
                n_at_sd_log_fstar=int(len(tr)))
    json.dump(res, open(od / "summary.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    if len(crit):
        pd.set_option("display.width", 250)
        print(crit.round(3).to_string(index=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--main", default="main34")
    ap.add_argument("--x0", default="")
    ap.add_argument("--weather", default="data/input/IUTFD")
    ap.add_argument("--raw", default="data/input/IUTFD")
    ap.add_argument("--win-days", type=int, default=14, help="comparison days within +- this many calendar days")
    a = ap.parse_args(argv)
    global WIN_DAYS
    WIN_DAYS = a.win_days
    return run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    sys.exit(main())
