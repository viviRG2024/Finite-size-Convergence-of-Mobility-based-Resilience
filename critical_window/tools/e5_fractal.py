"""E5: fractal geometry of the sensor layout and the road network, the old exponents, the uneven layout behind the
coverage law, and a block-averaging test of short-range correlation.

    python -m critical_window.tools.e5_fractal run      [--main main34] [--x0 DIR] [--iutfd DIR] [--workers 2]
    python -m critical_window.tools.e5_fractal summary  [--main main34]
    python -m critical_window.tools.e5_fractal deff     [--main main34] [--x0 DIR]

Per city (run, out/e5/cities/<city>.json):
  box counting   occupied boxes N(L) of the valid station positions and of the road lines (points every 20 m along the
                 segments), both inside the station extent; 16 grid origins, N averaged over origins;
                 D = -slope of log N on log L for L = 200..1000 m (the 17 scales of the old exponents) and 100..4000 m
  equal flows    alpha, beta of the old method with every station given the same flow (layout only), and the mean
                 ln(number of active nodes) over 200..1000 m
  blocks         z = log flow; square blocks of side b = 250..4000 m (8 origins, averaged); VIF(b) = between-block mean
                 square / total variance of z, observed and for 200 shuffles of z among stations (independence)
Summary (out/e5/):
  geometry.csv       per city: D_station, D_road, old exponents (observed, shuffled median, equal flows)
  geometry_corr.csv  Spearman correlations across the cities where the old exponents are estimable
  blocks.csv         per city and b: VIF observed, shuffled median and 95% range
  blocks_summary.csv per b: median VIF observed / shuffled, cities above the shuffled 97.5% quantile
  mixture.csv        per city and probe: law slope and f* from the observed P90, the median-count formula (E2) and the
                     window-count mixture: the P90 e of the error over the usable windows, each window w with its own
                     station count n_w and error |Z| s sqrt(1/n_w - 1/N), solves mean_w P(|Z| > e / sd_w) = 0.1
  mixture_summary.csv
deff (out/e5/): design effect of the spatial dependence. For each probe, the influence values of the stations (E2) are
  block-averaged as above (VIF(b) without shuffles); the window error is then predicted as the mixture P90 times
  sqrt(VIF(w)), w = sqrt(f x area) the window size (VIF interpolated on log b, constant beyond 250 / 4000 m). Compared
  with the observed P90, the shuffled-flow P90 of the main run (median over 2000 shuffles) and the mixture alone:
  deff.csv (per city and probe), deff_curves.csv, deff_summary.csv
"""
from __future__ import annotations

import argparse
import json
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..common.cfg import load_config, repo_path
from ..core import criteria, probes
from ..main.data import load_city
from ..resil import network_e4

OUT = Path(__file__).resolve().parents[1] / "out"
L_OLD = probes.LOCAL_L                                     # 200..1000 m, 17 scales
L_WIDE = np.geomspace(100, 4000, 17)
BLOCKS = np.array([250, 350, 500, 700, 1000, 1400, 2000, 2800, 4000], float)
N_ORIG_BOX, N_ORIG_BLOCK, N_SHUFFLE = 16, 8, 200
STEP = 20.0                                                 # road points every 20 m
PROBES_E2 = ("mean_log", "sd_log", "site_cv")


def _rng(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def box_counts(px, py, ext, Ls, rng, n_orig=N_ORIG_BOX):
    """Mean number of occupied boxes of side L (points inside ext), over n_orig random grid origins."""
    m = (px >= ext[0]) & (px <= ext[2]) & (py >= ext[1]) & (py <= ext[3])
    px, py = px[m], py[m]
    out = []
    for L in Ls:
        off = rng.uniform(0, L, (n_orig, 2))
        cnt = [np.unique(np.floor((px - ext[0] + o[0]) / L).astype(np.int64) * 1_000_003
                         + np.floor((py - ext[1] + o[1]) / L).astype(np.int64)).size for o in off]
        out.append(float(np.mean(cnt)))
    return np.array(out)


def dimension(Ls, N):
    ok = N > 0
    b, _ = np.polyfit(np.log(Ls[ok]), np.log(N[ok]), 1)
    return float(-b)


def road_points(seg, step=STEP):
    a, b, ln, _, _, vx, vy = seg
    m = np.maximum(np.ceil(ln / step).astype(int), 1)
    idx = np.repeat(np.arange(a.size), m)
    j = np.arange(idx.size) - np.repeat(np.cumsum(m) - m, m)
    t = (j + 0.5) / m[idx]
    return vx[a[idx]] + t * (vx[b[idx]] - vx[a[idx]]), vy[a[idx]] + t * (vy[b[idx]] - vy[a[idx]])


def vif_curve(x, y, Z, ext, rng):
    """Per block side b and column of Z (stations x T), averaged over block origins:
    VIF = between-block mean square / total variance (about 1 for values placed at random), and the one-way ANOVA
    intraclass correlation ICC = (MSB - MSW) / (MSB + (k0 - 1) MSW) with k0 the ANOVA mean block size."""
    N, T = Z.shape
    mu = Z.mean(0)
    var = Z.var(0, ddof=1)
    vif = np.full((BLOCKS.size, T), np.nan)
    icc = np.full((BLOCKS.size, T), np.nan)
    nblk = np.full(BLOCKS.size, np.nan)
    k0s = np.full(BLOCKS.size, np.nan)
    for i, b in enumerate(BLOCKS):
        v_, c_, nb, kk = [], [], [], []
        for o in rng.uniform(0, b, (N_ORIG_BLOCK, 2)):
            cid = np.floor((x - ext[0] + o[0]) / b).astype(np.int64) * 1_000_003 + np.floor((y - ext[1] + o[1]) / b).astype(np.int64)
            _, inv = np.unique(cid, return_inverse=True)
            B = inv.max() + 1
            if B < 3 or B >= N:
                continue
            k = np.bincount(inv, minlength=B).astype(float)
            S = np.stack([np.bincount(inv, Z[:, t], minlength=B) for t in range(T)], 1)
            mb = S / k[:, None]
            ssb = (k[:, None] * (mb - mu) ** 2).sum(0)
            msb = ssb / (B - 1)
            msw = ((N - 1) * var - ssb) / (N - B)
            k0 = (N - (k ** 2).sum() / N) / (B - 1)
            v_.append(msb / var)
            c_.append((msb - msw) / (msb + (k0 - 1) * msw))
            nb.append(B)
            kk.append(k0)
        if v_:
            vif[i], icc[i] = np.mean(v_, 0), np.mean(c_, 0)
            nblk[i], k0s[i] = float(np.mean(nb)), float(np.mean(kk))
    return vif, icc, nblk, k0s


def city_job(a):
    c, x0, iutfd, m = a
    cd = load_city(c, Path(x0), m)
    ext = cd.extent
    rng = _rng("e5", c)
    rec = dict(city=c, n_stations=int(cd.F.size), area_km2=cd.area_km2, short_side_m=cd.short_side)
    # stations
    rec["N_station_old"] = box_counts(cd.x, cd.y, ext, L_OLD, rng).tolist()
    rec["N_station_wide"] = box_counts(cd.x, cd.y, ext, L_WIDE, rng).tolist()
    # roads
    inv = pd.read_csv(Path(x0) / "network_inventory.csv").set_index("city")
    if c in inv.index and isinstance(inv.loc[c, "road_file"], str):
        qc = json.load(open(Path(x0) / "per_city" / c / "qc.json", encoding="utf-8"))
        seg = network_e4.load_segments_xy(Path(iutfd) / c / inv.loc[c, "road_file"].replace("\\", "/"), inv.loc[c, "road_source"], int(qc["epsg"]))
        px, py = road_points(seg)
        rec["N_road_old"] = box_counts(px, py, ext, L_OLD, rng).tolist()
        rec["N_road_wide"] = box_counts(px, py, ext, L_WIDE, rng).tolist()
    # old exponents with equal flows (layout only)
    ones = np.ones((cd.F.size, 1))
    ex = probes.exponents(cd.x, cd.y, ones, ext)
    rec["alpha_equal"], rec["beta_equal"] = float(ex["alpha"][0]), float(ex["beta"][0])
    lnn = []
    for L in L_OLD:
        _, H, _, nn = probes.grid_probe(cd.x, cd.y, ones, ext, L)
        lnn.append((float(np.log(max(nn[0], 1))), float(H[0])))
    rec["ln_nodes_old"], rec["H_equal_old"] = [u for u, _ in lnn], [v for _, v in lnn]
    # blocks: observed z and shuffles
    keep = cd.F > 0
    x, y, z = cd.x[keep], cd.y[keep], np.log(cd.F[keep])
    Z = np.column_stack([z] + [z[rng.permutation(z.size)] for _ in range(N_SHUFFLE)])
    v, ic, nb, k0 = vif_curve(x, y, Z, ext, rng)
    rec["icc_obs"] = ic[:, 0].tolist()
    rec["icc_shuf_q975"] = np.nanquantile(ic[:, 1:], 0.975, axis=1).tolist()
    rec["k0"] = k0.tolist()
    rec["vif_obs"] = v[:, 0].tolist()
    rec["vif_shuf_median"] = np.nanmedian(v[:, 1:], 1).tolist()
    rec["vif_shuf_q975"] = np.nanquantile(v[:, 1:], 0.975, axis=1).tolist()
    rec["vif_shuf_q025"] = np.nanquantile(v[:, 1:], 0.025, axis=1).tolist()
    rec["n_blocks"] = nb.tolist()
    od = OUT / "e5" / "cities"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(od / f"{c}.json", "w"))
    return c


def run(main_tag, x0, iutfd, workers):
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    todo = [c for c in cities if not (OUT / "e5" / "cities" / f"{c}.json").exists()]
    m = load_config()["main"]
    with Pool(workers) as pool:
        for c in pool.imap_unordered(city_job, [(c, str(x0), str(iutfd), m) for c in todo]):
            print(c, flush=True)


def mixture_p90(n_w, N, s, q=0.9):
    """P90 of |Z| s sqrt(1/n_w - 1/N) over windows (mixture of normals)."""
    sd = s * np.sqrt(np.maximum(1 / n_w - 1 / N, 0))
    sd = sd[sd > 0]
    if sd.size == 0:
        return 0.0
    g = lambda e: np.mean(2 * stats.norm.sf(e / sd)) - (1 - q)
    return float(optimize.brentq(g, 1e-12, 50 * sd.max()))


def _law(fs, st, eps):
    ok = np.isfinite(st) & (st > 0) & (fs < 1)
    if ok.sum() < 4:
        return np.nan, np.nan
    u = np.log((1 - fs[ok]) / fs[ok])
    b, a = np.polyfit(u, np.log(st[ok] / eps), 1)
    return float(b), float(np.exp(a))


def summary(main_tag):
    m = load_config()["main"]
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    R = {c: json.load(open(OUT / "e5" / "cities" / f"{c}.json")) for c in cities if (OUT / "e5" / "cities" / f"{c}.json").exists()}
    od = OUT / "e5"
    # ---- geometry and old exponents
    rows = []
    for c, r in R.items():
        js = json.load(open(OUT / main_tag / c / "city.json", encoding="utf-8"))
        x8 = js["x8"]
        row = dict(city=c, n_stations=r["n_stations"], area_km2=r["area_km2"], short_side_km=r["short_side_m"] / 1000,
                   D_station=dimension(L_OLD, np.array(r["N_station_old"])), D_station_wide=dimension(L_WIDE, np.array(r["N_station_wide"])),
                   D_road=dimension(L_OLD, np.array(r["N_road_old"])) if "N_road_old" in r else np.nan,
                   D_road_wide=dimension(L_WIDE, np.array(r["N_road_wide"])) if "N_road_wide" in r else np.nan,
                   estimable=bool(x8.get("estimable")))
        for k in ("alpha", "beta"):
            row[k] = x8.get(k, np.nan)
            row[f"{k}_shuffled"] = x8.get(f"{k}_perm_median", np.nan)
            row[f"{k}_equal"] = r[f"{k}_equal"] if row["estimable"] else np.nan
        lnn = np.array(r["ln_nodes_old"])
        row["beta_geo"] = -row["D_station"] / float(np.mean(lnn[lnn > 0])) if row["estimable"] else np.nan
        rows.append(row)
    g = pd.DataFrame(rows)
    g.to_csv(od / "geometry.csv", index=False)
    ge = g[g.estimable]
    corr = []
    pairs = [("alpha", "D_station"), ("beta", "D_station"), ("alpha", "D_road"), ("beta", "D_road"), ("D_station", "D_road"),
             ("alpha", "alpha_shuffled"), ("beta", "beta_shuffled"), ("alpha", "alpha_equal"), ("beta", "beta_equal"),
             ("beta", "beta_geo"), ("alpha_shuffled", "D_station"), ("beta_shuffled", "D_station"), ("alpha", "n_stations"), ("beta", "n_stations")]
    for u, v in pairs:
        d = ge[[u, v]].dropna()
        corr.append(dict(x=u, y=v, cities=len(d), spearman=float(stats.spearmanr(d[u], d[v])[0]), pearson=float(stats.pearsonr(d[u], d[v])[0]),
                         median_abs_diff=float((d[u] - d[v]).abs().median()) if u.split("_")[0] == v.split("_")[0] or v == "beta_geo" else np.nan))
    d = g[["D_station", "D_road"]].dropna()
    corr.append(dict(x="D_station", y="D_road", cities=len(d), spearman=float(stats.spearmanr(d.D_station, d.D_road)[0]),
                     pearson=float(stats.pearsonr(d.D_station, d.D_road)[0]), median_abs_diff=np.nan, note="all cities"))
    corr = pd.DataFrame(corr)
    corr.to_csv(od / "geometry_corr.csv", index=False)
    # ---- blocks
    brow = []
    for c, r in R.items():
        for i, b in enumerate(BLOCKS):
            brow.append(dict(city=c, b=b, n_blocks=r["n_blocks"][i], k0=r["k0"][i], icc_obs=r["icc_obs"][i], icc_shuf_q975=r["icc_shuf_q975"][i],
                             vif_obs=r["vif_obs"][i], vif_shuf_median=r["vif_shuf_median"][i],
                             vif_shuf_q025=r["vif_shuf_q025"][i], vif_shuf_q975=r["vif_shuf_q975"][i]))
    bl = pd.DataFrame(brow)
    bl["above"] = bl.vif_obs > bl.vif_shuf_q975
    bl.to_csv(od / "blocks.csv", index=False)
    bs = bl.groupby("b").agg(cities=("city", "count"), vif_obs_median=("vif_obs", "median"), vif_obs_q25=("vif_obs", lambda x: x.quantile(0.25)),
                             vif_obs_q75=("vif_obs", lambda x: x.quantile(0.75)), vif_shuf_median=("vif_shuf_median", "median"),
                             n_above=("above", "sum"), n_blocks_median=("n_blocks", "median"), k0_median=("k0", "median"),
                             icc_obs_median=("icc_obs", "median"), icc_q25=("icc_obs", lambda x: x.quantile(0.25)),
                             icc_q75=("icc_obs", lambda x: x.quantile(0.75))).reset_index()
    bs.to_csv(od / "blocks_summary.csv", index=False)
    # ---- window-count mixture vs observed P90 and the E2 median-count formula
    iqr = json.load(open(OUT / "e2" / "iqr.json"))["iqr"]
    mix, curves = [], []
    for c in cities:
        z = np.load(OUT / "e2" / "windows" / f"{c}.npz")
        s = json.loads(str(z["s"]))
        fs, N = z["levels"], int(z["N"])
        for p in PROBES_E2:
            eps = 0.25 * iqr[p]
            obs = np.full(fs.size, np.nan)
            pmix = np.full(fs.size, np.nan)
            pmed = np.full(fs.size, np.nan)
            n10 = np.full(fs.size, np.nan)
            for i in range(fs.size):
                n = z[f"n_{i}"]
                n10[i] = np.quantile(n, 0.1)
                if f"err_{p}_{i}" not in z.files:
                    continue
                nu = n[n >= int(m["n_w"])].astype(float)
                obs[i] = np.quantile(z[f"err_{p}_{i}"], 0.9)
                pmix[i] = mixture_p90(nu, N, s[p])
                pmed[i] = stats.norm.ppf(0.95) * s[p] * np.sqrt(max(1 / np.median(nu) - 1 / N, 0))
            b_obs, C_obs = _law(fs, obs, eps)
            b_mix, C_mix = _law(fs, pmix, eps)
            b_med, C_med = _law(fs, pmed, eps)
            okn = (n10 > 0) & (fs < 1)
            gam = float(np.polyfit(np.log(fs[okn]), np.log(n10[okn]), 1)[0]) if okn.sum() >= 4 else np.nan
            fo = criteria.critical_f(fs, obs, eps)
            fx = criteria.critical_f(fs, np.where(np.isfinite(obs), pmix, np.nan), eps)
            fm = criteria.critical_f(fs, np.where(np.isfinite(obs), pmed, np.nan), eps)
            mix.append(dict(city=c, probe=p, N=N, s=s[p], eps=eps, b_obs=b_obs, b_mix=b_mix, b_median_count=b_med, gamma_p10=gam,
                            C_obs=C_obs, C_mix=C_mix, f_obs=fo["f_star"], flag_obs=fo["flag"], f_mix=fx["f_star"], flag_mix=fx["flag"],
                            f_median_count=fm["f_star"], flag_median_count=fm["flag"],
                            ratio_obs_mix_median=float(np.nanmedian(obs / pmix)), ratio_obs_med_median=float(np.nanmedian(obs / pmed))))
            curves += [dict(city=c, probe=p, f=float(f), obs=float(o), mixture=float(x_), median_count=float(y_), n_p10=float(q_))
                       for f, o, x_, y_, q_ in zip(fs, obs, pmix, pmed, n10)]
    mix = pd.DataFrame(mix)
    mix.to_csv(od / "mixture.csv", index=False)
    pd.DataFrame(curves).to_csv(od / "mixture_curves.csv", index=False)
    ms = []
    geo = g.set_index("city")
    for p, gp in mix.groupby("probe"):
        ok = gp[gp.flag_obs == "ok"]
        for lab in ("mix", "median_count"):
            col = f"f_{lab}"
            okp = ok[ok[f"flag_{lab}"] == "ok"] if lab == "mix" else ok[ok[f"flag_{lab}"] == "ok"]
            d = ok[col] - ok.f_obs
            bcol = "b_mix" if lab == "mix" else "b_median_count"
            ms.append(dict(probe=p, predictor=lab, n_ok=len(ok), mae=float(d.abs().mean()), share_within_0p1=float((d.abs() <= 0.1).mean()),
                           spearman=float(stats.spearmanr(ok.f_obs, ok[col])[0]), b_obs_median=float(gp.b_obs.median()),
                           b_pred_median=float(gp[bcol].median()), spearman_b=float(stats.spearmanr(gp.b_obs, gp[bcol], nan_policy="omit")[0]),
                           ratio_obs_pred_median=float(gp[f"ratio_obs_{'mix' if lab == 'mix' else 'med'}_median"].median()),
                           n_pred_ok=len(okp)))
        d = gp.merge(geo[["D_station"]], left_on="city", right_index=True)
        ms.append(dict(probe=p, predictor="gamma_p10 vs b_obs", spearman_b=float(stats.spearmanr(d.gamma_p10, d.b_obs, nan_policy="omit")[0]),
                       b_pred_median=float((d.gamma_p10 / 2).median())))
        ms.append(dict(probe=p, predictor="D_station vs b_obs", spearman_b=float(stats.spearmanr(d.D_station, d.b_obs, nan_policy="omit")[0])))
    ms = pd.DataFrame(ms)
    ms.to_csv(od / "mixture_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(g.round(3).to_string(index=False))
    print(corr.round(3).to_string(index=False))
    print(bs.round(3).to_string(index=False))
    print(ms.round(3).to_string(index=False))


def influence_values(F):
    z = np.log(F)
    mu, sd = z.mean(), z.std(ddof=1)
    m, s = F.mean(), F.std(ddof=1)
    return dict(mean_log=z - mu, sd_log=((z - mu) ** 2 - sd ** 2) / (2 * sd),
                site_cv=((F - m) ** 2 - s ** 2) / (2 * s * m) - s * (F - m) / m ** 2)


def _vif_at(w, v):
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.nan
    lb, vv = np.log(BLOCKS[ok]), v[ok]
    return float(np.interp(np.log(w), lb, vv))           # constant beyond the first / last block size


def deff_stage(main_tag, x0):
    m = load_config()["main"]
    cities = json.load(open(OUT / main_tag / "summary.json", encoding="utf-8"))["cities"]
    iqr = json.load(open(OUT / "e2" / "iqr.json"))["iqr"]
    rows, curves = [], []
    for c in cities:
        cd = load_city(c, Path(x0), m)
        keep = cd.F > 0
        x, y, F = cd.x[keep], cd.y[keep], cd.F[keep]
        IF = influence_values(F)
        vif, icc, _, _ = vif_curve(x, y, np.column_stack([IF[p] for p in PROBES_E2]), cd.extent, _rng("e5", "deff", c))
        z = np.load(OUT / "e2" / "windows" / f"{c}.npz")
        s = json.loads(str(z["s"]))
        fs, N = z["levels"], int(z["N"])
        A = cd.area_km2 * 1e6
        a = np.load(OUT / main_tag / c / "city.npz")
        for j, p in enumerate(PROBES_E2):
            eps = 0.25 * iqr[p]
            obs, mix, blk = (np.full(fs.size, np.nan) for _ in range(3))
            perm = np.nanmedian(a[f"{p}__p90_perm"], 1)
            for i, f in enumerate(fs):
                if f"err_{p}_{i}" not in z.files:
                    continue
                n = z[f"n_{i}"]
                nu = n[n >= int(m["n_w"])].astype(float)
                obs[i] = np.quantile(z[f"err_{p}_{i}"], 0.9)
                mix[i] = mixture_p90(nu, N, s[p])
                blk[i] = mix[i] * np.sqrt(max(_vif_at(np.sqrt(f * A), vif[:, j]), 0.0))
                curves.append(dict(city=c, probe=p, f=float(f), obs=obs[i], perm=float(perm[i]), mixture=mix[i], block=blk[i],
                                   vif_w=_vif_at(np.sqrt(f * A), vif[:, j])))
            perm = np.where(np.isfinite(obs), perm, np.nan)
            r = dict(city=c, probe=p, N=N, eps=eps, vif_500=float(vif[2, j]), vif_1000=float(vif[4, j]), vif_2000=float(vif[6, j]),
                     icc_500=float(icc[2, j]), icc_1000=float(icc[4, j]),
                     ratio_obs_perm=float(np.nanmedian(obs / perm)), ratio_obs_mix=float(np.nanmedian(obs / mix)),
                     ratio_obs_block=float(np.nanmedian(obs / blk)), ratio_perm_mix=float(np.nanmedian(perm / mix)))
            for lab, cur in (("obs", obs), ("perm", perm), ("mix", mix), ("block", blk)):
                cr = criteria.critical_f(fs, cur, eps)
                r[f"f_{lab}"], r[f"flag_{lab}"] = cr["f_star"], cr["flag"]
                r[f"b_{lab}"] = _law(fs, cur, eps)[0]
            rows.append(r)
        print(c, flush=True)
    d = pd.DataFrame(rows)
    od = OUT / "e5"
    d.to_csv(od / "deff.csv", index=False)
    pd.DataFrame(curves).to_csv(od / "deff_curves.csv", index=False)
    srows = []
    for p, g in d.groupby("probe"):
        ok = g[g.flag_obs == "ok"]
        for lab in ("perm", "mix", "block"):
            dd = ok[f"f_{lab}"] - ok.f_obs
            srows.append(dict(probe=p, predictor=lab, n_ok=len(ok), mae=float(dd.abs().mean()), share_within_0p1=float((dd.abs() <= 0.1).mean()),
                              median_signed=float(dd.median()), spearman=float(stats.spearmanr(ok.f_obs, ok[f"f_{lab}"])[0]),
                              ratio_obs_pred_median=float(g[f"ratio_obs_{lab}"].median()),
                              ratio_q25=float(g[f"ratio_obs_{lab}"].quantile(0.25)), ratio_q75=float(g[f"ratio_obs_{lab}"].quantile(0.75)),
                              b_obs_median=float(g.b_obs.median()), b_pred_median=float(g[f"b_{lab}"].median()),
                              spearman_ratio_obs_perm_vs_vif1000=float(stats.spearmanr(g.ratio_obs_perm, g.vif_1000, nan_policy="omit")[0]),
                              vif_1000_median=float(g.vif_1000.median()), icc_500_median=float(g.icc_500.median())))
    sm = pd.DataFrame(srows)
    sm.to_csv(od / "deff_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(sm.round(3).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "summary", "deff"])
    ap.add_argument("--main", default="main34")
    ap.add_argument("--x0", default=str(OUT / "x0"))
    ap.add_argument("--iutfd", default="")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    (OUT / "e5").mkdir(parents=True, exist_ok=True)
    if a.what == "run":
        iutfd = Path(a.iutfd) if a.iutfd else repo_path(load_config()["paths"]["iutfd_dir"])
        run(a.main, a.x0, iutfd, a.workers)
    elif a.what == "deff":
        deff_stage(a.main, Path(a.x0))
    else:
        summary(a.main)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
