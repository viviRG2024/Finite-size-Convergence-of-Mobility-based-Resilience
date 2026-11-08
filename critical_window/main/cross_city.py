"""Cross-city stage: tolerance, critical windows, H1-H4, layout control, centre rule and the summary of the hypothesis tests."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..core import collapse, criteria, hypotheses, probes, windows
from .city_stage import PROBES


def _load(out_dir: Path, city: str):
    with open(out_dir / city / "city.json", encoding="utf-8") as fh:
        js = json.load(fh)
    arr = dict(np.load(out_dir / city / "city.npz"))
    return js, arr


def _dl_meta(y, se):
    """DerSimonian-Laird random-effects summary: mean, tau^2, I^2."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, v = y[ok], se[ok] ** 2
    if y.size < 3:
        return dict(n=int(y.size), mean=np.nan, tau2=np.nan, I2=np.nan)
    w = 1 / v
    mu_f = np.sum(w * y) / w.sum()
    Q = np.sum(w * (y - mu_f) ** 2)
    df = y.size - 1
    c = w.sum() - np.sum(w ** 2) / w.sum()
    tau2 = max(0.0, (Q - df) / c)
    ws = 1 / (v + tau2)
    return dict(n=int(y.size), mean=float(np.sum(ws * y) / ws.sum()), tau2=float(tau2),
                I2=float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0, Q=float(Q))


def cross_city(out_dir: Path, cities: list, m: dict, city_data: dict | None = None, tag: str = "main") -> dict:
    alpha = float(m["alpha"])
    J, A = {}, {}
    for c in cities:
        J[c], A[c] = _load(out_dir, c)
    levels = np.array(J[cities[0]]["levels"])
    rows, fstar_rows = [], []
    eps = {p: criteria.tolerance([J[c]["theta_E"][p] for c in cities], float(m["tol_frac"])) for p in PROBES}
    iqr = {p: eps[p] / float(m["tol_frac"]) for p in PROBES}

    # ---- f*, CI, conversions; H1 (Holm) and H2 per probe
    for p in PROBES:
        h1p = np.array([J[c]["tests"][p]["h1"]["p"] for c in cities], float)
        h1_adj = np.full(h1p.size, np.nan)
        okp = np.isfinite(h1p)
        h1_adj[okp] = criteria.holm(h1p[okp])
        for i, c in enumerate(cities):
            js, a = J[c], A[c]
            usable = a["lev__usable"] >= int(m["min_usable_windows"])
            p90 = np.where(usable, a[f"{p}__p90_obs"], np.nan)
            fs = criteria.critical_f(levels, p90, eps[p])
            boot = a[f"{p}__p90_boot"]
            fb_list = []
            for b in range(boot.shape[1]):
                r = criteria.critical_f(levels, np.where(usable, boot[:, b], np.nan), eps[p])
                fb_list.append(r["f_star"] if r["flag"] == "ok" else np.nan)
            fb_arr = np.array(fb_list)
            area = js["area_km2"]
            medn = a["lev__med_n"]
            xi = js["x1"]["xi"]
            ident = js["x1"]["xi_identifiable"]
            A_star = fs["f_star"] * area if fs["flag"] == "ok" else np.nan
            N_star = criteria.interp_at(levels, medn, fs["f_star"]) if fs["flag"] == "ok" else np.nan
            h2 = js["tests"][p]["h2"]
            # centre rule for the breakpoint (H2) and for f* (reported)
            f_full = js["x9"]["f_full"]

            def geometric(f_thr):
                if not (np.isfinite(f_thr) and np.isfinite(f_full)):
                    return False
                kt = int(np.argmin(np.abs(levels - f_thr)))
                kf = int(np.argmin(np.abs(levels - f_full)))
                if abs(kt - kf) > 1 or kt == 0:
                    return False
                pin = a[f"{p}__p90_cin"][kt - 1]
                return bool(np.isfinite(pin) and pin <= eps[p])

            rows.append(dict(probe=p, city=c, h1_stat=js["tests"][p]["h1"]["stat"], h1_p_raw=h1p[i], h1_p=h1_adj[i],
                             h2_p=h2["p"], dbic=h2["dbic"], f_b=h2["f_b"], geometric=geometric(h2["f_b"])))
            fstar_rows.append(dict(probe=p, city=c, eps=eps[p], theta_E=js["theta_E"][p], f_star=fs["f_star"], flag=fs["flag"],
                                   f_star_ci_lo=float(np.nanquantile(fb_arr, 0.025)) if np.isfinite(fb_arr).sum() >= 20 else np.nan,
                                   f_star_ci_hi=float(np.nanquantile(fb_arr, 0.975)) if np.isfinite(fb_arr).sum() >= 20 else np.nan,
                                   boot_ok_share=float(np.isfinite(fb_arr).mean()), A_star_km2=A_star, N_star=N_star,
                                   A_bound_km2=fs["f_star"] * area if fs["flag"] in ("above", "below") else np.nan,
                                   xi_m=xi, xi_identifiable=ident,
                                   A_over_xi2=(A_star * 1e6 / xi ** 2) if (ident and np.isfinite(A_star)) else np.nan,
                                   geometric_fstar=geometric(fs["f_star"]) if fs["flag"] == "ok" else False,
                                   p90_over_iqr_small=float(np.nanmean(p90[np.flatnonzero(usable)[:3]]) / iqr[p]) if usable.any() and iqr[p] > 0 else np.nan))
    res = pd.DataFrame(rows)
    fst = pd.DataFrame(fstar_rows)

    # ---- summary of H1 and H2
    decision = hypotheses.summarise(res.rename(columns={}), alpha=alpha, primary=tuple(m["primary_probes"]))

    # ---- H3: dispersion of critical values and collapse quality
    h3 = []
    for p in PROBES:
        t = fst[(fst.probe == p) & (fst.flag == "ok")]
        row = dict(probe=p, n_cities=len(t))
        for col, name in (("f_star", "f"), ("A_star_km2", "A"), ("N_star", "N"), ("A_over_xi2", "A/xi2")):
            v = t[col].to_numpy(float)
            v = v[np.isfinite(v) & (v > 0)]
            row[f"sd_log_{name}"] = float(np.std(np.log(v), ddof=1)) if v.size >= 3 else np.nan
            row[f"n_{name}"] = int(v.size)
        curves = {k: [] for k in ("f", "A", "N", "A/xi2")}
        curves_id = {k: [] for k in ("f", "A", "N", "A/xi2")}
        for c in cities:
            js, a = J[c], A[c]
            usable = a["lev__usable"] >= int(m["min_usable_windows"])
            y = np.log(np.where(usable, a[f"{p}__p90_obs"], np.nan))
            xs = dict(f=levels, A=levels * js["area_km2"], N=a["lev__med_n"],
                      **{"A/xi2": levels * js["area_km2"] * 1e6 / js["x1"]["xi"] ** 2 if js["x1"]["xi_identifiable"] else np.full(levels.size, np.nan)})
            for k in curves:
                curves[k].append((xs[k], y))
                if js["x1"]["xi_identifiable"]:
                    curves_id[k].append((xs[k], y))
        for k in curves:
            row[f"collapse_{k}"] = collapse.collapse_quality(curves[k]) if k != "A/xi2" else np.nan
            row[f"collapse_xi_cities_{k}"] = collapse.collapse_quality(curves_id[k]) if len(curves_id[k]) >= 3 else np.nan
        h3.append(row)
    h3 = pd.DataFrame(h3)

    # ---- X8 layout control (random-effects across cities)
    x8 = []
    for name in ("delta_beta", "delta_alpha", "beta", "alpha"):
        y = [J[c]["x8"].get(name, np.nan) for c in cities]
        se_key = f"{name}_se" if name.startswith("delta") else f"delta_{name}_se"
        se = [J[c]["x8"].get(se_key, np.nan) for c in cities]
        x8.append(dict(quantity=name, **_dl_meta(y, se), median=float(np.nanmedian(y)) if np.isfinite(y).any() else np.nan))
    x8 = pd.DataFrame(x8)

    # ---- X7 ranking stability (needs station data)
    x7 = []
    if city_data:
        reps = int(m["rank_reps"])
        for p in PROBES:
            t = fst[fst.probe == p].set_index("city")
            thE = np.array([J[c]["theta_E"][p] for c in cities])
            res_rho = {"at_A*": [], "at_A*/4": []}
            for r in range(reps):
                rng = np.random.default_rng(zlib.crc32(f"X7|{tag}|{p}|{r}".encode()))
                for key, div in (("at_A*", 1.0), ("at_A*/4", 4.0)):
                    vals = []
                    for c in cities:
                        cd = city_data[c]
                        fz = t.loc[c, "f_star"]
                        fz = 0.9 if t.loc[c, "flag"] == "above" else fz
                        f = float(np.clip(fz / div, levels[0], 0.9)) if np.isfinite(fz) else np.nan
                        if not np.isfinite(f):
                            vals.append(np.nan)
                            continue
                        b = windows.random_windows(cd.extent, f, 1, rng)[0]
                        mm = windows.in_box(cd.x, cd.y, b)
                        if mm.sum() < 2:
                            vals.append(np.nan)
                            continue
                        vals.append(probes.level_probes(cd.x[mm], cd.y[mm], cd.F[mm][:, None], b, float(m["grid_m"]))[p][0])
                    vals = np.array(vals, float)
                    ok = np.isfinite(vals) & np.isfinite(thE)
                    if ok.sum() >= 5:
                        res_rho[key].append(stats.spearmanr(vals[ok], thE[ok])[0])
            for key, v in res_rho.items():
                v = np.array(v, float)
                x7.append(dict(probe=p, window=key, n_reps=int(v.size), rho_median=float(np.nanmedian(v)) if v.size else np.nan,
                               rho_p05=float(np.nanquantile(v, 0.05)) if v.size else np.nan,
                               share_rho_ge_0p9=float(np.mean(v >= 0.9)) if v.size else np.nan))
    x7 = pd.DataFrame(x7)

    # ---- tables
    table1 = []
    for k, f in enumerate(levels):
        for p in PROBES:
            ratio, p90r, ov = [], [], []
            for c in cities:
                a = A[c]
                if a["lev__usable"][k] < int(m["min_usable_windows"]):
                    continue
                ratio.append(a[f"{p}__sd_obs"][k] / np.nanmedian(a[f"{p}__sd_perm"][k]))
                p90r.append(a[f"{p}__p90_obs"][k] / iqr[p] if iqr[p] > 0 else np.nan)
                ov.append(a["lev__overlap"][k])
            table1.append(dict(f=f, probe=p, n_cities=len(ratio), median_sd_ratio_obs_to_n1=float(np.nanmedian(ratio)) if ratio else np.nan,
                               median_p90_over_iqr=float(np.nanmedian(p90r)) if p90r else np.nan,
                               mean_overlap=float(np.nanmean(ov)) if ov else np.nan, overlap_analytic=windows.expected_overlap(f)))
    table1 = pd.DataFrame(table1)
    table2 = pd.DataFrame([dict(city=c, n_stations=J[c]["n_stations"], n_days=J[c]["n_days"], area_km2=J[c]["area_km2"],
                                short_km=J[c]["x1"]["short_side_m"] / 1000, xi_m=J[c]["x1"]["xi"],
                                xi_ci_lo=J[c]["x1"]["xi_ci"][0], xi_ci_hi=J[c]["x1"]["xi_ci"][1],
                                xi_identifiable=J[c]["x1"]["xi_identifiable"], sigma_z=J[c]["x1"]["sigma_z"],
                                f_full=J[c]["x9"]["f_full"], n_centres=len(J[c]["x9"]["centres"])) for c in cities])

    summary = dict(tag=tag, cities=cities, eps=eps, iqr=iqr, tests=decision)
    res.to_csv(out_dir / "h1_h2.csv", index=False)
    fst.to_csv(out_dir / "critical_windows.csv", index=False)
    h3.to_csv(out_dir / "h3_dispersion_collapse.csv", index=False)
    x8.to_csv(out_dir / "x8_layout_meta.csv", index=False)
    x7.to_csv(out_dir / "x7_ranking.csv", index=False)
    table1.to_csv(out_dir / "table1_levels.csv", index=False)
    table2.to_csv(out_dir / "table2_cities.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return summary
