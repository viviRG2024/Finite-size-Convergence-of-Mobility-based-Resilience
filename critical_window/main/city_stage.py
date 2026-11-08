"""Per-city stage of the main analysis (X1, X2, X3, X4 and X6c inputs, X8, X9).

Everything that does not need other cities is computed here and saved to <out>/<city>/city.npz + city.json.
Flow matrix columns: 0 = observed, then N1 permutations, N2 simulations, day-bootstrap replicates.
"""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np

from ..core import criteria, probes, synth, transition, windows
from .data import CityData

PROBES = ("cv500", "hn500", "sd_log", "mean_log", "site_cv")


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


# ------------------------------------------------------------------------------------------------ X1
def _directional_xi(x, y, z, max_d, rng, max_pts=3000) -> list:
    if z.size > max_pts:
        idx = rng.choice(z.size, max_pts, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    i, j = np.triu_indices(z.size, 1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    h = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    g = 0.5 * (z[i] - z[j]) ** 2
    var = np.var(z)
    bins = probes.default_bins(max_d)
    mid = 0.5 * (bins[:-1] + bins[1:])
    out = []
    for a0 in (0.0, 45.0, 90.0, 135.0):
        sec = np.minimum(np.abs(ang - a0), 180.0 - np.abs(ang - a0)) <= 22.5
        k = np.digitize(h[sec], bins) - 1
        use = (k >= 0) & (k < bins.size - 1)
        cnt = np.bincount(k[use], minlength=bins.size - 1)
        gam = np.bincount(k[use], g[sec][use], minlength=bins.size - 1) / np.maximum(cnt, 1)
        rho = np.where(cnt >= 30, 1 - gam / var, np.nan)
        xi, prev = np.nan, None
        for q in range(rho.size):
            if not np.isfinite(rho[q]):
                continue
            if rho[q] < np.exp(-1):
                xi = mid[q] if prev is None else prev[0] + (prev[1] - np.exp(-1)) / (prev[1] - rho[q]) * (mid[q] - prev[0])
                break
            prev = (mid[q], rho[q])
        out.append(float(xi))
    return out


def _residual(z, X):
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    return z - X @ coef


def x1_correlation_length(cd: CityData, m: dict, n_boot: int, tag: str) -> dict:
    x, y, z = cd.x, cd.y, np.log(cd.F)
    short = cd.short_side
    max_d = short / 2
    rng = _rng(cd.city, tag, "xi")
    xi = probes.corr_length(x, y, z, max_d, rng=rng)
    ident = bool(np.isfinite(xi) and xi <= short * float(m["xi_ident_frac"]))
    boot = []
    for Fb in cd.bootstrap(n_boot, _rng(cd.city, tag, "xi-boot")).T:
        boot.append(probes.corr_length(x, y, np.log(Fb), max_d, rng=rng))
    boot = np.array(boot, float)
    trend = _residual(z, np.column_stack([np.ones_like(x), x - x.mean(), y - y.mean()]))
    classes = np.unique(cd.fclass)
    cls = _residual(z, (cd.fclass[:, None] == classes[None, :]).astype(float)) if classes.size > 1 else z - z.mean()
    true, mean_est = probes.xi_bias_curve(x, y, cd.extent, _rng(cd.city, tag, "xi-bias"), reps=10)
    return dict(xi=float(xi), xi_identifiable=ident, xi_use=float(xi) if np.isfinite(xi) else float(max_d),
                xi_ci=[float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))] if np.isfinite(boot).any() else [np.nan, np.nan],
                xi_boot_found_share=float(np.isfinite(boot).mean()),
                xi_detrended=float(probes.corr_length(x, y, trend, max_d, rng=rng)),
                xi_class_adjusted=float(probes.corr_length(x, y, cls, max_d, rng=rng)),
                xi_bias_corrected=float(probes.xi_correct(xi, true, mean_est)),
                xi_directional=_directional_xi(x, y, z, max_d, rng),
                sigma_z=float(np.std(z, ddof=1)), mean_log=float(np.mean(z)), short_side_m=float(short))


# ------------------------------------------------------------------------------------------------ X9 centres
def x9_centres(cd: CityData, bandwidth: float, m: dict) -> dict:
    g = float(m["centre_grid_m"])
    x0, y0, x1, y1 = cd.extent
    gx = np.arange(x0, x1 + g, g)
    gy = np.arange(y0, y1 + g, g)
    GX, GY = np.meshgrid(gx, gy)
    dens = np.zeros_like(GX)
    h2 = 2 * bandwidth ** 2
    for xi_, yi_, w in zip(cd.x, cd.y, cd.F):
        dens += w * np.exp(-((GX - xi_) ** 2 + (GY - yi_) ** 2) / h2)
    pad = np.pad(dens, 1, constant_values=-np.inf)
    nb = np.max([pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)], axis=0)
    peaks = (dens >= nb) & (dens >= float(m["centre_peak_frac"]) * dens.max())
    py, px = np.nonzero(peaks)
    order = np.argsort(-dens[py, px])
    centres = [(float(GX[py[k], px[k]]), float(GY[py[k], px[k]]), float(dens[py[k], px[k]] / dens.max())) for k in order]
    return dict(centres=centres, main_centre=centres[0][:2] if centres else [np.nan, np.nan], bandwidth_m=float(bandwidth))


# ------------------------------------------------------------------------------------------------ main
def city_stage(cd: CityData, m: dict, out_dir: Path, tag: str, counts: dict) -> dict:
    t0 = time.time()
    fs = np.array(m["f_grid"], float)
    levels = fs[fs < 1.0]
    L = levels.size
    n_perm, n_n2, n_n3, n_boot = counts["n_perm"], counts["n_n2"], counts["n_n3"], counts["n_boot"]
    x, y, ext, n = cd.x, cd.y, cd.extent, cd.F.size
    q = float(m["stat_quantile"])
    n_w, n_rand, min_use = int(m["n_w"]), int(m["n_random"]), int(m["min_usable_windows"])

    x1 = x1_correlation_length(cd, m, counts["n_boot_xi"], tag)
    rng = _rng(cd.city, tag, "columns")
    perms = np.column_stack([cd.F[rng.permutation(n)] for _ in range(n_perm)])
    n2 = synth.stationary_flows(x, y, ext, x1["xi_use"], x1["sigma_z"], x1["mean_log"], n_n2, _rng(cd.city, tag, "n2"))
    boot = cd.bootstrap(n_boot, _rng(cd.city, tag, "boot"))
    F_all = np.column_stack([cd.F, perms, n2, boot])
    c_perm = slice(1, 1 + n_perm)
    c_n2 = slice(1 + n_perm, 1 + n_perm + n_n2)
    c_boot = slice(1 + n_perm + n_n2, F_all.shape[1])
    theta_E = probes.level_probes(x, y, F_all, ext, float(m["grid_m"]))

    cen = x9_centres(cd, max(x1["xi_use"], float(m.get("centre_min_bandwidth_m", 250.0))), m)
    mc = cen["main_centre"]

    shape = lambda *s: np.full(s, np.nan)
    res = {p: dict(p90_obs=shape(L), med_obs=shape(L), sd_obs=shape(L), p90_perm=shape(L, n_perm), sd_perm=shape(L, n_perm),
                   p90_n2=shape(L, n_n2), p90_boot=shape(L, n_boot), p90_n3=shape(L), p90_cin=shape(L), p90_cout=shape(L),
                   e_nested=shape(L), p90_tiles=shape(L)) for p in PROBES}
    lev = dict(usable=np.zeros(L, int), med_n=shape(L), overlap=shape(L), overlap_analytic=shape(L), p_contains=shape(L),
               n_cin=np.zeros(L, int), n_tiles=np.zeros(L, int))
    brng = _rng(cd.city, tag, "positions")
    for k, f in enumerate(levels):
        wins = windows.random_windows(ext, f, n_rand, _rng(cd.city, tag, "windows", f))
        Ns = windows.counts(x, y, wins)
        use = Ns >= n_w
        lev["usable"][k] = int(use.sum())
        lev["overlap_analytic"][k] = windows.expected_overlap(f)
        lev["p_contains"][k] = windows.p_contains(ext, f, mc) if np.isfinite(mc[0]) else np.nan
        if use.sum() < min_use:
            continue
        lev["med_n"][k] = float(np.median(Ns[use]))
        lev["overlap"][k] = windows.mean_pairwise_overlap(wins[use])
        vals = {p: [] for p in PROBES}
        for b in wins[use]:
            mm = windows.in_box(x, y, b)
            pr = probes.level_probes(x[mm], y[mm], F_all[mm], b, float(m["grid_m"]))
            for p in PROBES:
                vals[p].append(pr[p])
        cin = np.array([(b[0] <= mc[0] <= b[2]) and (b[1] <= mc[1] <= b[3]) for b in wins[use]]) if np.isfinite(mc[0]) else np.zeros(use.sum(), bool)
        lev["n_cin"][k] = int(cin.sum())
        idx = brng.integers(0, int(use.sum()), (n_boot, int(use.sum())))
        # nested window and tiles (observed column only)
        nb = windows.nested_window(ext, f)[0]
        mm = windows.in_box(x, y, nb)
        nested = probes.level_probes(x[mm], y[mm], F_all[mm][:, :1], nb, float(m["grid_m"])) if mm.sum() >= n_w else None
        tiles = [t for t in windows.tiling(ext, f) if windows.in_box(x, y, t).sum() >= n_w]
        lev["n_tiles"][k] = len(tiles)
        tile_vals = [probes.level_probes(x[windows.in_box(x, y, t)], y[windows.in_box(x, y, t)],
                                         F_all[windows.in_box(x, y, t)][:, :1], t, float(m["grid_m"])) for t in tiles]
        # N3: random subsets of the median usable N from the whole extent
        n3_n = int(round(lev["med_n"][k]))
        n3rng = _rng(cd.city, tag, "n3", f)
        n3_vals = {p: [] for p in PROBES}
        for _ in range(n_n3):
            sub = n3rng.choice(n, min(n3_n, n), replace=False)
            pr = probes.level_probes(x[sub], y[sub], F_all[sub][:, :1], ext, float(m["grid_m"]))
            for p in PROBES:
                n3_vals[p].append(pr[p][0])
        for p in PROBES:
            V = np.array(vals[p])                      # windows x columns
            E = np.abs(V - theta_E[p][None, :])
            r = res[p]
            r["p90_obs"][k] = np.nanquantile(E[:, 0], q)
            r["med_obs"][k] = np.nanmedian(E[:, 0])
            r["sd_obs"][k] = np.nanstd(V[:, 0], ddof=1)
            r["p90_perm"][k] = np.nanquantile(E[:, c_perm], q, axis=0)
            r["sd_perm"][k] = np.nanstd(V[:, c_perm], ddof=1, axis=0)
            r["p90_n2"][k] = np.nanquantile(E[:, c_n2], q, axis=0)
            Eb = E[:, c_boot]
            r["p90_boot"][k] = np.nanquantile(np.take_along_axis(Eb.T, idx, axis=1), q, axis=1)
            e3 = np.abs(np.array(n3_vals[p]) - theta_E[p][0])
            r["p90_n3"][k] = np.nanquantile(e3, q) if np.isfinite(e3).any() else np.nan
            if cin.sum() >= 5:
                r["p90_cin"][k] = np.nanquantile(E[cin, 0], q)
            if (~cin).sum() >= 5:
                r["p90_cout"][k] = np.nanquantile(E[~cin, 0], q)
            if nested is not None:
                r["e_nested"][k] = abs(nested[p][0] - theta_E[p][0])
            if len(tile_vals) >= 4:
                r["p90_tiles"][k] = np.nanquantile([abs(tv[p][0] - theta_E[p][0]) for tv in tile_vals], q)

    # ---- H1 and H2 per probe
    tests = {}
    usable = lev["usable"] >= min_use
    for p in PROBES:
        r = res[p]
        lv = np.flatnonzero(usable & np.isfinite(r["sd_obs"]))[: int(m["h1_levels"])]
        h1 = dict(stat=np.nan, p=np.nan, levels=levels[lv].tolist())
        if lv.size:
            med_perm = np.nanmedian(r["sd_perm"][lv], axis=1)
            stat = float(np.mean(np.log(r["sd_obs"][lv] / med_perm)))
            null = np.mean(np.log(r["sd_perm"][lv] / med_perm[:, None]), axis=0)
            h1.update(stat=stat, p=criteria.perm_pvalue(stat, null))
        d_obs, fb = transition.dbic(levels, np.where(usable, r["p90_obs"], np.nan))
        d_null = np.array([transition.dbic(levels, np.where(usable, r["p90_n2"][:, j], np.nan))[0] for j in range(n_n2)])
        tests[p] = dict(h1=h1, h2=dict(dbic=float(d_obs), f_b=float(fb), p=criteria.perm_pvalue(d_obs, d_null),
                                       dbic_null_median=float(np.nanmedian(d_null)) if np.isfinite(d_null).any() else np.nan))

    # ---- X8 exponents: full extent and large windows
    x8 = dict(estimable=cd.short_side >= float(m["exponent_min_short_m"]))
    if x8["estimable"]:
        nbe = counts["n_boot_exponents"]
        npe = counts["n_perm_exponents"]
        cols = np.r_[0, np.arange(1, 1 + npe), np.arange(c_boot.start, c_boot.start + nbe)]
        ex = probes.exponents(x, y, F_all[:, cols], ext)
        for name in ("alpha", "beta"):
            v = ex[name]
            med_perm = float(np.nanmedian(v[1:1 + npe]))
            dv_boot = v[1 + npe:] - med_perm
            x8[name] = float(v[0])
            x8[f"{name}_perm_median"] = med_perm
            x8[f"delta_{name}"] = float(v[0] - med_perm)
            perm_sd = float(np.nanstd(v[1:1 + npe], ddof=1)) if np.isfinite(v[1:1 + npe]).sum() > 2 else np.nan
            boot_sd = float(np.nanstd(dv_boot, ddof=1)) if np.isfinite(dv_boot).sum() > 2 else np.nan
            x8[f"{name}_perm_sd"] = perm_sd
            x8[f"delta_{name}_boot_se"] = boot_sd
            # SE of the layout-controlled exponent: day-sampling variance plus the spread of the exponent over station arrangements
            x8[f"delta_{name}_se"] = float(np.sqrt(np.nansum([boot_sd ** 2, perm_sd ** 2]))) if np.isfinite([boot_sd, perm_sd]).any() else np.nan
        big = []
        for f in levels[levels >= float(m["exponent_min_f"])]:
            s = np.sqrt(f)
            if min((ext[2] - ext[0]) * s, (ext[3] - ext[1]) * s) < float(m["exponent_min_short_m"]):
                continue
            for b in windows.random_windows(ext, f, n_rand, _rng(cd.city, tag, "windows", f)):
                mm = windows.in_box(x, y, b)
                if mm.sum() < n_w:
                    continue
                e = probes.exponents(x[mm], y[mm], F_all[mm][:, :1 + npe], b)
                big.append((f, e["beta"][0] - np.nanmedian(e["beta"][1:]), e["alpha"][0] - np.nanmedian(e["alpha"][1:])))
        if big:
            big = np.array(big)
            x8["window_levels"] = sorted(set(big[:, 0].tolist()))
            x8["window_delta_beta_median"] = [float(np.nanmedian(big[big[:, 0] == f, 1])) for f in x8["window_levels"]]
            x8["window_delta_alpha_median"] = [float(np.nanmedian(big[big[:, 0] == f, 2])) for f in x8["window_levels"]]

    f_full = np.nan
    pc = lev["p_contains"]
    ok = np.flatnonzero(np.isfinite(pc) & (pc >= 1 - 1e-9))
    if ok.size:
        f_full = float(levels[ok.min()])

    out = dict(city=cd.city, tag=tag, n_stations=int(n), n_days=int(cd.days.size), n_stuck_days=int(cd.n_stuck_days),
               area_km2=cd.area_km2, extent=list(map(float, ext)), levels=levels.tolist(), x1=x1, x9=dict(cen, f_full=f_full),
               theta_E={p: float(theta_E[p][0]) for p in PROBES},
               theta_E_boot={p: [float(np.nanquantile(theta_E[p][c_boot], 0.025)), float(np.nanquantile(theta_E[p][c_boot], 0.975))] for p in PROBES},
               tests=tests, x8=x8, counts=counts, seconds=round(time.time() - t0, 1))
    d = out_dir / cd.city
    d.mkdir(parents=True, exist_ok=True)
    arrays = {f"{p}__{k}": v for p in PROBES for k, v in res[p].items()}
    arrays.update({f"lev__{k}": v for k, v in lev.items()})
    np.savez_compressed(d / "city.npz", **arrays)
    with open(d / "city.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return out
