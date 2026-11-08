"""X0.4 regression and known-answer tests of the analysis core (no real flows are used).

    python -m critical_window.tests.test_core            # all tests, about 2 min on 2 cores
    python -m critical_window.tests.test_core T1 T4      # a subset

Layouts (sensor positions only) are in data/layouts; T1 runs only when the earlier code is placed in clean_code/.
Results are printed and written to critical_window/out/x04/.
"""
from __future__ import annotations

import sys
import time
import types
import zlib

import numpy as np
import pandas as pd

from ..common.cfg import REPO_DIR, PKG_DIR
from ..core import criteria, probes, synth, transition, windows

OUT = PKG_DIR / "out" / "x04"
LAYOUTS = PKG_DIR / "data" / "layouts"


def seed(*parts) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def layout(city: str):
    s = pd.read_csv(LAYOUTS / f"{city}_sensors.csv")
    x, y = s["x_m"].to_numpy(float), s["y_m"].to_numpy(float)
    return x, y, (x.min(), y.min(), x.max(), y.max())


def _load_ref(name: str):
    p = REPO_DIR / "clean_code" / "src" / "analysis" / name
    if not p.exists():
        return None
    src = p.read_text(encoding="utf-8").replace("from __future__ import annotations", "")
    mod = types.ModuleType(name[:-3])
    exec(compile(src, str(p), "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------------------------------------------ T1
def t1_equivalence() -> dict:
    ref = _load_ref("multi_scale.py")
    sf = _load_ref("scaling_fit.py")
    if ref is None or sf is None:
        return dict(test="T1", status="skipped", note="clean_code not found")
    rng = seed("T1")
    worst_cv = worst_h = 0.0
    n_eval = 0
    for _ in range(40):
        n = int(rng.integers(5, 400))
        W, H = rng.uniform(500, 8000), rng.uniform(500, 8000)
        x, y = rng.uniform(0, W, n), rng.uniform(0, H, n)
        F = np.exp(rng.standard_normal((n, 24)))
        F[rng.random((n, 24)) < 0.15] = 0.0
        segs = [{"centroid_x": x[i], "centroid_y": y[i], "flows": F[i]} for i in range(n)]
        for L in (50, 200, 450, 1000, 2500):
            r = ref.analyse_single_scale(segs, (0.0, 0.0, W, H), L)
            cv, Hh, _, _ = probes.grid_probe(x, y, F, (0.0, 0.0, W, H), L)
            rcv, rH = np.array(r["cv_per_hour"]), np.array(r["entropy_per_hour"])
            m1, m2 = rcv > 0, rH > 0
            if m1.any():
                worst_cv = max(worst_cv, float(np.max(np.abs(cv[m1] - rcv[m1]))))
            if m2.any():
                worst_h = max(worst_h, float(np.max(np.abs(Hh[m2] - rH[m2]))))
            n_eval += 1
    worst_fit = 0.0
    for _ in range(200):
        V = np.exp(rng.standard_normal((len(probes.GRID_SIZES), 1)) * 0.3) * probes.GRID_SIZES[:, None] ** rng.uniform(-0.6, 0.2)
        V[rng.random(V.shape) < 0.1] = np.nan
        a = probes.fit_loglog(probes.GRID_SIZES, V)[0]
        r = sf.fit_local_power_law(probes.GRID_SIZES, V[:, 0], 200, 1200, None, 4)["exponent"]
        if np.isfinite(a) or np.isfinite(r):
            worst_fit = max(worst_fit, abs(a - r))
    ok = worst_cv < 1e-10 and worst_h < 1e-10 and worst_fit < 1e-10
    return dict(test="T1", status="pass" if ok else "FAIL",
                note=f"{n_eval} scale evaluations: max |dCV| {worst_cv:.1e}, |dH| {worst_h:.1e}; exponent fit |d| {worst_fit:.1e}")


# ------------------------------------------------------------------------------------------------ T2
def t2_xi_recovery(cities=("london", "zurich", "bremen", "bern", "darmstadt"), reps: int = 20) -> dict:
    """Correlation length recovered on real layouts, raw and after the layout-specific bias correction."""
    rows = []
    for city in cities:
        x, y, ext = layout(city)
        short = min(ext[2] - ext[0], ext[3] - ext[1])
        true, mean_est = probes.xi_bias_curve(x, y, ext, seed("T2-bias", city))
        for xi in (500.0, 1000.0, 2000.0):
            rng = seed("T2", city, xi)
            est = np.array([probes.corr_length(x, y, synth.gaussian_field(x, y, ext, xi, rng), short / 2, rng=rng)
                            for _ in range(reps)])
            cor = np.array([probes.xi_correct(e, true, mean_est) for e in est])
            q = lambda v, a: float(np.nanquantile(v, a) / xi) if np.isfinite(v).any() else np.nan
            rows.append(dict(city=city, n=x.size, short_km=round(short / 1000, 2), xi_true=xi,
                             xi_over_short=round(xi / short, 3), n_found=int(np.isfinite(est).sum()),
                             raw_median=q(est, 0.5), raw_p10=q(est, 0.1), raw_p90=q(est, 0.9),
                             corrected_median=q(cor, 0.5), corrected_p10=q(cor, 0.1), corrected_p90=q(cor, 0.9)))
    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "t2_xi_recovery.csv", index=False)
    raw_ok = tab[tab.xi_over_short <= 1 / 6]
    cor_ok = tab[tab.xi_over_short <= 1 / 4]
    # raw xi_hat is reported as identifiable only when xi_hat <= short side / 6
    ok = bool(((raw_ok.raw_median - 1).abs() <= 0.15).all())
    return dict(test="T2", status="pass" if ok else "FAIL",
                note=(f"raw xi_hat/xi for xi <= short/6: {raw_ok.raw_median.min():.2f}-{raw_ok.raw_median.max():.2f}; "
                      f"raw up to short/4: {cor_ok.raw_median.min():.2f}-{cor_ok.raw_median.max():.2f}; "
                      f"bias-corrected up to short/4: {cor_ok.corrected_median.min():.2f}-{cor_ok.corrected_median.max():.2f}"))


# ------------------------------------------------------------------------------------------------ T3, T4
def t3_windows() -> dict:
    rng = seed("T3")
    ext = (0.0, 0.0, 12000.0, 8000.0)
    worst_ks = 0.0
    for f in (0.01, 0.1, 0.3, 0.6):
        w = windows.random_windows(ext, f, 5000, rng)
        assert (w[:, 0] >= ext[0] - 1e-6).all() and (w[:, 2] <= ext[2] + 1e-6).all()
        assert (w[:, 1] >= ext[1] - 1e-6).all() and (w[:, 3] <= ext[3] + 1e-6).all()
        area = (w[:, 2] - w[:, 0]) * (w[:, 3] - w[:, 1])
        assert np.allclose(area / (12000 * 8000), f)
        cx = (w[:, 0] + w[:, 2]) / 2
        lo, hi = ext[0] + (w[0, 2] - w[0, 0]) / 2, ext[2] - (w[0, 2] - w[0, 0]) / 2
        u = np.sort((cx - lo) / (hi - lo))
        worst_ks = max(worst_ks, float(np.max(np.abs(u - (np.arange(u.size) + 0.5) / u.size))))
    ok = worst_ks < 0.03
    return dict(test="T3", status="pass" if ok else "FAIL",
                note=f"windows inside the extent, exact area, centres uniform (max KS distance {worst_ks:.3f})")


def t4_overlap() -> dict:
    rng = seed("T4")
    ext = (0.0, 0.0, 10000.0, 6000.0)
    rows = []
    for f in (0.04, 0.16, 0.36, 0.64):
        mc = windows.mean_pairwise_overlap(windows.random_windows(ext, f, 400, rng))
        an = windows.expected_overlap(f)
        pt = (2500.0, 3000.0)
        w = windows.random_windows(ext, f, 20000, rng)
        mc_p = float(((w[:, 0] <= pt[0]) & (w[:, 2] >= pt[0]) & (w[:, 1] <= pt[1]) & (w[:, 3] >= pt[1])).mean())
        rows.append(dict(f=f, overlap_analytic=round(an, 3), overlap_mc=round(mc, 3),
                         p_contains_analytic=round(windows.p_contains(ext, f, pt), 3), p_contains_mc=round(mc_p, 3)))
    tab = pd.DataFrame(rows)
    ok = bool((abs(tab.overlap_analytic - tab.overlap_mc) < 0.02).all() and
              (abs(tab.p_contains_analytic - tab.p_contains_mc) < 0.02).all())
    return dict(test="T4", status="pass" if ok else "FAIL",
                note="expected overlap " + ", ".join(f"f={r.f}: {r.overlap_analytic:.2f}" for r in tab.itertuples()) +
                     "; analytic = Monte Carlo within 0.02")


# ------------------------------------------------------------------------------------------------ T5
def t5_critical_f() -> dict:
    fs = windows.F_GRID
    stat = 0.2 * np.sqrt((1 - fs) / fs)             # sampling-type curve, falls through eps
    res = []
    for eps in (0.6, 0.3, 0.15, 0.08):
        truth = 1.0 / (1.0 + (eps / 0.2) ** 2)        # solves 0.2 sqrt((1-f)/f) = eps
        r = criteria.critical_f(fs, stat, eps)
        res.append((eps, truth, r["f_star"], r["flag"]))
    ok = all(abs(np.log(fstar / t)) < 0.1 for _, t, fstar, fl in res if fl == "ok")
    below = criteria.critical_f(fs, stat, 5.0)["flag"] == "below"
    above = criteria.critical_f(fs, stat, 0.01)["flag"] == "above"
    return dict(test="T5", status="pass" if (ok and below and above) else "FAIL",
                note="; ".join(f"eps {e}: true {t:.3f} est {f:.3f}" for e, t, f, _ in res) + "; censoring flags correct")


# ------------------------------------------------------------------------------------------------ T6
def _error_curves(x, y, ext, F, rng, n_win: int = 50, probe: str = "mean_log"):
    """P90 error per coverage level for every column of F (one curve per realisation)."""
    theta_E = probes.level_probes(x, y, F, ext)[probe]
    out = np.full((windows.F_GRID.size, F.shape[1]), np.nan)
    for k, f in enumerate(windows.F_GRID[:-1]):
        errs = []
        for b in windows.random_windows(ext, f, n_win, rng):
            m = windows.in_box(x, y, b)
            if m.sum() < 20:
                continue
            errs.append(np.abs(probes.level_probes(x[m], y[m], F[m], b)[probe] - theta_E))
        if len(errs) >= n_win // 2:
            out[k] = np.quantile(np.array(errs), 0.9, axis=0)
    return out


def t6_transition(city: str = "bremen", n_null: int = 400, n_obs: int = 300) -> dict:
    x, y, ext = layout(city)
    rng = seed("T6", city)
    xi_true = 1000.0
    obs = synth.stationary_flows(x, y, ext, xi_true, 1.0, 5.0, n_obs, rng)
    mono = synth.monocentric_flows(x, y, ext, 500.0, 1.0, 5.0, 0.6, 40, rng)
    rows = []
    c_obs = _error_curves(x, y, ext, obs, np.random.default_rng(1))
    c_mono = _error_curves(x, y, ext, mono, np.random.default_rng(1))
    d_obs = np.array([transition.dbic(windows.F_GRID, c_obs[:, j])[0] for j in range(n_obs)])
    d_mono = np.array([transition.dbic(windows.F_GRID, c_mono[:, j]) for j in range(mono.shape[1])])
    for xi_null in (800.0, 1000.0, 1200.0):
        null = synth.stationary_flows(x, y, ext, xi_null, 1.0, 5.0, n_null, seed("T6-null", city, xi_null))
        c_null = _error_curves(x, y, ext, null, np.random.default_rng(1))
        d_null = np.array([transition.dbic(windows.F_GRID, c_null[:, j])[0] for j in range(n_null)])
        p_obs = np.array([criteria.perm_pvalue(d, d_null) for d in d_obs])
        p_mono = np.array([criteria.perm_pvalue(d, d_null) for d in d_mono[:, 0]])
        rows.append(dict(city=city, xi_true=xi_true, xi_null=xi_null,
                         false_alarm_calibrated=float(np.mean(p_obs < 0.05)),
                         false_alarm_fixed_dbic10=float(np.mean(d_obs > 10)),
                         power_monocentric=float(np.mean(p_mono < 0.05)),
                         monocentric_fb_median=float(np.nanmedian(d_mono[:, 1]))))
    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "t6_transition_calibration.csv", index=False)
    fa = tab.loc[tab.xi_null == xi_true, "false_alarm_calibrated"].iloc[0]
    ok = fa <= 0.10
    return dict(test="T6", status="pass" if ok else "FAIL",
                note=f"{city}: calibrated false alarm {fa:.2f} (xi_null = xi), "
                     f"{tab.false_alarm_calibrated.min():.2f}-{tab.false_alarm_calibrated.max():.2f} for xi_null 0.8-1.2 xi; "
                     f"fixed dBIC>10: {tab.false_alarm_fixed_dbic10.iloc[0]:.2f}; monocentric power "
                     f"{tab.power_monocentric.min():.2f}-{tab.power_monocentric.max():.2f}, breakpoint f {tab.monocentric_fb_median.iloc[0]:.2f}")





TESTS = dict(T1=t1_equivalence, T2=t2_xi_recovery, T3=t3_windows, T4=t4_overlap, T5=t5_critical_f,
             T6=t6_transition)


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(TESTS)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        t = time.time()
        r = TESTS[name]()
        r["seconds"] = round(time.time() - t, 1)
        rows.append(r)
        print(f"{r['test']} {r['status']:>7}  ({r['seconds']} s)  {r['note']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "test_core_summary.csv", index=False)
    return 0 if all(r["status"] in ("pass", "skipped") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
