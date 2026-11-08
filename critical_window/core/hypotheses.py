"""Cross-city summary of the hypothesis tests H1 and H2 for each diagnostic.

Input: one row per (probe, city) with
  h1_p       Holm-adjusted p-value of H1 (placement dispersion above the permutation null)
  h2_p       p-value of the change-of-slope test, calibrated with simulated stationary fields
  f_b        breakpoint coverage of that test
  geometric  True when the critical coverage of the city is attributed to the inclusion of its main centre
Output per probe: number of cities, share of cities with h1_p < alpha (H1), share with h2_p < alpha (H2), coefficient
of variation of the breakpoints among the latter and the share of them attributed to centre inclusion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY = ("cv500", "hn500")


def summarise_probe(tab: pd.DataFrame, alpha: float = 0.05) -> dict:
    n = len(tab)
    h1_share = float((tab["h1_p"] < alpha).mean()) if n else 0.0
    sig = tab[tab["h2_p"] < alpha]
    h2_share = len(sig) / n if n else 0.0
    fb = sig["f_b"].to_numpy(float)
    cv_fb = float(np.std(fb, ddof=1) / np.mean(fb)) if len(fb) > 1 and np.mean(fb) > 0 else np.nan
    geo_share = float(sig["geometric"].mean()) if len(sig) else np.nan
    h1 = h1_share > 0.5
    h2a = (h2_share >= 2 / 3) and np.isfinite(cv_fb) and (cv_fb < 0.3) and (geo_share <= 1 / 3)
    return dict(n_cities=n, h1_share=round(h1_share, 3), h1=h1, h2_share=round(h2_share, 3),
                cv_fb=None if not np.isfinite(cv_fb) else round(cv_fb, 3),
                geometric_share=None if not np.isfinite(geo_share) else round(geo_share, 3), h2a=h2a)


def summarise(results: pd.DataFrame, alpha: float = 0.05, primary=PRIMARY) -> dict:
    per = {p: summarise_probe(results[results["probe"] == p], alpha) for p in results["probe"].unique()}
    return dict(per_probe=per)
