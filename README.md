# Clean reproduction code — finite-size convergence of mobility-based resilience

This package reproduces the converged scaling exponents reported in **Table 2**
of:

> Lin, Lu, Batty, Mitoulis, Meng, Zhang, Chen.
> *Urban resilience quantification stabilises beyond a critical observation
> window.* (Nature Communications, submitted.)

The paper introduces a measurement-independent test for whether a
resilience study's **observation window** is large enough to represent the
city as a system. Applied to 34 cities of the Integrated Urban Traffic Flow
Dataset (IUTFD), the test locates a finite-size convergence near **40% of
the metropolitan area** (scale factor SF ≈ 0.4). Below the convergence,
identically sized windows in the same city can produce divergent, even
opposite-sign, exponents; above it, the exponents stabilise and 34 cities
collapse onto shared scaling curves.

This repository packages the per-city computation behind that result for
**London**, **Melbourne**, and **Los Angeles** (the cities shown in Fig. 3a
of the paper), and produces the converged α and β reported in Table 2 for
those three cities. The repository is intentionally minimal: no figures,
no diagnostics, just the end-to-end exponents.

## What the pipeline computes

For each city, for each of six nested observation windows centred on the
network (SF ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}) and each available date:

1. Tile the window with regular metric grids at side `L ∈ {10, 50, …, 10 000} m`
   (39 logarithmically-spaced scales).
2. At every hour `h`, compute the two diagnostic probes used by the paper:
   - `CV(L, h)` — **flow heterogeneity**: coefficient of variation of
     hourly flow across active cells (cells containing road segments with
     non-zero flow).
   - `H(L, h)` — **network complexity**: flow-weighted Shannon entropy of
     the cell adjacency graph, built under queen (8-) connectivity with
     geometric-mean edge weights `w_ij = √(f_i × f_j)`.
3. Fit the local power laws `CV(L) ~ L^α` and `H(L) ~ L^β` by OLS in
   log–log space on `L ∈ [200, 1200] m`, with the upper end of the fit
   capped at `0.25 × min(window dimensions)` to suppress grid-boundary
   artefacts at small windows.
4. Aggregate α and β over the **converged regime** (SF ≥ 0.4,
   i.e. `window_idx ∈ {2, 3, 4, 5}`) and workday hours 08:00–15:59,
   yielding the per-city median, inter-quartile range, and full range —
   the rows of Table 2.

## Layout

```
clean_code/
├── main.py                   # 4-stage idempotent pipeline (load -> compute -> fit -> aggregate)
├── config.yaml               # cities, scales, fit window, statistical params
├── requirements.txt
├── src/
│   ├── io/geojson_loader.py        # IUTFD GeoJSON load + unified metric CRS
│   ├── analysis/nested_windows.py  # nested windows at SF in {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}
│   ├── analysis/multi_scale.py     # CV(L), H(L) over 39 spatial scales
│   ├── analysis/scaling_fit.py     # local log-log fit, alpha and beta
│   ├── analysis/statistical_tests.py  # bootstrap CI, permutation, KS
│   └── utils/io.py                 # JSON helpers (numpy-aware)
├── data/input/                # IUTFD GeoJSONs for the 3 cities
├── results/                   # written by main.py
└── tests/test_pipeline.py     # ~30 s smoke test
```

## Install

```bash
pip install -r requirements.txt
```

`geopandas` requires the GDAL/PROJ system libraries; the easiest route on
Windows is `conda install -c conda-forge geopandas`.

## Data

The per-city IUTFD GeoJSONs are too large for GitHub and are hosted
separately on Google Drive:

> <https://drive.google.com/drive/folders/1Fn56WWYJmytDhbST4YJr7R5zwaT_qq-P?usp=sharing>

Download the three city folders (`london/`, `melbourne/`, `losangeles/`)
and place them under `data/input/` so the layout becomes:

```
clean_code/data/input/
├── london/      result_traffic_flow_hourly_YYYY_MM_DD.geojson
├── melbourne/   result_traffic_flow_hourly_YYYY_MM_DD.geojson
└── losangeles/  result_traffic_flow_hourly_YYYY_MM_DD.geojson
```

`main.py` discovers dates automatically from the filenames, so any subset
of the per-city files in the Drive folder is enough for a partial
reproduction.

## Run

Full pipeline (3 cities, 6 nested windows, 39 scales, 8 workday hours):

```bash
python main.py
```

Smoke test (~30 s, one Melbourne date, one window, one hour):

```bash
python tests/test_pipeline.py
```

Per-stage / per-city control:

```bash
python main.py --city melbourne --stage compute      # only CV/H
python main.py --city melbourne --stage fit          # only alpha/beta
python main.py --windows 2 3 4 5                     # restrict aggregation to the converged regime
python main.py --no-tests                            # skip bootstrap stage
python main.py --force                               # recompute everything
```

## Outputs

```
results/
├── <city>/
│   ├── windows_config.json                         # bbox of each nested SF window
│   ├── multi_scale/window<N>/<city>_<date>.json    # CV(L,h), H(L,h)
│   ├── scaling/window<N>/<city>_<date>.json        # alpha_by_hour, beta_by_hour
│   └── stats/window<N>/<city>_<date>.json          # bootstrap CIs (optional)
├── cross_city_summary.csv                          # converged exponents (Table 2)
└── cross_city_summary.json
```

`cross_city_summary.csv` aggregates α and β over the **converged windows**
(SF ≥ 0.4) and workday hours 08:00–15:59, reporting the per-city median,
inter-quartile range, and full range — the values reported as Table 2 in
the paper for London, Melbourne, and Los Angeles.

## What the convergence means

Mobility-based resilience metrics are computed over an observation window,
yet the field has had no quantitative criterion for when that window is
large enough to represent the city rather than the sub-region observed.
The two probes used here — flow heterogeneity (CV) and network complexity
(H) — are not themselves resilience metrics; they target the two
informational dimensions on which mobility-based resilience metrics depend
(how demand is distributed in space, and how the network is connected) and
ask whether those quantities have stabilised. The convergence near SF ≈ 0.4
identified in the paper provides the criterion: below it, measurements are
dominated by window-placement effects; above it, the probes reflect
stable, system-level properties of the city and become quantitatively
comparable across morphologies. This package reproduces the converged-regime
exponents for three representative cities; the position-dependence analysis
(Fig. 2), random-window validation (Fig. 3b), sensitivity sweep
(Fig. 3c), and 34-city cross-city collapse (Fig. 4) are documented in the
paper and SI Appendix.

## Reproducibility

All randomness is controlled by `random_seed: 42` in `config.yaml`. The
pipeline is idempotent at the per-task level: rerunning skips finished
work unless `--force` is passed.
