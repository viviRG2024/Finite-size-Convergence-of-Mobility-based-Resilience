# Urban Resilience Quantification Stabilises Beyond a Critical Observation Window

Reference implementation of the finite-size convergence framework
introduced in the paper of the same title. The pipeline computes two
diagnostic probes — flow heterogeneity `CV(L)` and network complexity
`H(L)` — across nested observation windows and 39 spatial resolutions,
fits the local scaling laws `CV(L) ~ L^α` and `H(L) ~ L^β`, and
aggregates the exponents over the converged regime (SF ≥ 0.4) to produce
the per-city values reported in Table 2 of the paper.

## Overview

Mobility-based urban resilience metrics — network-robustness measures,
flow-recovery rates, accessibility-based vulnerability indices, and the
indicator frameworks built on them — are always computed over an
*observation window*: the spatial area the metric covers. Across the
literature, that area varies by more than 300-fold, yet the field has had
no quantitative criterion for when a window is large enough for a result
to represent the city as a whole rather than the sub-region that happened
to be sampled.

The paper introduces a measurement-independent test built on two
dimensionless probes that target the two informational dimensions
mobility-based metrics ultimately depend on:

- `CV(L)` — *flow heterogeneity*: coefficient of variation of hourly cell
  flows; captures the spatial concentration of demand.
- `H(L)` — *network complexity*: flow-weighted Shannon entropy of the
  cell adjacency graph; captures the topological organisation of the
  network.

Power-law fits against the spatial resolution `L` yield exponents α and β.
Applied to 34 cities of the Integrated Urban Traffic Flow Dataset (IUTFD),
the framework locates a **finite-size convergence near SF ≈ 0.4** —
roughly 40 % of the metropolitan area. Below this coverage, equally sized
windows in the same city can produce divergent, sometimes opposite-sign,
exponents; above it, the exponents stabilise and 34 cities collapse onto
shared scaling curves.

This repository ships the per-city computation for **London**,
**Melbourne**, and **Los Angeles** — the cities shown in Fig. 3a of the
paper. The full 34-city analyses, position-dependence sweeps (Fig. 2),
random-window validation (Fig. 3b), sensitivity surfaces (Fig. 3c), and
cross-city collapse (Fig. 4) are documented in the paper and SI Appendix.

## What the pipeline computes

For each city × date × nested observation window
(SF ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}):

1. Tile the window with regular metric grids at side
   `L ∈ {10, 50, …, 10 000} m` (39 logarithmically-spaced scales).
2. At every hour `h`:
   - `CV(L, h)` = std / mean of hourly flow across active cells (cells
     with at least 10 segments and non-zero hourly flow).
   - `H(L, h)` = flow-weighted Shannon entropy of the cell adjacency
     graph, built under queen (8-) connectivity with geometric-mean edge
     weights `w_ij = √(f_i × f_j)`.
3. Fit the local power laws `CV(L) ~ L^α` and `H(L) ~ L^β` by OLS on
   log–log axes over `L ∈ [200, 1200] m`, with the upper end of the fit
   window capped at `0.25 × min(window dimensions)` to suppress
   grid-boundary artefacts at small windows.
4. Aggregate α and β over the **converged regime** (SF ≥ 0.4, i.e.
   `window_idx ∈ {2, 3, 4, 5}`) and workday hours 08:00–15:59, yielding
   the per-city median, inter-quartile range, and full range — the rows
   of Table 2.

## Quick start

```bash
pip install -r requirements.txt
# fetch the IUTFD GeoJSONs (see "Data") into data/input/
python main.py                                # full pipeline, 3 cities
python tests/test_pipeline.py                 # ~30 s smoke test
```

`geopandas` requires the GDAL/PROJ system libraries; on Windows the
easiest route is `conda install -c conda-forge geopandas`.

## Data

The per-city IUTFD GeoJSONs are too large for GitHub and are hosted on
Google Drive:

> <https://drive.google.com/drive/folders/1Fn56WWYJmytDhbST4YJr7R5zwaT_qq-P?usp=sharing>

Download the three city folders and place them so the layout becomes:

```
data/input/
├── london/      result_traffic_flow_hourly_YYYY_MM_DD.geojson
├── melbourne/   result_traffic_flow_hourly_YYYY_MM_DD.geojson
└── losangeles/  result_traffic_flow_hourly_YYYY_MM_DD.geojson
```

`main.py` discovers dates automatically from the filenames, so any subset
of the per-city files is enough for a partial reproduction.

The upstream IUTFD dataset and its full 34-city release are described in
the dataset's own documentation, which the paper cites.

## Repository layout

```
.
├── main.py                            # 4-stage idempotent pipeline (load → compute → fit → aggregate)
├── config.yaml                        # cities, scales, fit window, statistical params
├── requirements.txt
├── src/
│   ├── io/geojson_loader.py           # IUTFD GeoJSON load + unified metric CRS
│   ├── analysis/nested_windows.py     # nested windows at SF ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}
│   ├── analysis/multi_scale.py        # CV(L), H(L) over 39 spatial scales
│   ├── analysis/scaling_fit.py        # local log-log fit, α and β
│   ├── analysis/statistical_tests.py  # bootstrap CIs, permutation, KS
│   ├── utils/io.py                    # numpy-aware JSON helpers
│   └── viz/diagnostics.py             # opt-in process plots (--viz)
├── data/input/                        # IUTFD GeoJSONs (download from Google Drive)
├── results/                           # written by main.py
└── tests/test_pipeline.py             # ~30 s smoke test
```

## Usage

`main.py` runs four idempotent stages — `load → compute → fit → aggregate` —
and writes one JSON per `(city, date, window)`. Each stage skips work whose
output already exists unless `--force` is passed.

```bash
python main.py                                       # full pipeline, all cities
python main.py --city melbourne --stage compute      # only CV / H
python main.py --city melbourne --stage fit          # only α / β
python main.py --windows 2 3 4 5                     # restrict aggregation to the converged regime
python main.py --hours 8 9 10 11 12 13 14 15         # override workday hours
python main.py --no-tests                            # skip bootstrap / KS stage
python main.py --force                               # recompute outputs that already exist
python main.py --viz                                 # also write diagnostic PNGs (off by default)
```

Stages can also be invoked individually with `--stage {load,compute,fit,tests,aggregate,all}`.

## Outputs

```
results/
├── <city>/
│   ├── windows_config.json                       # bbox of each nested SF window
│   ├── multi_scale/window<N>/<city>_<date>.json  # CV(L,h), H(L,h)
│   ├── scaling/window<N>/<city>_<date>.json      # alpha_by_hour, beta_by_hour
│   ├── stats/window<N>/<city>_<date>.json        # bootstrap CIs (optional)
│   └── diagnostics/                              # written only with --viz
├── cross_city_summary.csv                        # converged exponents (Table 2)
└── cross_city_summary.json
```

`cross_city_summary.csv` aggregates α and β over the converged windows
(SF ≥ 0.4) and workday hours 08:00–15:59, reporting the per-city median,
inter-quartile range, and full range — the values reported as Table 2 of
the paper for London, Melbourne, and Los Angeles.

## Process visualisation (`--viz`)

Pass `--viz` and the pipeline writes diagnostic PNGs alongside the JSON
outputs:

```
results/<city>/diagnostics/
├── window<N>/<city>_<date>.png       # 6-panel per-(date, window) diagnostic
└── <city>_convergence_heatmap.png    # 24 h × window median α/β heatmap
```

The 6-panel diagnostic shows, at hour 10 (matching Fig. 3a of the paper):
the segment scatter with the SF window bbox overlaid; active-cell flow
heatmaps at `L = 200 m` and `L = 1000 m`; log-log scatter of `CV(L)` and
`H(L)` with the `[200, 1200] m` fit window shaded and the OLS line
overlaid; and a text summary of bbox, segment count, and the fitted α,
β, R² for that window. The per-city convergence heatmap shows the median
α and β over all dates as a function of `(window_idx, hour)` — the
converged regime (SF ≥ 0.4) occupies the upper four rows and should
settle to similar values, while `window_idx = 1` (SF = 0.2) deviates.

`--viz` is off by default and does not affect the numerics. It requires
`matplotlib` and adds roughly a few seconds per window.

## Reproducibility

- All randomness flows through `random_seed: 42` in `config.yaml`.
- The pipeline is idempotent at the per-task level: rerunning skips
  finished work unless `--force` is passed.
- `python tests/test_pipeline.py` is a ~30-second end-to-end smoke test
  that runs one Melbourne date, one window, one hour, and asserts the
  cross-city CSV is produced with finite values.