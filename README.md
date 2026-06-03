# Finite-size Convergence of Mobility-based Resilience — reproduction code

This package reproduces the multi-scale scaling-law analysis
(`CV(L) ~ L^alpha`, `H(L) ~ L^beta`) for three urban mobility networks:
**London**, **Melbourne**, and **Los Angeles**.

The repository is intentionally minimal: no figures, no diagnostics, just the
end-to-end computation that yields the scaling exponents reported in the
paper.

- **Source repository:** https://github.com/viviRG2024/Finite-size-Convergence-of-Mobility-based-Resilience
- **Data:** only the small demo subset (a cropped slice of the Melbourne
  network, ~2.7 MB) is bundled with the repository, under `data/input/demo/`.
  The full hourly traffic-flow GeoJSONs for **London**, **Melbourne**, and
  **Los Angeles** are too large to ship here and are hosted separately on
  Google Drive:
  <https://drive.google.com/drive/folders/1mcp3sbdLuMc3ytnTBnBsSR0VHAAQx8Fg?usp=drive_link>.
  Download a city's folder and place it under `data/input/<city>/` to
  reproduce the full results (see *Run (full reproduction)* below).

## Layout

```
clean_code/
├── main.py                   # one-shot pipeline (load -> compute -> fit -> aggregate)
├── config.yaml               # cities, scales, fit window, statistical params
├── requirements.txt
├── LICENSE                   # MIT
├── src/
│   ├── io/geojson_loader.py        # GeoJSON load + unified CRS handling
│   ├── analysis/nested_windows.py  # SF in {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}
│   ├── analysis/multi_scale.py     # CV(L), H(L) over 39 spatial scales
│   ├── analysis/scaling_fit.py     # local log-log fit, alpha and beta
│   ├── analysis/statistical_tests.py  # bootstrap CI, permutation, KS
│   └── utils/io.py                 # JSON helpers (numpy-aware)
├── data/input/                # demo subset bundled; full city data via Google Drive
├── results/                   # written by main.py
└── tests/test_pipeline.py     # one-minute smoke test
```

## System requirements

### Software dependencies

- **Python ≥ 3.9** (tested on 3.12.7).
- Python packages (see `requirements.txt`):
  `numpy>=1.21`, `scipy>=1.7`, `pandas>=1.3`, `geopandas>=0.10`,
  `shapely>=1.8`, `pyproj>=3.2`, `PyYAML>=5.4`.
- `geopandas` depends on the GDAL/PROJ system libraries (pulled in
  automatically by the conda route below).

### Versions the software has been tested on

| Component  | Tested version |
|------------|----------------|
| OS         | Windows 11 (also expected to run on Linux/macOS — pure-Python, no OS-specific code) |
| Python     | 3.12.7 |
| numpy      | 1.26.4 |
| scipy      | 1.13.1 |
| pandas     | 2.2.2  |
| geopandas  | 1.1.1  |
| shapely    | 2.1.1  |
| pyproj     | 3.7.1  |
| PyYAML     | 6.0.1  |

### Hardware

No non-standard hardware is required. A standard desktop or laptop is
sufficient; ~8 GB RAM is comfortable (the London GeoJSONs are ~60 MB each and
are loaded fully into memory). The pipeline is single-threaded; you can run
several cities in parallel by launching one process per `--city`.

## Installation

```bash
pip install -r requirements.txt
```

On Windows the easiest way to satisfy `geopandas`'s GDAL/PROJ dependency is:

```bash
conda install -c conda-forge geopandas
pip install -r requirements.txt   # the rest
```

**Typical install time on a normal desktop:** ~2–5 minutes via `pip` if
wheels are available; ~5–15 minutes if conda has to resolve and build the
GDAL/PROJ geospatial stack from scratch.

## Demo

A small bundled dataset (`data/input/demo/`, ~2.7 MB) runs the entire
pipeline — load → multi-scale CV/H → α/β fits → aggregation — in seconds.

```bash
python main.py --city demo --no-tests
```

**Expected output.** The run prints a short progress log and writes
`results/demo/...` plus a `demo` row in `results/cross_city_summary.csv`:

```
[load] demo: 1 dates, 6 windows, CRS=EPSG:32755
[compute] demo 2018_02_12
  loaded 4172 segments
  done in ~13s
[aggregate] wrote results/cross_city_summary.csv

city,n_alpha,n_beta,alpha_median,...,beta_median,...
demo,32,32,-0.122,...,-0.392,...
```

These values are deterministic (the α/β fits use no randomness; `--no-tests`
skips the only stochastic stage). A single `ConstantInputWarning` may appear —
it is harmless (one scale/hour on the small subset has a constant CV or H).

**Expected run time for the demo:** under ~30 seconds on a normal laptop
(~16 s observed).

An even faster, assertion-checked sanity test (~30 s):

```bash
python tests/test_pipeline.py
# OK: alpha_median=-0.346..., beta_median=-0.360...
# Smoke test passed.
```

## Run (full reproduction)

The full city datasets are not bundled with this repository. First download
them from Google Drive and place each city's folder under `data/input/`:

<https://drive.google.com/drive/folders/1mcp3sbdLuMc3ytnTBnBsSR0VHAAQx8Fg?usp=drive_link>

```
data/input/
├── demo/         # bundled with the repo
├── london/       # downloaded from Google Drive
├── melbourne/    # downloaded from Google Drive
└── losangeles/   # downloaded from Google Drive
```

Each folder holds the hourly GeoJSONs named
`result_traffic_flow_hourly_YYYY_MM_DD.geojson`. Then run the full pipeline on
the three study cities (6 nested windows, 39 scales, 8 workday hours):

```bash
python main.py --city london --city melbourne --city losangeles
```

(`python main.py` with no `--city` also runs the small `demo` city.)
The bootstrap/KS stage dominates the runtime; the whole reproduction takes on
the order of a few hours single-threaded, or far less with `--no-tests`.

Per-stage / per-city control:

```bash
python main.py --city melbourne --stage compute      # only CV/H
python main.py --city melbourne --stage fit          # only alpha/beta
python main.py --windows 2 3 4 5                     # restrict aggregation
python main.py --no-tests                            # skip bootstrap stage
python main.py --force                               # recompute everything
```

## Use on your own data

1. Place GeoJSON files named `result_traffic_flow_hourly_YYYY_MM_DD.geojson`
   under `data/input/<cityname>/`. Each feature must be a `LineString` with
   hourly flow fields `flow_0 … flow_23` (or per-mode fields `<mode>_<hour>`,
   which are combined using the `modal_weights` in `config.yaml`). The input
   CRS is auto-detected (WGS84 or a projected CRS).
2. Add a block under `cities:` in `config.yaml`:

   ```yaml
   cities:
     mycity:
       target_crs: null     # null = auto-pick a metric UTM zone; or e.g. EPSG:27700
       dates: null          # null = use every date found on disk
   ```

3. Run the pipeline for that city:

   ```bash
   python main.py --city mycity
   ```

## Outputs

```
results/
├── <city>/
│   ├── windows_config.json                         # bbox of each nested window
│   ├── multi_scale/window<N>/<city>_<date>.json    # CV(L,h), H(L,h)
│   ├── scaling/window<N>/<city>_<date>.json        # alpha_by_hour, beta_by_hour
│   └── stats/window<N>/<city>_<date>.json          # bootstrap CIs (optional)
├── cross_city_summary.csv                          # converged exponents (Table 2)
└── cross_city_summary.json
```

`cross_city_summary.csv` aggregates `alpha` and `beta` over the converged
windows (SF >= 0.4) and the workday hours 08:00-15:59, reporting the median,
inter-quartile range, and full range per city. This is the table reported in
the paper.

## Method (one paragraph)

For each city we tile the network bbox with regular grids of side
`L in {10, 50, 100, ..., 10 000} m`, aggregate per-segment hourly flows into
per-cell flows, and compute (a) the coefficient of variation `CV(L, h)`
across active cells and (b) the degree-based Shannon entropy `H(L, h)` of
the cell adjacency graph (8-connectivity, geometric-mean edge weights).
Power laws `CV(L) ~ L^alpha` and `H(L) ~ L^beta` are fit by ordinary
least squares in log-log space on the local window
`L in [200, 1200] m`, capped by `0.25 * win_min_dim` to avoid finite-size
artefacts at small windows. The fit is repeated for each combination of
date, nested window (six relative-bbox sub-regions), and hour, and the
exponents are aggregated over the converged windows (SF >= 0.4) and
workday hours. A complete, detailed description of this procedure
(pseudocode-level) is given in the **Methods** section of the manuscript.

## Reproducibility

All randomness is controlled by `random_seed: 42` in `config.yaml`.
The pipeline is idempotent at the per-task level: rerunning skips finished
work unless `--force` is passed.

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE).
Copyright (c) 2026 xuhui lin and qiuchen lu.
