# Urban resilience quantification stabilises beyond a critical observation window — analysis code

This repository contains the code, the configuration and the Source Data for the paper *Urban resilience quantification
stabilises beyond a critical observation window* (under review at Nature Communications). The processed data and the
per-city results of every analysis stage are attached to the GitHub release as `processed_results.zip`.

The analysis asks how much of a city must be observed for a measurement to represent the whole city. For each of 34
cities of the IUTF dataset, the **study area** is the smallest rectangle containing all traffic detector stations of the
city (called the *extent* in the code). Rectangular windows with the aspect ratio of the study area are placed at random
positions at 18 coverages *f* (window area / study area) from 0.01 to 0.9 (12 of them for the road-network metrics).
In every window the code computes traffic
diagnostics (the dispersion of log flows across stations, σz, and the coefficient of variation and entropy of flows in
500 m cells) and road-network resilience metrics (robustness to random failure and to targeted attack, efficiency loss),
compares each window value with the whole-city value, and determines the critical coverage *f\** from which the window
error stays within a tolerance. It also analyses rainfall events as real disruptions, and fits the error law from which
the critical coverage can be computed with a single measurement.

## Quick start

All commands are run from the repository root (the folder that contains `critical_window/`).

```bash
pip install -r requirements.txt

# 1. Known-answer tests on synthetic data (no data needed; about 2 min)
python -m critical_window.tests.test_core
python -m critical_window.tests.test_geo

# 2. Figures, tables and Source Data from the processed results (minutes; no raw data needed)
#    download processed_results.zip from the release and unzip it in the repository root
python -m critical_window.figures.make_all
python -m critical_window.figures.fig2_panels
python -m critical_window.figures.fig1a_panel
python -m critical_window.figures.paper_names

# 3. The full analysis from the raw IUTF data (hours; see "Data")
bash run_all.sh
```

Route 2 writes every figure, table and Source Data file to `critical_window/out/figures/`, and `paper_names` copies
them to `critical_window/out/figures/paper/` under the numbers of the figures and tables of the paper (see "Figures and
tables"). The maps of Fig. 2a,b
(London) and the station maps drawn by `figures/fig1.py` show the road network only when the IUTF road files are present
in `data/input/IUTFD/`; without them the maps show the stations, and every other panel, table and Source Data file is
unchanged.

Between routes 2 and 3, the main analysis and most further stages can be rerun from the processed station data in
`processed_results.zip` (`critical_window/out/x0/`) without downloading the raw data; the stages that need the raw files
are marked in the table of stages below. For example, the main analysis of all 34 cities (about 30 min on two cores):

```bash
python -m critical_window.run_main --cities "$(paste -sd, critical_window/data/cities_analysed.txt)" --out-tag main34
```

A quick check of the code on two cities, with reduced replicate counts, takes a few minutes:

```bash
python -m critical_window.run_main --cities basel,constance --quick --out-tag demo
```

The tolerance, and with it the critical coverage, is defined relative to the differences between the cities analysed
together, so a run on a subset of cities gives other values of *f\** than the 34-city run. Without `--cities`, `run_main`
uses the cities marked in `out/x0/inclusion.csv`; the paper analyses the 34 cities listed in
`critical_window/data/cities_analysed.txt`, which `run_all.sh` passes to every stage.

## Contents

| Path | Content |
|---|---|
| `critical_window/config.yaml` | the parameters of the data preparation and of the main analysis (analysis period, validity rules, coverage levels, windows, minimum numbers of stations and windows, tolerance, percentile); the replicate counts are set at the top of `run_main.py`, and the settings of the further stages at the top of each module |
| `critical_window/x0/`, `run_x0.py` | data preparation and quality control from the raw IUTF files |
| `critical_window/core/` | windows, traffic diagnostics, window error and critical coverage, simulated fields, tests |
| `critical_window/main/`, `run_main.py` | main analysis: per-city stage and cross-city stage |
| `critical_window/resil/` | road-network graphs and metrics, rainfall events |
| `critical_window/tools/` | further analysis stages (error law, grids, networks, robustness, rainfall, checks) |
| `critical_window/figures/` | figures, tables and Source Data |
| `critical_window/tests/` | known-answer and regression tests |
| `critical_window/data/` | city table, public holidays, the 34 cities analysed, station positions for the tests, SHA-256 checksums of the IUTF files used |
| `run_all.sh` | the full run, from the raw files to the figures, in order |
| `source_data/` | Source Data of every figure panel, named by the figures of the paper (`Fig1_a_windows.csv` to `SuppFig15.csv`; one CSV per panel, or per pair of panels plotted from the same values, each with a note; all compiled in `Source_Data.xlsx`), the tables (`Table1.csv`, `SuppTable5.csv`, `SuppTable6.csv`, `Tables.xlsx`) and `name_map.csv`, which lists the working name under which each file is written |
| `DATA_LICENCE.md` | sources and terms of the data and of the processed data |
| `processed_results.zip` (release asset) | `critical_window/out/`: processed station data and the results of every stage |

## Data

The analysis uses the IUTF dataset (Lin, X. et al. *Sci. Data* 13, 22 (2026)), openly available on figshare under a
CC BY 4.0 licence: <https://doi.org/10.6084/m9.figshare.30022807.v1>. Download `IUTFD.zip` (1.28 GB) and unzip it so
that the city folders are in `data/input/IUTFD/<city>/`, each with `sensors/`, `roads/`, `weather/` and `npz/`. The
traffic data in IUTF are derived from UTD19, the road networks from OpenStreetMap and the rainfall from ERA5; see
`DATA_LICENCE.md` for the sources, citations and terms. No other data are needed.

`run_x0` reads only these files and writes a manifest of every raw file used, with its size and SHA-256
(`out/x0/manifest.csv`). The checksums of the files used are also in `critical_window/data/iutfd_checksums.sha256`, so
that a downloaded copy of the data can be checked against the one analysed before anything is run:

```bash
cd data/input/IUTFD && sha256sum -c ../../../critical_window/data/iutfd_checksums.sha256
```

## Software

Python 3.10 or later with the packages in `requirements.txt` (numpy, pandas, pyarrow, scipy, python-igraph, shapely,
matplotlib, xlsxwriter). `numba` is optional and speeds up the road-network stage.

The results were computed with Python 3.11 and numpy 2.4 (main analysis and further stages) and Python 3.12 with
numpy 1.26 (data preparation). All random draws use seeds derived from fixed labels (city, metric, coverage level,
replicate), so every stage gives the same result when it is rerun with the same software; with other versions of
Python and numpy, the results agree to within floating-point rounding (see "Checks of reproducibility").

## Stages

`run_all.sh` runs the stages in this order. Each stage can also be run on its own; every module prints its options with
`--help`.

| Stage | Command | Needs raw data | Output |
|---|---|---|---|
| Data preparation | `run_x0` | yes | `out/x0/` |
| Main analysis: windows, traffic diagnostics, window errors, f\*, shuffled flows, simulated fields, bootstrap | `run_main`, then `tools.report_fixes` | no | `out/main34/` |
| Error law: sampling formula | `tools.e2_formula` | no | `out/e2/` |
| CV and H on 500 m cells, averaged over 32 grid origins | `tools.e3_grid` | no | `out/e3/` |
| Road-network resilience metrics | `tools.e4_network --kind e4b` | yes (roads) | `out/e4b/` |
| Station layout, block averaging, design effect (three steps of the error law) | `tools.e5_fractal` | yes (roads) for `run` | `out/e5/` |
| Robustness: periods, weather, outliers, window shapes, study-area definitions, cell sizes, tolerance | `tools.e6_robust` | yes (rainfall) | `out/e6/` |
| Criterion based on the stability of the scaling exponents | `tools.e7_bridge` | no | `out/e7/` |
| Window overlap and tiles, flow centres, dates, rankings | `tools.e8_tables` | no | `out/e8/` |
| Rainfall events, hour-by-hour profile, outlier check | `tools.e10_rain`, `tools.rain_periods`, `tools.rain_outlier_check` | yes (rainfall, detector occupancy) | `out/e10/` |
| Further checks and numbers given in the text | `tools.further_checks`, `tools.active_cells`, `tools.summary_stats` | no | `out/checks/` |
| Figures, tables and Source Data | `figures.make_all`, `figures.fig1a_panel`, `figures.fig2_panels`, `figures.paper_names` | road files for the maps only | `out/figures/`, `out/figures/paper/` |

On a computer with two cores, the main analysis of the 34 cities takes about 30 min and the road-network stage 2.5–4 h
(London alone about 70 min); each of the other stages takes from a few minutes to under an hour, and the figures a few
minutes. Some stages print numpy warnings about empty slices (coverage levels without usable windows in some cities);
they do not affect the results.

The labels X0–X11 and E2–E10 in the code and in the names of the output folders are the names of these stages.

## Where each part of the Methods is implemented

| Methods | Code |
|---|---|
| Data cleaning, stations, analysis period (non-holiday weekdays 07:00–19:59) | `x0/x02_stations.py`, `x0/x03_inclusion.py`; stuck detectors in `main/data.py`; parameters in `config.yaml` (`x0`) and `data/holidays.csv`; the same hours are used as constants in `main/data.py` (stuck detectors) and `resil/rain.py` |
| Study area and windows (coverage by area, random positions, usable windows and levels) | `core/windows.py` (`random_windows`); `main/city_stage.py`; `config.yaml` (`f_grid`, `n_random`, `n_w`, `min_usable_windows`) |
| Traffic diagnostics: σz, mean log flow, site CV | `core/probes.py` (`level_probes`) |
| CV and entropy H on 500 m cells, 32 grid origins | `tools/e3_grid.py` (`_cv_hn`; variant `sum_k32`) |
| Window error, tolerance, critical coverage, Kaplan–Meier medians | `core/criteria.py` (`p90`, `tolerance`, `critical_f`, `fstar_stats`); `main/cross_city.py` |
| 95% interval of f\* (weekdays and windows resampled) | `main/city_stage.py` (day bootstrap); `tools/report_fixes.py` |
| Shuffled flows, simulated stationary fields, test for an abrupt change | `main/city_stage.py`; `core/synth.py`; `core/transition.py`; `core/hypotheses.py` |
| Correlation length | `core/probes.py` (`correlogram`, `corr_length`); `main/city_stage.py` (`x1_correlation_length`) |
| Flow centres and the windows that contain them | `main/city_stage.py` (`x9_centres`); `tools/e8_tables.py`; `tools/summary_stats.py centres` |
| Consistency of rankings | `tools/e8_tables.py`; `tools/further_checks.py ranking`; `tools/summary_stats.py pairs` |
| Error law and the three steps | `tools/e2_formula.py` (step 1); `tools/e5_fractal.py deff` (steps 2 and 3); `figures/fig5.py`; `tools/summary_stats.py error_law` |
| Road-network graphs and resilience metrics | `resil/network.py` (graph); `resil/network_e4b.py` (metrics); `tools/e4_network.py` |
| Rainfall events, baseline, disruption and recovery | `resil/rain.py`; `tools/e10_rain.py`; `tools/rain_periods.py` |
| Robustness analyses | `tools/e6_robust.py`; `tools/further_checks.py` |
| Dependence of the grid-based diagnostics on cell size | `tools/further_checks.py segments`; `tools/active_cells.py`; `figures/si.py` (`s12`) |

## Figures and tables

The figure scripts write their outputs under working names; `figures/paper_names.py` copies them, with their Source Data,
to `out/figures/paper/` under the numbers of the paper, and `source_data/` in this repository holds these files. Every
panel writes the values it plots to a CSV file together with a note on how they were computed. The figures of the paper
were finished from these outputs (layout, labels and annotations); the plotted values are those of the Source Data.
Fig. 1a (left) and Fig. 1b are schematic illustrations, and the map of Fig. 1c was drawn from `Fig1_c.csv` (city
locations and numbers of stations and road segments).

| Paper | Script (working name) | Source Data |
|---|---|---|
| Fig. 1a (right) | `figures/fig1a_panel.py` | `Fig1_a_windows`, `Fig1_a_error` |
| Fig. 1c | map drawn from the Source Data | `Fig1_c` |
| Fig. 2 | `figures/fig2_panels.py` (a, b; panels c–f from `figures/fig2.py`) | `Fig2_a`–`Fig2_f` |
| Figs 3–5 | `figures/fig3.py`, `fig4.py`, `fig5.py` | `Fig3_a`–`Fig5_e` |
| Table 1, Supplementary Tables 5 and 6 | `figures/tables.py` | `Table1.csv`, `SuppTable5.csv`, `SuppTable6.csv` |
| Supplementary Fig. 1 | `figures/si.py` `s9` (SuppFig9) | `SuppFig1` |
| Supplementary Fig. 2 | `figures/fig1.py` (Fig1) | `SuppFig2_a`–`SuppFig2_d` |
| Supplementary Fig. 3 | `figures/si.py` `s1` (SuppFig1) | `SuppFig3` |
| Supplementary Fig. 4 | `figures/si.py` `s4` (SuppFig4) | `SuppFig4_a`, `SuppFig4_b` |
| Supplementary Fig. 5 | `figures/si.py` `s13` (SuppFig13) | `SuppFig5_a`, `SuppFig5_bc` |
| Supplementary Fig. 6 | `figures/si.py` `s2` (SuppFig2) | `SuppFig6` |
| Supplementary Fig. 7 | `figures/si.py` `s3` (SuppFig3) | `SuppFig7_a`–`SuppFig7_c` |
| Supplementary Fig. 8 | `figures/si.py` `s11` (SuppFig11) | `SuppFig8` |
| Supplementary Fig. 9 | `figures/si.py` `s10` (SuppFig10) | `SuppFig9_a`–`SuppFig9_ef` |
| Supplementary Fig. 10 | `figures/si.py` `s6` (SuppFig6) | `SuppFig10` |
| Supplementary Fig. 11 | `figures/si.py` `s7` (SuppFig7) | `SuppFig11` |
| Supplementary Fig. 12 | `figures/si.py` `s12` (SuppFig12) | `SuppFig12_ab`, `SuppFig12_c`, `SuppFig12_fits` |
| Supplementary Fig. 13 | `figures/fig6.py` (Fig6) | `SuppFig13_a`–`SuppFig13_d` |
| Supplementary Fig. 14 | `figures/si.py` `s5` (SuppFig5) | `SuppFig14_ab`–`SuppFig14_d` |
| Supplementary Fig. 15 | `figures/si.py` `s8` (SuppFig8) | `SuppFig15` |
| Supplementary Table 4 (city information) | `tools/e8_tables.py` | `out/e8/t_dates.csv`, `t_cells.csv` |
| Supplementary Table 7 (pairs of cities) | `tools/summary_stats.py pairs` | `out/checks/pairs_traffic.csv`, `pairs_network.csv` |
| Supplementary Table 8 (dispersion relative to shuffled flows) | `main/cross_city.py` | `out/main34/h1_h2.csv` |
| Supplementary Table 9 (active cells) | `tools/active_cells.py` | `out/checks/active_cells_summary.csv` |
| Flow centres per city | `tools/summary_stats.py centres` | `out/checks/centres_fstar.csv` |

## Checks of reproducibility

- **Full chain, four cities.** The full chain was rerun in a new folder containing only the code and the raw files of
  four cities (Basel, Constance, Essen, Augsburg). Every per-city output of every stage was identical to the reported one.
- **Main analysis, 34 cities.** The main analysis of all 34 cities was rerun from the processed station data with the code
  of this repository, with Python 3.13 and numpy 2.5 instead of Python 3.11 and numpy 2.4. The critical coverages and
  their bootstrap intervals were identical to the reported ones, and all other per-city and cross-city results agreed
  with them to within floating-point rounding (differences below 10<sup>-9</sup>). The results in
  `processed_results.zip` are those of this rerun.
- **Figures, tables and Source Data.** All figures, tables and Source Data were regenerated with this code in a new
  folder from `processed_results.zip` and the road files of the 34 cities (Python 3.13, numpy 2.5, pandas 3.0). The
  Source Data were identical to those in `source_data/`.
- **Data preparation and rainfall.** The data preparation was rerun from the raw files of Basel and Constance, and the
  rainfall profile and the rainfall analysis without the outlier rule from the raw files of all cities with rainfall
  events (Python 3.10, numpy 2.2); the results were identical to the reported ones.
- **Tests.** `tests/test_core.py` checks the windows, the critical coverage, the correlation length and the test for an
  abrupt change against synthetic data with known answers. Test T1 checks that the grid diagnostics reproduce the
  earlier release of this code (commit 8c12261 of
  <https://github.com/viviRG2024/Finite-size-Convergence-of-Mobility-based-Resilience>) to rounding error; it runs when
  that code is placed in `clean_code/`.

## Licence

The code is released under the MIT licence (`LICENSE`). The processed data and Source Data are processed from the IUTF
dataset and are released under CC BY 4.0 (`DATA_LICENCE.md`).
