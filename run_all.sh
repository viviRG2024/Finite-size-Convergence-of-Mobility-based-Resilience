#!/usr/bin/env bash
# Full run from the raw IUTF files to every figure, table and Source Data file.
# Run from the repository root, with the IUTF data in data/input/IUTFD/ (see README.md).
# Total time on a two-core computer: about 5-6 h, most of it in the road-network stage (step 5).
set -euo pipefail
rp="python -m critical_window"
CITIES=$(paste -sd, critical_window/data/cities_analysed.txt)
IUTFD=data/input/IUTFD

# 1. Data preparation and quality control (station flows, QC tables)            -> critical_window/out/x0
$rp.run_x0
# 2. Main analysis: random windows, traffic diagnostics, window errors, f*       -> out/main34
$rp.run_main --cities "$CITIES" --out-tag main34
$rp.tools.report_fixes --dir critical_window/out/main34                     # bootstrap interval of f*
# 3. Error law: sampling formula                                                -> out/e2
$rp.tools.e2_formula windows
$rp.tools.e2_formula formula
# 4. CV and H on 500 m cells over 32 grid origins                               -> out/e3
$rp.tools.e3_grid run
$rp.tools.e3_grid summary
# 5. Road-network resilience metrics (the long step)                            -> out/e4b
$rp.tools.e4_network run --kind e4b --iutfd $IUTFD
$rp.tools.e4_network summary --kind e4b
# $rp.tools.e4_network noise --kind e4b --iutfd $IUTFD    # optional: Monte-Carlo noise of the network metrics
# 6. Station layout, block averaging, design effect (three steps)               -> out/e5
$rp.tools.e5_fractal run --iutfd $IUTFD
$rp.tools.e5_fractal summary
$rp.tools.e5_fractal deff
# 7. Robustness: periods, weather, outliers, windows, study areas, cells, tolerance -> out/e6
$rp.tools.e6_robust run --weather $IUTFD
$rp.tools.e6_robust summary
# 8. Criterion based on the stability of the scaling exponents                   -> out/e7
$rp.tools.e7_bridge run
$rp.tools.e7_bridge run-glob
$rp.tools.e7_bridge summary
# 9. Window overlap, tiles, flow centres, dates, rankings                       -> out/e8
$rp.tools.e8_tables run
$rp.tools.e8_tables summary
# 10. Rainfall events                                                           -> out/e10
$rp.tools.e10_rain run --weather $IUTFD --raw $IUTFD
$rp.tools.e10_rain summary
$rp.tools.e10_rain run --weather $IUTFD --raw $IUTFD --thr 1.0
$rp.tools.e10_rain summary --thr 1.0
$rp.tools.e10_rain run --weather $IUTFD --raw $IUTFD --win-days 7
$rp.tools.e10_rain summary --win-days 7
$rp.tools.e10_rain run --weather $IUTFD --raw $IUTFD --win-days 1000
$rp.tools.e10_rain summary --win-days 1000
$rp.tools.rain_periods run --raw $IUTFD
$rp.tools.rain_periods summary
$rp.tools.rain_outlier_check --raw $IUTFD
# 11. Further checks and numbers given in the text                              -> out/checks
$rp.tools.further_checks ranking
$rp.tools.further_checks segments
$rp.tools.further_checks correlates
$rp.tools.further_checks units200
$rp.tools.active_cells
$rp.tools.summary_stats all
# 12. Figures, tables and Source Data                                           -> out/figures
$rp.figures.make_all
$rp.figures.fig1a_panel
$rp.figures.fig2_panels
$rp.figures.paper_names                                                    # numbering of the paper -> out/figures/paper
