"""Build all main figures, the tables and the Source Data workbook.

    python -m critical_window.figures.make_all

Outputs in critical_window/out/figures/, under working names: Fig1-6 and SuppFig1-13 (.pdf, .png), Table1.csv,
SuppTable5.csv, SuppTable6.csv, Tables.xlsx, source_data/<Fig>_<panel>.csv (+ .txt notes) and Source_Data.xlsx. Then run
figures.fig1a_panel, figures.fig2_panels and figures.paper_names, which copies everything to out/figures/paper/ under the
numbers of the paper (name_map.csv lists the correspondence).
"""
from __future__ import annotations

import pandas as pd

from . import common as C
from . import fig1, fig2, fig3, fig4, fig5, fig6, si, tables


def compile_source_data():
    files = sorted(C.SRC.glob("*.csv"))
    with pd.ExcelWriter(C.FIG / "Source_Data.xlsx", engine="xlsxwriter") as xw:
        wb = xw.book
        bold = wb.add_format({"bold": True})
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        for f in files:
            df = pd.read_csv(f)
            sheet = f.stem.replace("SuppFig", "Supp Fig ").replace("Fig", "Fig ").replace("Supp Fig  ", "Supp Fig ")[:31]
            note_f = f.with_suffix(".txt")
            note = note_f.read_text(encoding="utf-8") if note_f.exists() else ""
            df.to_excel(xw, sheet_name=sheet, index=False, startrow=3)
            ws = xw.sheets[sheet]
            ws.write(0, 0, f.stem.replace("_", " panel "), bold)
            ws.merge_range(1, 0, 1, max(6, df.shape[1] - 1), note, wrap)
            ws.set_row(1, 30)
            for j, col in enumerate(df.columns):
                ws.set_column(j, j, max(10, min(26, len(str(col)) + 2)))
    print(C.FIG / "Source_Data.xlsx", len(files), "panels")


def main():
    for m in (fig1, fig2, fig3, fig4, fig5, fig6):
        m.main()
    for w in si.ALL:
        C.style()
        getattr(si, w)()
    tables.build()
    compile_source_data()


if __name__ == "__main__":
    main()
