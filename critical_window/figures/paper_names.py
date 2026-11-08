"""Copy the figures and Source Data to the numbering of the paper.

    python -m critical_window.figures.paper_names      # after make_all, fig1a_panel and fig2_panels

The figure scripts write their outputs under working names (Fig1-Fig6, SuppFig1-SuppFig13). This step copies them to
out/figures/paper/ under the numbers of the published figures and tables, writes Source_Data.xlsx with one sheet per
panel, and lists the correspondence in out/figures/paper/name_map.csv.
"""
from __future__ import annotations

import shutil

import pandas as pd

from . import common as C

PAPER = C.FIG / "paper"
PSRC = PAPER / "source_data"

# working figure name -> figure in the paper
FIGURES = {
    "Fig1a_right": "Fig1a_right", "Fig2_panels": "Fig2", "Fig3": "Fig3", "Fig4": "Fig4", "Fig5": "Fig5",
    "SuppFig9": "SuppFig1", "Fig1": "SuppFig2", "SuppFig1": "SuppFig3", "SuppFig4": "SuppFig4", "SuppFig13": "SuppFig5",
    "SuppFig2": "SuppFig6", "SuppFig3": "SuppFig7", "SuppFig11": "SuppFig8", "SuppFig10": "SuppFig9", "SuppFig6": "SuppFig10",
    "SuppFig7": "SuppFig11", "SuppFig12": "SuppFig12", "Fig6": "SuppFig13", "SuppFig5": "SuppFig14", "SuppFig8": "SuppFig15",
}
# working Source Data file -> file in the paper (panel letters of the published figures)
SOURCE = {
    # Fig. 1: a, the data panel (Zurich); c, the map of the 34 cities
    "Fig1_c": ["Fig1_a_windows", "SuppFig2_c"], "Fig1_d": ["SuppFig2_d"], "Fig1_a": ["Fig1_c", "SuppFig2_a"],
    "Fig1_b": ["SuppFig2_b"],
    # Fig. 2: a, b from fig2_panels; c-f from fig2
    "Fig2_London_windows": ["Fig2_a"], "Fig2_London_two_windows": ["Fig2_b"],
    "Fig2_a": ["Fig2_c"], "Fig2_b": ["Fig2_d"], "Fig2_c": ["Fig2_e"], "Fig2_d": ["Fig2_f"],
    **{f"Fig{k}_{p}": [f"Fig{k}_{p}"] for k, ps in ((3, "abcd"), (4, "abcd")) for p in ps},
    "Fig5_ab": ["Fig5_ab"], "Fig5_c": ["Fig5_c"], "Fig5_d": ["Fig5_d"], "Fig5_e": ["Fig5_e"],
    # Supplementary figures
    "SuppFig9": ["SuppFig1"], "SuppFig1": ["SuppFig3"], "SuppFig4_a": ["SuppFig4_a"], "SuppFig4_b": ["SuppFig4_b"],
    "SuppFig13_a": ["SuppFig5_a"], "SuppFig13_bc": ["SuppFig5_bc"], "SuppFig2": ["SuppFig6"],
    "SuppFig3_a": ["SuppFig7_a"], "SuppFig3_b": ["SuppFig7_b"], "SuppFig3_c": ["SuppFig7_c"], "SuppFig11": ["SuppFig8"],
    "SuppFig10_a": ["SuppFig9_a"], "SuppFig10_b": ["SuppFig9_b"], "SuppFig10_c": ["SuppFig9_c"], "SuppFig10_d": ["SuppFig9_d"],
    "SuppFig10_ef": ["SuppFig9_ef"], "SuppFig6": ["SuppFig10"], "SuppFig7": ["SuppFig11"],
    "SuppFig12_ab": ["SuppFig12_ab"], "SuppFig12_c": ["SuppFig12_c"], "SuppFig12_fits": ["SuppFig12_fits"],
    "Fig6_a": ["SuppFig13_a"], "Fig6_b": ["SuppFig13_b"], "Fig6_c": ["SuppFig13_c"], "Fig6_d": ["SuppFig13_d"],
    "SuppFig5_ab": ["SuppFig14_ab"], "SuppFig5_c": ["SuppFig14_c"], "SuppFig5_d": ["SuppFig14_d"], "SuppFig8": ["SuppFig15"],
}
TABLES = ("Table1.csv", "SuppTable5.csv", "SuppTable6.csv", "Tables.xlsx")


def sheet_name(stem: str) -> str:
    s = stem.replace("SuppFig", "Supp Fig ").replace("Fig", "Fig ").replace("Supp Fig  ", "Supp Fig ")
    return s.replace("_", " ")[:31]


def main():
    if PAPER.exists():
        shutil.rmtree(PAPER)
    PSRC.mkdir(parents=True)
    rows = []
    for old, new in FIGURES.items():
        for ext in (".png", ".pdf"):
            if (C.FIG / f"{old}{ext}").exists():
                shutil.copy2(C.FIG / f"{old}{ext}", PAPER / f"{new}{ext}")
        rows.append(dict(kind="figure", working_name=old, paper_name=new))
    missing = sorted({f.stem for f in C.SRC.glob("*.csv")} - set(SOURCE))
    if missing:
        raise SystemExit(f"Source Data without a paper name: {missing}")
    for old, news in SOURCE.items():
        df = pd.read_csv(C.SRC / f"{old}.csv")
        note_f = C.SRC / f"{old}.txt"
        note = note_f.read_text(encoding="utf-8") if note_f.exists() else ""
        for new in news:
            d, n = df, note
            if new == "Fig1_a_windows":
                n = "Fig. 1a (right): " + note
            d.to_csv(PSRC / f"{new}.csv", index=False)
            if n:
                (PSRC / f"{new}.txt").write_text(n, encoding="utf-8")
            rows.append(dict(kind="source data", working_name=old, paper_name=new))
    # Fig. 1a insets: window error of sigma_z in Zurich
    d = pd.read_csv(C.SRC / "Fig1_d.csv")
    d[d.metric == "sd_log"].to_csv(PSRC / "Fig1_a_error.csv", index=False)
    (PSRC / "Fig1_a_error.txt").write_text("Fig. 1a (insets): " + (C.SRC / "Fig1_d.txt").read_text(encoding="utf-8"), encoding="utf-8")
    rows.append(dict(kind="source data", working_name="Fig1_d (sigma_z rows)", paper_name="Fig1_a_error"))
    for t in TABLES:
        if (C.FIG / t).exists():
            shutil.copy2(C.FIG / t, PAPER / t)
    pd.DataFrame(rows).to_csv(PAPER / "name_map.csv", index=False)
    files = sorted(PSRC.glob("*.csv"), key=lambda f: (not f.stem.startswith("Fig"), f.stem))
    with pd.ExcelWriter(PAPER / "Source_Data.xlsx", engine="xlsxwriter") as xw:
        bold = xw.book.add_format({"bold": True})
        wrap = xw.book.add_format({"text_wrap": True, "valign": "top"})
        for f in files:
            df = pd.read_csv(f)
            sheet = sheet_name(f.stem)
            note_f = f.with_suffix(".txt")
            note = note_f.read_text(encoding="utf-8") if note_f.exists() else ""
            df.to_excel(xw, sheet_name=sheet, index=False, startrow=3)
            ws = xw.sheets[sheet]
            ws.write(0, 0, sheet, bold)
            ws.merge_range(1, 0, 1, max(6, df.shape[1] - 1), note, wrap)
            ws.set_row(1, 30)
    print(PAPER, len(files), "Source Data files")


if __name__ == "__main__":
    main()
