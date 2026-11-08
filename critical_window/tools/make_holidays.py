"""Generate data/holidays.csv (public holidays per city) with the `holidays` package.

Run once by the maintainers; the pipeline itself only reads the CSV, so users do not need `holidays`.
    python critical_window/tools/make_holidays.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import holidays

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
YEARS = [2008] + list(range(2014, 2022))


def main() -> None:
    rows = []
    with open(PKG / "data" / "cities.csv", newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            cc, sub = rec["country"], rec["holiday_subdiv"] or None
            try:
                cal = holidays.country_holidays(cc, subdiv=sub, years=YEARS)
            except Exception as exc:  # unknown subdivision -> national calendar
                print(f"{rec['city']}: subdivision {sub!r} not available ({exc}); using national holidays")
                cal = holidays.country_holidays(cc, years=YEARS)
            for day, name in sorted(cal.items()):
                rows.append({"city": rec["city"], "date": day.isoformat(), "name": name})
    out = PKG / "data" / "holidays.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["city", "date", "name"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out} (holidays {holidays.__version__}, years {YEARS[0]}-{YEARS[-1]})")


if __name__ == "__main__":
    main()
