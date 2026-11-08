"""Configuration and paths shared by all pipeline stages."""
from __future__ import annotations

import csv
import json
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]          # .../critical_window
REPO_DIR = PKG_DIR.parent                               # repository root (044/)


def _scalar(tok: str):
    tok = tok.strip()
    if tok == "" or tok in ("null", "~"):
        return None
    if tok in ("true", "True"):
        return True
    if tok in ("false", "False"):
        return False
    if tok.startswith("[") and tok.endswith("]"):
        inner = tok[1:-1].strip()
        return [] if not inner else [_scalar(t) for t in inner.split(",")]
    if (tok[0] == tok[-1]) and tok[0] in "'\"":
        return tok[1:-1]
    for cast in (int, float):
        try:
            return cast(tok)
        except ValueError:
            pass
    return tok


def _mini_yaml(text: str) -> dict:
    """Parse the small YAML subset used in config.yaml (nested maps, scalars, inline lists)."""
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val.strip() == "":
            parent[key] = {}
            stack.append((indent, parent[key]))
        else:
            parent[key] = _scalar(val)
    return root


def load_config(path: Path | None = None) -> dict:
    path = Path(path) if path else PKG_DIR / "config.yaml"
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml(text)


def repo_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_DIR / p


def city_table() -> dict:
    with open(PKG_DIR / "data" / "cities.csv", newline="", encoding="utf-8") as fh:
        return {r["city"]: r for r in csv.DictReader(fh)}


def holidays_by_city() -> dict:
    out: dict = {}
    with open(PKG_DIR / "data" / "holidays.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["city"], set()).add(r["date"])
    return out


def read_metadata(city_dir: Path) -> dict:
    p = city_dir / f"{city_dir.name}_metadata.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
