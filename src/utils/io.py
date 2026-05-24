"""JSON helpers that handle numpy types and write atomically."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """Encode numpy scalars/arrays as native JSON."""

    def default(self, obj: Any) -> Any:  # noqa: D401
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Atomically write ``obj`` as JSON to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=NumpyJSONEncoder, indent=indent)
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    """Read JSON from ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
