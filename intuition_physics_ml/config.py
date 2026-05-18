from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        out = yaml.safe_load(f)
    if not isinstance(out, dict):
        raise ValueError(f"配置文件必须是 YAML 映射: {path}")
    return out
