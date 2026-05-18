from __future__ import annotations

from pathlib import Path
from typing import Literal


def repo_root() -> Path:
    """仓库根目录（含 pyproject.toml 的目录）。"""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_pinns_train_csv(dataset_source: Literal["algorithm", "human"]) -> Path:
    """PINN 默认训练表路径；human 文件尚未提供时使用约定文件名。"""
    root = repo_root()
    if dataset_source == "algorithm":
        return root / "data" / "raw" / "algorithm_prediction_data" / "ode_prediction_data.csv"
    return root / "data" / "raw" / "human_prediction_data" / "human_prediction_data.csv"
