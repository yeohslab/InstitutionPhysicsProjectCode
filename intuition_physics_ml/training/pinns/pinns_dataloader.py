"""从宽表 CSV 加载 PINN 监督样本 (t, l, theta0, g, theta)。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class PinnsCsvDataset(Dataset):
    """列：t, l, theta0, g, theta（弧度）；其余列忽略。"""

    _required = ("t", "l", "theta0", "g", "theta")

    def __init__(self, csv_path: Path | str) -> None:
        self._path = Path(csv_path)
        self._df = pd.read_csv(self._path)
        missing = [c for c in self._required if c not in self._df.columns]
        if missing:
            raise ValueError(f"CSV 缺少列 {missing}: {self._path}")

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._df.iloc[idx]
        return {
            "t": torch.tensor([float(row["t"])], dtype=torch.float32),
            "l": torch.tensor([float(row["l"])], dtype=torch.float32),
            "theta0": torch.tensor([float(row["theta0"])], dtype=torch.float32),
            "g": torch.tensor([float(row["g"])], dtype=torch.float32),
            "theta": torch.tensor([float(row["theta"])], dtype=torch.float32),
        }


def make_pinns_dataloader(
    csv_path: Path | str,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = PinnsCsvDataset(csv_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
