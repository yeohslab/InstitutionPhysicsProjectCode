"""PINN 输入 min-max 归一化到约 [0,1]；对 t 的可微仿射使 autograd 自动实现链式法则。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch


@dataclass(frozen=True)
class PinnsInputNormalizer:
    """各维独立：\\(\\hat{x}=(x-x_{\\min})/\\max(x_{\\max}-x_{\\min},\\epsilon)\\)。"""

    t_min: float
    t_max: float
    l_min: float
    l_max: float
    theta0_min: float
    theta0_max: float

    _eps: float = 1e-8

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> PinnsInputNormalizer:
        return cls(
            t_min=float(df["t"].min()),
            t_max=float(df["t"].max()),
            l_min=float(df["l"].min()),
            l_max=float(df["l"].max()),
            theta0_min=float(df["theta0"].min()),
            theta0_max=float(df["theta0"].max()),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "t_min": self.t_min,
            "t_max": self.t_max,
            "l_min": self.l_min,
            "l_max": self.l_max,
            "theta0_min": self.theta0_min,
            "theta0_max": self.theta0_max,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PinnsInputNormalizer:
        return cls(
            t_min=float(d["t_min"]),
            t_max=float(d["t_max"]),
            l_min=float(d["l_min"]),
            l_max=float(d["l_max"]),
            theta0_min=float(d["theta0_min"]),
            theta0_max=float(d["theta0_max"]),
        )

    def _norm(self, x: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
        span = max(x_max - x_min, self._eps)
        lo = torch.as_tensor(x_min, device=x.device, dtype=x.dtype)
        return (x - lo) / span

    def pack(self, t: torch.Tensor, l: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
        """拼接 \\((\\hat t,\\hat l,\\hat{\\theta}_0)\\)，形状 (N,1) 输入 → (N,3)。保留对 t,l,θ0 的计算图。"""
        t_hat = self._norm(t, self.t_min, self.t_max)
        l_hat = self._norm(l, self.l_min, self.l_max)
        th_hat = self._norm(theta0, self.theta0_min, self.theta0_max)
        return torch.cat([t_hat, l_hat, th_hat], dim=-1)
