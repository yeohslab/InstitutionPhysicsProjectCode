"""全连接 PINN：输入 (t, l, theta0)，输出 theta。"""

from __future__ import annotations

import torch
from torch import nn


class PendulumPINN(nn.Module):
    """3×128 全连接 + Tanh，与项目 TODO 一致。"""

    def __init__(
        self,
        hidden_dim: int = 128,
        n_hidden: int = 3,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3) 列为 t, l, theta0（弧度）。"""
        return self.net(x)
