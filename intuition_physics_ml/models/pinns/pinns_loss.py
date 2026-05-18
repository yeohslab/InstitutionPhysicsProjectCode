"""PINN 数据损失 + 单摆物理残差（非线性 / 小角度线性）。"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from intuition_physics_ml.models.pinns.pinns_input_normalizer import PinnsInputNormalizer

PhysicsMode = Literal["nonlinear", "linear"]


def theta_tt_wrt_t(
    net: nn.Module,
    t: torch.Tensor,
    l: torch.Tensor,
    theta0: torch.Tensor,
    input_normalizer: PinnsInputNormalizer,
) -> torch.Tensor:
    """对 batch 中每个样本计算 ∂²θ/∂t²（物理时间 t）；t 形状 (N,1)，需 requires_grad。

    网络输入为归一化坐标；\\(\\hat t(t)\\) 对 t 仿射可微，autograd 自动应用链式法则，
    故 \\(\\partial^2\\theta/\\partial t^2\\) 为物理时间下的二阶导，无需再手写缩放因子。
    """
    x = input_normalizer.pack(t, l, theta0)
    theta = net(x)
    (theta_t,) = torch.autograd.grad(
        outputs=theta.sum(),
        inputs=t,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    (theta_tt,) = torch.autograd.grad(
        outputs=theta_t.sum(),
        inputs=t,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    return theta_tt


def physics_residual(
    net: nn.Module,
    t: torch.Tensor,
    l: torch.Tensor,
    theta0: torch.Tensor,
    g: torch.Tensor,
    physics_mode: PhysicsMode,
    input_normalizer: PinnsInputNormalizer,
) -> torch.Tensor:
    """f = θ_tt + (g/l) sin θ 或 θ_tt + (g/l) θ。各张量形状 (N,1)。"""
    theta_tt = theta_tt_wrt_t(net, t, l, theta0, input_normalizer)
    gl = g / l
    theta = net(input_normalizer.pack(t, l, theta0))
    if physics_mode == "nonlinear":
        return theta_tt + gl * torch.sin(theta)
    return theta_tt + gl * theta


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def pinns_losses(
    net: nn.Module,
    t: torch.Tensor,
    l: torch.Tensor,
    theta0: torch.Tensor,
    g: torch.Tensor,
    theta_target: torch.Tensor,
    *,
    physics_mode: PhysicsMode,
    input_normalizer: PinnsInputNormalizer,
    lambda_data: float,
    lambda_physics: float,
    lambda_ic: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    数据项在 detach 的 t 上算监督；物理项在 t.clone().requires_grad_(True) 上算残差。

    当 ``lambda_ic > 0`` 时增加初始条件项：\\(\\theta(0,l,\\theta_0)=\\theta_0\\)，
    避免网络退化为恒为 0 的「假」物理解（此时 \\(\\theta_{tt}\\) 与 \\(\\sin\\theta\\) 均为 0，
    物理损失很小却与数据严重不符）。

    t, l, theta0, g, theta_target: (N,1)
    """
    x_data = input_normalizer.pack(t.detach(), l, theta0)
    pred_data = net(x_data)
    loss_data = mse(pred_data, theta_target)

    t_phys = t.detach().clone().requires_grad_(True)
    res = physics_residual(net, t_phys, l, theta0, g, physics_mode, input_normalizer)
    loss_physics = torch.mean(res**2)

    if lambda_ic > 0.0:
        t0 = torch.zeros_like(t)
        x_ic = input_normalizer.pack(t0, l, theta0)
        pred_ic = net(x_ic)
        loss_ic = mse(pred_ic, theta0)
    else:
        loss_ic = pred_data.new_zeros(())

    total = lambda_data * loss_data + lambda_physics * loss_physics + lambda_ic * loss_ic
    return total, loss_data, loss_physics, loss_ic
