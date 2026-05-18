"""PINN 损失与产物路径的轻量测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from intuition_physics_ml.models.pinns.pinns_input_normalizer import PinnsInputNormalizer
from intuition_physics_ml.models.pinns.pinns_loss import physics_residual, pinns_losses
from intuition_physics_ml.models.pinns.pinns_net import PendulumPINN
from intuition_physics_ml.paths import repo_root
from intuition_physics_ml.training.pinns.pinns_config import PinnsTrainConfig
from intuition_physics_ml.training.pinns.pinns_trainer import train_pinns


def _minmax_norm_from_tensors(
    t: torch.Tensor, l: torch.Tensor, theta0: torch.Tensor
) -> PinnsInputNormalizer:
    return PinnsInputNormalizer(
        t_min=float(t.detach().min()),
        t_max=float(t.detach().max()),
        l_min=float(l.detach().min()),
        l_max=float(l.detach().max()),
        theta0_min=float(theta0.detach().min()),
        theta0_max=float(theta0.detach().max()),
    )


def test_linear_ode_residual_analytical() -> None:
    """θ = cos(ω t)，ω²=g/l 满足 θ_tt + (g/l)θ = 0。"""
    g, l = 9.81, 1.0
    w = (g / l) ** 0.5
    t = torch.linspace(0.0, 1.0, 32, requires_grad=True).reshape(-1, 1)
    theta = torch.cos(w * t)
    theta_t = torch.autograd.grad(theta.sum(), t, create_graph=True)[0]
    theta_tt = torch.autograd.grad(theta_t.sum(), t, create_graph=True)[0]
    r = theta_tt + (g / l) * theta
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-5, rtol=0.0)


def test_pinns_run_dir_layout() -> None:
    cfg = PinnsTrainConfig(
        dataset_source="algorithm",
        train_csv=Path("dummy.csv"),
        physics_mode="nonlinear",
        batch_size=8,
        lr=1e-3,
        epochs=1,
        lambda_data=1.0,
        lambda_physics=1.0,
        lambda_ic=0.0,
        seed=0,
        artifact_base_dir=Path("/tmp/repo"),
    )
    assert cfg.run_dir() == Path("/tmp/repo") / "artifacts" / "pinns" / "algorithm" / "pinns_run" / "model01"
    cfg2 = PinnsTrainConfig(
        dataset_source="human",
        train_csv=Path("d.csv"),
        physics_mode="linear",
        batch_size=8,
        lr=1e-3,
        epochs=1,
        lambda_data=1.0,
        lambda_physics=1.0,
        lambda_ic=0.0,
        seed=0,
        artifact_base_dir=Path("/tmp/repo"),
    )
    assert cfg2.run_dir().name == "model02"
    assert cfg2.run_dir().parent.name == "pinns_run"
    assert cfg2.run_dir().parent.parent.name == "human"


def test_pinns_losses_finite() -> None:
    net = PendulumPINN(hidden_dim=16, n_hidden=2)
    n = 8
    t = torch.rand(n, 1, dtype=torch.float32)
    l = torch.full((n, 1), 1.0, dtype=torch.float32)
    theta0 = torch.randn(n, 1, dtype=torch.float32) * 0.1
    g = torch.full((n, 1), 9.81, dtype=torch.float32)
    theta = torch.randn(n, 1, dtype=torch.float32) * 0.1
    norm = _minmax_norm_from_tensors(t, l, theta0)
    for mode in ("nonlinear", "linear"):
        total, ld, lp, lic = pinns_losses(
            net,
            t,
            l,
            theta0,
            g,
            theta,
            physics_mode=mode,
            input_normalizer=norm,
            lambda_data=1.0,
            lambda_physics=0.5,
            lambda_ic=0.0,
        )
        assert (
            torch.isfinite(total)
            and torch.isfinite(ld)
            and torch.isfinite(lp)
            and torch.isfinite(lic)
        )


def test_physics_residual_shape_matches_net() -> None:
    net = PendulumPINN(hidden_dim=8, n_hidden=1)
    n = 4
    t = torch.linspace(0, 0.5, n, requires_grad=True).reshape(-1, 1)
    l = torch.ones(n, 1)
    theta0 = torch.zeros(n, 1)
    g = torch.full((n, 1), 9.81)
    norm = _minmax_norm_from_tensors(t, l, theta0)
    r = physics_residual(net, t, l, theta0, g, "nonlinear", norm)
    assert r.shape == (n, 1)


def test_train_pinns_smoke(tmp_path: Path) -> None:
    g, l0 = 9.81, 1.0
    rows = []
    for i in range(16):
        rows.append(
            {
                "t": float(i) * 0.05,
                "l": l0,
                "theta0": 0.2,
                "g": g,
                "theta": 0.2 * (1.0 - float(i) * 0.01),
            }
        )
    csv = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    cfg = PinnsTrainConfig(
        dataset_source="algorithm",
        train_csv=csv,
        physics_mode="linear",
        batch_size=8,
        lr=0.01,
        epochs=1,
        lambda_data=1.0,
        lambda_physics=0.1,
        lambda_ic=1.0,
        seed=0,
        n_hidden=2,
        artifact_base_dir=tmp_path,
    )
    train_pinns(cfg)
    run = cfg.run_dir()
    assert (run / "checkpoint.pt").is_file()
    assert (run / "run_meta.json").is_file()
    ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert "pinns_input_norm" in ck and "t_min" in ck["pinns_input_norm"]


def test_default_pinns_train_csv_algorithm() -> None:
    from intuition_physics_ml.paths import default_pinns_train_csv

    p = default_pinns_train_csv("algorithm")
    assert p == repo_root() / "data" / "raw" / "algorithm_prediction_data" / "ode_prediction_data.csv"
