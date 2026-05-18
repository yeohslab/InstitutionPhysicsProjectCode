"""PINN 训练循环与检查点（含 run_meta.json）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from intuition_physics_ml.models.pinns.pinns_input_normalizer import PinnsInputNormalizer
from intuition_physics_ml.models.pinns.pinns_loss import pinns_losses
from intuition_physics_ml.models.pinns.pinns_net import PendulumPINN
from intuition_physics_ml.paths import ensure_dir
from intuition_physics_ml.training.pinns.pinns_config import PinnsTrainConfig
from intuition_physics_ml.training.pinns.pinns_dataloader import make_pinns_dataloader


def train_pinns(cfg: PinnsTrainConfig) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    run_dir = cfg.run_dir()
    ensure_dir(run_dir)

    if not cfg.train_csv.is_file():
        raise FileNotFoundError(f"训练 CSV 不存在: {cfg.train_csv}")

    train_df = pd.read_csv(cfg.train_csv)
    input_normalizer = PinnsInputNormalizer.from_dataframe(train_df)

    loader = make_pinns_dataloader(
        cfg.train_csv,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    net = PendulumPINN(hidden_dim=cfg.hidden_dim, n_hidden=cfg.n_hidden)
    opt = Adam(net.parameters(), lr=cfg.lr)

    last_total = last_data = last_phys = last_ic = 0.0
    for epoch in range(cfg.epochs):
        net.train()
        epoch_tot = epoch_data = epoch_phys = epoch_ic = 0.0
        n_batches = 0
        for batch in loader:
            t = batch["t"]
            l = batch["l"]
            theta0 = batch["theta0"]
            g = batch["g"]
            theta = batch["theta"]

            opt.zero_grad(set_to_none=True)
            total, ld, lp, lic = pinns_losses(
                net,
                t,
                l,
                theta0,
                g,
                theta,
                physics_mode=cfg.physics_mode,
                input_normalizer=input_normalizer,
                lambda_data=cfg.lambda_data,
                lambda_physics=cfg.lambda_physics,
                lambda_ic=cfg.lambda_ic,
            )
            total.backward()
            opt.step()

            epoch_tot += float(total.detach())
            epoch_data += float(ld.detach())
            epoch_phys += float(lp.detach())
            epoch_ic += float(lic.detach())
            n_batches += 1

        last_total = epoch_tot / max(n_batches, 1)
        last_data = epoch_data / max(n_batches, 1)
        last_phys = epoch_phys / max(n_batches, 1)
        last_ic = epoch_ic / max(n_batches, 1)
        print(
            f"epoch {epoch + 1}/{cfg.epochs} "
            f"loss={last_total:.6e} data={last_data:.6e} phys={last_phys:.6e} ic={last_ic:.6e}"
        )

    ckpt_path = run_dir / "checkpoint.pt"
    norm_dict = input_normalizer.to_dict()
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "config": {
                "dataset_source": cfg.dataset_source,
                "physics_mode": cfg.physics_mode,
                "train_csv": str(cfg.train_csv),
                "hidden_dim": cfg.hidden_dim,
                "n_hidden": cfg.n_hidden,
                "lambda_ic": cfg.lambda_ic,
            },
            "pinns_input_norm": norm_dict,
        },
        ckpt_path,
    )

    meta: dict[str, Any] = {
        "dataset_source": cfg.dataset_source,
        "physics_mode": cfg.physics_mode,
        "train_csv": str(cfg.train_csv),
        "lambda_data": cfg.lambda_data,
        "lambda_physics": cfg.lambda_physics,
        "lambda_ic": cfg.lambda_ic,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "final_loss_total": last_total,
        "final_loss_data": last_data,
        "final_loss_physics": last_phys,
        "final_loss_ic": last_ic,
        "pinns_input_norm": norm_dict,
        "checkpoint": str(ckpt_path),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "loss_total": last_total,
        "loss_data": last_data,
        "loss_physics": last_phys,
        "loss_ic": last_ic,
    }
