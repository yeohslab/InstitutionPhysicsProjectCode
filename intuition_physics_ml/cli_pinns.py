"""命令行：按 YAML 训练 PINN（模型01 非线性 / 模型02 线性）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from intuition_physics_ml.config import load_yaml
from intuition_physics_ml.paths import repo_root
from intuition_physics_ml.training.pinns.pinns_config import PinnsTrainConfig
from intuition_physics_ml.training.pinns.pinns_trainer import train_pinns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train pendulum PINN from YAML config")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML 路径（默认 configs/pinns/algorithm_model01.yaml）",
    )
    args = parser.parse_args(argv)
    root = repo_root()
    cfg_path = args.config or (root / "configs" / "pinns" / "algorithm_model01.yaml")
    raw = load_yaml(cfg_path)
    cfg = PinnsTrainConfig.from_dict(raw, base_dir=root)
    metrics = train_pinns(cfg)
    print(f"Saved run to: {cfg.run_dir()}")
    print(f"Final losses: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
