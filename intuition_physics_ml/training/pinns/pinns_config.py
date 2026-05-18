"""PINN 训练配置：数据源 × 物理模式 → artifacts/pinns/<source>/pinns_run/model0x/。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from intuition_physics_ml.paths import repo_root

DatasetSource = Literal["algorithm", "human"]
PhysicsMode = Literal["nonlinear", "linear"]


@dataclass(frozen=True)
class PinnsTrainConfig:
    dataset_source: DatasetSource
    train_csv: Path
    physics_mode: PhysicsMode
    batch_size: int
    lr: float
    epochs: int
    lambda_data: float
    lambda_physics: float
    # 初始条件 θ(t=0)=θ0 的 MSE 权重；>0 可抑制「恒为零」退化解（仅靠物理项时 θ≡0 残差为 0）。
    lambda_ic: float
    seed: int
    hidden_dim: int = 128
    n_hidden: int = 3
    # 若指定，则产物根为 ``artifact_base_dir / artifacts/pinns/...``（便于测试隔离）。
    artifact_base_dir: Path | None = None

    def model_dir_name(self) -> str:
        return "model01" if self.physics_mode == "nonlinear" else "model02"

    def run_dir(self, base_dir: Path | None = None) -> Path:
        root = base_dir or self.artifact_base_dir or repo_root()
        return (
            root
            / "artifacts"
            / "pinns"
            / self.dataset_source
            / "pinns_run"
            / self.model_dir_name()
        )

    @staticmethod
    def from_dict(data: dict[str, Any], base_dir: Path | None = None) -> PinnsTrainConfig:
        root = base_dir or repo_root()
        csv = Path(data["train_csv"])
        if not csv.is_absolute():
            csv = root / csv
        ds = str(data["dataset_source"])
        if ds not in ("algorithm", "human"):
            raise ValueError(f"dataset_source 须为 algorithm 或 human，收到 {ds!r}")
        pm = str(data["physics_mode"])
        if pm not in ("nonlinear", "linear"):
            raise ValueError(f"physics_mode 须为 nonlinear 或 linear，收到 {pm!r}")
        ab_raw = data.get("artifact_base_dir")
        artifact_base: Path | None = None
        if ab_raw is not None:
            artifact_base = Path(ab_raw)
            if not artifact_base.is_absolute():
                artifact_base = root / artifact_base
        return PinnsTrainConfig(
            dataset_source=ds,  # type: ignore[arg-type]
            train_csv=csv,
            physics_mode=pm,  # type: ignore[arg-type]
            batch_size=int(data["batch_size"]),
            lr=float(data["lr"]),
            epochs=int(data["epochs"]),
            lambda_data=float(data["lambda_data"]),
            lambda_physics=float(data["lambda_physics"]),
            lambda_ic=float(data.get("lambda_ic", 1.0)),
            seed=int(data["seed"]),
            hidden_dim=int(data.get("hidden_dim", 128)),
            n_hidden=int(data.get("n_hidden", 3)),
            artifact_base_dir=artifact_base,
        )
