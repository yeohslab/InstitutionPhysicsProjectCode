# 直觉物理方向的机器学习项目

以**单摆 PINN** 为主：配置 → 宽表 CSV → 网络与物理损失 → 训练 → `artifacts/` 产物；配套 Jupyter 可视化。

## 环境

- 安装 [uv](https://docs.astral.sh/uv/) 后，在仓库根目录执行：

```bash
uv sync
```

- 开发依赖（含 pytest）：`uv sync --extra dev`

## 推荐阅读顺序（对应代码位置）

| 顺序 | 目录/文件 | 作用 |
|------|-----------|------|
| 1 | `configs/pinns/*.yaml` | PINN 超参、`dataset_source`、`physics_mode`、数据路径 |
| 2 | `intuition_physics_ml/paths.py` | 项目根目录与默认数据路径 |
| 3 | `intuition_physics_ml/config.py` | `load_yaml` |
| 4 | `intuition_physics_ml/ode_prediction_data.py` | ODE 宽表生成与默认 CSV 路径 |
| 5 | `intuition_physics_ml/models/pinns/` | `PendulumPINN` 与损失 |
| 6 | `intuition_physics_ml/training/pinns/` | 配置、`DataLoader`、训练循环 |
| 7 | `intuition_physics_ml/cli_pinns.py` | `train-pinns` 命令行入口 |

## 目录约定

- `data/raw/`：原始数据（默认不进 Git，仅保留 `.gitkeep`）
- `data/processed/`：清洗/特征工程后的中间表
- `artifacts/`：训练出的模型、日志等可再生产物（PINN 为 `artifacts/pinns/...`）
- `notebooks/`：探索性分析（Jupyter）
- `scripts/`：可重复执行的一键脚本
- `tests/`：自动化测试

## 运行

```bash
uv run train-pinns --config configs/pinns/algorithm_model01.yaml
# 或
uv run python main.py
# 或
uv run python -m intuition_physics_ml
```

以上入口均默认使用 `configs/pinns/algorithm_model01.yaml`（可用 `--config` 指定 `algorithm_model02.yaml` 等）。

## 测试

```bash
uv run pytest tests -q
```

## 依赖说明

`pyproject.toml` 中含 `torch`、`numpy`、`pandas`、`scipy`、`scikit-learn`（部分工具或遗留依赖）等；PINN 训练依赖 **PyTorch**。
