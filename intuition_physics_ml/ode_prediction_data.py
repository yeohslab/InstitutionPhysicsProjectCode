"""单摆 ODE 宽表：l、θ0 带测量噪声，ω0=0；t 取自首次过平衡至对侧顶点之间的开区间，每 case 多条时刻。

训练 CSV（``ode_prediction_data.csv``）与测试 CSV（``ode_prediction_test.csv``）共用
``ODE_PREDICTION_SHARED_KWARGS`` 中的物理与采样超参；仅 ``n_cases``、``n_times_per_case``、
``rng_seed`` 在测试入口上默认不同。重新生成二者::

    python -m intuition_physics_ml.ode_prediction_data
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from intuition_physics_ml.paths import repo_root

_IVP_RTOL = 1e-9
_IVP_ATOL = 1e-12
_IVP_METHOD = "DOP853"
_THETA0_ABS_MIN = 1e-4
_DT_MIN = 1e-10
_U_FRAC_LO = 1e-3
_U_FRAC_HI = 1.0 - 1e-3
_MAX_CASE_DRAW_TRIES = 128

# 训练/测试 CSV 共用物理与噪声范围；与 ``generate_ode_prediction_dataset`` 默认值同源。
ODE_PREDICTION_SHARED_KWARGS: dict[str, Any] = {
    "g": 9.81,
    "t_end": 10.0,
    "length_range": (0.5, 1.5),
    "theta0_deg_range": (-55.0, 55.0),
    "sigma_l": 0.01,
    "sigma_theta0": float(np.deg2rad(1.0)),
    "l_floor": None,
}


def _make_rhs(length: float, g: float):
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        theta, omega = y
        return np.array([omega, -(g / length) * np.sin(theta)], dtype=float)

    return rhs


def _tspan_upper(length: float, g: float, t_end: float) -> float:
    t_char = 2.0 * np.pi * np.sqrt(max(length, 1e-6) / max(g, 1e-9))
    return float(max(t_end, 10.0 * t_char))


def _first_equilibrium_crossing(
    rhs,
    *,
    theta0: float,
    omega0: float,
    t_span_max: float,
) -> tuple[float, np.ndarray] | None:
    if abs(theta0) < _THETA0_ABS_MIN:
        return None

    def event_theta(_t: float, y: np.ndarray) -> float:
        return float(y[0])

    event_theta.terminal = True
    event_theta.direction = -1.0 if theta0 > 0.0 else 1.0

    sol = solve_ivp(
        rhs,
        (0.0, t_span_max),
        np.array([theta0, omega0], dtype=float),
        events=event_theta,
        method=_IVP_METHOD,
        rtol=_IVP_RTOL,
        atol=_IVP_ATOL,
    )
    if not sol.success or sol.t_events is None or len(sol.t_events[0]) == 0:
        return None
    t_eq = float(sol.t_events[0][0])
    y_eq = np.asarray(sol.y_events[0][0], dtype=float).ravel()
    if t_eq <= 0.0:
        return None
    return t_eq, y_eq


def _first_omega_zero_after(rhs, *, t_start: float, y_start: np.ndarray, t_span_max: float) -> float | None:
    w0 = float(y_start[1])
    if abs(w0) < 1e-15:
        return None

    def event_omega(_t: float, y: np.ndarray) -> float:
        return float(y[1])

    event_omega.terminal = True
    event_omega.direction = 1.0 if w0 < 0.0 else -1.0

    sol = solve_ivp(
        rhs,
        (t_start, t_span_max),
        np.asarray(y_start, dtype=float).ravel(),
        events=event_omega,
        method=_IVP_METHOD,
        rtol=_IVP_RTOL,
        atol=_IVP_ATOL,
    )
    if not sol.success or sol.t_events is None or len(sol.t_events[0]) == 0:
        return None
    return float(sol.t_events[0][0])


def _case_equilibrium_to_apex_times(
    l_meas: float,
    g: float,
    theta0_meas: float,
    omega0_meas: float,
    t_end: float,
) -> tuple[float, float] | None:
    rhs = _make_rhs(l_meas, g)
    t_hi = _tspan_upper(l_meas, g, t_end)
    out = _first_equilibrium_crossing(rhs, theta0=theta0_meas, omega0=omega0_meas, t_span_max=t_hi)
    if out is None:
        return None
    t_eq, y_eq = out
    t_apex = _first_omega_zero_after(rhs, t_start=t_eq, y_start=y_eq, t_span_max=t_hi)
    if t_apex is None or not (t_apex > t_eq + _DT_MIN):
        return None
    return t_eq, t_apex


def _sample_t_in_open_interval(
    rng: np.random.Generator,
    t_eq: float,
    t_apex: float,
) -> float:
    span = t_apex - t_eq
    u = float(rng.uniform(_U_FRAC_LO, _U_FRAC_HI))
    return t_eq + u * span


def _theta_at_time(
    rhs,
    *,
    theta0: float,
    omega0: float,
    t_target: float,
) -> float:
    if t_target <= 0.0:
        return float(theta0)
    sol = solve_ivp(
        rhs,
        (0.0, float(t_target)),
        np.array([theta0, omega0], dtype=float),
        t_eval=np.array([float(t_target)], dtype=float),
        method=_IVP_METHOD,
        rtol=_IVP_RTOL,
        atol=_IVP_ATOL,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return float(sol.y[0, -1])


def generate_ode_prediction_dataset(
    *,
    n_cases: int = 2048,
    n_times_per_case: int = 8,
    g: float = ODE_PREDICTION_SHARED_KWARGS["g"],
    t_end: float = ODE_PREDICTION_SHARED_KWARGS["t_end"],
    rng_seed: int = 42,
    length_range: tuple[float, float] = ODE_PREDICTION_SHARED_KWARGS["length_range"],
    theta0_deg_range: tuple[float, float] = ODE_PREDICTION_SHARED_KWARGS["theta0_deg_range"],
    sigma_l: float = ODE_PREDICTION_SHARED_KWARGS["sigma_l"],
    sigma_theta0: float = ODE_PREDICTION_SHARED_KWARGS["sigma_theta0"],
    l_floor: float | None = ODE_PREDICTION_SHARED_KWARGS["l_floor"],
) -> pd.DataFrame:
    """生成宽表：每个物理 case 一组 (l, θ0)，连续 ``n_times_per_case`` 行共享该组参数；
    各行的 ``t`` 独立采样于 (t_过平衡, t_对侧顶点) 开区间；``traj_id`` 为全局唯一行号。
    """
    rng = np.random.default_rng(rng_seed)
    l_min, l_max = length_range
    th_min, th_max = theta0_deg_range
    floor = l_floor if l_floor is not None else max(1e-4, 0.01 * l_min)

    n_cases_i = int(n_cases)
    n_times_i = int(n_times_per_case)
    if n_cases_i < 1 or n_times_i < 1:
        raise ValueError("n_cases and n_times_per_case must be positive")

    rows: list[dict] = []
    traj_id = 0

    for case_id in range(n_cases_i):
        bounds_ok = False
        l_meas = theta0_meas = 0.0
        for _ in range(_MAX_CASE_DRAW_TRIES):
            l_nom = float(rng.uniform(l_min, l_max))
            theta0_nom = float(rng.uniform(np.deg2rad(th_min), np.deg2rad(th_max)))

            l_meas = float(l_nom + rng.normal(0.0, sigma_l))
            l_meas = max(floor, l_meas)

            theta0_meas = float(theta0_nom + rng.normal(0.0, sigma_theta0))
            omega0_meas = 0.0

            interval = _case_equilibrium_to_apex_times(l_meas, g, theta0_meas, omega0_meas, t_end)
            if interval is None:
                continue
            t_eq, t_apex = interval
            if t_apex - t_eq > _DT_MIN:
                bounds_ok = True
                break

        if not bounds_ok:
            raise RuntimeError(
                "Failed to sample a pendulum case with a valid (t_equilibrium, t_opposite_apex) "
                f"interval after {_MAX_CASE_DRAW_TRIES} tries (case_index={case_id})."
            )

        rhs = _make_rhs(l_meas, g)
        for _ in range(n_times_i):
            t_i = _sample_t_in_open_interval(rng, t_eq, t_apex)
            theta_f = _theta_at_time(rhs, theta0=theta0_meas, omega0=omega0_meas, t_target=t_i)

            rows.append(
                {
                    "traj_id": traj_id,
                    "t": float(t_i),
                    "l": l_meas,
                    "theta0": theta0_meas,
                    "omega0": omega0_meas,
                    "theta0_deg": float(np.degrees(theta0_meas)),
                    "g": float(g),
                    "theta": theta_f,
                    "theta_deg": float(np.degrees(theta_f)),
                }
            )
            traj_id += 1

    return pd.DataFrame(rows)


def default_ode_prediction_csv_path() -> Path:
    return repo_root() / "data" / "raw" / "algorithm_prediction_data" / "ode_prediction_data.csv"


def default_ode_prediction_test_csv_path() -> Path:
    return repo_root() / "data" / "test_dataset" / "ode_prediction_test.csv"


def generate_and_save_ode_prediction_data(
    csv_path: Path | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, Path]:
    df = generate_ode_prediction_dataset(**{**ODE_PREDICTION_SHARED_KWARGS, **kwargs})
    path = csv_path or default_ode_prediction_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df, path


def generate_and_save_ode_prediction_test_data(
    csv_path: Path | None = None,
    *,
    n_cases: int = 256,
    n_times_per_case: int = 4,
    rng_seed: int = 1337,
    **kwargs,
) -> tuple[pd.DataFrame, Path]:
    """与训练集相同的物理/``t`` 采样规则（见 ``ODE_PREDICTION_SHARED_KWARGS``）；默认更小
    ``n_cases``、``n_times_per_case`` 及独立 ``rng_seed``，避免与训练行重复。
    """
    path = csv_path or default_ode_prediction_test_csv_path()
    return generate_and_save_ode_prediction_data(
        csv_path=path,
        **{
            **ODE_PREDICTION_SHARED_KWARGS,
            "n_cases": n_cases,
            "n_times_per_case": n_times_per_case,
            "rng_seed": rng_seed,
            **kwargs,
        },
    )


def _cli_regenerate_default_csvs() -> None:
    _, train_path = generate_and_save_ode_prediction_data()
    _, test_path = generate_and_save_ode_prediction_test_data()
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    _cli_regenerate_default_csvs()
