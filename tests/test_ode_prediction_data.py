from pathlib import Path

import numpy as np
import pandas as pd

from intuition_physics_ml.ode_prediction_data import (
    ODE_PREDICTION_SHARED_KWARGS,
    _case_equilibrium_to_apex_times,
    default_ode_prediction_csv_path,
    generate_and_save_ode_prediction_data,
    generate_and_save_ode_prediction_test_data,
    generate_ode_prediction_dataset,
)


def test_generate_ode_prediction_dataset_shape_and_traj_ids() -> None:
    n_cases = 11
    n_times = 5
    t_end = 5.0
    df = generate_ode_prediction_dataset(
        n_cases=n_cases,
        n_times_per_case=n_times,
        t_end=t_end,
        rng_seed=0,
        sigma_l=0.0,
        sigma_theta0=0.0,
    )
    n_rows = n_cases * n_times
    assert len(df) == n_rows
    assert df["traj_id"].tolist() == list(range(n_rows))
    assert len(df["traj_id"].unique()) == n_rows
    for i in range(n_cases):
        blk = df.iloc[i * n_times : (i + 1) * n_times]
        assert len(blk) == n_times
        assert blk["l"].nunique() == 1
        assert blk["theta0"].nunique() == 1
    expected_cols = {
        "traj_id",
        "t",
        "l",
        "theta0",
        "omega0",
        "theta0_deg",
        "g",
        "theta",
        "theta_deg",
    }
    assert set(df.columns) == expected_cols
    assert (df["omega0"] == 0.0).all()


def test_t_values_strictly_between_equilibrium_and_opposite_apex() -> None:
    g = 9.81
    t_end = 8.0
    df = generate_ode_prediction_dataset(
        n_cases=20,
        n_times_per_case=3,
        t_end=t_end,
        rng_seed=3,
        sigma_l=0.0,
        sigma_theta0=0.0,
    )
    for _, row in df.iterrows():
        interval = _case_equilibrium_to_apex_times(
            float(row["l"]),
            g,
            float(row["theta0"]),
            float(row["omega0"]),
            t_end,
        )
        assert interval is not None
        t_eq, t_apex = interval
        t_i = float(row["t"])
        assert t_eq < t_i < t_apex, (t_eq, t_i, t_apex)


def test_consecutive_rows_per_case_share_l_and_theta0() -> None:
    n_cases = 12
    n_times = 4
    df = generate_ode_prediction_dataset(
        n_cases=n_cases,
        n_times_per_case=n_times,
        t_end=6.0,
        rng_seed=4,
        sigma_l=0.01,
        sigma_theta0=np.deg2rad(0.5),
    )
    for i in range(n_cases):
        blk = df.iloc[i * n_times : (i + 1) * n_times]
        assert blk["l"].nunique() == 1
        assert blk["theta0"].nunique() == 1
        assert blk["omega0"].nunique() == 1
        assert blk["g"].nunique() == 1


def test_generate_and_save_csv(tmp_path: Path) -> None:
    p = tmp_path / "ode_prediction_data.csv"
    df, out = generate_and_save_ode_prediction_data(
        csv_path=p,
        n_cases=5,
        n_times_per_case=1,
        t_end=10.0,
        rng_seed=0,
        sigma_l=0.0,
        sigma_theta0=0.0,
    )
    assert out == p
    assert p.is_file()
    assert len(df) == 5


def test_default_csv_path_name() -> None:
    path = default_ode_prediction_csv_path()
    assert path.name == "ode_prediction_data.csv"
    assert path.parent.name == "algorithm_prediction_data"


def test_shared_physics_keys_cover_train_merge() -> None:
    assert set(ODE_PREDICTION_SHARED_KWARGS.keys()) == {
        "g",
        "t_end",
        "length_range",
        "theta0_deg_range",
        "sigma_l",
        "sigma_theta0",
        "l_floor",
    }
    assert ODE_PREDICTION_SHARED_KWARGS["g"] == 9.81
    assert ODE_PREDICTION_SHARED_KWARGS["t_end"] == 10.0


def test_train_and_test_save_identical_when_case_scale_matches(tmp_path: Path) -> None:
    p_train = tmp_path / "train.csv"
    p_test = tmp_path / "test.csv"
    df_train, _ = generate_and_save_ode_prediction_data(
        csv_path=p_train,
        n_cases=4,
        n_times_per_case=2,
        rng_seed=99,
        sigma_l=0.0,
        sigma_theta0=0.0,
    )
    df_test, _ = generate_and_save_ode_prediction_test_data(
        csv_path=p_test,
        n_cases=4,
        n_times_per_case=2,
        rng_seed=99,
        sigma_l=0.0,
        sigma_theta0=0.0,
    )
    pd.testing.assert_frame_equal(df_train, df_test)
