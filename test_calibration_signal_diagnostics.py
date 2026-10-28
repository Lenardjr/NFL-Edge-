import numpy as np
import pandas as pd

from calibration_signal_diagnostics import (
    confidence_table,
    decile_table,
    discrimination_metrics,
    expected_calibration_error,
)


def sample_frame():
    probabilities = np.linspace(0.1, 0.9, 20)
    targets = np.array([0] * 10 + [1] * 10, dtype=float)
    return pd.DataFrame({"raw_probability": probabilities, "target": targets})


def test_auc_detects_rank_signal():
    metrics = discrimination_metrics(sample_frame())
    assert metrics["auc"] > 0.9


def test_ece_is_zero_for_perfect_binary_probabilities():
    p = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    assert expected_calibration_error(p, y) == 0.0


def test_deciles_account_for_all_games():
    table = decile_table(sample_frame())
    assert int(table.games.sum()) == 20


def test_confidence_table_accounts_for_all_games():
    table = confidence_table(sample_frame())
    assert int(table.games.sum()) == 20
    assert table.model_side_win_rate.between(0, 1).all()
