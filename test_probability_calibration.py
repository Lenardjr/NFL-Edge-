import numpy as np
import pandas as pd

from probability_calibration import (
    apply_calibrator,
    brier_score,
    fit_logit_calibrator,
    log_loss_score,
    prepare_calibration_frame,
    reliability_table,
)


def sample_history():
    return pd.DataFrame([
        {"season_eval": 2023, "home_cover_probability": .80, "market_spread": 3.0, "actual_margin": 0.0},
        {"season_eval": 2023, "home_cover_probability": .70, "market_spread": -2.0, "actual_margin": 4.0},
        {"season_eval": 2024, "home_cover_probability": .30, "market_spread": 1.0, "actual_margin": -3.0},
        {"season_eval": 2024, "home_cover_probability": .20, "market_spread": -1.0, "actual_margin": 2.0},
        {"season_eval": 2025, "home_cover_probability": .60, "market_spread": 2.0, "actual_margin": 2.0},
    ])


def test_prepare_excludes_pushes_and_scores_cover_direction():
    frame = prepare_calibration_frame(sample_history())
    assert len(frame) == 4
    assert frame.target.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_metrics_reward_better_probabilities():
    y = np.array([1.0, 0.0])
    good = np.array([.8, .2])
    bad = np.array([.2, .8])
    assert brier_score(good, y) < brier_score(bad, y)
    assert log_loss_score(good, y) < log_loss_score(bad, y)


def test_calibrator_returns_probabilities():
    frame = prepare_calibration_frame(sample_history())
    dev = frame[frame.season_eval.isin([2023, 2024])]
    model = fit_logit_calibrator(dev)
    calibrated = apply_calibrator(model, dev.raw_probability)
    assert len(calibrated) == len(dev)
    assert np.all((calibrated > 0) & (calibrated < 1))


def test_reliability_table_covers_ten_buckets():
    table = reliability_table(np.array([.05, .55, .95]), np.array([0.0, 1.0, 1.0]))
    assert len(table) == 10
    assert int(table.n.sum()) == 3
