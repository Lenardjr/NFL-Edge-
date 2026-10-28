import numpy as np
import pandas as pd

from ats_signal_root_cause import confidence_deciles, diagnostics, gap_deciles, prepare_frame


def sample_history():
    gaps = np.linspace(-10, 10, 20)
    market = np.zeros(20)
    projected = gaps.copy()
    actual = np.where(gaps < 0, -3.0, 3.0)
    probabilities = 1.0 / (1.0 + np.exp(-gaps / 3.0))
    return pd.DataFrame({
        "season_eval": [2023] * 10 + [2024] * 10,
        "projected_margin": projected,
        "actual_margin": actual,
        "market_spread": market,
        "home_cover_probability": probabilities,
    })


def test_prepare_builds_market_relative_gap_and_target():
    frame = prepare_frame(sample_history())
    assert len(frame) == 20
    assert np.allclose(frame.model_market_gap.to_numpy(), np.linspace(-10, 10, 20))
    assert frame.target.iloc[0] == 0.0
    assert frame.target.iloc[-1] == 1.0


def test_margin_gap_has_strong_rank_signal_in_synthetic_sample():
    metrics = diagnostics(prepare_frame(sample_history()))
    assert metrics["margin_vs_market_auc"] > 0.9
    assert metrics["cover_probability_auc"] > 0.9


def test_gap_deciles_account_for_all_games():
    table = gap_deciles(prepare_frame(sample_history()))
    assert int(table.games.sum()) == 20


def test_confidence_deciles_account_for_all_games():
    table = confidence_deciles(prepare_frame(sample_history()))
    assert int(table.games.sum()) == 20
    assert table.model_side_win_rate.between(0, 1).all()
