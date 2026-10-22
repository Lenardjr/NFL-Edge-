import pandas as pd
import pytest

from prospective_holdout import challenger_margin, score_frozen_predictions


def test_challenger_uses_correct_home_margin_sign():
    result = challenger_margin(pd.Series([4.0, -4.0]), pd.Series([2.0, -2.0]))
    assert result.tolist() == pytest.approx([3.4, -3.4])


def test_challenger_falls_back_without_market_line():
    result = challenger_margin(pd.Series([3.0]), pd.Series([float("nan")]))
    assert result.tolist() == pytest.approx([3.0])


def test_holdout_preserves_winner_definition_and_scores_margin():
    predictions = pd.DataFrame([
        {"game_id": "A", "engine_1_0_margin": 4.0, "market_spread": 2.0},
        {"game_id": "B", "engine_1_0_margin": -3.0, "market_spread": -1.0},
    ])
    results = pd.DataFrame([
        {"game_id": "A", "actual_home_score": 24, "actual_away_score": 21},
        {"game_id": "B", "actual_home_score": 17, "actual_away_score": 20},
    ])
    report = score_frozen_predictions(predictions, results)
    assert report["summary"]["games"] == 2
    assert report["summary"]["winner_definition_matches"] is True
    assert report["summary"]["engine_1_0_winner_accuracy"] == 1.0
    assert report["summary"]["engine_1_1_b_winner_accuracy"] == 1.0
