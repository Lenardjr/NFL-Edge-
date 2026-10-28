import pandas as pd

from edge_bucket_validation import _bucket_metrics, _side_rows, evaluate_split


def sample_history():
    return pd.DataFrame([
        {"season_eval": 2023, "week_eval": 1, "gameday": "2023-09-10", "away_team": "A", "home_team": "H", "home_cover_probability": .60, "market_spread": 2.5, "actual_margin": 7.0},
        {"season_eval": 2025, "week_eval": 1, "gameday": "2025-09-07", "away_team": "B", "home_team": "J", "home_cover_probability": .51, "market_spread": -3.0, "actual_margin": -4.0},
    ])


def test_side_rows_classify_and_score():
    bets = _side_rows(sample_history())
    assert len(bets) == 2
    assert bets.iloc[0].side == "HOME"
    assert bets.iloc[0].result == "WIN"


def test_split_keeps_validation_separate():
    bets = _side_rows(sample_history())
    assert evaluate_split(bets, (2023, 2024))["games"] == 1
    assert evaluate_split(bets, (2025,))["games"] == 1


def test_bucket_table_has_all_labels():
    table = _bucket_metrics(_side_rows(sample_history()))
    assert table.classification.tolist() == ["PASS", "LEAN", "EDGE", "STRONG_EDGE"]
