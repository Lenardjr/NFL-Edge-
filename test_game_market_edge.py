import pytest

from game_market_edge import (
    assess_game_markets,
    assess_moneyline,
    assess_spread,
    assess_total,
    best_available_edge,
)


def prediction():
    return {
        "home_win_prob": 0.60,
        "home_cover_prob": 0.58,
        "over_prob": 0.47,
    }


def test_moneyline_probabilities_are_complements():
    result = assess_moneyline(prediction(), 110, -130)
    assert result["home"]["model_probability"] == pytest.approx(0.60)
    assert result["away"]["model_probability"] == pytest.approx(0.40)


def test_spread_uses_cover_probability():
    result = assess_spread(prediction())
    assert result["home"]["model_probability"] == pytest.approx(0.58)
    assert result["away"]["model_probability"] == pytest.approx(0.42)
    assert result["home"]["edge"] == pytest.approx(0.08)


def test_total_uses_over_probability():
    result = assess_total(prediction())
    assert result["over"]["model_probability"] == pytest.approx(0.47)
    assert result["under"]["model_probability"] == pytest.approx(0.53)


def test_complete_snapshot_selects_best_non_pass_edge():
    result = assess_game_markets(
        prediction(),
        home_moneyline=110,
        away_moneyline=-130,
    )
    assert set(result["markets"]) == {"moneyline", "spread", "total"}
    assert result["best_edge"] is not None
    assert result["best_edge"]["classification"] != "PASS"


def test_best_edge_can_return_none():
    markets = {
        "spread": {
            "home": {"classification": "PASS", "expected_value": -0.01, "edge": 0.0},
            "away": {"classification": "PASS", "expected_value": -0.02, "edge": 0.0},
        }
    }
    assert best_available_edge(markets) is None
