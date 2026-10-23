import pandas as pd

from game_betting_lab import WIN_UNITS, recommendations, score_bets


def test_spread_and_total_grading():
    d = pd.DataFrame([{"week": 6, "gameday": "2026-10-18", "away_team": "A", "home_team": "H", "actual_margin": 3.0, "actual_total": 44.0, "market_spread": 2.5, "market_total": 40.5, "home_cover_probability": .61, "over_probability": .62}])
    bets = recommendations(d, .57)
    assert list(bets["result"]) == ["WIN", "WIN"]
    s = score_bets(bets)
    assert s["wins"] == 2
    assert round(s["units"], 4) == round(2 * WIN_UNITS, 4)


def test_away_cover_uses_home_margin_semantics():
    d = pd.DataFrame([{"week": 6, "gameday": "2026-10-18", "away_team": "A", "home_team": "H", "actual_margin": 1.0, "actual_total": 40.0, "market_spread": 3.0, "market_total": 40.0, "home_cover_probability": .40, "over_probability": .50}])
    bets = recommendations(d, .57)
    assert len(bets) == 1
    assert bets.iloc[0]["side"] == "AWAY"
    assert bets.iloc[0]["result"] == "WIN"


def test_threshold_filters_low_confidence():
    d = pd.DataFrame([{"week": 6, "gameday": "2026-10-18", "away_team": "A", "home_team": "H", "actual_margin": 0.0, "actual_total": 40.0, "market_spread": 0.0, "market_total": 40.0, "home_cover_probability": .53, "over_probability": .54}])
    assert recommendations(d, .55).empty
