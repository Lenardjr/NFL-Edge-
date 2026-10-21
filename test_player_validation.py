import numpy as np
import pandas as pd

from player_validation import normalize_player_stats, validate_player_feed, empirical_interval_errors, reliability_tier


def test_normalize_accepts_new_team_schema():
    d = pd.DataFrame({"season":[2026], "week":[1], "player_id":["p1"], "team_abbr":["LA"]})
    out = normalize_player_stats(d)
    assert out.loc[0, "recent_team"] == "LAR"


def test_feed_health_accepts_normalized_player_data():
    d = pd.DataFrame({"season":[2026], "week":[1], "player_id":["p1"], "club_code":["DET"]})
    health = validate_player_feed(d)
    assert health["ok"] is True
    assert health["teams"] == 1
    assert health["player_key"] == "player_id"


def test_error_metrics_are_deterministic():
    metrics = empirical_interval_errors([10, 20, 30], [12, 18, 33])
    assert metrics["n"] == 3
    assert np.isclose(metrics["mae"], 7/3)
    assert np.isclose(metrics["bias"], -1.0)
    assert metrics["correlation"] > 0.98


def test_reliability_gate_is_conservative():
    assert reliability_tier(300, 0.60, 0.50) == "STRONG"
    assert reliability_tier(180, 0.45, 0.70) == "USABLE"
    assert reliability_tier(100, 0.80, 0.20) == "RESEARCH"
