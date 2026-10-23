import pandas as pd

import spread_walkforward as sw


def test_walkforward_aggregates_weekly_results(monkeypatch):
    def fake_week(season, week, simulations=10000, seed=0):
        # Home cover probability .60 qualifies under the frozen .56 rule.
        actual_margin = 4.0 if week == 2 else 1.0
        return pd.DataFrame([{
            "gameday":"2026-09-01","away_team":"AWY","home_team":"HME",
            "market_spread":3.0,"market_total":45.0,"projected_margin":5.0,
            "projected_total":44.0,"home_cover_probability":.60,
            "over_probability":.50,"actual_margin":actual_margin,"actual_total":44.0,
        }])
    monkeypatch.setattr(sw,"run_week_retrospective",fake_week)
    summary,weekly,bets=sw.run_walkforward(2026,weeks=(2,3),simulations=10)
    assert summary["overall"]["bets"] == 2
    assert summary["overall"]["wins"] == 1
    assert summary["overall"]["losses"] == 1
    assert summary["profitable_weeks"] == 1
    assert summary["losing_weeks"] == 1
    assert len(weekly) == 2
    assert len(bets) == 2


def test_walkforward_rule_is_spread_only():
    assert sw.SPREAD_RULE["probability_threshold"] == .56
    assert sw.SPREAD_RULE["min_gap"] == 0.0
