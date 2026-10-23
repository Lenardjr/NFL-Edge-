import pandas as pd

from game_betting_multiseason import evaluate_fixed_spread, season_weeks


def sample_history():
    return pd.DataFrame([
        {"season_eval":2024,"week_eval":1,"week":1,"gameday":"2024-09-08","away_team":"A","home_team":"H","actual_margin":7.0,"actual_total":40.0,"projected_margin":6.0,"projected_total":42.0,"market_spread":3.0,"market_total":41.0,"home_cover_probability":.61,"over_probability":.50},
        {"season_eval":2025,"week_eval":2,"week":2,"gameday":"2025-09-14","away_team":"B","home_team":"J","actual_margin":-4.0,"actual_total":44.0,"projected_margin":-5.0,"projected_total":43.0,"market_spread":-1.0,"market_total":42.0,"home_cover_probability":.38,"over_probability":.50},
        {"season_eval":2025,"week_eval":2,"week":2,"gameday":"2025-09-14","away_team":"C","home_team":"K","actual_margin":1.0,"actual_total":39.0,"projected_margin":1.5,"projected_total":40.0,"market_spread":1.0,"market_total":40.0,"home_cover_probability":.53,"over_probability":.50},
    ])


def test_2026_stops_at_week6_but_prior_seasons_use_full_regular_season():
    assert list(season_weeks(2026)) == [1,2,3,4,5,6]
    assert list(season_weeks(2025))[-1] == 18


def test_fixed_rule_is_scored_by_season_without_retuning():
    overall,by_season,bets=evaluate_fixed_spread(sample_history())
    assert overall["bets"] == 2
    assert overall["wins"] == 2
    assert set(by_season.season) == {2024,2025}
    assert len(bets) == 2
