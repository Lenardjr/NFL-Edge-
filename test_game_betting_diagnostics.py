import pandas as pd

from game_betting_diagnostics import add_edges, calibration_table, market_bets


def sample():
    return pd.DataFrame([
        {"week":6,"gameday":"2026-10-18","away_team":"A","home_team":"H","actual_margin":3.0,"actual_total":44.0,"projected_margin":6.0,"projected_total":46.0,"market_spread":2.5,"market_total":40.5,"home_cover_probability":.61,"over_probability":.62},
        {"week":6,"gameday":"2026-10-18","away_team":"B","home_team":"J","actual_margin":-4.0,"actual_total":38.0,"projected_margin":-5.0,"projected_total":39.0,"market_spread":-1.0,"market_total":44.0,"home_cover_probability":.38,"over_probability":.35},
    ])


def test_add_edges_uses_absolute_model_market_gap():
    d=add_edges(sample())
    assert d.iloc[0].spread_market_gap==3.5
    assert d.iloc[0].total_market_gap==5.5


def test_market_filter_separates_spread_and_total():
    spread=market_bets(sample(),"SPREAD",.55,3.0)
    total=market_bets(sample(),"TOTAL",.55,5.0)
    assert len(spread)==2 and set(spread.market)=={"SPREAD"}
    assert len(total)==2 and set(total.market)=={"TOTAL"}


def test_calibration_reports_probability_vs_actual():
    bets=market_bets(sample(),"SPREAD",.50,0)
    c=calibration_table(bets)
    assert c.bets.sum()==2
    assert "calibration_error" in c.columns
