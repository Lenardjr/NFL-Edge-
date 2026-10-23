import pandas as pd

from spread_signal_research import BREAK_EVEN_WIN_RATE,enrich,segment_table,side_bets


def test_enrichment_and_spread_grading():
    d=pd.DataFrame([
        {"projected_margin":6.0,"market_spread":3.0,"actual_margin":7.0,"home_cover_probability":.62},
        {"projected_margin":-4.0,"market_spread":-1.0,"actual_margin":2.0,"home_cover_probability":.40},
    ])
    x=side_bets(enrich(d))
    assert list(x.model_side)==["HOME","AWAY"]
    assert list(x.result)==["WIN","LOSS"]
    assert list(x.model_market_gap)==[3.0,3.0]


def test_segment_table_marks_small_samples_ineligible():
    d=pd.DataFrame({"bucket":["A","A"],"result":["WIN","LOSS"],"units":[100/110,-1.0]})
    table=segment_table(d,"bucket",min_bets=20)
    assert table.iloc[0].bets==2
    assert not bool(table.iloc[0].eligible)
    assert abs(BREAK_EVEN_WIN_RATE-.5238095238)<1e-9
