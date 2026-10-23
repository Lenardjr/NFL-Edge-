import pandas as pd

from qb_stat_v04_context import qb_history_features, context_features


def test_qb_history_features_use_supplied_rows_only():
    h=pd.DataFrame({"season":[2026]*4,"week":[1,2,3,4],"attempts":[20,30,40,100],"completions":[10,18,28,90],"passing_yards":[100,180,300,900],"passing_tds":[1,1,2,6]})
    f=qb_history_features(h[h.week<4])
    assert f["att_mean_3"]==30
    assert f["att_mean_8"]==30
    assert f["ypa_8"] < 8


def test_context_market_sign_home_and_away():
    row=pd.Series({"home_team":"H","away_team":"A","spread_line":3.0,"total_line":47.0,"home_pf5":24,"home_pa5":20,"away_pf5":21,"away_pa5":23,"home_rest":7,"away_rest":6})
    home=context_features(row,"H");away=context_features(row,"A")
    assert home["market_margin"]==3.0
    assert away["market_margin"]==-3.0
    assert home["market_team_total"]==25.0
    assert away["market_team_total"]==22.0
    assert home["home"]==1.0 and away["home"]==0.0
