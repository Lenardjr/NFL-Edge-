import pandas as pd

from spread_challenger import choose_on_development, validate_frozen_candidate, wilson


def bets(n=50,wins=30):
    rows=[]
    for i in range(n):
        result="WIN" if i<wins else "LOSS"
        rows.append({
            "season_eval":2024,"week_eval":1+(i%18),"gameday":f"2024-10-{1+(i%20):02d}",
            "model_takes_dog":False,"market_favorite":"HOME","spread_bucket":"3-5","gap_bucket":"3-5",
            "result":result,"units":100/110 if result=="WIN" else -1.0,
        })
    return pd.DataFrame(rows)


def test_wilson_is_bounded():
    lo,hi=wilson(30,20)
    assert 0 < lo < .6 < hi < 1


def test_development_selection_uses_predeclared_eligible_regime():
    candidate,table=choose_on_development(bets())
    assert candidate is not None
    assert bool(table.iloc[0].eligible)


def test_validation_can_return_no_edge():
    val=bets(n=30,wins=12)
    status,m=validate_frozen_candidate(val,"MODEL_WITH_MARKET_FAVORITE")
    assert status=="NO_VALIDATED_EDGE"
    assert m["bets"]==30
