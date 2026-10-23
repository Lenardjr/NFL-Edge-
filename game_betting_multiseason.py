"""NFL Edge multi-season betting validation.

Tests already-defined betting rules across several historical seasons without
retuning them on the evaluation sample. This is evidence gathering, not a
profit guarantee or parameter search.
"""
from __future__ import annotations

import pandas as pd

from game_betting_diagnostics import market_bets
from game_betting_lab import score_bets
from week6_backtest import run_week_retrospective

SEASONS=(2023,2024,2025,2026)
# Rule carried forward from the existing 2026 diagnostic work. Do not tune it
# from the multi-season results below.
SPREAD_PROBABILITY_THRESHOLD=.56
SPREAD_MIN_GAP=0.0


def season_weeks(season:int):
    """Regular-season weeks available for evaluation, capped before future 2026 games."""
    return range(1,7) if season==2026 else range(1,19)


def load_history(seasons=SEASONS, simulations=5000) -> pd.DataFrame:
    rows=[]
    for season in seasons:
        for week in season_weeks(season):
            d=run_week_retrospective(season,week,simulations=simulations,seed=season*100+week)
            if d is None or d.empty:
                continue
            d=d.copy();d["season_eval"]=season;d["week_eval"]=week
            rows.append(d)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def evaluate_fixed_spread(history:pd.DataFrame) -> tuple[dict,pd.DataFrame,pd.DataFrame]:
    if history.empty:
        return score_bets(pd.DataFrame()),pd.DataFrame(),pd.DataFrame()
    bets=market_bets(history,"SPREAD",SPREAD_PROBABILITY_THRESHOLD,SPREAD_MIN_GAP)
    if bets.empty:
        return score_bets(bets),pd.DataFrame(),bets
    # recommendations preserves week but not evaluation season, so attach season
    # by stable game identity.
    keys=history[["season_eval","week_eval","gameday","away_team","home_team"]].drop_duplicates()
    bets=bets.merge(keys,on=["gameday","away_team","home_team"],how="left")
    rows=[]
    for season,g in bets.groupby("season_eval",dropna=False):
        s=score_bets(g)
        rows.append({"season":int(season),**s})
    by_season=pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    return score_bets(bets),by_season,bets


def run_multiseason(simulations=5000):
    history=load_history(simulations=simulations)
    overall,by_season,bets=evaluate_fixed_spread(history)
    summary={
        "seasons":list(SEASONS),
        "rule":{"market":"SPREAD","probability_threshold":SPREAD_PROBABILITY_THRESHOLD,"min_gap":SPREAD_MIN_GAP},
        "validation_mode":"FIXED_RULE_MULTI_SEASON_NO_RETUNING",
        "games_evaluated":int(len(history)),
        "overall":overall,
    }
    return summary,by_season,bets


if __name__=="__main__":
    summary,by_season,bets=run_multiseason()
    print("NFL EDGE MULTI-SEASON FIXED-RULE VALIDATION")
    print(summary)
    print("\nBY SEASON")
    print(by_season.to_string(index=False) if not by_season.empty else "No qualifying wagers")
    if not bets.empty:
        weekly=[]
        for (season,week),g in bets.groupby(["season_eval","week_eval"]):
            weekly.append({"season":int(season),"week":int(week),**score_bets(g)})
        print("\nBY WEEK")
        print(pd.DataFrame(weekly).to_string(index=False))
