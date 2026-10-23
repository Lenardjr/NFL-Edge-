"""NFL Edge historical walk-forward spread validation.

Evaluates the spread engine one week at a time using only information available
before each target week. This is a research diagnostic, not a profitability
claim. Totals are intentionally excluded after failing calibration diagnostics.
"""
from __future__ import annotations

import pandas as pd

from game_betting_diagnostics import market_bets
from game_betting_lab import score_bets
from week6_backtest import run_week_retrospective

SEASON = 2026
WEEKS = (2, 3, 4, 5, 6)
SPREAD_RULE = {"probability_threshold": .56, "min_gap": 0.0}


def run_walkforward(season: int = SEASON, weeks=WEEKS, simulations: int = 10000):
    """Grade a fixed spread rule independently for each historical week."""
    weekly=[]
    all_bets=[]
    for week in weeks:
        results=run_week_retrospective(season, week, simulations=simulations, seed=9200+week).copy()
        results["week"]=week
        bets=market_bets(results, "SPREAD", SPREAD_RULE["probability_threshold"], SPREAD_RULE["min_gap"])
        if not bets.empty:
            all_bets.append(bets)
        s=score_bets(bets)
        weekly.append({"season":season,"week":week,**s})
    weekly_df=pd.DataFrame(weekly)
    combined=pd.concat(all_bets,ignore_index=True) if all_bets else pd.DataFrame()
    overall=score_bets(combined)
    profitable_weeks=int((weekly_df["units"]>0).sum()) if not weekly_df.empty else 0
    losing_weeks=int((weekly_df["units"]<0).sum()) if not weekly_df.empty else 0
    summary={
        "season":season,
        "weeks":list(weeks),
        "market":"SPREAD",
        "rule":SPREAD_RULE,
        "validation_mode":"FIXED_RULE_WEEK_BY_WEEK_RETROSPECTIVE",
        "profitable_weeks":profitable_weeks,
        "losing_weeks":losing_weeks,
        "overall":overall,
    }
    return summary,weekly_df,combined


if __name__ == "__main__":
    summary,weekly,bets=run_walkforward()
    print("NFL EDGE FIXED-RULE SPREAD WALK-FORWARD")
    print(summary)
    print("\nWEEKLY RESULTS")
    print(weekly.to_string(index=False))
    print("\nALL QUALIFYING SPREAD WAGERS")
    print(bets.to_string(index=False) if not bets.empty else "No qualifying wagers")
