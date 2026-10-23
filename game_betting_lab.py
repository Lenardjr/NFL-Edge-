"""NFL Edge game betting laboratory.

Turns strict pre-week game forecasts into auditable ATS/total wagers. Thresholds
must be selected on earlier weeks; the target week is grading only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from week6_backtest import run_week_retrospective

PRICE = -110
WIN_UNITS = 100.0 / abs(PRICE)


def _grade(value: float, line: float, side: str) -> str:
    if side == "HOME": edge = value - line
    elif side == "AWAY": edge = line - value
    elif side == "OVER": edge = value - line
    elif side == "UNDER": edge = line - value
    else: raise ValueError(side)
    return "WIN" if edge > 0 else "LOSS" if edge < 0 else "PUSH"


def _units(result: str) -> float:
    return WIN_UNITS if result == "WIN" else -1.0 if result == "LOSS" else 0.0


def recommendations(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Create bets only when simulated win probability clears threshold."""
    rows=[]
    for _, r in results.iterrows():
        hp=float(r["home_cover_probability"]); op=float(r["over_probability"])
        if np.isfinite(hp) and np.isfinite(float(r["market_spread"])):
            p=max(hp,1-hp)
            if p >= threshold:
                side="HOME" if hp >= .5 else "AWAY"
                # nflverse spread_line is expected HOME margin, not home handicap.
                result=_grade(float(r["actual_margin"]),float(r["market_spread"]),side)
                rows.append({"week":int(r.get("week",0)),"gameday":r["gameday"],"away_team":r["away_team"],"home_team":r["home_team"],"market":"SPREAD","side":side,"line":float(r["market_spread"]),"model_probability":p,"result":result,"units":_units(result)})
        if np.isfinite(op) and np.isfinite(float(r["market_total"])):
            p=max(op,1-op)
            if p >= threshold:
                side="OVER" if op >= .5 else "UNDER"
                result=_grade(float(r["actual_total"]),float(r["market_total"]),side)
                rows.append({"week":int(r.get("week",0)),"gameday":r["gameday"],"away_team":r["away_team"],"home_team":r["home_team"],"market":"TOTAL","side":side,"line":float(r["market_total"]),"model_probability":p,"result":result,"units":_units(result)})
    return pd.DataFrame(rows)


def score_bets(bets: pd.DataFrame) -> dict:
    if bets is None or bets.empty:return {"bets":0,"wins":0,"losses":0,"pushes":0,"win_rate":None,"units":0.0,"roi":None}
    decided=bets[bets.result != "PUSH"]
    units=float(bets.units.sum())
    return {"bets":int(len(bets)),"wins":int((bets.result=="WIN").sum()),"losses":int((bets.result=="LOSS").sum()),"pushes":int((bets.result=="PUSH").sum()),"win_rate":float((decided.result=="WIN").mean()) if len(decided) else None,"units":units,"roi":units/len(bets)}


def tune_threshold(season: int=2026, train_weeks=(2,3,4,5), thresholds=(.52,.54,.55,.57,.60), simulations: int=10000) -> tuple[float,pd.DataFrame]:
    """Choose threshold using training weeks only. Never sees target Week 6."""
    rows=[]
    for threshold in thresholds:
        all_bets=[]
        for week in train_weeks:
            d=run_week_retrospective(season,week,simulations=simulations,seed=4200+week)
            if d.empty:continue
            d=d.copy();d["week"]=week
            b=recommendations(d,threshold)
            if not b.empty:all_bets.append(b)
        bets=pd.concat(all_bets,ignore_index=True) if all_bets else pd.DataFrame()
        s=score_bets(bets);rows.append({"threshold":threshold,**s})
    table=pd.DataFrame(rows)
    eligible=table[table.bets >= 20].copy()
    pool=eligible if not eligible.empty else table[table.bets > 0]
    if pool.empty:return .55,table
    # ROI is primary, then larger sample, then stricter threshold.
    best=pool.sort_values(["roi","bets","threshold"],ascending=[False,False,False]).iloc[0]
    return float(best.threshold),table


def run_week6_lab(simulations: int=20000) -> tuple[dict,pd.DataFrame,pd.DataFrame]:
    threshold,tuning=tune_threshold(simulations=max(5000,simulations//2))
    w6=run_week_retrospective(2026,6,simulations=simulations,seed=6006).copy()
    w6["week"]=6
    bets=recommendations(w6,threshold)
    summary={"season":2026,"target_week":6,"threshold":threshold,"threshold_selected_on_weeks":[2,3,4,5],"odds_assumption":"-110 flat price; nflverse closing spread/total benchmark","validation_mode":"RETROSPECTIVE_WEEK6_WITH_THRESHOLD_FROZEN_PREWEEK6","week6":score_bets(bets)}
    return summary,tuning,bets


if __name__ == "__main__":
    summary,tuning,bets=run_week6_lab()
    print("THRESHOLD TUNING — WEEKS 2-5 ONLY")
    print(tuning.to_string(index=False))
    print("\nWEEK 6 FROZEN-THRESHOLD REPORT")
    print(summary)
    print("\nWAGERS")
    print(bets.to_string(index=False) if not bets.empty else "No qualifying wagers")
