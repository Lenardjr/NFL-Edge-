"""NFL Edge betting diagnostics.

Diagnoses why raw simulated probabilities fail to translate into profitable
spread/total bets. All selection rules are learned on Weeks 2-5 and Week 6 is
reported only after those rules are frozen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from game_betting_lab import recommendations, score_bets
from week6_backtest import run_week_retrospective

TRAIN_WEEKS=(2,3,4,5)
TARGET_WEEK=6


def add_edges(results: pd.DataFrame) -> pd.DataFrame:
    """Attach confidence and model-vs-market disagreement without outcomes."""
    d=results.copy()
    d["spread_confidence"]=(d["home_cover_probability"]-.5).abs()+.5
    d["total_confidence"]=(d["over_probability"]-.5).abs()+.5
    d["spread_market_gap"]=(d["projected_margin"]-d["market_spread"]).abs()
    d["total_market_gap"]=(d["projected_total"]-d["market_total"]).abs()
    return d


def market_bets(results: pd.DataFrame, market: str, probability_threshold: float=.5, min_gap: float=0.0) -> pd.DataFrame:
    d=add_edges(results)
    market=market.upper()
    if market=="SPREAD":
        d=d[d.spread_market_gap >= min_gap]
    elif market=="TOTAL":
        d=d[d.total_market_gap >= min_gap]
    else:
        raise ValueError(market)
    bets=recommendations(d,probability_threshold)
    return bets[bets.market==market].reset_index(drop=True) if not bets.empty else bets


def calibration_table(bets: pd.DataFrame, bins=(.50,.55,.60,.65,.70,1.01)) -> pd.DataFrame:
    if bets is None or bets.empty:return pd.DataFrame(columns=["band","bets","avg_probability","actual_win_rate","calibration_error","units","roi"])
    x=bets[bets.result!="PUSH"].copy()
    x["won"]=(x.result=="WIN").astype(float)
    x["band"]=pd.cut(x.model_probability,bins=bins,right=False,include_lowest=True)
    rows=[]
    for band,g in x.groupby("band",observed=True):
        if g.empty:continue
        avg=float(g.model_probability.mean());actual=float(g.won.mean())
        s=score_bets(g)
        rows.append({"band":str(band),"bets":len(g),"avg_probability":avg,"actual_win_rate":actual,"calibration_error":actual-avg,"units":s["units"],"roi":s["roi"]})
    return pd.DataFrame(rows)


def load_weeks(weeks, simulations=10000) -> pd.DataFrame:
    rows=[]
    for week in weeks:
        d=run_week_retrospective(2026,week,simulations=simulations,seed=8100+week).copy()
        d["week"]=week;rows.append(d)
    return pd.concat(rows,ignore_index=True)


def tune_market(train: pd.DataFrame, market: str, probability_thresholds=(.50,.52,.54,.56,.58,.60), gaps=(0,1,2,3,4,5)) -> tuple[dict,pd.DataFrame]:
    rows=[]
    for p in probability_thresholds:
        for gap in gaps:
            bets=market_bets(train,market,p,gap);s=score_bets(bets)
            rows.append({"market":market,"probability_threshold":p,"min_gap":gap,**s})
    table=pd.DataFrame(rows)
    # Avoid selecting a rule from a tiny lucky sample. Require 20 training bets.
    eligible=table[table.bets>=20]
    if eligible.empty:eligible=table[table.bets>0]
    best=eligible.sort_values(["roi","bets","probability_threshold","min_gap"],ascending=[False,False,False,False]).iloc[0]
    rule={"market":market,"probability_threshold":float(best.probability_threshold),"min_gap":float(best.min_gap),"training_bets":int(best.bets),"training_roi":float(best.roi)}
    return rule,table


def run_diagnostics(simulations=10000):
    train=load_weeks(TRAIN_WEEKS,simulations)
    target=load_weeks((TARGET_WEEK,),simulations)
    rules={};search=[];target_bets=[];cal=[]
    for market in ("SPREAD","TOTAL"):
        rule,grid=tune_market(train,market);rules[market]=rule;search.append(grid)
        b=market_bets(target,market,rule["probability_threshold"],rule["min_gap"])
        target_bets.append(b)
        c=calibration_table(market_bets(train,market,.50,0));c["market"]=market;cal.append(c)
    bets=pd.concat(target_bets,ignore_index=True) if target_bets else pd.DataFrame()
    summary={"season":2026,"training_weeks":list(TRAIN_WEEKS),"target_week":TARGET_WEEK,"validation_mode":"WEEK6_FROZEN_AFTER_WEEKS2_5_RULE_SELECTION","rules":rules,"week6":score_bets(bets)}
    return summary,pd.concat(search,ignore_index=True),pd.concat(cal,ignore_index=True),bets


if __name__=="__main__":
    summary,grid,cal,bets=run_diagnostics()
    print("NFL EDGE BETTING DIAGNOSTICS")
    print(summary)
    print("\nTRAINING CALIBRATION")
    print(cal.to_string(index=False))
    print("\nTOP TRAINING RULES BY MARKET")
    print(grid.sort_values(["market","roi"],ascending=[True,False]).groupby("market").head(8).to_string(index=False))
    print("\nWEEK 6 FROZEN-RULE BETS")
    print(bets.to_string(index=False) if not bets.empty else "No qualifying wagers")
