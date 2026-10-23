"""NFL Edge spread signal research framework.

Development: 2023-2024 only.
Validation: 2025 only.
Final holdout: 2026 (never used for candidate selection here).

This module diagnoses where the existing game engine agrees/disagrees with the
market. It is intentionally descriptive first: candidate rules are selected on
development data, then evaluated unchanged on validation data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from game_betting_lab import score_bets
from game_betting_multiseason import load_history

DEV_SEASONS=(2023,2024)
VALIDATION_SEASON=2025
HOLDOUT_SEASON=2026
BREAK_EVEN_WIN_RATE=110/210
WIN_UNITS=100/110


def enrich(history:pd.DataFrame)->pd.DataFrame:
    d=history.copy()
    d["model_market_gap"]=(d["projected_margin"]-d["market_spread"]).abs()
    d["model_side"]=np.where(d["projected_margin"]>=d["market_spread"],"HOME","AWAY")
    d["market_favorite"]=np.where(d["market_spread"]>0,"HOME",np.where(d["market_spread"]<0,"AWAY","PICK"))
    d["model_takes_dog"]=((d.model_side!=d.market_favorite)&(d.market_favorite!="PICK"))
    d["spread_bucket"]=pd.cut(d.market_spread.abs(),[-.01,2.5,5,7.5,100],labels=["0-2.5","3-5","5.5-7.5","8+"])
    d["gap_bucket"]=pd.cut(d.model_market_gap,[-.01,1,2.5,5,100],labels=["0-1","1.5-2.5","3-5","5+"])
    return d


def side_bets(d:pd.DataFrame)->pd.DataFrame:
    x=d.copy()
    x["market"]="SPREAD"
    x["pick"]=x.model_side
    x["probability"]=np.where(x.model_side=="HOME",x.home_cover_probability,1-x.home_cover_probability)
    x["gap"]=x.model_market_gap
    x["result"]=np.where(x.actual_margin==x.market_spread,"PUSH",np.where((x.model_side=="HOME")==(x.actual_margin>x.market_spread),"WIN","LOSS"))
    x["units"]=np.where(x.result=="WIN",WIN_UNITS,np.where(x.result=="LOSS",-1.0,0.0))
    return x


def segment_table(bets:pd.DataFrame,column:str,min_bets:int=20)->pd.DataFrame:
    rows=[]
    for value,g in bets.groupby(column,dropna=False,observed=True):
        s=score_bets(g)
        decided=s["wins"]+s["losses"]
        rows.append({"segment":str(value),"bets":s["bets"],"wins":s["wins"],"losses":s["losses"],"pushes":s["pushes"],"decided_win_rate":s["wins"]/decided if decided else np.nan,"units":s["units"],"roi":s["roi"],"eligible":s["bets"]>=min_bets})
    return pd.DataFrame(rows).sort_values(["eligible","roi","bets"],ascending=[False,False,False]).reset_index(drop=True)


def research(simulations=5000):
    history=enrich(load_history(seasons=DEV_SEASONS+(VALIDATION_SEASON,),simulations=simulations))
    dev=side_bets(history[history.season_eval.isin(DEV_SEASONS)])
    val=side_bets(history[history.season_eval==VALIDATION_SEASON])
    dimensions=("model_side","market_favorite","model_takes_dog","spread_bucket","gap_bucket")
    diagnostics={k:segment_table(dev,k) for k in dimensions}
    return {
        "method":"2023_2024_DEVELOPMENT__2025_VALIDATION__2026_UNTOUCHED_HOLDOUT",
        "break_even_win_rate":BREAK_EVEN_WIN_RATE,
        "development_games":int(len(dev)),
        "validation_games":int(len(val)),
        "diagnostics":diagnostics,
    },dev,val


if __name__=="__main__":
    summary,dev,val=research()
    print("NFL EDGE SPREAD SIGNAL RESEARCH")
    print({k:v for k,v in summary.items() if k!="diagnostics"})
    for name,table in summary["diagnostics"].items():
        print(f"\nDEVELOPMENT SEGMENT: {name}")
        print(table.to_string(index=False))
    print("\n2025 is loaded as validation but is intentionally not used above to rank/select development segments.")
    print("2026 remains untouched by this research module.")
