"""NFL Edge spread challenger selection with strict temporal separation.

Candidate discovery/selection: 2023-2024 only.
Validation: 2025 only, after the candidate is frozen.
2026: never loaded here; reserved as final holdout.

The challenger may explicitly return NO_VALIDATED_EDGE rather than forcing bets.
"""
from __future__ import annotations

import math
import pandas as pd

from game_betting_lab import score_bets
from game_betting_multiseason import load_history
from spread_signal_research import DEV_SEASONS, VALIDATION_SEASON, enrich, side_bets

MIN_DEV_BETS = 40
MIN_VALIDATION_BETS = 25
BREAK_EVEN = 110 / 210
MIN_DEV_WIN_RATE = 0.54
MIN_DEV_ROI = 0.02
MIN_VALIDATION_WIN_RATE = BREAK_EVEN
MIN_VALIDATION_ROI = 0.0

# Predeclared simple regimes. No 2025 information is used to choose among them.
REGIMES = (
    ("MODEL_WITH_MARKET_FAVORITE", lambda d: ~d.model_takes_dog),
    ("HOME_MARKET_FAVORITE", lambda d: d.market_favorite.eq("HOME")),
    ("SPREAD_3_TO_5", lambda d: d.spread_bucket.astype(str).eq("3-5")),
    ("GAP_3_TO_5", lambda d: d.gap_bucket.astype(str).eq("3-5")),
)


def wilson(wins:int, losses:int, z:float=1.96):
    n=wins+losses
    if not n:
        return (math.nan, math.nan)
    p=wins/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return center-half, center+half


def metrics(bets:pd.DataFrame)->dict:
    s=score_bets(bets)
    decided=s["wins"]+s["losses"]
    lo,hi=wilson(s["wins"],s["losses"])
    ordered=bets.sort_values(["season_eval","week_eval","gameday"]) if not bets.empty else bets
    cumulative=ordered.units.cumsum() if not ordered.empty else pd.Series(dtype=float)
    peak=cumulative.cummax() if not cumulative.empty else cumulative
    drawdown=(cumulative-peak) if not cumulative.empty else cumulative
    weekly=[]
    if not bets.empty:
        for _,g in bets.groupby(["season_eval","week_eval"]):
            weekly.append(float(g.units.sum()))
    return {**s,
        "decided_win_rate":s["wins"]/decided if decided else math.nan,
        "wilson95_low":lo,"wilson95_high":hi,
        "max_drawdown_units":float(drawdown.min()) if not drawdown.empty else 0.0,
        "profitable_weeks":sum(x>0 for x in weekly),
        "losing_weeks":sum(x<0 for x in weekly),
    }


def apply_regime(bets:pd.DataFrame,name:str)->pd.DataFrame:
    for regime,fn in REGIMES:
        if regime==name:
            return bets.loc[fn(bets)].copy()
    raise KeyError(name)


def choose_on_development(dev:pd.DataFrame):
    rows=[]
    for name,_ in REGIMES:
        m=metrics(apply_regime(dev,name))
        eligible=(m["bets"]>=MIN_DEV_BETS and m["decided_win_rate"]>=MIN_DEV_WIN_RATE and m["roi"]>=MIN_DEV_ROI)
        rows.append({"regime":name,**m,"eligible":eligible})
    table=pd.DataFrame(rows).sort_values(["eligible","roi","bets"],ascending=[False,False,False]).reset_index(drop=True)
    eligible=table[table.eligible]
    return (None if eligible.empty else str(eligible.iloc[0].regime)),table


def validate_frozen_candidate(val:pd.DataFrame,candidate:str|None):
    if candidate is None:
        return "NO_VALIDATED_EDGE",None
    m=metrics(apply_regime(val,candidate))
    passed=(m["bets"]>=MIN_VALIDATION_BETS and m["decided_win_rate"]>=MIN_VALIDATION_WIN_RATE and m["roi"]>=MIN_VALIDATION_ROI)
    return ("VALIDATED_CANDIDATE" if passed else "NO_VALIDATED_EDGE"),m


def run_challenger(simulations=5000):
    # Deliberately do not load 2026.
    history=enrich(load_history(seasons=DEV_SEASONS+(VALIDATION_SEASON,),simulations=simulations))
    all_bets=side_bets(history)
    dev=all_bets[all_bets.season_eval.isin(DEV_SEASONS)].copy()
    val=all_bets[all_bets.season_eval.eq(VALIDATION_SEASON)].copy()
    candidate,dev_table=choose_on_development(dev)
    status,val_metrics=validate_frozen_candidate(val,candidate)
    return {
        "method":"2023_2024_SELECT__2025_VALIDATE__2026_UNTOUCHED",
        "candidate":candidate,
        "status":status,
        "development":dev_table,
        "validation":val_metrics,
        "rules":{
            "min_dev_bets":MIN_DEV_BETS,"min_validation_bets":MIN_VALIDATION_BETS,
            "min_dev_win_rate":MIN_DEV_WIN_RATE,"min_dev_roi":MIN_DEV_ROI,
            "min_validation_win_rate":MIN_VALIDATION_WIN_RATE,"min_validation_roi":MIN_VALIDATION_ROI,
        },
    }


if __name__=="__main__":
    r=run_challenger()
    print("NFL EDGE SPREAD CHALLENGER")
    print({k:v for k,v in r.items() if k not in ("development",)})
    print("\nDEVELOPMENT CANDIDATES")
    print(r["development"].to_string(index=False))
    print("\n2026 FINAL HOLDOUT: UNTOUCHED")
