"""Leakage-safe backtest for NFL Edge Player Production Challenger.

Only grades a statistic when the player's PRE-GAME history makes that market
plausibly prop-eligible. This prevents inactive/irrelevant zero rows from making
accuracy look artificially strong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from player_engine import normalize_player_schema
from player_stat_engine import STAT_COLUMNS, project_player_stats

PASS_STATS={"completions","attempts","passing_yards","passing_tds","interceptions"}
RUSH_STATS={"carries","rushing_yards","rushing_tds"}
REC_STATS={"targets","receptions","receiving_yards","receiving_tds"}


def _recent_mean(history: pd.DataFrame, column: str, lookback: int) -> float:
    if column not in history.columns:return 0.0
    x=pd.to_numeric(history[column],errors="coerce").fillna(0.0).tail(max(3,lookback))
    return float(x.mean()) if len(x) else 0.0


def stat_is_pregame_eligible(history: pd.DataFrame, position: str | None, stat: str, lookback: int=8) -> bool:
    """Eligibility uses history only; never target-week production."""
    pos=str(position or "").upper()
    if stat in PASS_STATS:
        return pos=="QB" and _recent_mean(history,"attempts",lookback)>=5.0
    if stat in RUSH_STATS:
        return pos in {"QB","RB","WR","TE"} and _recent_mean(history,"carries",lookback)>=2.0
    if stat in REC_STATS:
        return pos in {"RB","WR","TE"} and _recent_mean(history,"targets",lookback)>=2.0
    return False


def backtest_player_stats(stats: pd.DataFrame, season: int, start_week: int, end_week: int, lookback: int=8, min_games: int=3) -> pd.DataFrame:
    d=normalize_player_schema(stats)
    id_col=next((c for c in ("player_id","gsis_id","player_display_name","player_name") if c in d.columns),None)
    if id_col is None:return pd.DataFrame()
    rows=[]
    for week in range(int(start_week),int(end_week)+1):
        history=d[(d["season"]<season)|((d["season"]==season)&(d["week"]<week))]
        actuals=d[(d["season"]==season)&(d["week"]==week)]
        histories={pid:g for pid,g in history.groupby(id_col,dropna=True)}
        for _,actual in actuals.iterrows():
            pid=actual.get(id_col);h=histories.get(pid)
            if h is None or len(h)<min_games:continue
            p=project_player_stats(h,lookback=lookback)
            if not p:continue
            base={"season":season,"week":week,"player_id":str(pid),"player_name":p["player_name"],"team":p["team"],"position":p["position"]}
            for stat in STAT_COLUMNS:
                if stat not in actual.index or pd.isna(actual.get(stat)):continue
                if not stat_is_pregame_eligible(h,p["position"],stat,lookback=lookback):continue
                projected=float(p["stats"][stat]["projection"]);observed=float(actual.get(stat))
                rows.append({**base,"stat":stat,"projected":projected,"actual":observed,"error":projected-observed,"abs_error":abs(projected-observed),"floor":p["stats"][stat]["floor"],"ceiling":p["stats"][stat]["ceiling"],"covered":p["stats"][stat]["floor"]<=observed<=p["stats"][stat]["ceiling"]})
    return pd.DataFrame(rows)


def summarize_player_stats(results: pd.DataFrame) -> pd.DataFrame:
    if results is None or results.empty:return pd.DataFrame(columns=["stat","n","mae","rmse","bias","correlation","coverage"])
    rows=[]
    for stat,g in results.groupby("stat"):
        corr=g["projected"].corr(g["actual"]) if len(g)>1 else np.nan
        rows.append({"stat":stat,"n":int(len(g)),"mae":float(g["abs_error"].mean()),"rmse":float(np.sqrt(np.mean(g["error"]**2))),"bias":float(g["error"].mean()),"correlation":float(corr) if pd.notna(corr) else np.nan,"coverage":float(g["covered"].mean())})
    return pd.DataFrame(rows).sort_values("stat").reset_index(drop=True)
