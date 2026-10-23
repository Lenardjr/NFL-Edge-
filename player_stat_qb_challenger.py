"""NFL Edge QB production challenger v0.2.

Experimental, leakage-safe QB layer. Uses only the supplied pregame history and
models passing volume first, then completion rate / yards per attempt / TD and
INT rates. It does not replace the frozen player-production baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from player_engine import normalize_player_schema, _weighted_mean
from player_stat_engine import project_player_stats


def _series(history: pd.DataFrame, column: str, lookback: int) -> np.ndarray:
    if column not in history.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(history[column], errors="coerce").dropna().tail(max(3, lookback)).to_numpy(dtype=float)


def _weighted_rate(numer: np.ndarray, denom: np.ndarray, fallback: float=0.0) -> float:
    n=min(len(numer),len(denom))
    if n == 0:return fallback
    numer=numer[-n:];denom=denom[-n:]
    rates=np.divide(numer,denom,out=np.full(n,fallback,dtype=float),where=denom>0)
    return float(_weighted_mean(rates))


def project_qb_stats(history: pd.DataFrame, lookback: int=8) -> dict | None:
    """Project QB stats from pregame volume and efficiency components."""
    if history is None or history.empty:return None
    d=normalize_player_schema(history)
    if "position" in d.columns and not (d["position"].astype(str).str.upper()=="QB").any():return None
    sort_cols=[c for c in ("season","week") if c in d.columns]
    if sort_cols:d=d.sort_values(sort_cols)
    recent=d.tail(max(3,lookback))
    base=project_player_stats(recent,lookback=lookback)
    if not base:return None

    att=_series(recent,"attempts",lookback)
    comp=_series(recent,"completions",lookback)
    py=_series(recent,"passing_yards",lookback)
    ptd=_series(recent,"passing_tds",lookback)
    ints=_series(recent,"interceptions",lookback)
    if len(att)<3:return base

    att_proj=max(0.0,float(_weighted_mean(att)))
    comp_rate=float(np.clip(_weighted_rate(comp,att,0.62),0.35,0.85))
    ypa=float(np.clip(_weighted_rate(py,att,6.8),3.0,12.0))
    td_rate=float(np.clip(_weighted_rate(ptd,att,0.045),0.0,0.15))
    int_rate=float(np.clip(_weighted_rate(ints,att,0.025),0.0,0.12))

    projections={
        "attempts":att_proj,
        "completions":att_proj*comp_rate,
        "passing_yards":att_proj*ypa,
        "passing_tds":att_proj*td_rate,
        "interceptions":att_proj*int_rate,
    }
    for stat,value in projections.items():
        old=base["stats"].get(stat,{})
        floor=float(old.get("floor",0.0));ceiling=float(old.get("ceiling",value))
        base["stats"][stat]={**old,"projection":round(float(value),2),"floor":round(min(floor,value),2),"ceiling":round(max(ceiling,value),2)}
    base["model_version"]="qb-volume-efficiency-v0.2"
    return base
