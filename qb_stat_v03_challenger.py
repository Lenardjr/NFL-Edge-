"""NFL Edge QB production challenger v0.3.

Experimental only. Targets the largest Week-6 weakness: systematic passing-volume
underprediction. Uses only supplied pregame QB history. The adjustment is a
shrunken correction learned from a QB's own recent volume trend, not target-week data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from player_engine import normalize_player_schema, _weighted_mean
from player_stat_engine import project_player_stats


def _values(d: pd.DataFrame, col: str, n: int) -> np.ndarray:
    if col not in d.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(d[col], errors="coerce").dropna().tail(n).to_numpy(dtype=float)


def _rate(numer: np.ndarray, denom: np.ndarray, fallback: float) -> float:
    n=min(len(numer),len(denom))
    if n==0:return fallback
    numer=numer[-n:];denom=denom[-n:]
    good=denom>0
    if not good.any():return fallback
    return float(np.average(numer[good]/denom[good],weights=np.linspace(1.0,2.0,good.sum())))


def project_qb_v03(history: pd.DataFrame, lookback: int=8) -> dict | None:
    if history is None or history.empty:return None
    d=normalize_player_schema(history)
    sort_cols=[c for c in ("season","week") if c in d.columns]
    if sort_cols:d=d.sort_values(sort_cols)
    recent=d.tail(max(3,lookback))
    base=project_player_stats(recent,lookback=lookback)
    if not base:return None
    att=_values(recent,"attempts",lookback)
    if len(att)<3:return base

    # Blend weighted recent usage with the QB's longer-window mean. A modest
    # trend term lets rising/falling workloads move without hard-coding the
    # inspected Week-6 bias into the forecast.
    weighted=float(_weighted_mean(att))
    long_mean=float(np.mean(att))
    short_mean=float(np.mean(att[-3:]))
    trend=short_mean-long_mean
    att_proj=max(0.0,0.60*weighted+0.40*long_mean+0.25*trend)

    comp=_values(recent,"completions",lookback)
    yards=_values(recent,"passing_yards",lookback)
    tds=_values(recent,"passing_tds",lookback)
    ints=_values(recent,"interceptions",lookback)
    comp_rate=float(np.clip(_rate(comp,att,.62),.35,.85))
    ypa=float(np.clip(_rate(yards,att,6.8),3.0,12.0))
    td_rate=float(np.clip(_rate(tds,att,.045),0,.15))
    int_rate=float(np.clip(_rate(ints,att,.025),0,.12))
    vals={"attempts":att_proj,"completions":att_proj*comp_rate,"passing_yards":att_proj*ypa,"passing_tds":att_proj*td_rate,"interceptions":att_proj*int_rate}
    for stat,val in vals.items():
        old=base["stats"].get(stat,{})
        floor=float(old.get("floor",0));ceil=float(old.get("ceiling",val))
        base["stats"][stat]={**old,"projection":round(float(val),2),"floor":round(min(floor,val),2),"ceiling":round(max(ceil,val),2)}
    base["model_version"]="qb-volume-trend-v0.3"
    return base
