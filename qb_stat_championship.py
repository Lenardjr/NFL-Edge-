"""Strict Week 6 comparison: Player Production baseline vs QB v0.2 challenger."""
from __future__ import annotations
import numpy as np
import pandas as pd
from player_engine import normalize_player_schema
from player_stat_engine import project_player_stats
from player_stat_qb_challenger import project_qb_stats

URL="https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
SEASON=2026; WEEK=6; LOOKBACK=8; MIN_GAMES=3
QB_STATS=("attempts","completions","passing_yards","passing_tds","interceptions")

def load_stats():
    frames=[pd.read_csv(URL.format(season=s),low_memory=False) for s in (2025,2026)]
    return normalize_player_schema(pd.concat(frames,ignore_index=True))

def _metrics(rows):
    d=pd.DataFrame(rows);out=[]
    for stat,g in d.groupby("stat"):
        out.append({"stat":stat,"n":len(g),"baseline_mae":g.baseline_abs.mean(),"challenger_mae":g.challenger_abs.mean(),"mae_improvement":g.baseline_abs.mean()-g.challenger_abs.mean(),"baseline_bias":g.baseline_error.mean(),"challenger_bias":g.challenger_error.mean()})
    return pd.DataFrame(out).sort_values("stat")

def run():
    d=load_stats()
    hist=d[(d.season<SEASON)|((d.season==SEASON)&(d.week<WEEK))]
    actual=d[(d.season==SEASON)&(d.week==WEEK)]
    id_col=next(c for c in ("player_id","gsis_id","player_display_name","player_name") if c in d.columns)
    histories={pid:g for pid,g in hist.groupby(id_col,dropna=True)};rows=[]
    for _,a in actual.iterrows():
        if str(a.get("position","")).upper()!="QB":continue
        h=histories.get(a.get(id_col))
        if h is None or len(h)<MIN_GAMES:continue
        # Require established pregame passing volume; target-week usage is never consulted.
        att=pd.to_numeric(h.get("attempts"),errors="coerce").fillna(0).tail(LOOKBACK)
        if att.mean()<5:continue
        b=project_player_stats(h,LOOKBACK);c=project_qb_stats(h,LOOKBACK)
        if not b or not c:continue
        for stat in QB_STATS:
            if stat not in a.index or pd.isna(a.get(stat)):continue
            y=float(a[stat]);bp=float(b["stats"][stat]["projection"]);cp=float(c["stats"][stat]["projection"])
            rows.append({"player_id":str(a.get(id_col)),"stat":stat,"actual":y,"baseline":bp,"challenger":cp,"baseline_error":bp-y,"challenger_error":cp-y,"baseline_abs":abs(bp-y),"challenger_abs":abs(cp-y)})
    if not rows:raise RuntimeError("No QB championship rows")
    return pd.DataFrame(rows),_metrics(rows)

if __name__=="__main__":
    rows,summary=run()
    print("NFL EDGE QB PRODUCTION CHALLENGER v0.2 — STRICT 2026 WEEK 6")
    print("Target-week results excluded from every projection feature.")
    print(summary.to_string(index=False))
    improved=int((summary.mae_improvement>0).sum())
    print(f"Markets improved: {improved}/{len(summary)}")
