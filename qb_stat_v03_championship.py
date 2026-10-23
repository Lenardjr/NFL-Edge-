"""Retrospective multi-week QB production comparison for v0.3.

Weeks 2-6 are replayed independently. Every forecast receives only rows from
strictly before its target week. This is development/backtesting, not a blind
prospective validation.
"""
from __future__ import annotations
import pandas as pd
from player_engine import normalize_player_schema
from player_stat_engine import project_player_stats
from qb_stat_v03_challenger import project_qb_v03

URL="https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
SEASON=2026; WEEKS=range(2,7); LOOKBACK=8; MIN_GAMES=3
QB_STATS=("attempts","completions","passing_yards","passing_tds","interceptions")

def load_stats():
    return normalize_player_schema(pd.concat([pd.read_csv(URL.format(season=s),low_memory=False) for s in (2025,2026)],ignore_index=True))

def run():
    d=load_stats();id_col=next(c for c in ("player_id","gsis_id","player_display_name","player_name") if c in d.columns);rows=[]
    for week in WEEKS:
        hist=d[(d.season<SEASON)|((d.season==SEASON)&(d.week<week))]
        actual=d[(d.season==SEASON)&(d.week==week)]
        histories={pid:g for pid,g in hist.groupby(id_col,dropna=True)}
        for _,a in actual.iterrows():
            if str(a.get("position","")).upper()!="QB":continue
            h=histories.get(a.get(id_col))
            if h is None or len(h)<MIN_GAMES:continue
            att=pd.to_numeric(h.get("attempts"),errors="coerce").fillna(0).tail(LOOKBACK)
            if att.mean()<5:continue
            b=project_player_stats(h,LOOKBACK);c=project_qb_v03(h,LOOKBACK)
            if not b or not c:continue
            for stat in QB_STATS:
                if stat not in a.index or pd.isna(a.get(stat)):continue
                y=float(a[stat]);bp=float(b["stats"][stat]["projection"]);cp=float(c["stats"][stat]["projection"])
                rows.append({"week":week,"player_id":str(a.get(id_col)),"stat":stat,"actual":y,"baseline":bp,"challenger":cp,"baseline_abs":abs(bp-y),"challenger_abs":abs(cp-y),"baseline_error":bp-y,"challenger_error":cp-y})
    r=pd.DataFrame(rows)
    if r.empty:raise RuntimeError("No QB v0.3 comparison rows")
    summary=r.groupby("stat").agg(n=("actual","size"),baseline_mae=("baseline_abs","mean"),challenger_mae=("challenger_abs","mean"),baseline_bias=("baseline_error","mean"),challenger_bias=("challenger_error","mean")).reset_index()
    summary["mae_improvement"]=summary.baseline_mae-summary.challenger_mae
    weekly=r.groupby(["week","stat"]).agg(n=("actual","size"),baseline_mae=("baseline_abs","mean"),challenger_mae=("challenger_abs","mean")).reset_index();weekly["mae_improvement"]=weekly.baseline_mae-weekly.challenger_mae
    return r,summary,weekly

if __name__=="__main__":
    rows,summary,weekly=run()
    print("NFL EDGE QB v0.3 — RETROSPECTIVE WEEKS 2-6")
    print("Strict cutoff: each target week is excluded from its own features.")
    print(summary.to_string(index=False))
    print("\nBY WEEK")
    print(weekly.to_string(index=False))
    print(f"\nOverall stat markets improved: {(summary.mae_improvement>0).sum()}/{len(summary)}")
