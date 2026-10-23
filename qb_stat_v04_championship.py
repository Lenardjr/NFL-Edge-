"""NFL Edge QB v0.4 game-context retrospective championship.

Builds one row per established QB start by joining strictly pregame QB history
with Game Engine pregame team/opponent/market context. For each 2026 target week,
v0.4 is fit only on earlier games and then compared with the frozen weighted-mean
player-stat baseline. Week 6 remains retrospective development data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from game_engine import load_games, pregame_features, TEAM_ALIASES
from player_engine import normalize_player_schema
from player_stat_engine import project_player_stats
from qb_stat_v04_context import qb_history_features, context_features, fit_models, predict, TARGETS

STATS_URL="https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
GAMES_URL="https://github.com/nflverse/nfldata/raw/master/data/games.csv"
SEASON=2026; WEEKS=range(2,7); LOOKBACK=8; MIN_GAMES=3


def load_stats():
    d=normalize_player_schema(pd.concat([pd.read_csv(STATS_URL.format(season=s),low_memory=False) for s in range(2022,2027)],ignore_index=True))
    if "recent_team" in d.columns:d["recent_team"]=d["recent_team"].map(lambda x: TEAM_ALIASES.get(str(x),str(x)))
    return d


def build_rows(stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    gf=pregame_features(games);game_map={}
    for _,g in gf.iterrows():
        try:key=(int(g.season),int(g.week))
        except (TypeError,ValueError):continue
        game_map.setdefault(key,[]).append(g)
    id_col=next(c for c in ("player_id","gsis_id","player_display_name","player_name") if c in stats.columns)
    qbs=stats[stats.get("position","").astype(str).str.upper()=="QB"].copy();rows=[]
    for pid,p in qbs.groupby(id_col,dropna=True):
        p=p.sort_values(["season","week"])
        for _,a in p.iterrows():
            try:season=int(a.season);week=int(a.week)
            except (TypeError,ValueError):continue
            prior=p[(p.season<season)|((p.season==season)&(p.week<week))]
            if len(prior)<MIN_GAMES or pd.to_numeric(prior.get("attempts"),errors="coerce").fillna(0).tail(LOOKBACK).mean()<5:continue
            team=str(a.get("recent_team",""));matches=[g for g in game_map.get((season,week),[]) if team in {str(g.home_team),str(g.away_team)}]
            if not matches:continue
            f={**qb_history_features(prior.tail(LOOKBACK)),**context_features(matches[0],team)}
            row={"season":season,"week":week,"player_id":str(pid),"team":team,**f}
            for t in TARGETS:row[t]=pd.to_numeric(pd.Series([a.get(t)]),errors="coerce").iloc[0]
            rows.append(row)
    return pd.DataFrame(rows)


def run():
    stats=load_stats();games=load_games(GAMES_URL);table=build_rows(stats,games);out=[]
    id_col=next(c for c in ("player_id","gsis_id","player_display_name","player_name") if c in stats.columns)
    for week in WEEKS:
        train=table[(table.season<SEASON)|((table.season==SEASON)&(table.week<week))]
        test=table[(table.season==SEASON)&(table.week==week)]
        models=fit_models(train)
        for _,r in test.iterrows():
            pred=predict(models,r.to_dict());hist=stats[(stats[id_col].astype(str)==str(r.player_id)) & ((stats.season<SEASON)|((stats.season==SEASON)&(stats.week<week)))]
            base=project_player_stats(hist,LOOKBACK)
            if not base:continue
            for stat in TARGETS:
                if stat not in pred or pd.isna(r.get(stat)):continue
                y=float(r[stat]);b=float(base["stats"][stat]["projection"]);c=float(pred[stat])
                out.append({"week":week,"stat":stat,"actual":y,"baseline":b,"v04":c,"baseline_abs":abs(b-y),"v04_abs":abs(c-y),"baseline_error":b-y,"v04_error":c-y})
    d=pd.DataFrame(out)
    if d.empty:raise RuntimeError("No v0.4 rows")
    s=d.groupby("stat").agg(n=("actual","size"),baseline_mae=("baseline_abs","mean"),v04_mae=("v04_abs","mean"),baseline_bias=("baseline_error","mean"),v04_bias=("v04_error","mean")).reset_index();s["mae_improvement"]=s.baseline_mae-s.v04_mae
    w=d.groupby(["week","stat"]).agg(n=("actual","size"),baseline_mae=("baseline_abs","mean"),v04_mae=("v04_abs","mean")).reset_index();w["mae_improvement"]=w.baseline_mae-w.v04_mae
    return d,s,w

if __name__=="__main__":
    rows,summary,weekly=run()
    print("NFL EDGE QB v0.4 GAME-CONTEXT CHALLENGER — RETROSPECTIVE WEEKS 2-6")
    print("Each target is predicted from strictly earlier QB/game rows.")
    print(summary.to_string(index=False));print("\nBY WEEK");print(weekly.to_string(index=False))
    print(f"\nStat markets improved: {(summary.mae_improvement>0).sum()}/{len(summary)}")
