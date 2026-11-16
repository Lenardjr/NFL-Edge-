"""Attach strictly historical play-by-play EPA features to scheduled games.

The caller supplies completed-game team aggregates from v3_pbp_features.
Same-day games NEVER inform each other, regardless of row order.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from game_engine import TEAM_ALIASES

def attach_prior_epa(games:pd.DataFrame,team_games:pd.DataFrame,window:int=5)->pd.DataFrame:
    required_games={"game_id","gameday","season","home_team","away_team"}
    required_epa={"game_id","team","offense_epa_per_play","offense_success_rate"}
    if required_games-set(games):raise ValueError(f"Missing game columns: {sorted(required_games-set(games))}")
    if required_epa-set(team_games):raise ValueError(f"Missing EPA columns: {sorted(required_epa-set(team_games))}")
    if window<1:raise ValueError("window must be positive")
    g=games.copy()
    if g.game_id.isna().any() or g.game_id.duplicated().any():raise ValueError("Missing/duplicate schedule game IDs")
    g["gameday"]=pd.to_datetime(g.gameday,errors="coerce")
    if g.gameday.isna().any():raise ValueError("Invalid game dates")
    g["home_team"]=g.home_team.map(lambda x:TEAM_ALIASES.get(str(x),str(x)))
    g["away_team"]=g.away_team.map(lambda x:TEAM_ALIASES.get(str(x),str(x)))
    t=team_games.copy()
    t["team"]=t.team.map(lambda x:TEAM_ALIASES.get(str(x),str(x)))
    if t.duplicated(["game_id","team"]).any():raise ValueError("Duplicate EPA team-game rows")
    if not set(t.game_id).issubset(set(g.game_id)):raise ValueError("EPA game IDs absent from schedule")
    for col in ("offense_epa_per_play","offense_success_rate"):
        t[col]=pd.to_numeric(t[col],errors="coerce")
    if not np.isfinite(t[["offense_epa_per_play","offense_success_rate"]].to_numpy()).all():
        raise ValueError("Invalid EPA metrics")
    if ((t.offense_success_rate<0)|(t.offense_success_rate>1)).any():
        raise ValueError("Invalid success rates")
    g=g.sort_values(["gameday","game_id"],kind="stable").reset_index(drop=True)
    by_game={gid:part for gid,part in t.groupby("game_id")}
    history=defaultdict(list)
    rows=[]
    current_season=None
    for day,day_games in g.groupby("gameday",sort=True):
        seasons=day_games.season.unique()
        if len(seasons)!=1:raise ValueError("Multiple seasons on same date")
        season=int(seasons[0])
        if current_season!=season:
            history.clear()
            current_season=season
        # Snapshot every game on this date before updating any history.
        for _,game in day_games.iterrows():
            r=game.to_dict()
            for side in ("home","away"):
                prior=history[game[f"{side}_team"]][-window:]
                r[f"{side}_epa_prior{window}"]=float(np.mean([x[0] for x in prior])) if prior else 0.0
                r[f"{side}_success_prior{window}"]=float(np.mean([x[1] for x in prior])) if prior else .45
                r[f"{side}_epa_sample_games"]=len(prior)
            r[f"epa_diff_prior{window}"]=r[f"home_epa_prior{window}"]-r[f"away_epa_prior{window}"]
            r[f"success_diff_prior{window}"]=r[f"home_success_prior{window}"]-r[f"away_success_prior{window}"]
            rows.append(r)
        for _,game in day_games.iterrows():
            for _,record in by_game.get(game.game_id,pd.DataFrame()).iterrows():
                if record.team not in (game.home_team,game.away_team):
                    raise ValueError("EPA team not participating in matched schedule game")
                history[record.team].append((float(record.offense_epa_per_play),
                                             float(record.offense_success_rate)))
    return pd.DataFrame(rows)
