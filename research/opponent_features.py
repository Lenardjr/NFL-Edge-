"""Leakage-safe opponent-strength features for a next-generation NFL model.

Each game on a calendar date uses only games from earlier dates. Opponent
strength is calculated from *pre-game* rolling points-for/against information;
it never uses same-day or future results.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from research.date_safe_features import date_safe_features
from game_engine import TEAM_ALIASES

V2_FEATURES=[
    "elo_diff","pf_matchup","form_diff","winform_diff",
    "home_pf5","away_pf5","home_pa5","away_pa5",
    "home_games_prior","away_games_prior","rest_diff","home_rest","away_rest",
    "home_offense_adj","away_offense_adj","home_defense_adj","away_defense_adj",
    "adj_net_diff",
]

def opponent_adjusted_features(games:pd.DataFrame)->pd.DataFrame:
    base=date_safe_features(games)
    if base.empty:return base
    # The adjustment is the team's previous game points scored minus the
    # opponent's pregame defensive baseline (and vice versa). Values are
    # appended to state only AFTER all same-day features have been computed.
    base=base.sort_values(["gameday","home_team","away_team"],kind="stable").copy()
    histories={}
    rows=[]
    season_last=None
    def history(team):
        return histories.setdefault(team,{"off":[],"def":[]})
    for date,group in base.groupby("gameday",sort=True):
        seasons=group.season.unique()
        if len(seasons)!=1:raise ValueError("mixed seasons on one date")
        season=seasons[0]
        if season_last is not None and season!=season_last:
            histories.clear()
        season_last=season
        updates=[]
        for idx,g in group.iterrows():
            h=history(g.home_team);a=history(g.away_team)
            def avg(x):return float(np.mean(x[-5:])) if x else 0.0
            ho,ao=avg(h["off"]),avg(a["off"])
            hd,ad=avg(h["def"]),avg(a["def"])
            rows.append((idx,ho,ao,hd,ad,(ho-hd)-(ao-ad)))
            if pd.notna(g.home_score) and pd.notna(g.away_score):
                # Opponent baseline is frozen before kickoff; defensive
                # adjustment is points prevented vs opponent's pregame offense.
                updates.append((g.home_team,g.away_team,
                    float(g.home_score)-float(g.away_pa5),
                    float(g.away_score)-float(g.home_pa5),
                    float(g.away_pf5)-float(g.home_score),
                    float(g.home_pf5)-float(g.away_score)))
        for ht,at,ho,ao,hd,ad in updates:
            history(ht)["off"].append(ho);history(at)["off"].append(ao)
            history(ht)["def"].append(hd);history(at)["def"].append(ad)
    extra=pd.DataFrame(rows,columns=["index","home_offense_adj","away_offense_adj",
       "home_defense_adj","away_defense_adj","adj_net_diff"]).set_index("index")
    return base.join(extra)
