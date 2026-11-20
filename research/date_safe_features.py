"""Leakage-resistant game features with conservative calendar-day availability.

All games on a given date receive features from games on *earlier* dates only.
This intentionally excludes same-day information even when it might have been
available before a later kickoff. For intraday use, require completion timestamps.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from game_engine import FEATURES, TEAM_ALIASES


def date_safe_features(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    required = {"gameday", "season", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(g.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    if g["gameday"].isna().any():
        raise ValueError("missing or invalid gameday")
    for c in ("home_score", "away_score"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    for c in ("home_team", "away_team"):
        g[c] = g[c].map(lambda x: TEAM_ALIASES.get(str(x), str(x)))
    g = g.sort_values(["gameday", "home_team", "away_team"], kind="stable").reset_index(drop=True)
    state = {}
    current_season = None
    rows = []

    def team(name):
        return state.setdefault(name, dict(games=0, pf=[], pa=[], margin=[], wins=[], elo=1500.0, last_date=None))

    def avg(values, n, default):
        return float(np.mean(values[-n:])) if values else default

    for day, day_games in g.groupby("gameday", sort=True):
        seasons = day_games["season"].dropna().unique()
        if len(seasons) != 1:
            raise ValueError("multiple seasons on same calendar date")
        season = int(seasons[0])
        if current_season is not None and season != current_season:
            for s in state.values():
                s.update(games=0, pf=[], pa=[], margin=[], wins=[],
                         elo=1500.0 + .67 * (s["elo"] - 1500.0), last_date=None)
        current_season = season

        # Snapshot every matchup BEFORE applying any result from this date.
        for _, game in day_games.iterrows():
            h, a = team(game.home_team), team(game.away_team)
            def rest(s):
                return 7.0 if s["last_date"] is None else float(np.clip((day - s["last_date"]).days, 3, 21))
            hr, ar = rest(h), rest(a)
            r = game.to_dict()
            r.update(home_elo=h["elo"], away_elo=a["elo"], elo_diff=h["elo"]-a["elo"],
                     home_pf5=avg(h["pf"],5,22), away_pf5=avg(a["pf"],5,22),
                     home_pa5=avg(h["pa"],5,22), away_pa5=avg(a["pa"],5,22),
                     home_margin5=avg(h["margin"],5,0), away_margin5=avg(a["margin"],5,0),
                     home_win5=avg(h["wins"],5,.5), away_win5=avg(a["wins"],5,.5),
                     home_games_prior=h["games"], away_games_prior=a["games"],
                     home_rest=hr, away_rest=ar, rest_diff=hr-ar)
            rows.append(r)

        # Only after every pregame row is frozen may the day's results update state.
        # Missing results never update state.
        for _, game in day_games.iterrows():
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            h, a = team(game.home_team), team(game.away_team)
            hs, aw = float(game.home_score), float(game.away_score)
            margin = hs-aw
            expected = 1/(1+10**((a["elo"]-(h["elo"]+55))/400))
            actual = 1 if margin>0 else (.5 if margin==0 else 0)
            delta = 20*math.log(abs(margin)+1)*(actual-expected)
            h["elo"] += delta
            a["elo"] -= delta
            for s,pf,pa,m,w in ((h,hs,aw,margin,actual),(a,aw,hs,-margin,1-actual)):
                s["games"] += 1
                s["pf"].append(pf);s["pa"].append(pa);s["margin"].append(m);s["wins"].append(w)
                s["last_date"] = day

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["target_margin"] = out.home_score - out.away_score
    out["target_total"] = out.home_score + out.away_score
    out["pf_matchup"] = (out.home_pf5+out.away_pa5)-(out.away_pf5+out.home_pa5)
    out["form_diff"] = out.home_margin5-out.away_margin5
    out["winform_diff"] = out.home_win5-out.away_win5
    assert set(FEATURES).issubset(out.columns)
    return out
