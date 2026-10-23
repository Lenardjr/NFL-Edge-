"""NFL Edge QB production challenger v0.4.

Game-context-aware research layer. Uses only features available before kickoff
and remains separate from frozen production baselines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

TARGETS=("attempts","completions","passing_yards","passing_tds")
FEATURES=("att_mean_3","att_mean_8","comp_rate_8","ypa_8","td_rate_8","team_pf_5","team_pa_5","opp_pf_5","opp_pa_5","home","rest_days","market_team_total","market_game_total","market_margin")


def _mean(s,n,default=np.nan):
    x=pd.to_numeric(s,errors="coerce").dropna().tail(n)
    return float(x.mean()) if len(x) else default


def _rate(num,den,n,default=np.nan):
    a=pd.to_numeric(num,errors="coerce").tail(n);b=pd.to_numeric(den,errors="coerce").tail(n)
    good=a.notna() & b.notna() & (b>0)
    return float(a[good].sum()/b[good].sum()) if good.any() else default


def qb_history_features(history: pd.DataFrame) -> dict:
    h=history.copy();sort=[c for c in ("season","week") if c in h.columns]
    if sort:h=h.sort_values(sort)
    return {"att_mean_3":_mean(h["attempts"],3),"att_mean_8":_mean(h["attempts"],8),"comp_rate_8":_rate(h["completions"],h["attempts"],8,.62),"ypa_8":_rate(h["passing_yards"],h["attempts"],8,6.8),"td_rate_8":_rate(h["passing_tds"],h["attempts"],8,.045)}


def context_features(game_row: pd.Series, team: str) -> dict:
    home=str(game_row.get("home_team"))==str(team);prefix="home" if home else "away";opp="away" if home else "home"
    spread=pd.to_numeric(pd.Series([game_row.get("spread_line")]),errors="coerce").iloc[0];total=pd.to_numeric(pd.Series([game_row.get("total_line")]),errors="coerce").iloc[0]
    team_margin=spread if home else -spread;team_total=(total+team_margin)/2 if pd.notna(total) and pd.notna(team_margin) else np.nan
    return {"team_pf_5":game_row.get(f"{prefix}_pf5",np.nan),"team_pa_5":game_row.get(f"{prefix}_pa5",np.nan),"opp_pf_5":game_row.get(f"{opp}_pf5",np.nan),"opp_pa_5":game_row.get(f"{opp}_pa5",np.nan),"home":1.0 if home else 0.0,"rest_days":game_row.get(f"{prefix}_rest",np.nan),"market_team_total":team_total,"market_game_total":total,"market_margin":team_margin}


def fit_models(training: pd.DataFrame) -> dict:
    models={};X=training.reindex(columns=FEATURES)
    for target in TARGETS:
        y=pd.to_numeric(training[target],errors="coerce");good=y.notna()
        if good.sum()<50:continue
        model=make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingRegressor(max_depth=3,learning_rate=.05,max_iter=150,l2_regularization=2.0,random_state=42))
        model.fit(X.loc[good],y.loc[good]);models[target]=model
    return models


def predict(models: dict, features: dict) -> dict:
    X=pd.DataFrame([{k:features.get(k,np.nan) for k in FEATURES}]);out={}
    for stat,model in models.items():out[stat]=max(0.0,float(model.predict(X)[0]))
    if "attempts" in out and "completions" in out:out["completions"]=min(out["completions"],out["attempts"])
    return out
