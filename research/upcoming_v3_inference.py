"""Pregame V3 inference using strictly earlier calendar-day completed games.

Research only. Requires a caller-supplied frozen schedule and EPA team-game
snapshot. Does not automatically label any quote executable or place wagers.
"""
from __future__ import annotations
import hashlib,json
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from game_engine import FEATURES
from research.date_safe_features import date_safe_features
from research.v3_date_safe_epa import attach_prior_epa
from research.run_nfl_v3 import EPA_FEATURES
from research.live_prediction_bridge import record_live_inference
from research.prospective_prediction_log import utc

def predict_upcoming(games,team_epa,*,asof_utc,now_utc,minimum_train=300):
    asof=utc(asof_utc);now=utc(now_utc)
    if asof>now:raise ValueError("Snapshot is later than prediction time")
    g=games.copy()
    required={"game_id","season","week","gameday","gametime","home_team","away_team","home_score","away_score","kickoff_utc"}
    if required-set(g):raise ValueError("Missing required schedule fields")
    if g.game_id.isna().any() or g.game_id.duplicated().any():raise ValueError("Duplicate game IDs")
    g["gameday"]=pd.to_datetime(g.gameday,errors="raise")
    # Historical completed games may lack a published kickoff clock time.\n    # They are used for training only; upcoming games require real kickoff.\n    g["kickoff_utc"]=g.kickoff_utc.map(lambda x: utc(x) if pd.notna(x) else pd.NaT)
    g["home_score"]=pd.to_numeric(g.home_score,errors="coerce")
    g["away_score"]=pd.to_numeric(g.away_score,errors="coerce")
    # We cannot prove when same-day final scores became available. Only use
    # games from calendar dates strictly earlier than the snapshot UTC date.
    historical=g.gameday.dt.date<asof.date()
    completed=g.home_score.notna() & g.away_score.notna()
    if (completed & ~historical).any():
        raise ValueError("Snapshot contains final scores on or after as-of date")
    # Future or unfinished historical games must not update EPA history.
    eligible=set(g.loc[historical & completed,"game_id"])
    if not set(team_epa.game_id).issubset(eligible):
        raise ValueError("EPA includes games not completed before snapshot date")
    f=date_safe_features(g)
    e=attach_prior_epa(g,team_epa)
    f=f.merge(e[["game_id"]+EPA_FEATURES],on="game_id",validate="one_to_one")
    train=f.loc[(f.gameday.dt.date<asof.date())&f.target_margin.notna()&
                (f.home_games_prior>=2)&(f.away_games_prior>=2)]
    if len(train)<minimum_train:raise ValueError("Insufficient mature historical training games")
    upcoming=f.loc[f.game_id.isin(g.loc[g.kickoff_utc>now,"game_id"])&
                   f.target_margin.isna()&
                   (f.home_games_prior>=2)&(f.away_games_prior>=2)].copy()
    if upcoming.empty:raise ValueError("No eligible upcoming games")
    columns=list(FEATURES)+EPA_FEATURES
    model=Pipeline([("impute",SimpleImputer(strategy="median")),
                    ("scale",StandardScaler()),
                    ("model",LogisticRegression(C=.45,max_iter=2000))])
    model.fit(train[columns],(train.target_margin>0).astype(int))
    upcoming["home_win_probability"]=model.predict_proba(upcoming[columns])[:,1]
    upcoming["input_fingerprint"]=upcoming.apply(
        lambda r:hashlib.sha256(json.dumps({k:float(r[k]) for k in columns},sort_keys=True).encode()).hexdigest(),axis=1)
    upcoming["model_version"]="live-v3-epa-research"
    upcoming["features_asof_utc"]=asof.isoformat()
    upcoming["model_trained_through_utc"]=pd.Timestamp(train.gameday.max()).tz_localize("UTC").isoformat()
    return upcoming[["game_id","kickoff_utc","home_team","away_team","home_win_probability",
                     "input_fingerprint","model_version","features_asof_utc","model_trained_through_utc"]].copy()

def log_upcoming(predictions,ledger,*,now_utc):
    records=[]
    for _,r in predictions.iterrows():
        records.append(record_live_inference(ledger,game_id=r.game_id,kickoff_utc=r.kickoff_utc,
            model_version=r.model_version,home_win_probability=r.home_win_probability,
            input_fingerprint=r.input_fingerprint,features_asof_utc=r.features_asof_utc,
            model_trained_through_utc=r.model_trained_through_utc,now=utc(now_utc)))
    return records
