"""V6 research-only nonlinear-rest ablation against original V3 predictions.

Chronological training, identical eligibility and cutoffs to run_nfl_v3.
Rest is ALREADY in V3; only threshold transformations are experimental.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from research.run_nfl_v3 import (SCHEDULE_URL,EPA_FEATURES,load_epa)
from research.date_safe_features import date_safe_features
from research.v3_date_safe_epa import attach_prior_epa
from game_engine import FEATURES

EXTRA=["home_rest_disadvantage_3plus","home_rest_advantage_3plus",
       "either_team_short_rest","rest_imbalance_excess_3"]
def add_rest_transforms(f):
    d=f.copy()
    diff=d.rest_diff.astype(float)
    d["home_rest_disadvantage_3plus"]=(diff<=-3).astype(float)
    d["home_rest_advantage_3plus"]=(diff>=3).astype(float)
    d["either_team_short_rest"]=(d[["home_rest","away_rest"]].min(axis=1)<=5).astype(float)
    d["rest_imbalance_excess_3"]=np.maximum(diff.abs()-3,0)
    return d

def evaluate(games,team_epa,v3_predictions,first_test_season=2022):
    g=games.copy()
    g["gameday"]=pd.to_datetime(g.gameday,errors="coerce")
    for col in ("home_score","away_score","home_moneyline","away_moneyline"):
        if col in g:g[col]=pd.to_numeric(g[col],errors="coerce")
    g=g.sort_values([c for c in ("gameday","gametime","game_id") if c in g],kind="stable")
    f=date_safe_features(g).merge(attach_prior_epa(g,team_epa)[["game_id"]+EPA_FEATURES],
                                  on="game_id",how="left",validate="one_to_one")
    if f[EPA_FEATURES].isna().any().any():raise ValueError("Missing EPA features")
    f=add_rest_transforms(f)
    done=f.dropna(subset=["target_margin","target_total"]).copy()
    test=done[(done.season>=first_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    cols=list(FEATURES)+EPA_FEATURES
    for (season,week),group in test.groupby(["season","week"],sort=True):
        cutoff=group.gameday.min()
        train=done[(done.gameday<cutoff)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(train)<300:continue
        outputs=[]
        for feature_set in (cols,cols+EXTRA):
            pipe=Pipeline([("impute",SimpleImputer(strategy="median")),
                ("scale",StandardScaler()),
                ("model",LogisticRegression(C=.45,max_iter=2000))])
            pipe.fit(train[feature_set],(train.target_margin>0).astype(int))
            outputs.append(pipe.predict_proba(group[feature_set])[:,1])
        for (_,game),v3,v6 in zip(group.iterrows(),*outputs):
            if game.target_margin==0:continue
            rows.append({"game_id":str(game.game_id),"season":int(season),"week":int(week),
                         "home_win":int(game.target_margin>0),"v3_reproduced_prob":float(v3),
                         "v6_rest_prob":float(v6)})
    out=pd.DataFrame(rows)
    required={"game_id","season","week","home_win","v3_prob","market_prob"}
    if required-set(v3_predictions):raise ValueError("Missing V3 baseline columns")
    if out.game_id.duplicated().any() or v3_predictions.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs")
    paired=out.merge(v3_predictions[list(required)],on=["game_id","season","week","home_win"],
                     validate="one_to_one",indicator=True)
    if len(paired)!=len(out) or len(paired)!=len(v3_predictions) or not (paired._merge=="both").all():
        raise ValueError("V3 and V6 game alignment mismatch")
    if not np.allclose(paired.v3_reproduced_prob,paired.v3_prob,atol=1e-9,rtol=0):
        raise ValueError("V3 reproduction mismatch; cannot compare models")
    if paired.market_prob.isna().any():raise ValueError("Missing paired market probabilities")
    def metrics(d):
        y=d.home_win.to_numpy(dtype=float)
        result={"games":len(d)}
        for name,col in [("v3","v3_prob"),("v6_rest","v6_rest_prob"),("market","market_prob")]:
            p=d[col].to_numpy(dtype=float)
            result[name+"_brier"]=float(np.mean((p-y)**2))
            result[name+"_log_loss"]=float(np.mean(-(y*np.log(np.clip(p,1e-15,1))+
                                                (1-y)*np.log(np.clip(1-p,1e-15,1)))))
            result[name+"_accuracy"]=float(np.mean((p>=.5)==y))
        result["v6_minus_v3_brier"]=result["v6_rest_brier"]-result["v3_brier"]
        result["v6_minus_market_brier"]=result["v6_rest_brier"]-result["market_brier"]
        return result
    report={"overall":metrics(paired),"seasons":{str(int(s)):metrics(d) for s,d in paired.groupby("season")},
            "experiment":"Nonlinear rest thresholds added to existing V3 rest features; no market input to model",
            "promotion":"Not eligible without stable out-of-sample improvement and further verification",
            "caution":"Retrospective historical odds not timestamp-verified or executable"}
    return report,paired.drop(columns="_merge")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--games-csv",default=SCHEDULE_URL)
    p.add_argument("--pbp-dir",default=None)
    p.add_argument("--first-pbp-season",type=int,default=2018)
    p.add_argument("--last-pbp-season",type=int,default=2026)
    p.add_argument("--v3-predictions",default="research_outputs/nfl_v3_predictions.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    games=pd.read_csv(a.games_csv)
    epa=load_epa(range(a.first_pbp_season,a.last_pbp_season+1),a.pbp_dir)
    report,preds=evaluate(games,epa,pd.read_csv(a.v3_predictions))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"nfl_v6_rest_summary.json").write_text(json.dumps(report,indent=2))
    preds.to_csv(out/"nfl_v6_rest_predictions.csv",index=False)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
