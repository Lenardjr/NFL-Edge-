"""Chronological V3 EPA experiment against V1 and no-vig market.

Uses the exact baseline eligibility/training cutoffs from run_nfl_v2.
Only lagged EPA features are used; contemporaneous game EPA is forbidden.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss,log_loss
from game_engine import FEATURES
from research.date_safe_features import date_safe_features
from research.v3_pbp_features import aggregate_games
from research.v3_date_safe_epa import attach_prior_epa
from research.market_validation import no_vig_two_way

EPA_FEATURES=["epa_diff_prior5","success_diff_prior5","home_epa_sample_games","away_epa_sample_games"]
PBP_URL="https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
SCHEDULE_URL="https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"

def load_epa(seasons,local_dir=None):
    parts=[]
    for season in seasons:
        path=Path(local_dir)/f"play_by_play_{season}.parquet" if local_dir else PBP_URL.format(season=season)
        cols=["game_id","posteam","defteam","epa","play_type","season","week"]
        part=pd.read_parquet(path,columns=cols)
        parts.append(aggregate_games(part))
    return pd.concat(parts,ignore_index=True)

def evaluate(games,team_epa,first_test_season=2022):
    g=games.copy()
    g["gameday"]=pd.to_datetime(g.gameday,errors="coerce")
    for c in ("home_score","away_score","home_moneyline","away_moneyline"):
        if c in g:g[c]=pd.to_numeric(g[c],errors="coerce")
    g=g.sort_values([c for c in ("gameday","gametime","game_id") if c in g],kind="stable")
    base=date_safe_features(g)
    epa=attach_prior_epa(g,team_epa)
    # Explicit one-to-one ID alignment; never rely on row ordering.
    cols=["game_id"]+EPA_FEATURES
    f=base.merge(epa[cols],on="game_id",how="left",validate="one_to_one")
    if f[EPA_FEATURES].isna().any().any():raise ValueError("EPA feature join incomplete")
    done=f.dropna(subset=["target_margin","target_total"]).copy()
    test=done[(done.season>=first_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    for (season,week),group in test.groupby(["season","week"],sort=True):
        cutoff=group.gameday.min()
        train=done[(done.gameday<cutoff)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(train)<300:continue
        preds=[]
        for columns in (FEATURES,list(FEATURES)+EPA_FEATURES):
            pipe=Pipeline([("impute",SimpleImputer(strategy="median")),
                ("scale",StandardScaler()),
                ("model",LogisticRegression(C=.45,max_iter=2000))])
            pipe.fit(train[columns],(train.target_margin>0).astype(int))
            preds.append(pipe.predict_proba(group[columns])[:,1])
        for (_,game),p1,p3 in zip(group.iterrows(),*preds):
            if game.target_margin==0:continue
            h,a=game.get("home_moneyline"),game.get("away_moneyline")
            market=(no_vig_two_way(int(h),int(a))[0] if pd.notna(h) and pd.notna(a)
                    and abs(h)>=100 and abs(a)>=100 else np.nan)
            rows.append(dict(game_id=str(game.game_id),season=int(season),week=int(week),
                home_win=int(game.target_margin>0),baseline_prob=float(p1),
                v3_prob=float(p3),market_prob=market))
    predictions=pd.DataFrame(rows)
    if predictions.empty:raise ValueError("Insufficient historical games")
    def score(data):
        y=data.home_win
        result={"games":len(data)}
        for name in ("baseline","v3","market"):
            p=data[f"{name}_prob"]
            valid=p.notna()
            result[name+"_brier"]=float(brier_score_loss(y[valid],p[valid])) if valid.any() else None
            result[name+"_log_loss"]=float(log_loss(y[valid],p[valid],labels=[0,1])) if valid.any() else None
            result[name+"_accuracy"]=float(((p[valid]>=.5)==y[valid]).mean()) if valid.any() else None
        return result
    summary={"overall":score(predictions),
        "seasons":{str(s):score(d) for s,d in predictions.groupby("season")},
        "status":"historical research only; no executable quote or ROI validation"}
    return summary,predictions

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--games-csv",default=SCHEDULE_URL)
    ap.add_argument("--pbp-dir",default=None)
    ap.add_argument("--first-pbp-season",type=int,default=2018)
    ap.add_argument("--last-pbp-season",type=int,default=2026)
    ap.add_argument("--output-dir",default="research_outputs")
    args=ap.parse_args()
    games=pd.read_csv(args.games_csv)
    epa=load_epa(range(args.first_pbp_season,args.last_pbp_season+1),args.pbp_dir)
    summary,p=evaluate(games,epa)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"nfl_v3_summary.json").write_text(json.dumps(summary,indent=2))
    p.to_csv(out/"nfl_v3_predictions.csv",index=False)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
