"""Chronological head-to-head evaluation of baseline vs opponent-adjusted V2.

Runs the same weekly training cutoffs and test games for both models.
Outputs paired predictions and scores, not betting ROI.
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
from research.opponent_features import opponent_adjusted_features,V2_FEATURES
from research.market_validation import no_vig_two_way

def evaluate(games,first_test_season=2022):
    g=games.copy()
    g["gameday"]=pd.to_datetime(g.gameday,errors="coerce")
    for c in ["home_score","away_score","home_moneyline","away_moneyline"]:
        if c in g:g[c]=pd.to_numeric(g[c],errors="coerce")
    g=g.sort_values([c for c in ["gameday","gametime","game_id"] if c in g],kind="stable")
    f=opponent_adjusted_features(g)
    done=f.dropna(subset=["target_margin","target_total"]).copy()
    test=done[(done.season>=first_test_season)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
    rows=[]
    for (season,week),group in test.groupby(["season","week"],sort=True):
        cutoff=group.gameday.min()
        train=done[(done.gameday<cutoff)&(done.home_games_prior>=2)&(done.away_games_prior>=2)]
        if len(train)<300:continue
        predictions=[]
        for cols in (FEATURES,V2_FEATURES):
            pipe=Pipeline([("impute",SimpleImputer(strategy="median")),
                ("scale",StandardScaler()),
                ("model",LogisticRegression(C=.45,max_iter=2000))])
            pipe.fit(train[cols],(train.target_margin>0).astype(int))
            predictions.append(pipe.predict_proba(group[cols])[:,1])
        for (_,game),p0,p2 in zip(group.iterrows(),*predictions):
            if game.target_margin==0:continue
            h,a=game.get("home_moneyline"),game.get("away_moneyline")
            market=(no_vig_two_way(int(h),int(a))[0]
                if pd.notna(h) and pd.notna(a) and abs(h)>=100 and abs(a)>=100 else np.nan)
            rows.append(dict(game_id=str(game.get("game_id","")),season=int(season),
                week=int(week),gameday=str(game.gameday.date()),home_win=int(game.target_margin>0),
                baseline_prob=float(p0),v2_prob=float(p2),market_prob=market))
    p=pd.DataFrame(rows)
    if p.empty:return {"status":"insufficient history"},p
    def metrics(frame):
        y=frame.home_win
        out={"games":len(frame)}
        for name in ("baseline","v2","market"):
            probs=frame[f"{name}_prob"]
            mask=probs.notna()
            out[name+"_brier"]=float(brier_score_loss(y[mask],probs[mask])) if mask.any() else None
            out[name+"_log_loss"]=float(log_loss(y[mask],probs[mask])) if mask.any() else None
        return out
    summary={"status":"research benchmark only; not a betting ROI test",
             "overall":metrics(p),"seasons":{str(s):metrics(d) for s,d in p.groupby("season")}}
    return summary,p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--games-csv",default="https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv")
    ap.add_argument("--output-dir",default="research_outputs")
    args=ap.parse_args()
    summary,p=evaluate(pd.read_csv(args.games_csv))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"nfl_v2_summary.json").write_text(json.dumps(summary,indent=2))
    p.to_csv(out/"nfl_v2_predictions.csv",index=False)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
