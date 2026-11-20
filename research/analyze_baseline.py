"""Analyze exported baseline predictions without retraining or changing the data.

Run: python -m research.analyze_baseline --predictions research_outputs/nfl_baseline_predictions.csv
This is diagnostic analysis, not evidence of executable betting returns.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

def metrics(df):
    y=df.home_win.to_numpy()
    model=df.model_home_prob.to_numpy()
    market=df.market_home_prob.to_numpy()
    return dict(games=len(df), model_accuracy=float(np.mean((model>=.5)==y)),
                market_accuracy=float(np.mean((market>=.5)==y)),
                model_brier=float(brier_score_loss(y,model)),
                market_brier=float(brier_score_loss(y,market)),
                model_log_loss=float(log_loss(y,model)),
                market_log_loss=float(log_loss(y,market)))

def analyze(df):
    required={"season","gameday","home_win","model_home_prob","market_home_prob"}
    if required-set(df.columns):
        raise ValueError(f"missing required fields: {sorted(required-set(df.columns))}")
    df=df.dropna(subset=list(required)).copy()
    df["gameday"]=pd.to_datetime(df.gameday)
    df=df.sort_values(["gameday"],kind="stable")
    seasons={str(s):metrics(g) for s,g in df.groupby("season")}
    df["disagreement"]=(df.model_home_prob-df.market_home_prob).abs()
    disagreement={str(t):metrics(df[df.disagreement>=t]) for t in (0,.05,.10,.15,.20)}
    # A blend weight is learned from strictly earlier seasons, never the test season.
    # 0 is market only; 1 is model only.
    oos=[]
    for season in sorted(df.season.unique()):
        train=df[df.season<season]
        test=df[df.season==season]
        if len(train)<150:continue
        weights=np.arange(0,1.00001,.025)
        w=min(weights,key=lambda v:brier_score_loss(
            train.home_win,(1-v)*train.market_home_prob+v*train.model_home_prob))
        q=(1-w)*test.market_home_prob+w*test.model_home_prob
        oos.append(dict(season=int(season),games=len(test),model_weight=float(w),
                        blend_brier=float(brier_score_loss(test.home_win,q)),
                        market_brier=float(brier_score_loss(test.home_win,test.market_home_prob)),
                        blend_log_loss=float(log_loss(test.home_win,q)),
                        market_log_loss=float(log_loss(test.home_win,test.market_home_prob))))
    return dict(all_games=metrics(df),seasons=seasons,disagreement=disagreement,
                chronological_blend=oos,
                interpretation="Closing-line benchmark only; not verified tradable odds or strategy ROI.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",required=True)
    p.add_argument("--output",default="research_outputs/nfl_diagnostic_report.json")
    a=p.parse_args()
    result=analyze(pd.read_csv(a.predictions))
    from pathlib import Path
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
