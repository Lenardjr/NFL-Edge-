"""Strict prior-season benchmark: market calibration versus market plus V3.

Both fitted competitors use identical historical training seasons, games,
regularization, and probability transformation. No current-season outcomes train
a current-season model. Historical odds lack executable quote timestamps.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss

def _logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-5,1-1e-5)
    return np.log(p/(1-p))

def _metrics(d):
    y=d.home_win.to_numpy(dtype=int)
    out={"games":len(d)}
    for name in ("raw_market","market_calibrated","market_plus_v3"):
        p=d[name].to_numpy(dtype=float)
        out[name+"_brier"]=float(brier_score_loss(y,p))
        out[name+"_log_loss"]=float(log_loss(y,p,labels=[0,1]))
        out[name+"_accuracy"]=float(((p>=.5)==y).mean())
    out["incremental_brier_vs_calibrated_market"]=out["market_plus_v3_brier"]-out["market_calibrated_brier"]
    out["incremental_brier_vs_raw_market"]=out["market_plus_v3_brier"]-out["raw_market_brier"]
    return out

def evaluate(data):
    required={"game_id","season","home_win","v3_prob","market_prob"}
    if required-set(data):raise ValueError("Missing columns")
    if data.game_id.duplicated().any():raise ValueError("Duplicate game IDs")
    d=data.copy()
    for c in ("season","home_win","v3_prob","market_prob"):
        d[c]=pd.to_numeric(d[c],errors="coerce")
    if d[list(required-{"game_id"})].isna().any().any():raise ValueError("Missing values")
    if not d.home_win.isin([0,1]).all():raise ValueError("Invalid outcomes")
    if not d.v3_prob.between(0,1).all() or not d.market_prob.between(0,1).all():
        raise ValueError("Invalid probabilities")
    d["market_logit"]=_logit(d.market_prob)
    d["v3_logit"]=_logit(d.v3_prob)
    rows=[]
    for season in sorted(d.season.unique()):
        train=d[d.season<season]
        test=d[d.season==season]
        if len(train)<100 or train.home_win.nunique()<2:continue
        # Fixed regularization and feature specifications; no selection using test season.
        base=LogisticRegression(C=.5,max_iter=2000)
        combo=LogisticRegression(C=.5,max_iter=2000)
        base.fit(train[["market_logit"]],train.home_win.astype(int))
        combo.fit(train[["market_logit","v3_logit"]],train.home_win.astype(int))
        t=test[["game_id","season","home_win"]].copy()
        t["raw_market"]=test.market_prob.to_numpy()
        t["market_calibrated"]=base.predict_proba(test[["market_logit"]])[:,1]
        t["market_plus_v3"]=combo.predict_proba(test[["market_logit","v3_logit"]])[:,1]
        rows.append(t)
    if not rows:raise ValueError("No held-out seasons")
    p=pd.concat(rows,ignore_index=True)
    summary={"overall":_metrics(p),
             "by_season":{str(int(s)):_metrics(g) for s,g in p.groupby("season")},
             "train_rule":"All strictly prior seasons; same training sample for both fitted competitors",
             "models":"Logistic regression C=0.5, market logit alone versus market and V3 logits",
             "status":"Research only; odds quote timestamps and executable prices not verified; no ROI"}
    return summary,p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="research_outputs/nfl_v3_predictions.csv")
    ap.add_argument("--output-dir",default="research_outputs")
    a=ap.parse_args()
    summary,p=evaluate(pd.read_csv(a.predictions))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"nfl_market_incremental_summary.json").write_text(json.dumps(summary,indent=2))
    p.to_csv(out/"nfl_market_incremental_predictions.csv",index=False)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
