"""Outcome-blind Week 18 versus other weeks diagnostic for V3 and market."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss,log_loss

def summarize(df):
    result={"games":len(df)}
    for model in ("v3","market"):
        p=pd.to_numeric(df[f"{model}_prob"],errors="coerce")
        y=pd.to_numeric(df.home_win,errors="coerce")
        valid=p.notna()&y.notna()
        result[f"{model}_games"]=int(valid.sum())
        result[f"{model}_accuracy"]=float(((p[valid]>=.5)==y[valid]).mean()) if valid.any() else None
        result[f"{model}_brier"]=float(brier_score_loss(y[valid],p[valid])) if valid.any() else None
        result[f"{model}_log_loss"]=float(log_loss(y[valid],p[valid],labels=[0,1])) if valid.any() else None
    both=df[["v3_prob","market_prob"]].notna().all(axis=1)
    result["large_gap_games"]=int((abs(df.loc[both,"v3_prob"]-df.loc[both,"market_prob"])>=.20-1e-12).sum())
    result["large_gap_rate"]=float(result["large_gap_games"]/both.sum()) if both.any() else None
    result["v3_minus_market_brier"]=(
        result["v3_brier"]-result["market_brier"]
        if result["v3_games"]==result["market_games"] and result["v3_brier"] is not None
        and result["market_brier"] is not None else None)
    return result

def audit(predictions):
    required={"game_id","season","week","home_win","v3_prob","market_prob"}
    if required-set(predictions):raise ValueError("Missing prediction fields")
    if predictions.game_id.duplicated().any():raise ValueError("Duplicate game IDs")
    p=predictions.copy()
    for c in ("season","week","home_win","v3_prob","market_prob"):
        p[c]=pd.to_numeric(p[c],errors="coerce")
    if p[["season","week","home_win","v3_prob"]].isna().any().any():
        raise ValueError("Missing mandatory values")
    if not p.home_win.isin([0,1]).all():raise ValueError("Invalid outcomes")
    for col in ("v3_prob","market_prob"):
        if (~p[col].dropna().between(0,1)).any():raise ValueError("Invalid probabilities")
    p["segment"]=np.where(p.week==18,"week_18","other_weeks")
    summary={"overall":summarize(p),"segments":{},
        "by_season":{},"method":"Week 18 defined as schedule week == 18; descriptive, not causal",
        "warning":"No pregame QB availability assumed; no betting ROI or deployable edge inferred."}
    for name,g in p.groupby("segment"):
        summary["segments"][name]=summarize(g)
    for season,g in p.groupby("season"):
        summary["by_season"][str(int(season))]={name:summarize(x) for name,x in g.groupby("segment")}
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="research_outputs/nfl_v3_predictions.csv")
    ap.add_argument("--output",default="research_outputs/nfl_week18_audit.json")
    args=ap.parse_args()
    summary=audit(pd.read_csv(args.predictions))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
