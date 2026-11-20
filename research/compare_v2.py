"""Pre-specified paired V2-vs-V1 evaluation on identical held-out games.

Outputs uncertainty and season consistency. A lower mean score in one exploratory
backtest is NOT sufficient evidence of a deployable or profitable betting edge.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def compare(predictions:pd.DataFrame,seed:int=2026,draws:int=5000)->dict:
    required={"season","game_id","home_win","baseline_prob","v2_prob","market_prob"}
    missing=required-set(predictions.columns)
    if missing:raise ValueError(f"Missing columns: {sorted(missing)}")
    df=predictions.dropna(subset=list(required)).copy()
    if df.empty:raise ValueError("No common games with all three predictions")
    if df.game_id.duplicated().any():raise ValueError("Duplicate game_id")
    if not df.home_win.isin([0,1]).all():raise ValueError("Invalid outcomes")
    for col in ("baseline_prob","v2_prob","market_prob"):
        if not df[col].between(0,1).all():raise ValueError(f"Invalid {col}")
    y=df.home_win.to_numpy(dtype=float)
    scores={}
    for label,col in (("baseline","baseline_prob"),("v2","v2_prob"),("market","market_prob")):
        p=np.clip(df[col].to_numpy(dtype=float),1e-12,1-1e-12)
        scores[label]={"brier":float(np.mean((p-y)**2)),
                       "log_loss":float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
                       "accuracy":float(np.mean((p>=.5)==y))}
    diff=(df.v2_prob.to_numpy()-y)**2-(df.baseline_prob.to_numpy()-y)**2
    market_diff=(df.v2_prob.to_numpy()-y)**2-(df.market_prob.to_numpy()-y)**2
    years=df.season.to_numpy();unique=np.unique(years)
    rng=np.random.default_rng(seed);sample=[]
    for _ in range(draws):
        selected=rng.choice(unique,len(unique),replace=True)
        idx=np.concatenate([np.flatnonzero(years==s) for s in selected])
        sample.append([float(diff[idx].mean()),float(market_diff[idx].mean())])
    intervals=np.quantile(sample,[.025,.975],axis=0)
    seasons={}
    for season,g in df.groupby("season"):
        t=g.home_win.to_numpy()
        seasons[str(season)]={"games":len(g),
            "v2_minus_baseline_brier":float(np.mean((g.v2_prob-t)**2-(g.baseline_prob-t)**2)),
            "v2_minus_market_brier":float(np.mean((g.v2_prob-t)**2-(g.market_prob-t)**2))}
    return {"games":len(df),"scores":scores,
        "v2_minus_baseline_brier":float(diff.mean()),
        "v2_minus_market_brier":float(market_diff.mean()),
        "season_cluster_bootstrap_95pct_intervals":{
            "v2_minus_baseline":[float(x) for x in intervals[:,0]],
            "v2_minus_market":[float(x) for x in intervals[:,1]]},
        "seasons":seasons,
        "decision":"research only; require consistent improvements on future untouched games",
        "caution":"Season bootstrap with few seasons has limited reliability; closing market odds are not executable historical prices."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--output",default="research_outputs/nfl_v2_comparison.json")
    a=ap.parse_args()
    report=compare(pd.read_csv(a.predictions))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
