"""Paired season-block uncertainty audit for incremental market benchmark.

Resamples complete seasons, not individual games. Four season clusters,
including a partial season, make these intervals exploratory only.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def analyze(d:pd.DataFrame,draws:int=5000,seed:int=2026):
    required={"game_id","season","home_win","raw_market","market_calibrated","market_plus_v3"}
    if required-set(d):raise ValueError("Missing columns")
    if d.game_id.duplicated().any():raise ValueError("Duplicate game IDs")
    if d[list(required)].isna().any().any():raise ValueError("Missing data")
    if not d.home_win.isin([0,1]).all():raise ValueError("Invalid outcomes")
    for c in ("raw_market","market_calibrated","market_plus_v3"):
        if not d[c].between(0,1).all():raise ValueError("Invalid probability")
    if draws<1:raise ValueError("Draws must be positive")
    y=d.home_win.to_numpy(dtype=float)
    p=d.market_plus_v3.to_numpy(dtype=float)
    market=d.raw_market.to_numpy(dtype=float)
    calibrated=d.market_calibrated.to_numpy(dtype=float)
    diff_raw=(p-y)**2-(market-y)**2
    diff_cal=(p-y)**2-(calibrated-y)**2
    seasons=d.season.to_numpy()
    unique=np.unique(seasons)
    blocks=[np.flatnonzero(seasons==s) for s in unique]
    rng=np.random.default_rng(seed)
    samples=np.empty((draws,2))
    for i in range(draws):
        indices=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])
        samples[i]=[diff_raw[indices].mean(),diff_cal[indices].mean()]
    def detail(x):
        return {"mean":float(x.mean()),"season_cluster_bootstrap_95pct":np.quantile(x,[.025,.975]).tolist(),
                "seasons_improved":int(sum(x[seasons==s].mean()<0 for s in unique)),
                "seasons_total":len(unique),
                "by_season":{str(s):{"games":int((seasons==s).sum()),"mean":float(x[seasons==s].mean())} for s in unique}}
    return {"paired_games":len(d),"seasons":len(unique),
            "combined_minus_raw_market_brier":{**detail(diff_raw),"season_cluster_bootstrap_95pct":np.quantile(samples[:,0],[.025,.975]).tolist()},
            "combined_minus_calibrated_market_brier":{**detail(diff_cal),"season_cluster_bootstrap_95pct":np.quantile(samples[:,1],[.025,.975]).tolist()},
            "decision":"Exploratory uncertainty only; no production promotion",
            "caveat":"Only four season clusters, one partial; bootstrap cannot establish a durable edge. Historical market quotes are not timestamped executable prices."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="research_outputs/nfl_market_incremental_predictions.csv")
    ap.add_argument("--output",default="research_outputs/nfl_market_incremental_uncertainty.json")
    a=ap.parse_args()
    report=analyze(pd.read_csv(a.predictions))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
