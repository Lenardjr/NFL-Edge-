"""Outcome-blind segmentation of held-out market-incremental predictions.

Diagnoses where a combined model helps or hurts without selecting a betting
strategy. Bins and season labels are fixed before looking at outcomes.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

BINS=((0.0,.025,"under_2_5pp"),(.025,.05,"2_5_to_5pp"),(.05,.10,"5_to_10pp"),(.10,1.01,"10pp_plus"))
def analyze(predictions:pd.DataFrame,original:pd.DataFrame):
    cols={"game_id","season","home_win","raw_market","market_calibrated","market_plus_v3"}
    if cols-set(predictions):raise ValueError("Missing prediction columns")
    if {"game_id","v3_prob","market_prob"}-set(original):raise ValueError("Missing original columns")
    if predictions.game_id.duplicated().any() or original.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs")
    d=predictions.merge(original[["game_id","v3_prob","market_prob"]],on="game_id",how="left",validate="one_to_one")
    if d.isna().any().any():raise ValueError("Unmatched or missing predictions")
    if not d.home_win.isin([0,1]).all():raise ValueError("Invalid outcomes")
    for c in ("raw_market","market_calibrated","market_plus_v3","v3_prob","market_prob"):
        if not d[c].between(0,1).all():raise ValueError("Invalid probabilities")
    if not np.allclose(d.raw_market,d.market_prob,atol=1e-10,rtol=0):
        raise ValueError("Raw market probabilities do not match")
    d["gap"]=abs(d.v3_prob-d.market_prob)
    def stats(g):
        y=g.home_win.to_numpy(dtype=float)
        raw=g.raw_market.to_numpy(dtype=float)
        calibrated=g.market_calibrated.to_numpy(dtype=float)
        combined=g.market_plus_v3.to_numpy(dtype=float)
        return {"games":len(g),
            "combined_minus_raw_market_brier":float(np.mean((combined-y)**2-(raw-y)**2)),
            "combined_minus_calibrated_market_brier":float(np.mean((combined-y)**2-(calibrated-y)**2)),
            "combined_minus_raw_market_log_loss":float(np.mean(-(y*np.log(np.clip(combined,1e-10,1))+(1-y)*np.log(np.clip(1-combined,1e-10,1)) )+
                (y*np.log(np.clip(raw,1e-10,1))+(1-y)*np.log(np.clip(1-raw,1e-10,1))))),
            "mean_absolute_v3_market_gap":float(g.gap.mean())}
    segments={}
    for low,high,label in BINS:
        g=d[(d.gap>=low)&(d.gap<high)]
        segments[label]=stats(g) if len(g) else {"games":0}
    return {"overall":stats(d),"gap_bins":segments,
        "by_season":{str(int(s)):stats(g) for s,g in d.groupby("season")},
        "segmentation":"Absolute V3-minus-market gap; fixed 2.5, 5, and 10 percentage-point boundaries, no outcome-based selection",
        "caution":"Exploratory diagnostic only; slices are not validated wagering filters; historical odds not verified executable."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",default="research_outputs/nfl_market_incremental_predictions.csv")
    p.add_argument("--original",default="research_outputs/nfl_v3_predictions.csv")
    p.add_argument("--output",default="research_outputs/nfl_market_incremental_gap_diagnostics.json")
    a=p.parse_args()
    report=analyze(pd.read_csv(a.predictions),pd.read_csv(a.original))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
