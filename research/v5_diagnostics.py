"""Pre-specified diagnostic slices of held-out V3 vs market forecasts.

Descriptive only: subgroup differences are exploratory, not evidence of edge.
All slices use only pregame probabilities and season, never outcomes to select.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

BINS=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0000001]
EDGE_BINS=[0,.025,.05,.10,.20,float("inf")]

def analyze(df:pd.DataFrame)->dict:
    required={"game_id","season","home_win","v3_prob","market_prob"}
    missing=required-set(df.columns)
    if missing:raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.game_id.duplicated().any():raise ValueError("Duplicate game_id")
    d=df.dropna(subset=list(required)).copy()
    if d.empty or not d.home_win.isin([0,1]).all():raise ValueError("Invalid outcomes")
    for col in ("v3_prob","market_prob"):
        if not d[col].between(0,1).all():raise ValueError("Invalid probability "+col)
    d["residual"]=d.v3_prob-d.market_prob
    d["absolute_residual"]=d.residual.abs()
    d["disagreement"]=(d.v3_prob>=.5)!=(d.market_prob>=.5)
    d["market_bin"]=pd.cut(d.market_prob,BINS,include_lowest=True,right=False)
    d["residual_bin"]=pd.cut(d.absolute_residual,EDGE_BINS,include_lowest=True,right=False)
    def score(g):
        if g.empty:return None
        y=g.home_win.to_numpy(dtype=float)
        m=g.market_prob.to_numpy(dtype=float)
        v=g.v3_prob.to_numpy(dtype=float)
        return {"games":int(len(g)),
                "market_brier":float(np.mean((m-y)**2)),
                "v3_brier":float(np.mean((v-y)**2)),
                "v3_minus_market_brier":float(np.mean((v-y)**2-(m-y)**2)),
                "market_accuracy":float(np.mean((m>=.5)==y)),
                "v3_accuracy":float(np.mean((v>=.5)==y)),
                "market_mean_probability":float(m.mean()),
                "v3_mean_probability":float(v.mean()),
                "home_win_rate":float(y.mean())}
    def groups(column):
        return [{"group":str(key),**score(g)} for key,g in d.groupby(column,observed=True,sort=True)]
    return {"overall":score(d),
            "market_probability_bins":groups("market_bin"),
            "absolute_model_market_gap_bins":groups("residual_bin"),
            "disagreement":groups("disagreement"),
            "season":groups("season"),
            "method":"Fixed probability and gap bins; same paired held-out games; no outcome-driven subgroup selection",
            "decision":"Exploratory diagnostics only; no subgroup or V5 deployment without future untouched validation",
            "limitations":"No injury, quarterback-status, travel, or timestamped executable quote inputs in this dataset."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="research_outputs/nfl_v3_predictions.csv")
    ap.add_argument("--output",default="research_outputs/nfl_v5_diagnostics.json")
    a=ap.parse_args()
    report=analyze(pd.read_csv(a.predictions))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
