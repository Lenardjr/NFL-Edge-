"""Descriptive rest vs extreme-disagreement audit on ALL eligible games.

Never use outcomes to define cohorts. Compare flagged games with nonflagged
controls; no causal attribution or betting filter inferred from slices.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from research.v5_matchup_enrichment import enrich,SCHEDULE_URL

def analyze(predictions:pd.DataFrame,schedules:pd.DataFrame)->dict:
    required={"game_id","season","week","home_win","v3_prob","market_prob"}
    if required-set(predictions):raise ValueError("Missing prediction columns")
    if predictions.game_id.isna().any() or predictions.game_id.duplicated().any():
        raise ValueError("Duplicate or null predictions")
    if predictions[list(required)].isna().any().any():raise ValueError("Missing paired observations")
    if not predictions.home_win.isin([0,1]).all():raise ValueError("Invalid outcome")
    for col in ("v3_prob","market_prob"):
        v=predictions[col].to_numpy(dtype=float)
        if not np.isfinite(v).all() or not ((v>=0)&(v<=1)).all():
            raise ValueError("Invalid probability")
    d=enrich(predictions,schedules)
    d["gap"]=(d.v3_prob-d.market_prob).abs()
    d["gap_group"]=pd.cut(d.gap,[-.000001,.10,.20,np.inf],
                           labels=["under_10pt","10_to_20pt","20pt_plus"],right=False)
    d["rest_group"]=np.select([
        d.rest_advantage_home<=-3,d.rest_advantage_home>=3,
        d.rest_advantage_home.abs()<3],
        ["home_disadvantage_3plus","home_advantage_3plus","balanced_under_3"],
        default="unknown")
    d["short_rest"]=np.where(d[["home_rest_days","away_rest_days"]].isna().any(axis=1),
                             "unknown",np.where(d[["home_rest_days","away_rest_days"]].min(axis=1)<=4,
                             "either_team_4_or_fewer","neither_team_4_or_fewer"))
    d["diff"]=(d.v3_prob-d.home_win)**2-(d.market_prob-d.home_win)**2
    def metrics(g):
        return {"games":int(len(g)),"v3_minus_market_brier":float(g["diff"].mean()),
                "v3_accuracy":float(((g.v3_prob>=.5)==g.home_win).mean()),
                "market_accuracy":float(((g.market_prob>=.5)==g.home_win).mean())}
    def group(cols):
        return [{**dict(zip(cols,k if isinstance(k,tuple) else (k,))),**metrics(g)}
                for k,g in d.groupby(cols,observed=True,sort=True)]
    return {"total_games":len(d),"overall":metrics(d),
            "gap_groups":group(["gap_group"]),
            "rest_groups":group(["rest_group"]),
            "gap_by_rest":group(["gap_group","rest_group"]),
            "gap_by_short_rest":group(["gap_group","short_rest"]),
            "gap_by_season":group(["season","gap_group"]),
            "warnings":["All eligible games are included as comparison controls",
                        "Rest group cutoffs were pre-specified; exploratory subgroup comparisons only",
                        "No injury or QB evidence; no causal attribution",
                        "Market prices lack verified pregame timestamps; no executable ROI claims"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="research_outputs/nfl_v3_predictions.csv")
    ap.add_argument("--schedules",default=SCHEDULE_URL)
    ap.add_argument("--output",default="research_outputs/nfl_v5_rest_comparison.json")
    a=ap.parse_args()
    r=analyze(pd.read_csv(a.predictions),pd.read_csv(a.schedules))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))
if __name__=="__main__":main()
