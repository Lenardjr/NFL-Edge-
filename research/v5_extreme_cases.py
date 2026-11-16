"""Export auditable extreme model-vs-market disagreements for manual pregame research.

Selection is based only on model and market probabilities, never outcomes.
Outcomes are included as evaluation labels, not as evidence of why a miss occurred.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

REVIEW_FIELDS=["game_id","season","week","home_win","v3_prob","market_prob",
               "signed_gap","absolute_gap","v3_pick_home","market_pick_home",
               "v3_brier","market_brier","brier_difference","review_priority",
               "qb_home_status","qb_away_status","injury_notes","rest_travel_notes",
               "source_urls","source_published_at","review_status"]
def build_cases(predictions:pd.DataFrame,threshold:float=.10):
    required={"game_id","season","week","home_win","v3_prob","market_prob"}
    missing=required-set(predictions)
    if missing:raise ValueError("Missing columns: "+str(sorted(missing)))
    if predictions.game_id.isna().any() or predictions.game_id.duplicated().any():
        raise ValueError("Missing or duplicate game IDs")
    if predictions[list(required)].isna().any().any():raise ValueError("Missing observations")
    if not predictions.home_win.isin([0,1]).all():raise ValueError("Invalid outcomes")
    for col in ("v3_prob","market_prob"):
        p=predictions[col].to_numpy(dtype=float)
        if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
            raise ValueError("Invalid probability "+col)
    if not 0<threshold<=1:raise ValueError("Invalid threshold")
    d=predictions.copy()
    d["signed_gap"]=d.v3_prob-d.market_prob
    d["absolute_gap"]=d.signed_gap.abs()
    d=d[d.absolute_gap>=threshold].copy()
    d["v3_pick_home"]=d.v3_prob>=.5
    d["market_pick_home"]=d.market_prob>=.5
    d["v3_brier"]=(d.v3_prob-d.home_win)**2
    d["market_brier"]=(d.market_prob-d.home_win)**2
    d["brier_difference"]=d.v3_brier-d.market_brier
    d["review_priority"]=np.where(d.absolute_gap>=.20,"20+ percentage points","10-20 percentage points")
    for col in ("qb_home_status","qb_away_status","injury_notes","rest_travel_notes",
                "source_urls","source_published_at"):
        d[col]=""
    d["review_status"]="unreviewed"
    d=d.sort_values(["absolute_gap","season","week","game_id"],
                    ascending=[False,True,True,True],kind="stable")
    return d[REVIEW_FIELDS].reset_index(drop=True)

def summarize(cases:pd.DataFrame)->dict:
    by_season=[]
    for season,g in cases.groupby("season",sort=True):
        by_season.append({"season":int(season),"cases":len(g),
            "v3_minus_market_brier":float(g.brier_difference.mean()),
            "v3_correct":int(((g.v3_pick_home)==g.home_win).sum()),
            "market_correct":int(((g.market_pick_home)==g.home_win).sum())})
    return {"cases":len(cases),"threshold_absolute_probability_gap":.10,
            "by_season":by_season,
            "selection_rule":"Absolute V3-minus-market probability >= 10 percentage points; outcome-blind",
            "review_rule":"Fill historical status only with independently verified sources published before kickoff; leave unknown blank",
            "caution":"Retrospective error review is hypothesis-generating, not a validated betting filter. Historical odds lack verified quote times."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",default="research_outputs/nfl_v3_predictions.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    cases=build_cases(pd.read_csv(a.predictions))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    cases.to_csv(out/"nfl_v5_extreme_case_review.csv",index=False)
    report=summarize(cases)
    (out/"nfl_v5_extreme_case_summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
