"""Generate an outcome-blind, season-balanced research queue for extreme games.

The queue is for manual source verification, not a deployable betting strategy.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def make_queue(cases:pd.DataFrame,limit_per_season:int=10)->tuple[pd.DataFrame,dict]:
    required={"game_id","season","week","gameday","home_team","away_team",
              "v3_prob","market_prob","absolute_gap","home_rest_days","away_rest_days"}
    missing=required-set(cases)
    if missing:raise ValueError("Missing case columns: "+str(sorted(missing)))
    if cases.game_id.isna().any() or cases.game_id.duplicated().any():
        raise ValueError("Invalid game IDs")
    if limit_per_season<1:raise ValueError("Invalid limit")
    if cases[list(required-{"home_rest_days","away_rest_days"})].isna().any().any():
        raise ValueError("Missing game metadata")
    if not np.isfinite(cases[["v3_prob","market_prob","absolute_gap"]].to_numpy(dtype=float)).all():
        raise ValueError("Invalid probabilities")
    if not cases[["v3_prob","market_prob"]].apply(lambda s:s.between(0,1)).all().all():
        raise ValueError("Probabilities outside [0,1]")
    if not np.allclose((cases.v3_prob-cases.market_prob).abs(),cases.absolute_gap,atol=1e-8):
        raise ValueError("Probability gap inconsistent")
    d=cases[cases.absolute_gap>=.20].copy()
    d=d.sort_values(["season","absolute_gap","game_id"],ascending=[True,False,True],kind="stable")
    d=d.groupby("season",group_keys=False,sort=True).head(limit_per_season)
    d["rest_advantage_home"]=d.home_rest_days-d.away_rest_days
    d["source_verification_status"]="not_started"
    d["qb_home_source"]="";d["qb_away_source"]=""
    d["injury_source"]="";d["kickoff_time_source"]=""
    d["pregame_publication_verified"]=False
    d["review_notes"]=""
    cols=["game_id","season","week","gameday","away_team","home_team",
          "v3_prob","market_prob","absolute_gap","home_rest_days","away_rest_days",
          "rest_advantage_home","source_verification_status","qb_home_source",
          "qb_away_source","injury_source","kickoff_time_source",
          "pregame_publication_verified","review_notes"]
    summary={"eligible_extreme_games":int((cases.absolute_gap>=.20).sum()),
             "queued":len(d),"limit_per_season":limit_per_season,
             "queued_by_season":{str(int(k)):len(g) for k,g in d.groupby("season")},
             "notes":["Queue selection depends only on pregame-model/market disagreement, not game results",
                      "QB and injury fields are intentionally empty until dated sources are checked",
                      "Historical odds lack verified capture times; no actionable ROI inference"]}
    return d[cols].reset_index(drop=True),summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",default="research_outputs/nfl_v5_extreme_cases_with_matchups.csv")
    ap.add_argument("--output-dir",default="research_outputs")
    a=ap.parse_args()
    queue,summary=make_queue(pd.read_csv(a.cases))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    queue.to_csv(out/"nfl_v5_extreme_pregame_research_queue.csv",index=False)
    (out/"nfl_v5_extreme_pregame_queue_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
