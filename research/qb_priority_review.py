"""Outcome-blind priority queue for verified QB source investigation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def prioritize(cases:pd.DataFrame,qbs:pd.DataFrame,threshold:float=.20):
    required={"game_id","v3_prob","market_prob","gameday","home_team","away_team"}
    if required-set(cases):raise ValueError("Missing case fields")
    if cases.game_id.duplicated().any():raise ValueError("Duplicate case IDs")
    if qbs.duplicated(["game_id","side"]).any():raise ValueError("Duplicate QB sides")
    c=cases.copy()
    c["gap"]=abs(pd.to_numeric(c.v3_prob)-pd.to_numeric(c.market_prob))
    if c.gap.isna().any():raise ValueError("Missing probability gap")
    c=c[c.gap>=threshold-1e-12].sort_values(["gap","game_id"],ascending=[False,True])
    cols=["game_id","side","historical_qb_name","verification","eligible_for_model"]
    if set(cols)-set(qbs):raise ValueError("Missing QB reference fields")
    rows=c.merge(qbs[cols],on="game_id",how="left",validate="one_to_many")
    if len(rows)!=2*len(c) or rows.side.isna().any():raise ValueError("Incomplete QB sides")
    if rows.groupby("game_id").side.nunique().ne(2).any():raise ValueError("Invalid QB sides")
    # Historical game results are deliberately omitted from the triage output.
    output=["game_id","gameday","home_team","away_team","gap","v3_prob","market_prob",
            "side","historical_qb_name","verification","eligible_for_model"]
    rows=rows[output].copy()
    rows["pregame_source_url"]=""
    rows["pregame_source_published_at"]=""
    rows["pregame_source_archived_at"]=""
    rows["kickoff_timestamp"]=""
    rows["source_review_notes"]=""
    rows["source_review_status"]="needs_independent_pregame_evidence"
    rows["eligible_for_model"]=False
    return rows,{"priority_games":len(c),"team_game_reviews":len(rows),
                 "verified_pregame":0,"model_eligible":0,
                 "selection":"absolute V3 versus market gap >= 0.20, independent of game outcome",
                 "warning":"Retrospective QB names are leads, not verified pregame information."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cases",default="research_outputs/nfl_v5_extreme_cases_with_matchups.csv")
    p.add_argument("--qbs",default="research_outputs/nfl_qb_retrospective_reference.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    rows,summary=prioritize(pd.read_csv(a.cases),pd.read_csv(a.qbs))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows.to_csv(out/"nfl_qb_priority_source_review.csv",index=False)
    (out/"nfl_qb_priority_source_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
