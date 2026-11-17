"""Collect retrospective QB reference labels; never mark them pregame verified.

The nflverse schedule's home_qb_name/away_qb_name identify retrospective
starters, NOT time-stamped pre-kickoff declarations.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from research.run_nfl_v3 import SCHEDULE_URL

def enrich(queue:pd.DataFrame,schedule:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    required={"game_id","side","verification","eligible_for_model"}
    if required-set(queue):raise ValueError("Missing queue columns")
    if queue.duplicated(["game_id","side"]).any():raise ValueError("Duplicate team-game queue rows")
    fields={"game_id","home_qb_name","away_qb_name"}
    if fields-set(schedule):raise ValueError("Missing schedule QB reference fields")
    if schedule.game_id.duplicated().any():raise ValueError("Duplicate schedule IDs")
    merged=queue.merge(schedule[list(fields)],on="game_id",how="left",validate="many_to_one",indicator=True)
    if not (merged._merge=="both").all():raise ValueError("Unmatched game IDs")
    merged["historical_qb_name"]=merged.apply(
        lambda r:r.home_qb_name if r.side=="home" else r.away_qb_name,axis=1)
    merged["historical_qb_name"]=merged.historical_qb_name.fillna("")
    merged["historical_qb_source"]="nflverse schedules (retrospective, no pregame timestamp)"
    merged["pregame_source_verified"]=False
    merged["verification"]="unreviewed"
    merged["eligible_for_model"]=False
    merged=merged.drop(columns=["home_qb_name","away_qb_name","_merge"])
    filled=int(merged.historical_qb_name.astype(str).str.strip().ne("").sum())
    summary={"team_games":len(merged),"retrospective_qb_labels":filled,
        "missing_retrospective_labels":len(merged)-filled,"verified_pregame":0,
        "model_eligible":0,"warning":"Retrospective starting QB is a research lead, never a pregame feature."}
    return merged,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--queue",default="research_outputs/nfl_qb_pregame_verification_queue.csv")
    p.add_argument("--schedule",default=SCHEDULE_URL)
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    enriched,summary=enrich(pd.read_csv(a.queue),pd.read_csv(a.schedule))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    enriched.to_csv(out/"nfl_qb_retrospective_reference.csv",index=False)
    (out/"nfl_qb_reference_coverage.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
