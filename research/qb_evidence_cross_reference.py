"""Cross-reference documented QB announcements with prioritized research games.

This is evidence triage, not a verified historical model feature.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

def cross_reference(priority:pd.DataFrame,evidence:pd.DataFrame):
    required_priority={"game_id","priority_rank","absolute_gap","season","week"}
    required_evidence={"game_id","side","qb_name","source_url","published_date","game_date"}
    if required_priority-set(priority) or required_evidence-set(evidence):
        raise ValueError("Missing required evidence or priority columns")
    if priority.game_id.isna().any() or priority.game_id.duplicated().any():
        raise ValueError("Invalid priority game IDs")
    if evidence.duplicated(["game_id","side"]).any():
        raise ValueError("Duplicate evidence game-side")
    if evidence[["game_id","side"]].isna().any().any():
        raise ValueError("Missing evidence identity")
    e=evidence.copy()
    for url in e.source_url:
        parsed=urlparse(str(url))
        if parsed.scheme!="https" or not parsed.hostname:
            raise ValueError("Invalid evidence URL")
    pub=pd.to_datetime(e.published_date,errors="coerce")
    game=pd.to_datetime(e.game_date,errors="coerce")
    e["publication_date_before_game"]=pub.notna() & game.notna() & pub.lt(game)
    e["archive_capture_verified"]=False
    e["prediction_cutoff_verified"]=False
    e["model_eligible"]=False
    e["review_status"]="requires_capture_and_decision_timestamp"
    e.loc[~e.publication_date_before_game,"review_status"]="publication_date_not_before_game"
    result=e.merge(priority[["game_id","priority_rank","absolute_gap","season","week"]],
                   on="game_id",how="left",validate="many_to_one",indicator=True)
    result["in_priority_queue"]=result["_merge"].eq("both")
    result=result.drop(columns=["_merge"])
    result=result.sort_values(["in_priority_queue","priority_rank","game_id","side"],
                              ascending=[False,True,True,True],na_position="last").reset_index(drop=True)
    summary={"documented_qb_examples":len(result),
             "examples_matching_priority_games":int(result.in_priority_queue.sum()),
             "examples_outside_priority_games":int((~result.in_priority_queue).sum()),
             "publication_dates_before_game":int(result.publication_date_before_game.sum()),
             "archive_capture_verified":0,"prediction_cutoff_verified":0,"model_eligible":0,
             "warning":"A publication date alone cannot establish availability before a model prediction or contemporaneous odds."}
    return result,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--priority",default="research_outputs/nfl_2024_qb_priority_games.csv")
    p.add_argument("--evidence",default="research_outputs/nfl_qb_documented_pregame_examples.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    result,summary=cross_reference(pd.read_csv(a.priority),pd.read_csv(a.evidence))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/"nfl_qb_documented_priority_cross_reference.csv",index=False)
    (out/"nfl_qb_documented_priority_cross_reference_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
