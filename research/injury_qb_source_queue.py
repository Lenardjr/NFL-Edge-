"""Prepare source-verification queue for 2024 QB injury reports in extreme-gap games.

A report timestamp alone is not evidence of contemporaneous public availability.
This module deliberately does not assign verified status or model eligibility.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def build_queue(reviews:pd.DataFrame):
    required={"game_id","team","side","date_modified","kickoff_utc","lead_hours","timing_class"}
    if required-set(reviews):raise ValueError("Missing review fields")
    if reviews.empty:
        return pd.DataFrame(columns=["game_id","team","side","date_modified","kickoff_utc","lead_hours",
            "timing_class","player","report_status","source_url","source_published_at",
            "archive_capture_url","archive_capture_at","source_verification","model_eligible"]),{
            "review_rows":0,"review_games":0,"independently_verified":0,"model_eligible":0}
    d=reviews.copy()
    if d[["game_id","team","side"]].isna().any().any():raise ValueError("Missing review identity")
    if not d.side.isin(["home","away"]).all():raise ValueError("Invalid side")
    if d.duplicated(["game_id","team","side","date_modified","full_name"] if "full_name" in d else ["game_id","team","side","date_modified"]).any():
        raise ValueError("Duplicate QB review row")
    d["player"]=d.full_name.fillna("").astype(str) if "full_name" in d else ""
    if "report_status" not in d:d["report_status"]=""
    d["source_url"]=""
    d["source_published_at"]=""
    d["archive_capture_url"]=""
    d["archive_capture_at"]=""
    d["source_verification"]="pending_independent_source"
    d["model_eligible"]=False
    d=d.sort_values(["game_id","team","date_modified","player"],kind="stable")
    cols=["game_id","team","side","player","report_status","date_modified","kickoff_utc",
          "lead_hours","timing_class","source_url","source_published_at",
          "archive_capture_url","archive_capture_at","source_verification","model_eligible"]
    summary={"review_rows":len(d),"review_games":int(d.game_id.nunique()),
             "independently_verified":0,"model_eligible":0,
             "next_step":"Find contemporaneous reporting and independent capture provenance; verify against kickoff and prediction decision time before feature use."}
    return d[cols],summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--review",default="research_outputs/nfl_2024_qb_extreme_injury_review.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    queue,summary=build_queue(pd.read_csv(a.review))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    queue.to_csv(out/"nfl_2024_qb_source_verification_queue.csv",index=False)
    (out/"nfl_2024_qb_source_verification_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
