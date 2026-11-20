"""Compare discovered injurybot asset upload times with actual game kickoffs.

A pre-kickoff upload is necessary but insufficient for historical model eligibility.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def evaluate(assets:pd.DataFrame, qb_queue:pd.DataFrame):
    required={"game_id","asset_found","asset_uploaded_at","asset_name","asset_url"}
    if required-set(assets):raise ValueError("Missing asset metadata")
    if not {"game_id","kickoff_utc"}<=set(qb_queue):raise ValueError("Missing QB kickoff fields")
    if assets.game_id.isna().any() or assets.game_id.duplicated().any():
        raise ValueError("Invalid asset game IDs")
    k=qb_queue[["game_id","kickoff_utc"]].drop_duplicates()
    if k.game_id.duplicated().any():raise ValueError("Conflicting kickoff values")
    d=assets.merge(k,on="game_id",how="left",validate="one_to_one",indicator=True)
    if not d._merge.eq("both").all():raise ValueError("Unmatched game kickoff")
    d=d.drop(columns=["_merge"])
    d["kickoff_parsed"]=pd.to_datetime(d.kickoff_utc,utc=True,errors="coerce")
    d["upload_parsed"]=pd.to_datetime(d.asset_uploaded_at,utc=True,errors="coerce")
    if d.kickoff_parsed.isna().any():raise ValueError("Unparseable kickoff")
    if (d.asset_found & d.upload_parsed.isna()).any():raise ValueError("Matched asset missing upload time")
    d["upload_hours_before_kickoff"]=(d.kickoff_parsed-d.upload_parsed).dt.total_seconds()/3600
    d["uploaded_before_kickoff"]=d.asset_found & d.upload_hours_before_kickoff.gt(0)
    d["uploaded_at_or_after_kickoff"]=d.asset_found & d.upload_hours_before_kickoff.le(0)
    d["content_inspected"]=False
    d["independent_capture_verified"]=False
    d["decision_time_verified"]=False
    d["model_eligible"]=False
    summary={"priority_games":len(d),"matched_assets":int(d.asset_found.sum()),
      "uploaded_before_kickoff":int(d.uploaded_before_kickoff.sum()),
      "uploaded_at_or_after_kickoff":int(d.uploaded_at_or_after_kickoff.sum()),
      "unmatched_assets":int((~d.asset_found).sum()),"model_eligible":0,
      "warning":"Release upload times do not prove file content was available before prediction decision time; no automatic feature promotion."}
    d=d.drop(columns=["kickoff_parsed","upload_parsed"])
    return d,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",default="research_outputs/nfl_2024_qb_asset_metadata.csv")
    p.add_argument("--qb-queue",default="research_outputs/nfl_2024_qb_source_verification_queue.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=evaluate(pd.read_csv(a.assets),pd.read_csv(a.qb_queue))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_2024_qb_asset_kickoff_timing.csv",index=False)
    (out/"nfl_2024_qb_asset_kickoff_timing_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
