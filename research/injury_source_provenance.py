"""Audit historical injury-source provenance claims before allowing feature use.

Asset upload times and release dates are leads, not proof that report content
was known at a model's actual decision timestamp.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

SOURCE_CANDIDATES=[
 {"source":"nflverse_2024_season_csv","url":"https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2024.csv",
  "historical_snapshot":False,"archive_verified":False,
  "note":"Retrospective season CSV; final asset uploaded after 2024 season"},
 {"source":"nflverse_injurybot_game_assets","url":"https://github.com/nflverse/nflverse-injurybot/releases",
  "historical_snapshot":False,"archive_verified":False,
  "note":"Individual game assets may have upload times; verify exact asset, content, matchup, and capture before using"},
 {"source":"contemporaneous_team_report","url":"",
  "historical_snapshot":False,"archive_verified":False,
  "note":"Require independent dated report and verifiable availability before prediction cutoff"}
]
def audit(queue:pd.DataFrame):
    required={"game_id","season","week","priority_rank","absolute_gap"}
    if required-set(queue):raise ValueError("Missing priority fields")
    if queue.game_id.isna().any() or queue.game_id.duplicated().any():
        raise ValueError("Invalid priority IDs")
    q=queue.copy()
    q["source_candidate"]="nflverse_injurybot_game_assets"
    q["source_asset_url"]=""
    q["source_asset_uploaded_at"]=""
    q["source_content_timestamp"]=""
    q["independent_archive_url"]=""
    q["independent_archive_at"]=""
    q["prediction_decision_at"]=""
    q["source_verified_before_decision"]=False
    q["model_eligible"]=False
    summary={"games_queued":len(q),"source_candidates":SOURCE_CANDIDATES,
      "verified_before_prediction":0,"model_eligible":0,
      "requirements":["Match exact game and team","Verify asset content and upload time",
        "Verify an independent capture or trustworthy contemporaneous publication",
        "Establish prediction decision time and compare timestamps",
        "Exclude retrospective or post-decision data"],
      "warning":"A source upload timestamp alone does not authenticate the historical content or prove availability before prediction."}
    return q,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--priority",default="research_outputs/nfl_2024_qb_priority_games.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    queue,summary=audit(pd.read_csv(a.priority))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    queue.to_csv(out/"nfl_2024_qb_provenance_investigation.csv",index=False)
    (out/"nfl_2024_qb_provenance_readiness.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({"games_queued":summary["games_queued"],"verified_before_prediction":0,"model_eligible":0},indent=2))
if __name__=="__main__":main()
