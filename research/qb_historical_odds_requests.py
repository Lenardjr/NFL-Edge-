"""Prepare historical moneyline snapshot requests for documented QB announcements.

Produces a research request manifest only; never fabricates or downloads odds.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def build_manifest(review:pd.DataFrame):
    required={"game_id","side","publisher_displayed_utc","kickoff_utc","in_priority_queue"}
    if required-set(review):raise ValueError("Missing publisher review fields")
    if review.duplicated(["game_id","side"]).any():raise ValueError("Duplicate game-side")
    rows=[]
    for _,r in review.iterrows():
        pub=pd.to_datetime(r.publisher_displayed_utc,utc=True,errors="coerce")
        kick=pd.to_datetime(r.kickoff_utc,utc=True,errors="coerce")
        if pd.isna(pub) or pd.isna(kick) or pub>=kick:continue
        for label,offset in [("pre_24h",-24),("pre_1h",-1),("post_1h",1),("post_24h",24)]:
            at=pub+pd.Timedelta(hours=offset)
            if at>=kick:continue
            rows.append({"game_id":r.game_id,"qb_side":r.side,
                         "in_priority_queue":str(r.in_priority_queue).lower()=="true",
                         "snapshot_window":label,"requested_snapshot_utc":at.isoformat(),
                         "announcement_displayed_utc":pub.isoformat(),"kickoff_utc":kick.isoformat(),
                         "market":"h2h","sport":"americanfootball_nfl",
                         "bookmaker":"","home_american_odds":"","away_american_odds":"",
                         "source_snapshot_url":"","source_capture_verified":False,
                         "quote_executable_verified":False,"model_eligible":False,
                         "status":"request_only_no_historical_quote"})
    d=pd.DataFrame(rows,columns=["game_id","qb_side","in_priority_queue","snapshot_window",
       "requested_snapshot_utc","announcement_displayed_utc","kickoff_utc","market","sport",
       "bookmaker","home_american_odds","away_american_odds","source_snapshot_url",
       "source_capture_verified","quote_executable_verified","model_eligible","status"])
    summary={"announcement_cases":int(review.shape[0]),"cases_with_requested_snapshots":int(d.game_id.nunique()),
             "snapshot_requests":len(d),"priority_snapshot_requests":int(d.in_priority_queue.sum()) if len(d) else 0,
             "verified_quotes":0,"model_eligible":0,
             "warning":"Requested times are research targets, not retrieved historical snapshots. No bookmaker prices, execution or ROI implied."}
    return d,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--review",default="research_outputs/nfl_qb_publisher_timestamp_review.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=build_manifest(pd.read_csv(a.review))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_qb_historical_odds_request_manifest.csv",index=False)
    (out/"nfl_qb_historical_odds_request_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
