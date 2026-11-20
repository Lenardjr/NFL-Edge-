"""Create credential-free The Odds API request plan from QB research manifest.

Never contacts an API, includes secrets, or represents requested snapshots as odds.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from urllib.parse import urlencode
import pandas as pd

BASE="https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds"
def plan(manifest:pd.DataFrame):
    needed={"game_id","snapshot_window","requested_snapshot_utc","in_priority_queue"}
    if needed-set(manifest):raise ValueError("Missing manifest fields")
    rows=[]
    for _,r in manifest.iterrows():
        t=pd.to_datetime(r.requested_snapshot_utc,utc=True,errors="coerce")
        if pd.isna(t):raise ValueError("Invalid requested time")
        stamp=t.strftime("%Y-%m-%dT%H:%M:%SZ")
        params={"regions":"us","markets":"h2h","oddsFormat":"american","dateFormat":"iso","date":stamp}
        rows.append({"game_id":r.game_id,"snapshot_window":r.snapshot_window,
                     "in_priority_queue":str(r.in_priority_queue).lower()=="true",
                     "requested_snapshot_utc":stamp,
                     "request_url_without_key":BASE+"?"+urlencode(params),
                     "estimated_quota_credits":10,
                     "response_retrieved":False,"verified_quotes":0})
    d=pd.DataFrame(rows)
    unique=int(d.requested_snapshot_utc.nunique()) if len(d) else 0
    s={"requests_in_manifest":len(d),"distinct_timestamp_requests":unique,
       "estimated_credits_without_deduplication":len(d)*10,
       "estimated_credits_with_timestamp_deduplication":unique*10,
       "provider":"The Odds API","requires_paid_historical_access":True,
       "network_calls_made":0,"quotes_retrieved":0,
       "note":"One US region and h2h market: 10 quota credits per historical timestamp request. No API key stored or transmitted."}
    return d,s

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",default="research_outputs/nfl_qb_historical_odds_request_manifest.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=plan(pd.read_csv(a.manifest))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_historical_odds_provider_request_plan.csv",index=False)
    (out/"nfl_historical_odds_provider_request_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
