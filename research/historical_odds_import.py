"""Parse saved The Odds API historical NFL h2h snapshots without network calls.

Raw API JSON must be provided by the researcher. Parsing is not verification of
availability, execution, independent capture, or profitable model edge.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def parse_snapshot(payload, requested_at):
    if not isinstance(payload,dict) or not isinstance(payload.get("data"),list):
        raise ValueError("Expected historical odds response with data array")
    snap=pd.to_datetime(payload.get("timestamp"),utc=True,errors="coerce")
    req=pd.to_datetime(requested_at,utc=True,errors="coerce")
    if pd.isna(snap) or pd.isna(req):raise ValueError("Invalid snapshot/request timestamp")
    if snap>req:raise ValueError("Snapshot timestamp after requested time")
    rows=[]
    for game in payload["data"]:
        for book in game.get("bookmakers",[]):
            for market in book.get("markets",[]):
                if market.get("key")!="h2h":continue
                prices={o.get("name"):o.get("price") for o in market.get("outcomes",[])}
                home=game.get("home_team");away=game.get("away_team")
                if not home or not away or home==away or home not in prices or away not in prices:continue
                for side,team in [("home",home),("away",away)]:
                    price=prices[team]
                    if isinstance(price,bool) or not isinstance(price,(int,float)) or not float(price).is_integer() or -100<int(price)<100:
                        raise ValueError("Invalid American odds")
                    rows.append({"event_id":game.get("id"),"home_team":home,"away_team":away,
                      "commence_time":game.get("commence_time"),"side":side,
                      "bookmaker_key":book.get("key"),"bookmaker_last_update":book.get("last_update"),
                      "market_last_update":market.get("last_update"),"american_odds":int(price),
                      "snapshot_utc":snap.isoformat(),"requested_utc":req.isoformat(),
                      "source":"the_odds_api_historical_json","source_capture_verified":False,
                      "quote_executable_verified":False,"model_eligible":False})
    return pd.DataFrame(rows,columns=["event_id","home_team","away_team","commence_time","side",
      "bookmaker_key","bookmaker_last_update","market_last_update","american_odds",
      "snapshot_utc","requested_utc","source","source_capture_verified",
      "quote_executable_verified","model_eligible"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--json",required=True,help="Previously saved provider JSON; never API credentials")
    p.add_argument("--requested-at",required=True)
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d=parse_snapshot(json.loads(Path(a.json).read_text()),a.requested_at)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_historical_odds_unverified.csv",index=False)
    summary={"rows":len(d),"distinct_events":int(d.event_id.nunique()),"verified_quotes":0,
      "model_eligible":0,"warning":"Provider snapshot data is not independent verification of execution, event identity matching, or historical model decision availability."}
    (out/"nfl_historical_odds_import_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
