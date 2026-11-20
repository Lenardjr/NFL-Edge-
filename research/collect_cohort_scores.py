"""Collect ESPN scoreboard results only when ESPN marks a game completed.

Research-only: no bets, no edits to frozen predictions, no assumed final scores.
ESPN is an external scoreboard provider; independent league verification is separate.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pandas as pd

URL="https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ALIASES={"WSH":"WAS","JAC":"JAX","LA":"LAR"}
def norm(x):
    x=str(x).upper()
    return ALIASES.get(x,x)
def parse_events(payload):
    out=[]
    for event in payload.get("events",[]):
        for competition in event.get("competitions",[]):
            sides={}
            for team in competition.get("competitors",[]):
                role=team.get("homeAway")
                if role in ("home","away"):sides[role]=team
            if set(sides)!={"home","away"}:continue
            completed=bool(competition.get("status",event.get("status",{})).get("type",{}).get("completed",False))
            date=competition.get("date",event.get("date"))
            if not date:continue
            home=norm(sides["home"]["team"]["abbreviation"])
            away=norm(sides["away"]["team"]["abbreviation"])
            def score(s):
                try:return int(s["score"])
                except (KeyError,TypeError,ValueError):return None
            out.append({"event_id":str(event.get("id","")),"kickoff":pd.Timestamp(date).tz_convert("UTC"),
                        "home_team":home,"away_team":away,
                        "home_score":score(sides["home"]),"away_score":score(sides["away"]),
                        "completed":completed})
    return out

def match_results(cohort,events,*,now_utc):
    now=pd.Timestamp(now_utc)
    if now.tzinfo is None:raise ValueError("now_utc must be timezone aware")
    now=now.tz_convert("UTC")
    if cohort.game_id.isna().any() or cohort.game_id.duplicated().any():raise ValueError("Invalid cohort IDs")
    rows=[];used=set()
    for row in cohort.itertuples(index=False):
        kickoff=pd.Timestamp(row.kickoff_utc)
        if kickoff.tzinfo is None:raise ValueError("Cohort kickoff must be timezone aware")
        kickoff=kickoff.tz_convert("UTC")
        home,away=norm(row.home_team),norm(row.away_team)
        matches=[e for e in events if e["home_team"]==home and e["away_team"]==away
                 and abs(e["kickoff"]-kickoff)<=pd.Timedelta(hours=24)]
        if len(matches)>1:raise ValueError("Ambiguous scoreboard match: "+str(row.game_id))
        m=matches[0] if matches else None
        if m and m["event_id"] in used:raise ValueError("Scoreboard event matched twice")
        if m:used.add(m["event_id"])
        final=bool(m and m["completed"] and m["home_score"] is not None
                   and m["away_score"] is not None and m["kickoff"]<now)
        rows.append({"game_id":row.game_id,"home_team":row.home_team,"away_team":row.away_team,
                     "home_score":m["home_score"] if final else None,
                     "away_score":m["away_score"] if final else None,
                     "status":"final" if final else "unverified",
                     "source":"ESPN scoreboard API" if m else "",
                     "source_event_id":m["event_id"] if m else ""})
    return pd.DataFrame(rows)

def collect(*,cohort_csv,output_dir,now_utc=None):
    path=Path(cohort_csv);raw=path.read_bytes();cohort=pd.read_csv(path)
    now=pd.Timestamp(now_utc or datetime.now(timezone.utc).isoformat())
    # Query each distinct UTC date plus adjacent dates to handle timezone shifts.
    dates=set()
    for kickoff in pd.to_datetime(cohort.kickoff_utc,utc=True):
        for offset in (-1,0,1):dates.add((kickoff+pd.Timedelta(days=offset)).strftime("%Y%m%d"))
    events=[];sources=[]
    for date in sorted(dates):
        url=URL+"?dates="+date+"&limit=100"
        req=urllib.request.Request(url,headers={"User-Agent":"NFL-Edge-research/1.0"})
        with urllib.request.urlopen(req,timeout=20) as response:body=response.read()
        payload=json.loads(body)
        events.extend(parse_events(payload))
        sources.append({"date":date,"url":url,"sha256":hashlib.sha256(body).hexdigest()})
    # Multiple adjacent daily queries can return the same event; dedupe by event ID.
    unique={}
    for e in events:
        if e["event_id"] and e["event_id"] in unique:
            if e!=unique[e["event_id"]]:raise ValueError("Conflicting scoreboard responses")
        else:unique[e["event_id"]]=e
    results=match_results(cohort,list(unique.values()),now_utc=now)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    results.to_csv(out/"cohort_verified_results.csv",index=False)
    manifest={"retrieved_at_utc":now.isoformat(),"cohort_sha256":hashlib.sha256(raw).hexdigest(),
              "source":"ESPN public scoreboard API","source_queries":sources,
              "completed_games":int((results.status=="final").sum()),
              "games_in_cohort":len(results),
              "league_independently_verified":False,"market_prices_verified":False,
              "note":"Only ESPN completed events with matching teams and kickoff are scored; provider snapshots are not a sportsbook market."}
    (out/"scoreboard_source_manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest

def main():
    a=argparse.ArgumentParser();a.add_argument("--cohort",required=True);a.add_argument("--output-dir",required=True)
    args=a.parse_args();print(json.dumps(collect(cohort_csv=args.cohort,output_dir=args.output_dir),indent=2))
if __name__=="__main__":main()
