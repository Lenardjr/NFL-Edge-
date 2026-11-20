"""Research-only NFL moneyline snapshot from The Odds API.

Never print the API key. Never overwrite a previous snapshot. No betting execution.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,urllib.parse,urllib.request
from pathlib import Path
import pandas as pd

BASE="https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
ALIASES={"WSH":"WAS","JAC":"JAX","LA":"LAR"}
TEAMS={"ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens",
"BUF":"Buffalo Bills","CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals",
"CLE":"Cleveland Browns","DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions",
"GB":"Green Bay Packers","HOU":"Houston Texans","IND":"Indianapolis Colts",
"JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs","LV":"Las Vegas Raiders",
"LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","MIA":"Miami Dolphins",
"MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints",
"NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles",
"PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks","SF":"San Francisco 49ers",
"TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders"}
def normalize(x):return ALIASES.get(str(x).upper(),str(x).upper())
def implied(american):
    a=float(american)
    if a==0:raise ValueError("Zero American odds")
    return 100/(a+100) if a>0 else -a/(-a+100)
def profit(american):
    a=float(american)
    if a==0:raise ValueError("Zero American odds")
    return a/100 if a>0 else 100/-a
def parse_snapshot(cohort,events,now):
    now=pd.Timestamp(now)
    if now.tzinfo is None:raise ValueError("Timestamp must be timezone-aware")
    if cohort.game_id.duplicated().any():raise ValueError("Duplicate cohort IDs")
    rows=[]
    for p in cohort.itertuples(index=False):
        kick=pd.Timestamp(p.kickoff_utc)
        if kick.tzinfo is None:raise ValueError("Invalid cohort kickoff")
        if now>=kick:continue # Never claim post-kickoff prices are pregame
        home=TEAMS[normalize(p.home_team)];away=TEAMS[normalize(p.away_team)]
        matches=[e for e in events if e.get("home_team")==home and e.get("away_team")==away
                 and abs(pd.Timestamp(e["commence_time"])-kick)<=pd.Timedelta(hours=24)]
        if len(matches)>1:raise ValueError("Ambiguous odds event "+str(p.game_id))
        if not matches:continue
        e=matches[0]
        prob=float(p.home_win_probability)
        if not 0<prob<1:raise ValueError("Invalid model probability")
        for book in e.get("bookmakers",[]):
            for market in book.get("markets",[]):
                if market.get("key")!="h2h":continue
                prices={o.get("name"):o.get("price") for o in market.get("outcomes",[])}
                if home not in prices or away not in prices:continue
                try:
                    hp,ap=implied(prices[home]),implied(prices[away])
                    if hp+ap<=0:continue
                    he=prob*profit(prices[home])-(1-prob)
                    ae=(1-prob)*profit(prices[away])-prob
                except (TypeError,ValueError,ZeroDivisionError):continue
                rows.append({"game_id":p.game_id,"kickoff_utc":p.kickoff_utc,
                  "home_team":p.home_team,"away_team":p.away_team,
                  "model_home_win_probability":prob,"sportsbook":book.get("key",""),
                  "sportsbook_title":book.get("title",""),
                  "market_last_update_utc":market.get("last_update",book.get("last_update","")),
                  "home_moneyline_american":prices[home],"away_moneyline_american":prices[away],
                  "home_implied_probability":hp,"away_implied_probability":ap,
                  "home_no_vig_probability":hp/(hp+ap),
                  "home_ev_per_unit":he,"away_ev_per_unit":ae,
                  "retrieved_at_utc":now.isoformat(),"odds_event_id":e.get("id",""),
                  "research_only":True})
    return pd.DataFrame(rows)

def collect(cohort_path,output_dir):
    key=os.environ.get("ODDS_API_KEY")
    if not key:raise RuntimeError("ODDS_API_KEY GitHub Actions secret is missing")
    raw=Path(cohort_path).read_bytes();cohort=pd.read_csv(cohort_path)
    now=pd.Timestamp(datetime.datetime.now(datetime.timezone.utc))
    params=urllib.parse.urlencode({"apiKey":key,"regions":"us","markets":"h2h",
                                    "oddsFormat":"american","dateFormat":"iso"})
    req=urllib.request.Request(BASE+"?"+params,headers={"User-Agent":"NFL-Edge-research/1.0"})
    with urllib.request.urlopen(req,timeout=30) as response:
        body=response.read()
        quota={k:response.headers.get(k) for k in ("x-requests-remaining","x-requests-used","x-requests-last")}
    events=json.loads(body)
    if not isinstance(events,list):raise ValueError("Unexpected odds API response")
    rows=parse_snapshot(cohort,events,now)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    stamp=now.strftime("%Y%m%dT%H%M%S%fZ")
    rows.to_csv(out/("moneyline_snapshot_"+stamp+".csv"),index=False)
    manifest={"retrieved_at_utc":now.isoformat(),"cohort_sha256":hashlib.sha256(raw).hexdigest(),
      "odds_response_sha256":hashlib.sha256(body).hexdigest(),
      "source":"The Odds API v4 NFL US h2h American",
      "matched_quotes":len(rows),"unique_games":int(rows.game_id.nunique()) if len(rows) else 0,
      "quota":quota,"key_redacted":True,"pregame_only":True,
      "note":"Model EV is hypothetical and uncalibrated; it does not establish profitable edge."}
    (out/("moneyline_manifest_"+stamp+".json")).write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cohort",required=True);p.add_argument("--output-dir",required=True)
    a=p.parse_args();collect(a.cohort,a.output_dir)
