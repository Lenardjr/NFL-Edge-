"""Auditable market disagreement report; no recommendations or wagers."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def summarize(quotes,*,max_age_minutes=90):
    required={"game_id","home_team","away_team","kickoff_utc","retrieved_at_utc",
      "market_last_update_utc","sportsbook","model_home_win_probability",
      "home_no_vig_probability","home_ev_per_unit","away_ev_per_unit"}
    if required-set(quotes.columns):raise ValueError("Missing quote fields: "+str(sorted(required-set(quotes.columns))))
    rows=[]
    for game,g in quotes.groupby("game_id",sort=True):
        if g.sportsbook.duplicated().any():raise ValueError("Duplicate sportsbook quote for "+str(game))
        if g.model_home_win_probability.nunique()!=1:raise ValueError("Inconsistent model probabilities")
        if g.kickoff_utc.nunique()!=1:raise ValueError("Inconsistent kickoff")
        now=pd.to_datetime(g.retrieved_at_utc,utc=True)
        update=pd.to_datetime(g.market_last_update_utc,utc=True,errors="coerce")
        kick=pd.to_datetime(g.kickoff_utc,utc=True)
        fresh=(update.notna() & (update<=now) & ((now-update)<=pd.Timedelta(minutes=max_age_minutes))
               & (now<kick))
        x=g.loc[fresh].copy()
        model=float(g.model_home_win_probability.iloc[0])
        consensus=float(x.home_no_vig_probability.median()) if len(x) else None
        rows.append({"game_id":game,"matchup":str(g.away_team.iloc[0])+" @ "+str(g.home_team.iloc[0]),
          "kickoff_utc":g.kickoff_utc.iloc[0],"model_home_probability":model,
          "fresh_books":len(x),"total_books":len(g),
          "market_consensus_home_no_vig":consensus,
          "model_minus_market_pp":round(100*(model-consensus),2) if consensus is not None else None,
          "best_home_ev_estimate":float(x.home_ev_per_unit.max()) if len(x) else None,
          "best_away_ev_estimate":float(x.away_ev_per_unit.max()) if len(x) else None,
          "best_home_book":str(x.loc[x.home_ev_per_unit.idxmax(),"sportsbook"]) if len(x) else None,
          "best_away_book":str(x.loc[x.away_ev_per_unit.idxmax(),"sportsbook"]) if len(x) else None,
          "review_flag":"INSUFFICIENT_FRESH_QUOTES" if len(x)<3 else
             ("LARGE_MODEL_MARKET_GAP" if abs(model-consensus)>=.15 else "MONITOR"),
          "profitable_edge_verified":False})
    return pd.DataFrame(rows)

def build(folder):
    folder=Path(folder)
    snapshots=sorted(folder.glob("moneyline_snapshot_*.csv"))
    if not snapshots:raise ValueError("No snapshot CSV in "+str(folder))
    latest=snapshots[-1]
    quotes=pd.read_csv(latest)
    if quotes.empty:raise ValueError("Snapshot has no matching quotes")
    report=summarize(quotes)
    report.to_csv(folder/"market_disagreement_report.csv",index=False)
    summary={"snapshot_file":latest.name,"games_with_quotes":int(report.shape[0]),
      "games_with_at_least_three_fresh_books":int((report.fresh_books>=3).sum()),
      "large_disagreement_games":int((report.review_flag=="LARGE_MODEL_MARKET_GAP").sum()),
      "market_consensus_method":"median no-vig home win probability across books with market updates <=90 minutes old",
      "freshness_minutes":90,"model_calibration_verified":False,"profitable_edge_verified":False,
      "warning":"Hypothetical EV is NOT a betting recommendation. Large gaps require independent model calibration and source checks."}
    (folder/"market_disagreement_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--snapshot-dir",required=True)
    a=p.parse_args();build(a.snapshot_dir)
