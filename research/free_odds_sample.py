"""Validate the free The Odds Gap NFL moneyline sample; no credentials or network.

The sample is October 19-25 2026 only. Captured odds are NOT evidence that
our model predicted a game before that capture or that a bet was executed.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

REQUIRED={"snapshot_ts","sport","away","home","commence_time","market",
          "side","book","american_odds"}
START=pd.Timestamp("2026-10-19",tz="UTC")
END=pd.Timestamp("2026-10-27",tz="UTC")

def validate_sample(frame:pd.DataFrame):
    missing=REQUIRED-set(frame.columns)
    if missing:raise ValueError("Missing odds columns: "+",".join(sorted(missing)))
    d=frame.copy()
    d=d.loc[(d.sport=="americanfootball_nfl")&(d.market=="ml")&
            (d.side.isin(["home","away"]))].copy()
    for c in ("snapshot_ts","commence_time"):
        d[c]=pd.to_datetime(d[c],utc=True,errors="coerce")
    d["american_odds"]=pd.to_numeric(d.american_odds,errors="coerce")
    valid=(d.snapshot_ts.notna()&d.commence_time.notna()&
           (d.snapshot_ts<d.commence_time)&
           (d.american_odds.le(-100)|d.american_odds.ge(100))&
           (d.american_odds%1==0)&d.book.notna()&
           d.home.notna()&d.away.notna())
    # The provider free window is based on Eastern game date, not UTC date.
    local=d.commence_time.dt.tz_convert("America/New_York").dt.date
    in_week=(local>=pd.Timestamp("2026-10-19").date())&(local<=pd.Timestamp("2026-10-25").date())
    accepted=d.loc[valid&in_week].copy()
    accepted["source"]="the_odds_gap_free_sample"
    accepted["source_capture_verified"]=False
    accepted["quote_executable_verified"]=False
    accepted["model_eligible"]=False
    summary={"input_rows":len(frame),"nfl_moneyline_rows":len(d),
             "accepted_pregame_sample_rows":len(accepted),
             "distinct_games":int(accepted[["home","away","commence_time"]].drop_duplicates().shape[0]),
             "distinct_books":int(accepted.book.nunique()),
             "rejected_rows":int(len(d)-len(accepted)),
             "model_eligible":0,"network_calls":0,
             "warning":"Sample source timestamps have not been independently authenticated. No model decisions or executable wagers established."}
    return accepted,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True,help="Downloaded free sample CSV")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=validate_sample(pd.read_csv(a.csv))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_free_sample_moneylines_unverified.csv",index=False)
    (out/"nfl_free_sample_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
