"""Normalize Odds Gap opening/closing XLSX sample into research-only quotes.

Input is a manually provided spreadsheet. Source timestamps and prices are
not independently authenticated; neither quote execution nor ROI is inferred.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

REQUIRED={"sport","away","home","commence_time","market","side","book",
          "earliest_ts","earliest_american_odds","close_ts","close_american_odds"}
def normalize(frame:pd.DataFrame):
    missing=REQUIRED-set(frame.columns)
    if missing:raise ValueError("Missing columns: "+",".join(sorted(missing)))
    d=frame.loc[(frame.sport=="americanfootball_nfl")&
                (frame.market.isin(["ml","spread","total"]))&
                (frame.side.isin(["home","away","over","under"]))].copy()
    rows=[]
    for _,r in d.iterrows():
        kickoff=pd.to_datetime(r.commence_time,utc=True,errors="coerce")
        if pd.isna(kickoff):continue
        for phase,tcol,pcol in (("earliest","earliest_ts","earliest_american_odds"),
                                ("close","close_ts","close_american_odds")):
            t=pd.to_datetime(r[tcol],utc=True,errors="coerce")
            p=pd.to_numeric(r[pcol],errors="coerce")
            if pd.isna(t) or pd.isna(p) or t>=kickoff or p%1 or not (p<=-100 or p>=100):continue
            rows.append({"game_id":str(r.get("game_id") or (str(r.away)+"_at_"+str(r.home)+"_"+kickoff.strftime("%Y%m%d"))),"away":r.away,"home":r.home,
                         "commence_time_utc":kickoff.isoformat(),"market":r.market,
                         "side":r.side,"line":r.get("line"),"book":r.book,
                         "quote_phase":phase,"quote_timestamp_utc":t.isoformat(),
                         "american_odds":int(p),"source":"oddsgap_manual_xlsx",
                         "source_capture_verified":False,
                         "quote_executable_verified":False,"model_eligible":False})
    out=pd.DataFrame(rows)
    summary={"input_rows":len(frame),"eligible_market_rows":len(d),
             "valid_pregame_quotes":len(out),
             "earliest_quotes":int((out.quote_phase=="earliest").sum()) if len(out) else 0,
             "closing_quotes":int((out.quote_phase=="close").sum()) if len(out) else 0,
             "distinct_games":int(out.game_id.nunique()) if len(out) else 0,
             "distinct_books":int(out.book.nunique()) if len(out) else 0,
             "model_eligible":0,
             "warning":"Only earliest and close snapshots, not full intraday history. Source capture not independently verified; no profitability claim."}
    return out,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--xlsx",required=True)
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=normalize(pd.read_excel(a.xlsx))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_oddsgap_open_close_unverified.csv",index=False)
    (out/"nfl_oddsgap_open_close_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
