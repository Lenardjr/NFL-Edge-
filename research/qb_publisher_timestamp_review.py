"""Curated publisher-displayed timestamps for historical QB announcements.

Displayed publisher timestamps are leads, not independently archived captures.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

SOURCES=[
 {"game_id":"2024_18_KC_DEN","side":"away","publisher":"Associated Press via ClickOnDetroit","publisher_displayed_at":"2025-01-01 02:01 PM","publisher_timezone":"America/Detroit","source_url":"https://www.clickondetroit.com/sports/2025/01/01/chiefs-to-start-carson-wentz-at-qb-against-denver-which-would-clinch-a-playoff-berth-with-a-win/","qb_name":"Carson Wentz"},
 {"game_id":"2023_18_PIT_BAL","side":"home","publisher":"Baltimore Ravens","publisher_displayed_at":"2024-01-03 03:42 PM","publisher_timezone":"America/New_York","source_url":"https://www.baltimoreravens.com/news/lamar-jackson-won-t-play-vs-steelers","qb_name":"Tyler Huntley"},
 {"game_id":"2024_18_SEA_LA","side":"home","publisher":"Los Angeles Rams","publisher_displayed_at":"2025-01-01 01:42 PM","publisher_timezone":"America/Los_Angeles","source_url":"https://www.therams.com/news/rams-jimmy-garoppolo-to-start-at-quarterback-week-18-seahawks","qb_name":"Jimmy Garoppolo"},
]
def review(timeline:pd.DataFrame, sources=None):
    required={"game_id","side","kickoff_utc","in_priority_queue"}
    if required-set(timeline):raise ValueError("Missing timeline columns")
    if timeline.duplicated(["game_id","side"]).any():raise ValueError("Duplicate timeline game-side")
    sources=SOURCES if sources is None else sources
    e=pd.DataFrame(sources)
    if e.duplicated(["game_id","side"]).any():raise ValueError("Duplicate source game-side")
    if not {"game_id","side","publisher_displayed_at","publisher_timezone","source_url"}<=set(e):
        raise ValueError("Missing source metadata")
    parsed=[]
    for a in e.to_dict("records"):
        stamp=pd.Timestamp(a["publisher_displayed_at"])
        if stamp.tzinfo is not None:raise ValueError("Expected publisher local timestamp")
        utc=stamp.tz_localize(a["publisher_timezone"],ambiguous="raise",nonexistent="raise").tz_convert("UTC")
        parsed.append(utc.isoformat())
    e["publisher_displayed_utc"]=parsed
    d=timeline.merge(e[["game_id","side","publisher","publisher_displayed_at","publisher_timezone","publisher_displayed_utc","source_url"]],on=["game_id","side"],how="left",suffixes=("","_publisher"),validate="one_to_one")
    kickoff=pd.to_datetime(d.kickoff_utc,utc=True,errors="coerce")
    publication=pd.to_datetime(d.publisher_displayed_utc,utc=True,errors="coerce")
    d["publisher_timestamp_before_kickoff"]=publication.notna() & kickoff.notna() & publication.lt(kickoff)
    d["publisher_timestamp_independently_captured"]=False
    d["source_verified_at_decision"]=False
    d["market_quote_verified_at_decision"]=False
    d["model_eligible"]=False
    s={"cases":len(d),"publisher_timestamps_found":int(publication.notna().sum()),
       "publisher_timestamps_before_kickoff":int(d.publisher_timestamp_before_kickoff.sum()),
       "priority_cases_with_publisher_timestamp":int((publication.notna() & d.in_priority_queue.astype(str).str.lower().eq("true")).sum()),
       "independent_captures_verified":0,"verified_market_quotes":0,"model_eligible":0,
       "warning":"Publisher-displayed local time is not an independently authenticated archival capture or evidence of quote availability."}
    return d,s

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--timeline",default="research_outputs/nfl_qb_announcement_decision_timeline.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=review(pd.read_csv(a.timeline))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_qb_publisher_timestamp_review.csv",index=False)
    (out/"nfl_qb_publisher_timestamp_review_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
