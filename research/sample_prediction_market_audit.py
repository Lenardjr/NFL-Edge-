"""Audit a single historical game against supplied sportsbook moneyline quotes.

Research-only: historical model rows lack a verified prediction publication
timestamp, so no positive-EV wager, realized ROI, or promotion is asserted.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def implied(american):
    p=float(american)
    if not (p<=-100 or p>=100):raise ValueError("Invalid American odds")
    return (-p)/(-p+100) if p<0 else 100/(p+100)

def compare(quotes, predictions, game_id):
    match=predictions.loc[predictions.game_id==game_id]
    if len(match)!=1:raise ValueError("Expected exactly one historical prediction")
    row=match.iloc[0]
    q=quotes.loc[(quotes.market=="ml")&(quotes.side.isin(["home","away"]))].copy()
    q=q.loc[q.quote_phase.isin(["earliest","close"])]
    q["implied_prob"]=q.american_odds.map(implied)
    pairs=q.pivot_table(index=["book","quote_phase"],columns="side",values="implied_prob",aggfunc="first").dropna()
    pairs["home_no_vig"]=pairs.home/(pairs.home+pairs.away)
    pairs=pairs.reset_index()
    summary={"game_id":game_id,"historical_model_home_prob":float(row.model_home_prob),
             "historical_market_home_prob":float(row.market_home_prob),
             "home_win":int(row.home_win),"books_with_paired_quotes":int(pairs.book.nunique()),
             "opening_home_no_vig_median":float(pairs.loc[pairs.quote_phase=="earliest","home_no_vig"].median()),
             "closing_home_no_vig_median":float(pairs.loc[pairs.quote_phase=="close","home_no_vig"].median()),
             "historical_prediction_timestamp_verified":False,"executable_quote_verified":False,
             "backtest_wagers":0,"realized_roi":None,
             "warning":"Retrospective model probability is not a timestamp-verified pre-quote decision. Opening quotes vary in capture date; one game cannot establish edge."}
    return pairs,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--quotes",required=True)
    p.add_argument("--predictions",required=True)
    p.add_argument("--game-id",default="2026_04_NYJ_CHI")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=compare(pd.read_csv(a.quotes),pd.read_csv(a.predictions),a.game_id)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_sample_bookmaker_novig_comparison.csv",index=False)
    (out/"nfl_sample_prediction_market_audit.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
