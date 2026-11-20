"""Compare frozen NFL predictions with pre-kickoff market consensus on completed games.

Research only. Strict game-ID join; no betting, no inferred historical prices.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def compare(cohort,results,quotes,max_age_minutes=90):
    required={"game_id","home_win_probability","kickoff_utc"}
    if required-set(cohort):raise ValueError("Cohort missing columns")
    if {"game_id","status","home_score","away_score"}-set(results):raise ValueError("Results missing columns")
    if {"game_id","home_no_vig_probability","retrieved_at_utc","market_last_update_utc","sportsbook"}-set(quotes):
        raise ValueError("Quotes missing columns")
    for label,df in (("cohort",cohort),("results",results)):
        if df.game_id.duplicated().any():raise ValueError("Duplicate "+label+" game ID")
    p=cohort[["game_id","home_win_probability","kickoff_utc"]].copy()
    r=results[["game_id","status","home_score","away_score"]].copy()
    merged=p.merge(r,on="game_id",how="left",validate="one_to_one")
    final=merged.status.astype(str).str.lower().isin(["final","closed"])
    home=pd.to_numeric(merged.home_score,errors="coerce")
    away=pd.to_numeric(merged.away_score,errors="coerce")
    completed=final&home.notna()&away.notna()&(home!=away)
    games=merged.loc[completed].copy()
    games["actual_home_win"]=(home.loc[completed]>away.loc[completed]).astype(int)
    q=quotes.copy()
    q["retrieved"]=pd.to_datetime(q.retrieved_at_utc,utc=True,errors="coerce")
    q["updated"]=pd.to_datetime(q.market_last_update_utc,utc=True,errors="coerce")
    kicks=p.set_index("game_id").kickoff_utc
    q["kickoff"]=pd.to_datetime(q.game_id.map(kicks),utc=True,errors="coerce")
    q["market_probability"]=pd.to_numeric(q.home_no_vig_probability,errors="coerce")
    q=q.loc[q.retrieved.notna()&q.updated.notna()&q.kickoff.notna()
            &(q.updated<=q.retrieved)&(q.retrieved<q.kickoff)
            &((q.retrieved-q.updated)<=pd.Timedelta(minutes=max_age_minutes))
            &q.market_probability.between(0,1,inclusive="neither")]
    # Earliest eligible snapshot per game; no choosing hindsight-friendly market prices.
    if len(q):
        q=q.sort_values(["game_id","retrieved","sportsbook"])
        first=q.groupby("game_id").retrieved.transform("min")
        q=q.loc[q.retrieved==first]
        market=q.groupby("game_id").agg(market_home_probability=("market_probability","median"),
                                        fresh_books=("sportsbook","nunique"),
                                        market_snapshot_utc=("retrieved","min")).reset_index()
        market=market.loc[market.fresh_books>=3]
    else:
        market=pd.DataFrame(columns=["game_id","market_home_probability","fresh_books","market_snapshot_utc"])
    games=games.merge(market,on="game_id",how="left",validate="one_to_one")
    games["model_brier"]=(games.home_win_probability-games.actual_home_win)**2
    games["market_brier"]=(games.market_home_probability-games.actual_home_win)**2
    paired=games.market_brier.notna()
    summary={"cohort_games":len(cohort),"completed_scored_games":len(games),
      "paired_games_with_fresh_pregame_market":int(paired.sum()),
      "model_brier_on_paired":float(games.loc[paired,"model_brier"].mean()) if paired.any() else None,
      "market_brier_on_paired":float(games.loc[paired,"market_brier"].mean()) if paired.any() else None,
      "model_minus_market_brier":float((games.loc[paired,"model_brier"]-games.loc[paired,"market_brier"]).mean()) if paired.any() else None,
      "market_outperformance_verified":bool(paired.sum()>=100 and (games.loc[paired,"model_brier"]-games.loc[paired,"market_brier"]).mean()<0),
      "profitability_verified":False,
      "note":"A small sample or lower Brier score alone does not establish a durable edge. Market source is historical pregame snapshots."}
    return games,summary

def main():
    p=argparse.ArgumentParser()
    for arg in ("cohort","results","quotes","output-dir"):p.add_argument("--"+arg,required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    games,summary=compare(pd.read_csv(a.cohort),pd.read_csv(a.results),pd.read_csv(a.quotes))
    games.to_csv(out/"paired_market_evaluation.csv",index=False)
    (out/"paired_market_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
