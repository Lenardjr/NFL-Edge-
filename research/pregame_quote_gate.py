"""Fail-closed pregame moneyline quote gate for future paper-trading research.

This does NOT retrieve historical odds, authenticate source URLs, or prove a
quote was executable. Only independently reviewed quotes can pass this gate.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from research.market_validation import american_to_decimal,expected_roi,validate_snapshot

FIELDS={"game_id","side","american_odds","sportsbook","source","source_url",
        "captured_at","kickoff_at","verification"}
def validate_quotes(quotes:pd.DataFrame):
    if FIELDS-set(quotes):raise ValueError("Missing quote fields: "+str(sorted(FIELDS-set(quotes))))
    if quotes.duplicated(["game_id","side","sportsbook","captured_at"]).any():
        raise ValueError("Duplicate quote key")
    rows=[]
    for _,r in quotes.iterrows():
        reason=""
        try:
            if pd.isna(r.game_id) or not str(r.game_id).strip():raise ValueError("missing game ID")
            if r.side not in ("home","away"):raise ValueError("invalid side")
            if not isinstance(r.american_odds,(int,float)) or pd.isna(r.american_odds) or not float(r.american_odds).is_integer():
                raise ValueError("invalid American odds")
            american_to_decimal(int(r.american_odds))
            if pd.isna(r.source_url) or not str(r.source_url).startswith("https://"):
                raise ValueError("missing HTTPS source URL")
            validate_snapshot(captured_at=str(r.captured_at),kickoff_at=str(r.kickoff_at),
                              sportsbook=str(r.sportsbook),source=str(r.source))
            if r.verification!="independently_verified_pregame":
                raise ValueError("source and quote not independently verified")
        except (ValueError,TypeError,OverflowError) as e:
            reason=str(e)
        rows.append(reason)
    result=quotes.copy()
    result["exclusion_reason"]=rows
    result["eligible_for_paper_research"]=[not x for x in rows]
    return result,{"quotes":len(result),"eligible":int(result.eligible_for_paper_research.sum()),
                   "excluded":int((~result.eligible_for_paper_research).sum()),
                   "limitation":"Metadata gate only. Independent archival/source verification and quote executability are not established by this code."}

def evaluate_paper_candidates(predictions:pd.DataFrame,quotes:pd.DataFrame):
    """Outcome-blind candidate evaluation. No bankroll allocation or bet placement."""
    needed={"game_id","home_probability","prediction_at"}
    if needed-set(predictions):raise ValueError("Missing prediction fields")
    if predictions.game_id.duplicated().any():raise ValueError("Duplicate predictions")
    validated,summary=validate_quotes(quotes)
    if not validated.eligible_for_paper_research.any():
        return pd.DataFrame(columns=["game_id","side","sportsbook","american_odds","home_probability","model_probability","expected_roi"]),summary
    accepted=validated[validated.eligible_for_paper_research].copy()
    merged=accepted.merge(predictions[list(needed)],on="game_id",how="inner",validate="many_to_one")
    if merged.home_probability.isna().any() or not merged.home_probability.between(0,1).all():
        raise ValueError("Invalid model probabilities")
    if merged.prediction_at.isna().any():raise ValueError("Missing prediction timestamp")
    from datetime import datetime,timezone
    def utc(s):
        t=datetime.fromisoformat(str(s).replace("Z","+00:00"))
        if t.tzinfo is None or t.utcoffset() is None:raise ValueError("Prediction timestamp requires timezone")
        return t.astimezone(timezone.utc)
    for _,r in merged.iterrows():
        if utc(r.prediction_at)>utc(r.captured_at):
            raise ValueError("Prediction was not frozen before quote capture")
    merged["model_probability"]=merged.apply(lambda r:float(r.home_probability) if r.side=="home" else 1-float(r.home_probability),axis=1)
    merged["expected_roi"]=merged.apply(lambda r:expected_roi(float(r.model_probability),int(r.american_odds)),axis=1)
    summary["paired_candidates"]=len(merged)
    summary["warning"]="Expected ROI is model-implied only; no verified profitability, executable quote or settled bets."
    return merged[["game_id","side","sportsbook","american_odds","home_probability","model_probability","expected_roi"]],summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--quotes",default="")
    ap.add_argument("--output-dir",default="research_outputs")
    a=ap.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if a.quotes:
        quotes=pd.read_csv(a.quotes)
        checked,summary=validate_quotes(quotes)
        checked.to_csv(out/"nfl_pregame_quote_gate.csv",index=False)
    else:
        summary={"quotes":0,"eligible":0,"excluded":0,
            "status":"No independently captured and verified pregame odds dataset supplied",
            "limitation":"Cannot calculate executable ROI or claim an edge from historical no-vig market probabilities."}
    (out/"nfl_pregame_quote_readiness.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
