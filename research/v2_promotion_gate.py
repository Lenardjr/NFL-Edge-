"""Conservative V2 research promotion gate.

A historical backtest is exploratory: passing these criteria only qualifies a
model for a prospective shadow test, never production or real-money wagering.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

def decision(report):
    scores=report["scores"]
    seasons=report["seasons"]
    intervals=report["season_cluster_bootstrap_95pct_intervals"]
    # Require both probability metrics, season consistency, and a bootstrap
    # interval entirely below zero for improvements against BOTH benchmarks.
    conditions={
        "v2_brier_better_than_v1":scores["v2"]["brier"]<scores["baseline"]["brier"],
        "v2_brier_better_than_market":scores["v2"]["brier"]<scores["market"]["brier"],
        "v2_logloss_better_than_v1":scores["v2"]["log_loss"]<scores["baseline"]["log_loss"],
        "v2_logloss_better_than_market":scores["v2"]["log_loss"]<scores["market"]["log_loss"],
        "at_least_three_seasons":len(seasons)>=3,
        "better_than_v1_in_majority_of_seasons":sum(
            s["v2_minus_baseline_brier"]<0 for s in seasons.values())>len(seasons)/2,
        "better_than_market_in_majority_of_seasons":sum(
            s["v2_minus_market_brier"]<0 for s in seasons.values())>len(seasons)/2,
        "v1_interval_excludes_zero":intervals["v2_minus_baseline"][1]<0,
        "market_interval_excludes_zero":intervals["v2_minus_market"][1]<0,
    }
    qualifies=all(conditions.values())
    return {"qualifies_for_prospective_shadow_testing":qualifies,
            "conditions":conditions,
            "interpretation":("Candidate for future untouched-game shadow testing only"
                if qualifies else "Do not promote; keep V1 and market benchmarks"),
            "warning":"This is not a betting ROI, calibration, or executable-odds validation."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--comparison",required=True)
    p.add_argument("--output",default="research_outputs/nfl_v2_decision.json")
    a=p.parse_args()
    report=decision(json.loads(Path(a.comparison).read_text()))
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
