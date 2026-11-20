"""Research-only pregame feature availability gate.

Candidate player, injury, scheme, and weather features enter experiments only
when their source publication time is at/before the prediction snapshot.
No feature is promoted to production by this module.
"""
from __future__ import annotations
import pandas as pd

CATEGORIES={"player","injury","scheme","weather","schedule","market"}
REQUIRED={"game_id","feature_name","feature_value","category","published_at_utc","source_id"}
def eligible_feature_snapshot(features:pd.DataFrame,*,asof_utc,allow_market=False):
    missing=REQUIRED-set(features.columns)
    if missing:raise ValueError(f"Missing feature fields: {sorted(missing)}")
    cutoff=pd.Timestamp(asof_utc)
    if cutoff.tzinfo is None:raise ValueError("asof_utc must be timezone-aware")
    f=features.copy()
    if f[list(REQUIRED)].isna().any().any():raise ValueError("Null feature or provenance field")
    if not f.category.isin(CATEGORIES).all():raise ValueError("Unknown feature category")
    if not allow_market and (f.category=="market").any():
        raise ValueError("Market-derived features require explicit experiment opt-in")
    f["published_at_utc"]=pd.to_datetime(f.published_at_utc,utc=True,errors="raise")
    if f.published_at_utc.isna().any():raise ValueError("Invalid publication timestamp")
    # Reject instead of silently discarding future-published rows.
    if (f.published_at_utc>cutoff).any():raise ValueError("Feature published after prediction snapshot")
    if f.duplicated(["game_id","feature_name"]).any():
        raise ValueError("Ambiguous duplicate feature for game")
    f["feature_value"]=pd.to_numeric(f.feature_value,errors="raise")
    if not f.feature_value.map(lambda x:pd.notna(x) and abs(x)!=float("inf")).all():
        raise ValueError("Nonfinite feature value")
    return f.sort_values(["game_id","feature_name"]).reset_index(drop=True)

def promotion_decision(*,baseline_brier,candidate_brier,market_brier,
                       paired_games,ci_upper_delta,source_verified,
                       min_games=500):
    """Conservative research gate: paired CI must exclude zero in our favor."""
    if paired_games<min_games:return {"promote":False,"reason":"insufficient paired games"}
    if not source_verified:return {"promote":False,"reason":"source timing not verified"}
    if ci_upper_delta>=0:return {"promote":False,"reason":"improvement not statistically supported"}
    if candidate_brier>=baseline_brier:return {"promote":False,"reason":"no baseline improvement"}
    if candidate_brier>=market_brier:return {"promote":False,"reason":"does not beat market benchmark"}
    return {"promote":True,"reason":"passes preliminary statistical gate; manual review required"}
