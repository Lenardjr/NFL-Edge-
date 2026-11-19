"""Safely connect *future* live model inference to the prospective ledger.

The historical V3 evaluator cannot serve live predictions: it filters to games
with final scores. Call record_live_inference only after a separately validated
pregame inference path produces a probability and frozen input fingerprint.
"""
from __future__ import annotations
from datetime import datetime,timezone
from research.prospective_prediction_log import append,utc

def record_live_inference(ledger,*,game_id,kickoff_utc,model_version,home_win_probability,
                          input_fingerprint,features_asof_utc,model_trained_through_utc,
                          now=None):
    now=now or datetime.now(timezone.utc)
    asof=utc(features_asof_utc)
    trained=utc(model_trained_through_utc)
    kickoff=utc(kickoff_utc)
    if asof>now:raise ValueError("Feature snapshot cannot be in the future")
    if trained>asof:raise ValueError("Training cutoff exceeds feature snapshot")
    if now>=kickoff:raise ValueError("Kickoff has already occurred")
    if not model_version.startswith("live-"):raise ValueError("Historical models cannot be logged as live")
    return append(ledger,{"game_id":game_id,"kickoff_utc":kickoff_utc,
        "model_version":model_version,"home_win_probability":home_win_probability,
        "input_fingerprint":input_fingerprint,
        "source_provenance":"live_inference_unverified"},now=now)
