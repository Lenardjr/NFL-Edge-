"""Append-only prospective NFL prediction log (research branch only).

A timestamped entry is evidence of logging, NOT independent proof that model
features, sportsbook prices, or execution were available at that time.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

FIELDS=["logged_at_utc","game_id","kickoff_utc","model_version","home_win_probability","input_fingerprint","prediction_fingerprint","source_provenance","independent_timestamp_verified","quote_executable_verified"]
def utc(value):
    d=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if d.tzinfo is None:raise ValueError("Timezone-aware timestamp required")
    return d.astimezone(timezone.utc)
def prepare(entry,now=None):
    now=now or datetime.now(timezone.utc)
    kickoff=utc(entry["kickoff_utc"])
    if kickoff<=now:raise ValueError("Cannot log a prediction at or after kickoff")
    prob=float(entry["home_win_probability"])
    if not 0<prob<1:raise ValueError("Probability must be strictly between zero and one")
    for key in ["game_id","model_version","input_fingerprint"]:
        if not str(entry.get(key,"")).strip():raise ValueError("Missing "+key)
    if not (len(str(entry["input_fingerprint"]))==64 and all(c in "0123456789abcdef" for c in str(entry["input_fingerprint"]))):
        raise ValueError("input_fingerprint must be a SHA256 hex digest")
    result={"logged_at_utc":now.isoformat(),"game_id":str(entry["game_id"]),"kickoff_utc":kickoff.isoformat(),
            "model_version":str(entry["model_version"]),"home_win_probability":format(prob,".12g"),
            "input_fingerprint":str(entry["input_fingerprint"]),"source_provenance":str(entry.get("source_provenance","unverified")),
            "independent_timestamp_verified":"false","quote_executable_verified":"false"}
    payload=json.dumps(result,sort_keys=True,separators=(",",":"))
    result["prediction_fingerprint"]=hashlib.sha256(payload.encode()).hexdigest()
    return result
def append(path,entry,now=None):
    path=Path(path)
    if path.exists():
        with path.open(newline="") as f: rows=list(csv.DictReader(f))
        if any(r["game_id"]==str(entry["game_id"]) and r["model_version"]==str(entry["model_version"]) for r in rows):
            raise ValueError("Duplicate game/model prediction: prospective log is immutable")
    else:rows=[]
    record=prepare(entry,now)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Do not overwrite any prior prediction; write a complete replacement atomically.
    import os,tempfile
    fd,tmp=tempfile.mkstemp(dir=str(path.parent),prefix=".prospective_",suffix=".csv")
    try:
        with os.fdopen(fd,"w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows);writer.writerow(record)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return record
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prediction-json",required=True,help="JSON object with game_id, kickoff_utc, model_version, home_win_probability, input_fingerprint")
    p.add_argument("--ledger",default="research_outputs/nfl_prospective_predictions.csv")
    args=p.parse_args()
    record=append(args.ledger,json.loads(Path(args.prediction_json).read_text()))
    print(json.dumps(record,indent=2))
if __name__=="__main__":main()
