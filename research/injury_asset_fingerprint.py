"""Download and fingerprint matched historical injury assets for subsequent content review.

Asset bytes are not parsed as QB injury claims here. SHA256 is reproducible
file integrity evidence, not independent publication-time verification.
"""
from __future__ import annotations
import argparse,hashlib,json,urllib.request
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

MAX_BYTES=25_000_000
def fingerprint(assets:pd.DataFrame,fetcher=None):
    needed={"game_id","asset_found","asset_url","asset_uploaded_at","uploaded_before_kickoff"}
    if needed-set(assets):raise ValueError("Missing asset fields")
    if assets.game_id.isna().any() or assets.game_id.duplicated().any():
        raise ValueError("Invalid game IDs")
    fetcher=fetcher or download
    rows=[]
    for item in assets.to_dict("records"):
        found=str(item["asset_found"]).lower()=="true"
        pre=str(item["uploaded_before_kickoff"]).lower()=="true"
        url=item["asset_url"] if isinstance(item["asset_url"],str) else ""
        row={"game_id":item["game_id"],"asset_url":url,"uploaded_before_kickoff":pre,
            "download_status":"not_attempted","byte_size":0,"sha256":"",
            "file_header_hex":"","content_parsed":False,"independent_capture_verified":False,
            "model_eligible":False}
        if found and pre:
            if urlparse(url).scheme!="https" or urlparse(url).hostname not in {"github.com","objects.githubusercontent.com","release-assets.githubusercontent.com"}:
                row["download_status"]="invalid_host"
            else:
                try:
                    data=fetcher(url)
                    if not isinstance(data,bytes) or not data or len(data)>MAX_BYTES:
                        raise ValueError("Invalid asset size")
                    row.update(download_status="downloaded",byte_size=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),file_header_hex=data[:12].hex())
                except Exception as exc:
                    row["download_status"]="error_"+type(exc).__name__
        rows.append(row)
    d=pd.DataFrame(rows,columns=["game_id","asset_url","uploaded_before_kickoff",
        "download_status","byte_size","sha256","file_header_hex","content_parsed",
        "independent_capture_verified","model_eligible"])
    summary={"games":len(d),"downloaded":int(d.download_status.eq("downloaded").sum()),
      "download_errors":int(d.download_status.str.startswith("error_").sum()),
      "content_parsed":0,"independently_verified":0,"model_eligible":0,
      "warning":"Hashes establish the bytes downloaded now, not the bytes available at historical prediction time."}
    return d,summary

def download(url):
    request=urllib.request.Request(url,headers={"User-Agent":"NFL-Edge-research"})
    with urllib.request.urlopen(request,timeout=30) as response:
        data=response.read(MAX_BYTES+1)
    if len(data)>MAX_BYTES:raise ValueError("Oversize asset")
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",default="research_outputs/nfl_2024_qb_asset_kickoff_timing.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=fingerprint(pd.read_csv(a.assets))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_2024_qb_asset_fingerprints.csv",index=False)
    (out/"nfl_2024_qb_asset_fingerprint_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
