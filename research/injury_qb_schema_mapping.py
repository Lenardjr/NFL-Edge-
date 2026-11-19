"""Compare observed RDS schema with QB extractor mappings, without guessing field semantics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from research.injury_qb_rds_extract import extract

FIELD_ALIASES={
"player":("player_name","player","full_name","name"),
"team":("team","team_abbr","team_abbreviation"),
"injury":("injury","injury_type","injury_description"),
"practice_status":("practice_status","practice"),
"game_status":("game_status","status","report_status"),
"report_date":("report_date","date","date_modified"),
}
HINTS={"injury":("injur","body","reason"),"practice_status":("practic","participat"),
       "game_status":("status","designat","availab","game"),"report_date":("date","time","update","report")}
def inspect_schema(schema:pd.DataFrame,extraction:pd.DataFrame):
    if not {"game_id","status","columns_json"}<=set(schema):raise ValueError("Missing schema fields")
    if not {"game_id","status","mapped_columns_json"}<=set(extraction):raise ValueError("Missing extraction fields")
    if schema.game_id.duplicated().any() or extraction.game_id.duplicated().any():raise ValueError("Duplicate game IDs")
    rows=[]
    for a in schema.to_dict("records"):
        if a["status"]!="schema_inspected":continue
        try:
            columns=json.loads(a["columns_json"])
            if not isinstance(columns,list) or not all(isinstance(x,str) for x in columns):raise ValueError("Invalid column list")
            lower={x.strip().lower():x for x in columns}
            for field,aliases in FIELD_ALIASES.items():
                matched=next((lower[x] for x in aliases if x in lower),None)
                hints=[x for x in columns if any(h in x.lower() for h in HINTS.get(field,()))]
                rows.append({"game_id":a["game_id"],"field":field,"mapped_column":matched or "",
                             "candidate_columns_json":json.dumps(hints),
                             "mapping_status":"mapped" if matched else ("review_possible_alias" if hints else "no_obvious_column"),
                             "manual_semantics_required":True,"model_eligible":False})
        except (ValueError,TypeError,KeyError) as exc:
            rows.append({"game_id":a["game_id"],"field":"schema","mapped_column":"",
                         "candidate_columns_json":"[]","mapping_status":"schema_error_"+type(exc).__name__,
                         "manual_semantics_required":True,"model_eligible":False})
    d=pd.DataFrame(rows,columns=["game_id","field","mapped_column","candidate_columns_json","mapping_status","manual_semantics_required","model_eligible"])
    counts=d.groupby(["field","mapping_status"]).size().to_dict() if len(d) else {}
    column_counts={}
    for item in schema.to_dict("records"):
        if item["status"]!="schema_inspected":continue
        try:
            cols=json.loads(item["columns_json"])
            if isinstance(cols,list):
                for col in set(x for x in cols if isinstance(x,str)):
                    column_counts[col]=column_counts.get(col,0)+1
        except (ValueError,TypeError):
            pass
    schema_columns=[{"column":k,"files":v} for k,v in sorted(column_counts.items(),key=lambda x:(-x[1],x[0]))]
    s={"schema_files_inspected":int(schema.status.eq("schema_inspected").sum()),
       "extractor_files":int(extraction.status.eq("qb_rows_extracted").sum()),
       "field_checks":len(d),
       "potential_unmapped_aliases":int(d.mapping_status.eq("review_possible_alias").sum()),
       "no_obvious_column":int(d.mapping_status.eq("no_obvious_column").sum()),
       "mapping_counts":{str(k[0])+":"+str(k[1]):int(v) for k,v in counts.items()},
       "observed_columns_by_file_count":schema_columns,
       "unmapped_field_review":d.loc[d.mapping_status.ne("mapped"),["field","candidate_columns_json","mapping_status"]].drop_duplicates().to_dict("records"),
       "independently_verified":0,"model_eligible":0,
       "warning":"Column-name hints are not evidence of field meaning or historical availability. Manual review required before modifying mappings."}
    return d,s

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--schema",default="research_outputs/nfl_2024_qb_rds_schema.csv")
    p.add_argument("--extraction",default="research_outputs/nfl_2024_qb_rds_extraction_audit.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    d,s=inspect_schema(pd.read_csv(a.schema),pd.read_csv(a.extraction))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"nfl_2024_qb_rds_field_mapping_review.csv",index=False)
    (out/"nfl_2024_qb_rds_field_mapping_summary.json").write_text(json.dumps(s,indent=2))
    print(json.dumps(s,indent=2))
if __name__=="__main__":main()
