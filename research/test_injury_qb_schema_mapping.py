import unittest,json
import pandas as pd
from research.injury_qb_schema_mapping import inspect_schema
class SchemaMappingTests(unittest.TestCase):
    def sample(self,columns):
        s=pd.DataFrame([{"game_id":"g","status":"schema_inspected","columns_json":json.dumps(columns)}])
        e=pd.DataFrame([{"game_id":"g","status":"qb_rows_extracted","mapped_columns_json":"{}"}])
        return s,e
    def test_detects_candidate_alias_without_promotion(self):
        d,s=inspect_schema(*self.sample(["position","player_name","injury_report_text"]))
        row=d[d.field.eq("injury")].iloc[0]
        self.assertEqual(row.mapping_status,"review_possible_alias")
        self.assertEqual(s["model_eligible"],0)
    def test_existing_mapping(self):
        d,s=inspect_schema(*self.sample(["injury","report_date"]))
        self.assertEqual(d[d.field.eq("injury")].mapping_status.iloc[0],"mapped")
    def test_no_obvious_column(self):
        d,s=inspect_schema(*self.sample(["position"]))
        self.assertEqual(d[d.field.eq("report_date")].mapping_status.iloc[0],"no_obvious_column")
    def test_reject_duplicate(self):
        a,b=self.sample(["position"])
        with self.assertRaisesRegex(ValueError,"Duplicate"):inspect_schema(pd.concat([a,a]),b)
    def test_skips_failed_schema(self):
        a,b=self.sample(["position"]);a.loc[0,"status"]="error_ValueError"
        d,s=inspect_schema(a,b)
        self.assertEqual(len(d),0)
