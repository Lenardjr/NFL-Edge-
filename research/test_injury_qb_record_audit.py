import unittest
import pandas as pd
from research.injury_qb_record_audit import audit
class RecordAuditTests(unittest.TestCase):
    def candidate(self):
        return pd.DataFrame([{"game_id":"g","player":"Q","team":"NYJ","injury":"knee","practice_status":"","game_status":"Out","report_date":"2024-12-01T10:00:00-05:00"}])
    def timing(self):
        return pd.DataFrame([{"game_id":"g","kickoff_utc":"2024-12-01T18:00:00Z"}])
    def test_before_kickoff_not_verified(self):
        d,s=audit(self.candidate(),self.timing())
        self.assertEqual(s["report_timestamp_before_kickoff_unverified"],1)
        self.assertEqual(s["model_eligible"],0)
        self.assertFalse(d.model_eligible.iloc[0])
    def test_date_only_not_timestamp(self):
        c=self.candidate();c.loc[0,"report_date"]="2024-12-01"
        d,s=audit(c,self.timing())
        self.assertEqual(s["report_timestamp_parseable_with_offset"],0)
    def test_missing_identity(self):
        c=self.candidate();c.loc[0,"player"]=""
        d,s=audit(c,self.timing())
        self.assertEqual(d.review_status.iloc[0],"identity_fields_incomplete")
    def test_duplicate_rows(self):
        c=self.candidate()
        d,s=audit(pd.concat([c,c]),self.timing())
        self.assertEqual(s["exact_duplicate_rows"],2)
    def test_duplicate_kickoff_rejected(self):
        t=self.timing()
        with self.assertRaisesRegex(ValueError,"Invalid kickoff"):audit(self.candidate(),pd.concat([t,t]))
