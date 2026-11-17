import unittest
import pandas as pd
from research.qb_retrospective_reference import enrich
class QBReferenceTests(unittest.TestCase):
    def test_reference_is_not_pregame_evidence(self):
        q=pd.DataFrame([{"game_id":"g1","side":"home","verification":"unreviewed",
                          "eligible_for_model":False},
                         {"game_id":"g1","side":"away","verification":"unreviewed",
                          "eligible_for_model":False}])
        s=pd.DataFrame([{"game_id":"g1","home_qb_name":"Starter A","away_qb_name":"Starter B"}])
        out,summary=enrich(q,s)
        self.assertEqual(summary["retrospective_qb_labels"],2)
        self.assertEqual(summary["verified_pregame"],0)
        self.assertFalse(out.eligible_for_model.any())
        self.assertEqual(set(out.historical_qb_name),{"Starter A","Starter B"})
    def test_missing_schedule_rejected(self):
        q=pd.DataFrame([{"game_id":"g1","side":"home","verification":"unreviewed",
                          "eligible_for_model":False}])
        s=pd.DataFrame([{"game_id":"g2","home_qb_name":"A","away_qb_name":"B"}])
        with self.assertRaisesRegex(ValueError,"Unmatched"):enrich(q,s)
