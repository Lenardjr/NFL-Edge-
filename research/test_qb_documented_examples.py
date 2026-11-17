import unittest
from research.qb_documented_examples import EVIDENCE,validate
class QBExamplesTests(unittest.TestCase):
    def test_pregame_publication_dates(self):
        e=validate(EVIDENCE)
        self.assertEqual(len(e),3)
        self.assertFalse(e.eligible_for_model.any())
    def test_postgame_source_rejected(self):
        rows=[dict(EVIDENCE[0],published_date="2025-01-06")]
        with self.assertRaisesRegex(ValueError,"precede"):validate(rows)
