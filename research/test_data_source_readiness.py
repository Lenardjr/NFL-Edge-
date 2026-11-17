import unittest
from research.data_source_readiness import SOURCES,audit
class DataSourceReadinessTests(unittest.TestCase):
    def test_candidates_not_claimed_verified(self):
        r=audit(SOURCES)
        self.assertEqual(r["candidate_sources"],3)
        self.assertEqual(r["verified_pregame_sources"],0)
    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):audit([SOURCES[0],SOURCES[0]])
    def test_unavailable_cannot_be_verified(self):
        x=dict(SOURCES[0],pregame_verified=True)
        with self.assertRaisesRegex(ValueError,"absent"):audit([x])
    def test_https_required(self):
        x=dict(SOURCES[0],url="http://example.com")
        with self.assertRaisesRegex(ValueError,"HTTPS"):audit([x])
