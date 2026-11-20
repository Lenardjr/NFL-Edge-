import unittest
import pandas as pd
from research.diagnose_disagreement import diagnose

class DisagreementTests(unittest.TestCase):
    def test_counts_and_error_direction(self):
        data=pd.DataFrame({"home_win":[1,0,1,0],
            "model_home_prob":[.9,.8,.2,.1],
            "market_home_prob":[.8,.2,.7,.1],
            "season":[2022,2022,2023,2023]})
        report=diagnose(data)
        self.assertEqual(report["overall"]["games"],4)
        self.assertEqual(report["same_favorite"]["games"],2)
        self.assertEqual(report["opposite_favorites"]["games"],2)
        self.assertEqual(sum(b["games"] for b in report["calibration_bins"]),4)

    def test_invalid_probabilities_rejected(self):
        data=pd.DataFrame({"home_win":[1],"model_home_prob":[1.2],"market_home_prob":[.5]})
        with self.assertRaises(ValueError): diagnose(data)

if __name__=="__main__":
    unittest.main()
