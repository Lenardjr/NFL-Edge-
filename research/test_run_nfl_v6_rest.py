import unittest
import pandas as pd
from research.run_nfl_v6_rest import add_rest_transforms,EXTRA
from game_engine import FEATURES

class RestAblationTests(unittest.TestCase):
    def test_v3_already_includes_rest(self):
        self.assertTrue({"home_rest","away_rest","rest_diff"}.issubset(FEATURES))
    def test_threshold_boundaries(self):
        f=pd.DataFrame({"rest_diff":[-3,-2,0,3,4],"home_rest":[4,6,7,9,10],
                        "away_rest":[7,8,7,6,6]})
        d=add_rest_transforms(f)
        self.assertEqual(d.home_rest_disadvantage_3plus.tolist(),[1,0,0,0,0])
        self.assertEqual(d.home_rest_advantage_3plus.tolist(),[0,0,0,1,1])
        self.assertEqual(d.either_team_short_rest.tolist(),[1,0,0,0,0])
        self.assertEqual(d.rest_imbalance_excess_3.tolist(),[0,0,0,0,1])
    def test_no_original_mutation(self):
        f=pd.DataFrame({"rest_diff":[0],"home_rest":[7],"away_rest":[7]})
        add_rest_transforms(f)
        self.assertTrue(set(EXTRA).isdisjoint(f.columns))
