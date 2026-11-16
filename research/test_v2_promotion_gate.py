import unittest
from research.v2_promotion_gate import decision

def report(v2_brier=.19,v2_log=.56,good=True):
    diff=-.01 if good else .01
    return {"scores":{"baseline":{"brier":.21,"log_loss":.61},
        "market":{"brier":.20,"log_loss":.59},
        "v2":{"brier":v2_brier,"log_loss":v2_log}},
        "seasons":{str(s):{"v2_minus_baseline_brier":diff,
          "v2_minus_market_brier":diff} for s in (2022,2023,2024,2025)},
        "season_cluster_bootstrap_95pct_intervals":{
            "v2_minus_baseline":[-.02,-.001] if good else [-.01,.02],
            "v2_minus_market":[-.02,-.001] if good else [-.01,.02]}}
class PromotionGateTests(unittest.TestCase):
    def test_strong_candidate_only_qualifies_for_shadow(self):
        result=decision(report())
        self.assertTrue(result["qualifies_for_prospective_shadow_testing"])
        self.assertIn("shadow",result["interpretation"].lower())
    def test_worse_than_market_fails(self):
        self.assertFalse(decision(report(v2_brier=.205))["qualifies_for_prospective_shadow_testing"])
    def test_uncertainty_fails(self):
        self.assertFalse(decision(report(good=False))["qualifies_for_prospective_shadow_testing"])
if __name__=="__main__":unittest.main()
