import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from placement_sim import make_synthetic_trace, allocations, validate, tier_work_proxy

class TestSyntheticPlacement(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'configs/synthetic_demo.json').read_text())
        self.cfg.update(layers=3, experts_per_layer=24, parameters_per_expert=1200,
                        spark_total_gb=0.000010, spark_reserved_gb=0.000004,
                        remote_available_gb=0.000100)
        self.experts = make_synthetic_trace(3,24,1200,0.85,17)

    def test_probabilities_sum_to_one_by_layer(self):
        for l in range(3):
            self.assertAlmostEqual(sum(e.probability for e in self.experts if e.layer == l), 1.0, places=10)

    def test_every_expert_once_and_shared_memory(self):
        plan = allocations(self.experts, self.cfg)
        result = validate(self.experts, plan, self.cfg)
        self.assertEqual(sum(result['counts'].values()), len(self.experts))
        self.assertLessEqual(result['spark_shared_expert_used_GB'], self.cfg['spark_total_gb'])
        self.assertTrue(all(x.tier != 'spark_gpu' or x.quant.startswith('gpu_') for x in plan))

    def test_deterministic(self):
        a = allocations(self.experts, self.cfg)
        b = allocations(self.experts, self.cfg)
        self.assertEqual([(p.tier,p.quant) for p in a], [(p.tier,p.quant) for p in b])
        self.assertIn('SYNTHETIC', tier_work_proxy(self.experts,a,self.cfg)['warning'].upper())

    def test_remote_capacity_failure(self):
        self.cfg['remote_available_gb']=0.000000001
        with self.assertRaises(ValueError):
            allocations(self.experts, self.cfg)

if __name__=='__main__':
    unittest.main()
