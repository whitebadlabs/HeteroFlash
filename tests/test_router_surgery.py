import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from router_surgery import route

class RouterMathTest(unittest.TestCase):
    def test_baseline_scale_and_selection(self):
        logits=[float(x) for x in range(12)]
        result=route(logits,[0]*12,k_active=8,k_reference=8)
        self.assertEqual(result['expert_ids'],list(range(11,3,-1)))
        self.assertAlmostEqual(result['sum_weights'],2.5)

    def test_reference_normalization_lowers_mass(self):
        logits=[-.3,0,.1,.7,1,2,3,4,5,6]
        bias=[0]*len(logits)
        base=route(logits,bias,k_active=4,k_reference=4)
        modified=route(logits,bias,k_active=4,k_reference=8)
        self.assertEqual(base['expert_ids'],modified['expert_ids'])
        self.assertGreater(base['sum_weights'],modified['sum_weights'])

    def test_bias_affects_selection_but_not_gate_magnitude(self):
        out=route([0.,4.,3.,2.],[5.,0.,0.,0.],k_active=2,k_reference=2)
        self.assertEqual(out['expert_ids'],[0,1])
        s0=.5; s1=1/(1+math.exp(-4))
        self.assertAlmostEqual(out['weights'][0],2.5*s0/(s0+s1))

    def test_group_guard(self):
        with self.assertRaises(ValueError):
            route([1.,2.,3.,4.],[0.]*4,k_active=3,k_reference=3,num_groups=2,topk_groups=1)

if __name__=='__main__': unittest.main()
