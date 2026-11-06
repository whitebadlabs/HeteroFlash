import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paired_analysis import summarize

class TestAnalysis(unittest.TestCase):
    def test_identical_paired_accuracy(self):
        rows=[]
        for i in range(50):
            rows += [dict(pair_id=str(i),system='T',accuracy='1',decode_tps='25'),
                     dict(pair_id=str(i),system='S1',accuracy='1',decode_tps='30')]
        r=summarize(rows,'T','S1',replicates=100)
        self.assertTrue(r['metrics']['accuracy']['passes_exploratory_lower_bound'])
        self.assertAlmostEqual(r['metrics']['decode_tps']['mean_candidate_minus_reference'],5.0)

    def test_disjoint_rows_not_paired(self):
        r=summarize([dict(pair_id='a',system='T',accuracy='1'),
                     dict(pair_id='b',system='S1',accuracy='1')],'T','S1',replicates=10)
        self.assertEqual(r['metrics'],{})

if __name__=='__main__': unittest.main()
