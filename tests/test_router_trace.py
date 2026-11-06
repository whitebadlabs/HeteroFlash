import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from router_trace import (TraceMeta, TraceWriter, read_trace, selection_counts, coverage, hot_set,
                          all_hot_probability, window_union, empirical_probabilities, summarize)
from router_surgery import route
import placement_sim


def meta(topn=0):
    return TraceMeta(model_id='test', model_revision='r0', engine='unit', source='synthetic',
                     num_experts=8, top_k=2, moe_layers=[3, 4], routed_scaling_factor=2.5,
                     norm_topk_prob=True, topn_recorded=topn)


class TestRouterTrace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 't.jsonl.gz'

    def tearDown(self):
        self.tmp.cleanup()

    def write_simple(self):
        with TraceWriter(self.path, meta()) as w:
            # request a, decode, 4 tokens, layer 3 uses experts {0,1},{0,2},{0,1},{5,6}
            w.record(request_id='a', domain='prose', phase='decode', layer=3, token_start=0,
                     expert_ids=[[0, 1], [0, 2], [0, 1], [5, 6]], gate_weights=[[1.25, 1.25]] * 4)
            w.record(request_id='a', domain='prose', phase='decode', layer=4, token_start=0,
                     expert_ids=[[7, 6]] * 4, gate_weights=[[1.25, 1.25]] * 4)
            w.record(request_id='a', domain='prose', phase='prefill', layer=3, token_start=0,
                     expert_ids=[[3, 4]], gate_weights=[[1.25, 1.25]])

    def test_roundtrip_and_counts(self):
        self.write_simple()
        m, recs = read_trace(self.path)
        recs = list(recs)
        self.assertEqual(m.top_k, 2)
        self.assertEqual(len(recs), 3)
        c = selection_counts(m, recs, phase='decode')
        self.assertEqual(c[3], [3, 2, 1, 0, 0, 1, 1, 0])
        self.assertEqual(c[4][7], 4)
        self.assertEqual(selection_counts(m, recs, phase='prefill')[3][3], 1)

    def test_coverage_hot_and_all_hot(self):
        self.write_simple()
        m, recs = read_trace(self.path)
        recs = [r for r in recs if r['phase'] == 'decode']
        c = selection_counts(m, recs)
        self.assertAlmostEqual(coverage(c[3], 0.25), 5 / 8)       # top-2 of 8 experts = ids 0,1
        self.assertEqual(hot_set(c[3], 2), [0, 1])
        p = all_hot_probability(m, recs, {3: {0, 1}, 4: {6, 7}})
        self.assertAlmostEqual(p[3], 0.5)                          # tokens {0,1},{0,1} of 4
        self.assertAlmostEqual(p[4], 1.0)

    def test_window_union(self):
        self.write_simple()
        m, recs = read_trace(self.path)
        u = window_union(m, list(recs), 2)
        self.assertAlmostEqual(u[3], (3 + 4) / 2)                  # {0,1,2} and {0,1,5,6}
        self.assertAlmostEqual(u[4], 2.0)

    def test_validation_rejects_bad_records(self):
        with TraceWriter(self.path, meta()) as w:
            with self.assertRaises(ValueError):
                w.record(request_id='a', domain='x', phase='decode', layer=5, token_start=0,
                         expert_ids=[[0, 1]], gate_weights=[[1, 1]])   # layer not MoE
            with self.assertRaises(ValueError):
                w.record(request_id='a', domain='x', phase='decode', layer=3, token_start=0,
                         expert_ids=[[0, 0]], gate_weights=[[1, 1]])   # duplicate expert
            with self.assertRaises(ValueError):
                w.record(request_id='a', domain='x', phase='warmup', layer=3, token_start=0,
                         expert_ids=[[0, 1]], gate_weights=[[1, 1]])
        with self.assertRaises(ValueError):
            TraceWriter(self.path, TraceMeta('m', 'r', 'e', 'made-up', 8, 2, [3], 2.5, True))

    def test_topn_and_router_surgery_parity(self):
        logits = [0.3, -1.2, 2.0, 0.9, -0.1, 1.5, 0.0, -2.2]
        bias = [0.0] * 8
        r = route(logits, bias, k_active=2, scale=2.5)
        with TraceWriter(self.path, meta(topn=4)) as w:
            w.record(request_id='p', domain='x', phase='decode', layer=3, token_start=0,
                     expert_ids=[r['expert_ids']], gate_weights=[r['weights']],
                     topn_ids=[route(logits, bias, k_active=4, scale=2.5)['expert_ids']],
                     topn_scores=[[0.9, 0.8, 0.7, 0.6]])
        m, recs = read_trace(self.path)
        rec = next(recs)
        self.assertEqual(rec['expert_ids'][0], [2, 5])
        self.assertAlmostEqual(sum(rec['gate_weights'][0]), 2.5, places=5)
        self.assertEqual(rec['topn_ids'][0][:2], [2, 5])

    def test_placement_from_trace_and_calibration(self):
        self.write_simple()
        experts, m = placement_sim.experts_from_trace(str(self.path), 1000, phase='decode')
        probs = empirical_probabilities(selection_counts(m, list(read_trace(self.path)[1]), phase='decode'))
        self.assertAlmostEqual(sum(e.probability for e in experts if e.layer == 3), 1.0)
        self.assertEqual(len(experts), 16)
        self.assertAlmostEqual(probs[3][0], 3 / 8)
        cfg = json.loads((ROOT / 'configs/synthetic_demo.json').read_text())
        prov = placement_sim.apply_calibration(cfg, json.loads((ROOT / 'configs/calibration/2026-11-06.json').read_text()))
        self.assertEqual(prov['assumed_remote_roundtrip_ms']['status'], 'measured')
        self.assertEqual(prov['assumed_spark_bw_GBps']['status'], 'measured')
        self.assertEqual(cfg['assumed_remote_roundtrip_ms'], 0.54)
        with self.assertRaises(ValueError):
            placement_sim.apply_calibration(cfg, {'schema': 'other', 'measurements': {}})

    def test_summarize(self):
        self.write_simple()
        s = summarize(self.path)
        self.assertEqual(s['source'], 'synthetic')
        self.assertEqual(s['tokens_by_phase']['decode'], 4)


if __name__ == '__main__':
    unittest.main()
