import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import ttft_bench


class Fake(BaseHTTPRequestHandler):
    cache = set()          # exact prefixes (tuples) that are "cached"
    leak_cache = False     # if True, cold prompts report cached tokens (should be flagged)
    spec = {'drafts': 0, 'draft_tokens': 0, 'accepted': 0}

    def do_GET(self):
        if self.path == '/metrics':
            s = Fake.spec
            txt = (f"# HELP x\nvllm:spec_decode_num_drafts_total{{model_name=\"m\"}} {s['drafts']}\n"
                   f"vllm:spec_decode_num_draft_tokens_total{{model_name=\"m\"}} {s['draft_tokens']}\n"
                   f"vllm:spec_decode_num_accepted_tokens_total{{model_name=\"m\"}} {s['accepted']}\n").encode()
            self.send_response(200); self.send_header('Content-Length', str(len(txt))); self.end_headers(); self.wfile.write(txt)

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/tokenize':
            text = body['prompt'] if 'prompt' in body else ''.join(m['content'] for m in body['messages'])
            out = {'tokens': [1000 + (ord(c) % 50) for c in text]}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
            return
        ids = tuple(body['prompt'])
        cached = 0
        for p in Fake.cache:
            if ids[:len(p)] == p:
                cached = max(cached, len(p))
        if Fake.leak_cache:
            cached = len(ids)
        new = len(ids) - cached
        time.sleep(0.002 + new * 0.00002)            # "prefill"
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        cont = (body.get('stream_options') or {}).get('continuous_usage_stats')
        k, emitted = 0, 0
        while emitted < body['max_tokens']:
            n = min(3 if cont else 1, body['max_tokens'] - emitted)      # spec decoding: 3 tokens per verify
            emitted += n
            ev = {'choices': [{'text': 'x' * n}]}
            if cont:
                ev['usage'] = {'completion_tokens': emitted}
                Fake.spec['drafts'] += 1; Fake.spec['draft_tokens'] += 3; Fake.spec['accepted'] += n - 1
            self.wfile.write(b'data: ' + json.dumps(ev).encode() + b'\n\n'); self.wfile.flush()
            time.sleep(0.003)
        usage = {'prompt_tokens': len(ids), 'completion_tokens': body['max_tokens'], 'prompt_tokens_details': {'cached_tokens': cached}}
        self.wfile.write(b'data: ' + json.dumps({'choices': [], 'usage': usage}).encode() + b'\n\n')
        self.wfile.write(b'data: [DONE]\n\n')
        if body['max_tokens'] == 1:
            Fake.cache.add(ids)


class TestTTFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(('127.0.0.1', 0), Fake)
        cls.base = f'http://127.0.0.1:{cls.srv.server_address[1]}'
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_stream_parse_and_cache_state(self):
        Fake.cache = set(); Fake.leak_cache = False
        r = ttft_bench.stream_completion(self.base, 'm', list(range(2000)), max_tokens=3)
        self.assertEqual(r['prompt_tokens'], 2000)
        self.assertEqual(r['cached_tokens'], 0)
        self.assertEqual(r['n_chunks'], 3)
        self.assertLess(r['t_first'], r['t_done'])
        ttft_bench.stream_completion(self.base, 'm', list(range(2000)), max_tokens=1)
        r2 = ttft_bench.stream_completion(self.base, 'm', list(range(2000)) + [7] * 32)
        self.assertEqual(r2['cached_tokens'], 2000)
        self.assertLess(r2['t_first'], r['t_first'])

    def test_full_run_flags_and_summary(self):
        Fake.cache = set(); Fake.leak_cache = False
        with tempfile.TemporaryDirectory() as d:
            corpus = Path(d) / 'c.txt'; corpus.write_text('abcdefghij' * 400)
            out = Path(d) / 'o.jsonl'
            sys.argv = ['x', '--base', self.base, '--model', 'm', '--corpus', str(corpus), '--n', '3000',
                        '--suffixes', '32', '--repeats', '4', '--block-size', '16', '--out', str(out)]
            ttft_bench.main()
            recs = [json.loads(l) for l in out.read_text().splitlines()]
            trials = [r for r in recs if r['type'] == 'trial']
            self.assertEqual(len(trials), 8)
            self.assertFalse(any(t['violation'] for t in trials))
            summ = recs[-1]
            self.assertEqual(summ['cold_3000']['n'], 4)
            self.assertIn('pass_p95_upper_ci', summ['cached_3000+32'])
            self.assertLess(summ['cached_3000+32']['p50'], summ['cold_3000']['p50'])

    def test_cold_cache_leak_is_flagged(self):
        Fake.cache = set(); Fake.leak_cache = True
        with tempfile.TemporaryDirectory() as d:
            corpus = Path(d) / 'c.txt'; corpus.write_text('abcdefghij' * 400)
            out = Path(d) / 'o.jsonl'
            sys.argv = ['x', '--base', self.base, '--model', 'm', '--corpus', str(corpus), '--n', '3000',
                        '--suffixes', '32', '--repeats', '2', '--block-size', '16', '--out', str(out)]
            ttft_bench.main()
            trials = [json.loads(l) for l in out.read_text().splitlines() if '"trial"' in l]
            self.assertTrue(all(t['violation'] for t in trials if t['cond'].startswith('cold')))
        Fake.leak_cache = False


class TestDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(('127.0.0.1', 0), Fake)
        cls.base = f'http://127.0.0.1:{cls.srv.server_address[1]}'
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_decode_tpot_itl_and_acceptance(self):
        Fake.cache = set(); Fake.leak_cache = False
        with tempfile.TemporaryDirectory() as d:
            pf = Path(d) / 'p.jsonl'
            pf.write_text('\n'.join(json.dumps({'prompt_id': f'w{i}', 'domain': 'writing_chat', 'split': 'holdout',
                                                 'messages': [{'role': 'user', 'content': 'hello world'}]}) for i in range(3)))
            out = Path(d) / 'd.jsonl'
            ttft_bench.decode_main(['--base', self.base, '--model', 'm', '--prompts', str(pf), '--max-tokens', '30',
                                    '--out', str(out)])
            recs = [json.loads(l) for l in out.read_text().splitlines()]
            trials = [r for r in recs if r['type'] == 'trial']
            summ = recs[-1]
            self.assertEqual(len(trials), 3)
            self.assertEqual(trials[0]['completion_tokens'], 30)
            self.assertAlmostEqual(trials[0]['tokens_per_chunk_mean'], 3.0)
            self.assertGreater(trials[0]['itl_chunk_p50'], 0.002)
            self.assertAlmostEqual(summ['spec']['committed_tokens_per_verify'], 3.0)
            self.assertAlmostEqual(summ['spec']['draft_acceptance_rate'], 2 / 3, places=4)
            self.assertIn('committed_tokens_needed_at_this_step_for_target', summ)


if __name__ == '__main__':
    unittest.main()
