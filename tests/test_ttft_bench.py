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

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/tokenize':
            out = {'tokens': [1000 + (ord(c) % 50) for c in body['prompt']]}
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
        for k in range(body['max_tokens']):
            self.wfile.write(b'data: ' + json.dumps({'choices': [{'text': 'x'}]}).encode() + b'\n\n'); self.wfile.flush()
            time.sleep(0.001)
        usage = {'prompt_tokens': len(ids), 'prompt_tokens_details': {'cached_tokens': cached}}
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


if __name__ == '__main__':
    unittest.main()
