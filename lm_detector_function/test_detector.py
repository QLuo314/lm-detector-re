"""Integration tests: real official CLI and a local fake Chat Completions server."""
import json
import random
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lm_detector_function import DetectionError, detect_model


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.headers.get("Authorization") == "Bearer failure-test-key":
            self.send_response(401)
            body = {"error": {"message": "invalid failure-test-key"}}
        else:
            self.send_response(200)
            rng = random.Random(42)
            text = " ".join(str(rng.randint(1, 355)) for _ in range(500))
            body = {"id": "test", "object": "chat.completion", "model": "mock",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": "stop"}]}
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_real_cli_multiround(self):
        report = detect_model(self.url, "local-test-key", "mock", api="cc", stream=False,
                              count=1, repeat=2, total_timeout=60)
        self.assertEqual(report["scored_rounds"], 2)
        self.assertEqual(len(report["rounds"]), 2)
        self.assertTrue(report["rounds"][0]["analysis"]["results"])
        self.assertNotIn("local-test-key", json.dumps(report))

    def test_failed_request_preserves_report(self):
        with self.assertRaises(DetectionError) as caught:
            detect_model(self.url, "failure-test-key", "mock", api="cc", stream=False,
                         count=1, total_timeout=60)
        self.assertIsInstance(caught.exception.report, dict)
        self.assertNotIn("failure-test-key", str(caught.exception) + json.dumps(caught.exception.report))

    def test_total_timeout(self):
        with self.assertRaisesRegex(DetectionError, "total_timeout"):
            detect_model(self.url, "local-test-key", "mock", total_timeout=0.001)

    def test_validation(self):
        for kwargs in ({"count": 4}, {"repeat": 0}, {"timeout": float("nan")},
                       {"api": "invalid"}, {"strict": True, "count": 1}):
            with self.assertRaises(ValueError):
                detect_model(self.url, "local-test-key", "mock", **kwargs)


if __name__ == "__main__":
    unittest.main()
