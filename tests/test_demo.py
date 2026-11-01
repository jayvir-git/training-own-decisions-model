"""CPU-only contract tests for the demo server: fake backend, no GPU, no llama-server."""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import HTTPServer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "demo"))
import app as demo

PROFILE_MARKER = "PRIVATE-PROFILE-MARKER-ZQ7"
POSTING = {"title": "Junior Software Developer", "company": "Example Tools Ltd",
           "location": "Remote, United States",
           "description": "Build Python tools and SQL reports. Entry-level applicants welcome. "
                          "No minimum years of experience stated."}


class FakeLlama:
    """LlamaBackend stand-in: same decide(state, questions) -> raw typed answers shape."""
    load_seconds = 0.5
    state = None

    def describe(self):
        return {"kind": "fake"}

    def decide(self, state, questions):
        FakeLlama.state = state
        return {
            "verdict": {"probabilities": {"strong": .71, "stretch": .2, "reject": .09}},
            "needs_verification": {"noul": .9},
            "sponsorship_blocked": {"noul": .04},
            "clearance_or_citizenship_required": {"noul": .03},
            "min_years": {"probabilities": {"0": .8, "1": .15, "2": .04, "3": .01}},
        }


def fake_explainer():
    def explain_posting(posting, decide):
        base = decide(posting)
        return {"base": base,
                "sentences": [
                    {"index": 0, "text": "First sentence.", "in_state": True,
                     "delta_reject": .2, "delta_good": 0, "verdict_if_removed": "stretch"},
                    {"index": 1, "text": "Second sentence.", "in_state": False,
                     "delta_reject": 0, "delta_good": .1, "verdict_if_removed": "strong"}],
                "top_reject_drivers": [0], "top_good_drivers": [1]}
    return explain_posting


def make_app(folder):
    profile = Path(folder) / "profile.md"
    profile.write_text("Candidate profile. " + PROFILE_MARKER +
                       " Secret details that must never leave the server.", encoding="utf-8")
    builder = demo.StateBuilder(profile=profile, count=lambda s: max(1, len(s) // 4))
    return demo.DemoApp(builder, FakeLlama())


class DemoUnitTests(unittest.TestCase):
    def test_validate_posting(self):
        demo.validate_posting(POSTING)
        for bad in (None, [], {}, "x", {**POSTING, "extra": 1},
                    {k: v for k, v in POSTING.items() if k != "location"},
                    {**POSTING, "description": 3}, {**POSTING, "description": "  "}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                demo.validate_posting(bad)

    def test_examples_are_valid_fictional_postings(self):
        self.assertGreaterEqual(len(demo.EXAMPLES), 3)
        ids = set()
        for example in demo.EXAMPLES:
            self.assertNotIn(example["id"], ids)
            ids.add(example["id"])
            demo.validate_posting(example["posting"])
            self.assertIn(example["expect"], ("strong", "stretch", "reject"))

    def test_decode_answers(self):
        questions = {"n": {"type": "noul"}, "c": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
                     "s": {"type": "score", "criteria": ["low", "high"]}}
        answers = demo.decode_answers(questions, {"n": {"noul": .8},
                                                  "c": {"probabilities": {"a": .2, "b": .8}},
                                                  "s": {"probabilities": {"0": .1, "1": .9}}})
        self.assertIs(answers["n"]["answer"], True)
        self.assertEqual(answers["n"]["probabilities"]["true"], .8)
        self.assertAlmostEqual(answers["n"]["probabilities"]["false"], .2)
        self.assertEqual(answers["c"]["answer"], "b")
        self.assertEqual(answers["s"]["answer"], 1)
        self.assertIsInstance(answers["s"]["answer"], int)
        self.assertEqual(demo.decode_answers(questions, {"n": {"noul": 1.4},
                                                         "c": {"probabilities": {"a": .2, "b": .8}},
                                                         "s": {"probabilities": {"0": .1, "1": .9}}})
                         ["n"]["probabilities"]["true"], 1.0)
        with self.assertRaises(ValueError):
            demo.decode_answers(questions, {"n": {"noul": .5}, "c": {"probabilities": {"a": 1.0}},
                                            "s": {"probabilities": {"0": .1, "1": .9}}})


class DemoHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = make_app(self.tmp.name)
        self.server = HTTPServer(("127.0.0.1", 0),
                                 demo.handler_for(self.app, explainer_loader=fake_explainer))
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.bodies = []

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        client = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        client.request(method, path, body, headers or {})
        response = client.getresponse()
        status, raw = response.status, response.read()
        client.close()
        self.bodies.append(raw)
        return status, raw, response

    def json_request(self, method, path, body):
        status, raw, _ = self.request(method, path, json.dumps(body),
                                      {"Content-Type": "application/json"})
        return status, json.loads(raw)

    def test_decide_shape(self):
        status, result = self.json_request("POST", "/api/decide", POSTING)
        self.assertEqual(status, 200)
        self.assertEqual(set(result), {"answers", "state_truncated", "latency_ms"})
        self.assertEqual(set(result["answers"]),
                         {"verdict", "sponsorship_blocked", "clearance_or_citizenship_required", "min_years"})
        verdict = result["answers"]["verdict"]
        self.assertIn(verdict["answer"], ("strong", "stretch", "reject"))
        self.assertAlmostEqual(sum(verdict["probabilities"].values()), 1.0, places=5)
        gate = result["answers"]["sponsorship_blocked"]
        self.assertIs(gate["answer"], False)
        self.assertEqual(set(gate["probabilities"]), {"true", "false"})
        years = result["answers"]["min_years"]
        self.assertIsInstance(years["answer"], int)
        self.assertIn(years["answer"], (0, 1, 2, 3))
        self.assertEqual(set(years["probabilities"]), {"0", "1", "2", "3"})
        self.assertGreaterEqual(result["latency_ms"], 0)
        self.assertIsInstance(result["state_truncated"], bool)
        self.assertFalse(result["state_truncated"])

    def test_long_posting_reports_truncation(self):
        long_posting = dict(POSTING, description="Must have 3 years of experience. " * 400)
        status, result = self.json_request("POST", "/api/decide", long_posting)
        self.assertEqual(status, 200)
        self.assertTrue(result["state_truncated"])

    def test_explain_passthrough_shape(self):
        status, result = self.json_request("POST", "/api/explain", POSTING)
        self.assertEqual(status, 200)
        self.assertEqual(set(result), {"base", "sentences", "top_reject_drivers", "top_good_drivers"})
        self.assertIn("verdict", result["base"])
        self.assertEqual(len(result["sentences"]), 2)
        for sentence in result["sentences"]:
            self.assertEqual(set(sentence),
                             {"index", "text", "in_state", "delta_reject", "delta_good", "verdict_if_removed"})

    def test_explain_missing_module_returns_501(self):
        server = HTTPServer(("127.0.0.1", 0),
                            demo.handler_for(self.app, explainer_loader=self.missing))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            client.request("POST", "/api/explain", json.dumps(POSTING),
                           {"Content-Type": "application/json"})
            response = client.getresponse()
            self.assertEqual(response.status, 501)
            body = response.read()
            self.bodies.append(body)
            client.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    @staticmethod
    def missing():
        raise ImportError("explain_decision is not available")

    def test_bad_requests(self):
        self.assertEqual(self.request("POST", "/api/decide", "{")[0], 400)
        self.assertEqual(self.request("POST", "/api/decide", "null")[0], 400)
        self.assertEqual(self.json_request("POST", "/api/decide", {"title": "x"})[0], 400)
        self.assertEqual(self.json_request("POST", "/api/decide",
                                           {**POSTING, "description": ""})[0], 400)
        self.assertEqual(self.request("POST", "/api/decide", "",
                                      {"Content-Length": str(demo.MAX_BODY + 1)})[0], 413)
        self.assertEqual(self.json_request("POST", "/api/other", POSTING)[0], 404)
        self.assertEqual(self.request("GET", "/nope")[0], 404)

    def test_html_served(self):
        status, raw, response = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", response.getheader("Content-Type"))
        self.assertIn(b"decide-btn", raw)
        self.assertIn(b"api/decide", raw)
        self.assertNotIn(b"http://", raw)  # no external resources

    def test_examples_endpoint(self):
        status, raw, _ = self.request("GET", "/api/examples")
        examples = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertEqual(len(examples), len(demo.EXAMPLES))

    def test_health(self):
        status, raw, _ = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["status"], "ok")

    def test_profile_never_appears_in_responses(self):
        # The built state really contains the private profile text...
        self.app.decide_posting(POSTING)
        self.assertIn(PROFILE_MARKER, FakeLlama.state)
        # ...but no endpoint, error or success, ever leaks it.
        self.json_request("POST", "/api/decide", POSTING)
        self.json_request("POST", "/api/explain", POSTING)
        self.request("GET", "/")
        self.request("GET", "/health")
        self.request("GET", "/api/examples")
        self.request("POST", "/api/decide", "{")
        self.json_request("POST", "/api/decide", {"title": "x"})
        self.request("GET", "/missing")
        for raw in self.bodies:
            self.assertNotIn(PROFILE_MARKER.encode("utf-8"), raw)


if __name__ == "__main__":
    unittest.main()
