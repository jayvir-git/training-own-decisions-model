import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import HTTPServer
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serve_decision as serve
import export_model as export


REQUEST = {"state": "Fictional example", "questions": {
    "eligible": {"type": "noul", "instructions": "Is the role remote?"}}}


class FakeEngine:
    precision = "4bit"
    load_seconds = 1.0

    def vram(self):
        return {"allocated_mib": 0}

    def decide(self, body):
        serve.validate_request(body)
        return {"answers": {"eligible": {"answer": True, "probabilities": {"true": .8, "false": .2}}}}


class ServingTests(unittest.TestCase):
    def test_request_contract(self):
        serve.validate_request(REQUEST)
        serve.validate_request({"posting": {"title": "Developer", "company": "Example Co",
            "location": "Remote", "description": "Build sample tools."}})
        serve.validate_request({"state": "x", "questions": {"expanded": {"type": "choice",
            "instructions": "Choose", "criteria": {str(n): "option" for n in range(8)}}}})
        for bad in (None, [], {}, {"state": 3, "questions": {}}, {**REQUEST, "posting": {}},
                    {"state": "x", "questions": {"x": {"type": "score", "instructions": "x", "criteria": []}}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                serve.validate_request(bad)

    def test_output_containment(self):
        self.assertEqual(serve.local_output(serve.REPO / "outputs/run/a.json").name, "a.json")
        for path in (serve.REPO, serve.REPO / "outputs", serve.REPO / "outputs/../training/file"):
            with self.assertRaises(ValueError):
                serve.local_output(path)

    def test_http(self):
        server = HTTPServer(("127.0.0.1", 0), serve.handler_for(FakeEngine()))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            def request(method, path, body=None, headers=None):
                client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                client.request(method, path, body, headers or {})
                r = client.getresponse()
                status, data = r.status, json.loads(r.read())
                client.close()
                return status, data
            self.assertEqual(request("GET", "/health")[0], 200)
            self.assertEqual(request("GET", "/bad")[0], 404)
            status, result = request("POST", "/decide", json.dumps(REQUEST))
            self.assertEqual(status, 200)
            self.assertIs(result["answers"]["eligible"]["answer"], True)
            self.assertEqual(request("POST", "/decide", "{")[0], 400)
            self.assertEqual(request("POST", "/decide", "null")[0], 400)
            self.assertEqual(request("POST", "/decide", "", {"Content-Length": str(serve.MAX_BODY + 1)})[0], 413)
            self.assertEqual(request("POST", "/other", "{}")[0], 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_lock_reuse(self):
        with patch("gpu_lock.holder", return_value={"pid": 123}), patch("gpu_lock.alive", return_value=True), \
                patch("psutil.Process") as process, patch.dict("os.environ", {}, clear=True):
            process.return_value.parents.return_value = [type("Parent", (), {"pid": 123})()]
            serve.under_gpu_lock()
            self.assertEqual(serve.os.environ["DECISION_GPU_LOCK_PID"], "123")

    def test_foreign_lock_is_not_reused_from_environment(self):
        with patch("gpu_lock.holder", return_value={"pid": 123}), patch("psutil.Process") as process, \
                patch.dict("os.environ", {"DECISION_GPU_LOCK_PID": "123"}), \
                patch("serve_decision.subprocess.call", return_value=0) as call:
            process.return_value.parents.return_value = []
            with self.assertRaises(SystemExit):
                serve.under_gpu_lock()
            self.assertIn("gpu_lock.py", call.call_args.args[0][2])


class ExportTests(unittest.TestCase):
    def row(self, answer=True, probability=.8):
        return {"posting_id": "fake-1", "pred": {"eligible": answer},
                "probabilities": {"eligible": {"true": probability, "false": 1 - probability}}}

    def test_agreement(self):
        r = export.compare([self.row()], [self.row()])
        self.assertEqual(r["answer_agreement"], 1)
        self.assertEqual(r["max_probability_difference"], 0)
        r = export.compare([self.row()], [self.row(False, .2)])
        self.assertEqual(r["answer_agreement"], 0)
        self.assertAlmostEqual(r["max_probability_difference"], .6)
        self.assertEqual(export.compare([self.row()], [self.row(1)])["answer_agreement"], 0)

    def test_incompatible_reference(self):
        for actual in ([], [self.row(), self.row()], [{**self.row(), "posting_id": "other"}],
                       [{**self.row(), "pred": {"new": True}}], [self.row(probability=float("nan"))]):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                export.compare([self.row()], actual)

    def test_gguf_preflight_does_not_import_unsloth(self):
        with patch.dict("os.environ", {"UNSLOTH_LLAMA_CPP_SCRIPTS_DIR": str(serve.REPO / "outputs/nonexistent-tools")}), \
                patch("export_model.shutil.which", return_value=None):
            _, _, missing = export.gguf_prerequisites()
            self.assertTrue(any("clef.py" in s for s in missing))
            self.assertNotIn("unsloth", sys.modules)

    def test_native_typed_response(self):
        row = {"posting_id": "fake", "questions": {
            "n": {"type": "noul"}, "c": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            "s": {"type": "score", "criteria": ["low", "high"]}}}
        result = export.native_prediction(row, {"answers": {"n": {"noul": .8},
            "c": {"probabilities": {"a": .2, "b": .8}}, "s": {"probabilities": {"0": .1, "1": .9}}}})
        self.assertEqual(result["pred"], {"n": True, "c": "b", "s": 1})
        self.assertIsInstance(result["pred"]["s"], int)

    def test_export_http_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            export.smoke_http(FakeEngine(), Path(folder))
            result = json.loads((Path(folder) / "http-smoke.json").read_text())
            self.assertEqual(set(result), {"health", "state", "posting"})


if __name__ == "__main__":
    unittest.main()
