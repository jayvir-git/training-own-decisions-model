import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import explain_decision as explain


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}


def posting(description):
    return {"title": "Developer", "company": "Example Co", "location": "Remote", "description": description}


def verdict(reject):
    return {"verdict": {"strong": (1 - reject) * .75, "stretch": (1 - reject) * .25, "reject": reject}}


class SentenceTests(unittest.TestCase):
    def test_split_sentences_handles_punctuation_newlines_and_bullets(self):
        text = "Build tools. Fix bugs!\n• Write tests\n* Document APIs\nLast line"
        self.assertEqual(explain.split_sentences(text), [
            "Build tools.", "Fix bugs!", "Write tests", "Document APIs", "Last line"])
        self.assertEqual(explain.split_sentences(""), [])

    @patch.object(explain, "_shared_text_tokenizer", return_value=CharacterTokenizer())
    def test_explain_measures_signed_occlusion_effects(self, _):
        original = posting("Build Python tools. Requires 5 years of experience. Entry-level applicants welcome.")

        def decide(p):
            reject = .2
            if "Requires 5 years" in p["description"]:
                reject += .6
            if "Entry-level" not in p["description"]:
                reject += .2
            return verdict(reject)

        before = copy.deepcopy(original)
        result = explain.explain_posting(original, decide)
        self.assertEqual(original, before)
        self.assertAlmostEqual(result["base"]["reject"], .8)
        years = result["sentences"][1]
        self.assertAlmostEqual(years["delta_reject"], .6)
        self.assertAlmostEqual(years["delta_good"], -.6)
        self.assertEqual(years["verdict_if_removed"], "strong")
        entry = result["sentences"][2]
        self.assertAlmostEqual(entry["delta_reject"], -.2)
        self.assertAlmostEqual(entry["delta_good"], .2)
        self.assertEqual(result["top_reject_drivers"][0]["index"], 1)
        self.assertEqual(result["top_good_drivers"][0]["index"], 2)

    @patch.object(explain.bdd, "truncate_posting", return_value=("First sentence.", False))
    @patch.object(explain, "_shared_text_tokenizer", return_value=CharacterTokenizer())
    def test_out_of_state_sentence_is_not_scored(self, _, __):
        calls = []

        def decide(p):
            calls.append(p["description"])
            return verdict(.7)

        result = explain.explain_posting(posting("First sentence. Hidden sentence."), decide)
        self.assertEqual(len(calls), 2)  # base plus the sole in-state occlusion
        self.assertTrue(result["sentences"][0]["in_state"])
        self.assertFalse(result["sentences"][1]["in_state"])
        self.assertEqual(result["sentences"][1]["delta_reject"], 0)
        self.assertEqual(result["sentences"][1]["verdict_if_removed"], "reject")

    @patch.object(explain, "_shared_text_tokenizer", return_value=CharacterTokenizer())
    def test_bad_scorer_contract_fails(self, _):
        with self.assertRaises(ValueError):
            explain.explain_posting(posting("One sentence."), lambda _: {"verdict": {"reject": 1}})


class SummaryTests(unittest.TestCase):
    def test_gate_and_suspicious_driver_accounting(self):
        results = [
            {"posting_id": "gate", "base": verdict(.9)["verdict"], "sentences": [
                {"index": 0, "text": "Requires 5 years of experience.", "in_state": True, "delta_reject": .4}]},
            {"posting_id": "eeo", "base": verdict(.8)["verdict"], "sentences": [
                {"index": 0, "text": "We are an equal opportunity employer.", "in_state": True, "delta_reject": .2}]},
            {"posting_id": "good", "base": verdict(.1)["verdict"], "sentences": []},
        ]
        summary = explain.summarize(results)
        self.assertEqual(summary["reject_postings_with_top_driver"], 2)
        self.assertEqual(summary["rejects_top_driver_gate_count"], 1)
        self.assertEqual(summary["rejects_top_driver_gate_fraction"], .5)
        self.assertEqual(summary["suspicious_reject_top_driver_count"], 1)
        self.assertEqual(summary["suspicious_cases"][0]["posting_id"], "eeo")


class PostingResolutionTests(unittest.TestCase):
    def test_state_only_rows_resolve_by_id_and_source_id(self):
        rows = [{"posting_id": "one", "state": "first"},
                {"posting_id": "export:two", "source_id": "source-two", "state": "second"}]
        sources = [posting("First description.") | {"posting_id": "one"},
                   posting("Second description.") | {"posting_id": "other", "source_id": "source-two"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "postings.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8")
            resolved = explain.load_postings(Path("val.jsonl"), rows, [path])
        self.assertEqual([p["description"] for _, p in resolved], ["First description.", "Second description."])


class NativeDecodeTests(unittest.TestCase):
    def test_native_response_drops_verification_and_validates_options(self):
        client = explain.GGUFClient()
        client.base_url = "http://127.0.0.1:1"
        client._request = lambda path, body: {"answers": {
            "verdict": {"probabilities": {"strong": .1, "stretch": .2, "reject": .7}},
            "sponsorship_blocked": {"noul": .1},
            "clearance_or_citizenship_required": {"noul": .2},
            "min_years": {"probabilities": {"0": .1, "1": .2, "2": .3, "3": .4}},
        }}
        response = client.decide_state("fictional state")
        self.assertEqual(response["answers"]["verdict"]["probabilities"]["reject"], .7)
        client.stop()


if __name__ == "__main__":
    unittest.main()
