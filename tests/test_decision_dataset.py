import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("build_decision_dataset", SCRIPTS / "build_decision_dataset.py")
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)
from build_dataset import normalize_employer  # noqa: E402


def posting(pid, employer, text="Build backend services in Python."):
    return {"posting_id": pid, "employer": employer, "normalized_employer": normalize_employer(employer),
            "employer_aliases": [employer], "title": "Software Engineer", "locations": ["Remote, US"],
            "description": text}


def label(verdict, **kw):
    base = {"verdict": verdict, "needs_verification": "no", "sponsorship_blocked": "unknown",
            "clearance_or_citizenship_required": "no", "min_years": None}
    return base | kw


class GoldTests(unittest.TestCase):
    def test_label_to_gold_types(self):
        gold = dd.label_to_gold(label("stretch", sponsorship_blocked="yes", min_years=3))
        self.assertEqual(gold, {"verdict": "stretch", "needs_verification": False, "sponsorship_blocked": True,
                                "clearance_or_citizenship_required": False, "min_years": 2})
        self.assertFalse(dd.label_to_gold(label("reject"))["sponsorship_blocked"])  # unknown -> False

    def test_years_buckets_follow_the_rule_boundaries(self):
        self.assertEqual([dd.years_bucket(y) for y in (None, 0, 1, 2, 3, 4, 5, 12)], [0, 0, 1, 1, 2, 2, 3, 3])

    def test_latest_receipt_skips_needs_information(self):
        receipts = [
            {"posting_id": "a", "verdict": "reject", "reviewed_at": "2026-10-01"},
            {"posting_id": "a", "verdict": "stretch", "reviewed_at": "2026-10-05"},
            {"posting_id": "a", "verdict": "needs-information", "reviewed_at": "2026-10-09"},
            {"posting_id": "b", "verdict": "needs-information", "reviewed_at": "2026-10-09"},
        ]
        self.assertEqual(dd.latest_receipts(receipts), {"a": "stretch"})

    def test_bulk_final_rows_flatten_and_pending_rows_have_no_verdict(self):
        final = {"posting_id": "b", "weight": "2.5", "stratum": "uniform", "status": "final",
                 "label_source": "luna+glm agree", "verdict": "stretch",
                 "final_label": label("stretch", min_years="4", sponsorship_blocked="yes")}
        flat = dd.flatten_bulk(final)
        self.assertEqual(dd.label_to_gold(flat)["min_years"], 2)
        self.assertTrue(dd.label_to_gold(flat)["sponsorship_blocked"])
        self.assertEqual(flat["weight"], "2.5")
        pending = final | {"status": "pending_tiebreak", "verdict": None, "final_label": None}
        self.assertNotIn("verdict", dd.label_to_gold(dd.flatten_bulk(pending)))
        self.assertEqual(dd.flatten_bulk(label("reject"))["verdict"], "reject")  # bulk-labels format

    def test_questions_are_valid_decision_questions(self):
        self.assertEqual(set(dd.QUESTIONS["verdict"]["criteria"]), set(dd.VERDICTS))
        self.assertEqual(len(dd.QUESTIONS["min_years"]["criteria"]), 4)
        for q in dd.QUESTIONS.values():
            self.assertIn(q["type"], {"choice", "noul", "score"})


class TruncationTests(unittest.TestCase):
    def test_short_text_untouched(self):
        self.assertEqual(dd.truncate_posting("Short posting.", 100), ("Short posting.", False))

    def test_gate_sentences_survive_over_boilerplate(self):
        filler = " ".join(f"Our team builds great product number {i}." for i in range(80))
        eeo = " ".join("We are an equal opportunity employer and celebrate everyone." for _ in range(20))
        gate = "Candidates must be US citizens and we will not sponsor visas."
        text, truncated = dd.truncate_posting(f"About us. {filler} {gate} {eeo}", 200)
        self.assertTrue(truncated)
        self.assertIn(gate, text)
        self.assertNotIn("equal opportunity", text)
        self.assertTrue(text.startswith("About us."))
        self.assertLessEqual(dd.char_counter(text), 200 + 20)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.splits = {"train": ["t1", "t2", "t3"], "val": ["v1", "v2"], "test": ["x1"],
                       "employer_assignment": {"acme": "train", "valco": "val", "testco": "test"}}
        self.postings = {p["posting_id"]: p for p in [
            posting("t1", "Acme"), posting("t2", "Acme"), posting("t3", "Acme"), posting("v1", "ValCo"),
            posting("v2", "ValCo"), posting("x1", "TestCo"), posting("b1", "NewCo"),
            posting("b2", "ValCo, Inc."), posting("b3", "OtherCo")]}
        self.labeller = {"t1": ("v3-astra-eval", label("strong")), "v1": ("v4-luna-eval", label("stretch")),
                         "v2": ("v3-astra-eval", label("reject")), "x1": ("v3-astra-eval", label("reject"))}
        self.receipts = {"t1": "reject", "t2": "stretch", "v1": "reject"}
        self.bulk = {"b1": label("stretch") | {"weight": 2.0, "stratum": "uniform"},
                     "b2": label("reject"), "v2": label("reject"), "t1": label("reject"), "b3": label("strong")}

    def build(self):
        return dd.build_rows(self.splits, self.postings, self.labeller, self.receipts, self.bulk, "v4-luna-bulk",
                             "PROFILE", 1000, dd.char_counter, normalize_employer)

    def test_gold_priorities(self):
        rows, stats = self.build()
        train = {r["posting_id"]: r for r in rows["train"]}
        self.assertEqual(train["t1"]["gold"]["verdict"], "strong")  # labeller beats receipt in train
        self.assertEqual(train["t1"]["label_source"], "v3-astra-eval")
        self.assertEqual(train["t2"]["gold"], {"verdict": "stretch"})  # receipt fallback, verdict only
        self.assertEqual(set(train["t2"]["questions"]), {"verdict"})
        self.assertNotIn("t3", train)
        self.assertEqual(stats["train_no_label"], 1)
        val = {r["posting_id"]: r for r in rows["val"]}
        self.assertEqual(val["v1"]["gold"]["verdict"], "reject")  # receipt beats labeller in val
        self.assertEqual(val["v1"]["label_source"], "receipt")
        self.assertEqual(val["v1"]["labeller_gold"]["verdict"], "stretch")
        self.assertIn("sponsorship_blocked", val["v1"]["gold"])
        self.assertEqual(val["v2"]["label_source"], "v3-astra-eval")

    def test_bulk_never_leaks_heldout_ids_or_employers(self):
        rows, stats = self.build()
        self.assertEqual(sorted(r["posting_id"] for r in rows["bulk"]), ["b1", "b3"])
        self.assertEqual(stats["bulk_dropped_heldout_id"], 1)
        self.assertEqual(stats["bulk_dropped_heldout_employer"], 1)
        self.assertEqual(stats["bulk_dropped_already_labelled"], 1)
        self.assertEqual(rows["bulk"][0]["weight"], 2.0)
        dd.check_disjoint(rows, self.splits, self.postings, normalize_employer)

    def test_check_disjoint_catches_leak(self):
        rows, _ = self.build()
        rows["train"].append({"posting_id": "v1"})
        with self.assertRaises(AssertionError):
            dd.check_disjoint(rows, self.splits, self.postings, normalize_employer)

    def test_state_contains_profile_and_posting(self):
        rows, _ = self.build()
        state = rows["train"][0]["state"]
        self.assertTrue(state.startswith("PROFILE"))
        self.assertIn("Title: Software Engineer", state)
        self.assertIn("Build backend services", state)


if __name__ == "__main__":
    unittest.main()
