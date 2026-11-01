import copy
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("rule_tests", SCRIPTS / "rule_tests.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def posting(text="Build software services. 3+ years of experience required."):
    return {"title": "Software Engineer", "employer": "Acme", "locations": ["Remote, US"],
            "description": text}


class EditTests(unittest.TestCase):
    def test_years_up_is_at_description_head(self):
        edited, marker = rt.edit_posting(posting("Build software."), "years_up")
        self.assertTrue(edited["description"].startswith(marker))
        self.assertIn("Build software.", edited["description"])

    def test_required_years_uses_lower_bound_and_skips_preferred(self):
        self.assertEqual(rt.required_years("At least 3-5 years of experience required.")["low"], 3)
        self.assertIsNone(rt.required_years("5+ years preferred."))

    def test_years_down_replaces_only_required_sentence(self):
        edited, marker = rt.edit_posting(posting("Build software. 3+ years of experience required. Benefits included."), "years_down")
        self.assertTrue(marker)
        self.assertIn("Requires 1-2 years of relevant experience.", edited["description"])
        self.assertIn("Build software.", edited["description"])
        self.assertIn("Benefits included.", edited["description"])

    def test_controls_do_not_mutate_input(self):
        original = posting()
        snapshot = copy.deepcopy(original)
        edited, kind = rt.control_edit(original, 0)
        self.assertEqual(original, snapshot)
        self.assertEqual(kind, "company")
        self.assertNotEqual(edited["employer"], original["employer"])


class ScoringTests(unittest.TestCase):
    def test_gate_scores_require_typed_gate_and_reject(self):
        base = {"pred": {"verdict": "strong", "min_years": 0, "sponsorship_blocked": False,
                          "clearance_or_citizenship_required": False}}
        self.assertTrue(rt.score_case("years_up", base, {"pred": {"verdict": "reject", "min_years": 3}})[0])
        self.assertTrue(rt.score_case("sponsorship", base, {"pred": {"verdict": "reject", "sponsorship_blocked": True}})[0])
        self.assertFalse(rt.score_case("clearance", base, {"pred": {"verdict": "reject", "clearance_or_citizenship_required": False}})[0])

    def test_invariance_and_preferred_scores(self):
        base = {"pred": {"verdict": "stretch", "min_years": 1}}
        self.assertTrue(rt.score_case("invariance", base, {"pred": {"verdict": "stretch"}})[0])
        self.assertTrue(rt.score_case("preferred", base, {"pred": {"verdict": "stretch", "min_years": 1}})[0])
        self.assertFalse(rt.score_case("invariance", base, {"pred": {"verdict": "reject"}})[0])

    def test_wilson_handles_zero_and_extremes(self):
        self.assertEqual(rt.wilson_interval(0, 0), (0.0, 0.0))
        low, high = rt.wilson_interval(10, 10)
        self.assertEqual(low > 0, True)
        self.assertEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
