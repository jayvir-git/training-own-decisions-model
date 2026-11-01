import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_dataset.py"
spec = importlib.util.spec_from_file_location("build_dataset", SCRIPT)
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


class IdentityTests(unittest.TestCase):
    def test_employer_and_requisition_normalization(self):
        left = {"employer": " Acme, Inc. ", "job_id": "JR-12345"}
        right = {"employer": "ACME LLC", "job_id": "jr12345"}
        self.assertEqual(dataset.dedupe_key(left), dataset.dedupe_key(right))
        self.assertNotEqual(dataset.dedupe_key(left), dataset.dedupe_key(right | {"job_id": "JR12346"}))
        self.assertEqual(dataset.normalize_employer("Smith & Sons, Corporation"), "smith and sons")

    def test_url_fallback_and_job_id_extraction(self):
        left = {"url": "http://example.org/role/123/?utm_source=mail#apply"}
        right = {"url": "https://www.example.org/role/123"}
        self.assertEqual(dataset.dedupe_key(left), dataset.dedupe_key(right))
        self.assertEqual(dataset.url_job_id("https://acme.wd1.myworkdayjobs.com/en-US/jobs/job/NY/Engineer_JR-12345-1"), "jr12345")
        self.assertEqual(dataset.url_job_id("https://www.example.com/company/jobs?gh_jid=1234567890"), "1234567890")
        self.assertEqual(dataset.url_job_id("https://example.wd5.myworkdayjobs.com/Example_External_Career_Site"), "")
        self.assertEqual(dataset.url_job_id("https://file.squarespace-cdn.com/assets/00000000-0000-4000-8000-000000000000/font.woff2"), "")

    def test_tracking_removed_identity_parameters_kept(self):
        url = "https://www.example.org/jobs/?utm_source=mail&gh_jid=123&jobId=456&cid=tenant&gh_src=foo#apply"
        self.assertEqual(dataset.canonical_url(url), "https://example.org/jobs?cid=tenant&gh_jid=123&jobid=456")
        self.assertNotEqual(dataset.canonical_url(url), dataset.canonical_url(url.replace("gh_jid=123", "gh_jid=789")))
        self.assertEqual(dataset.canonical_url("https://boards.greenhouse.io/acme/jobs/123?gh_src=x"),
                         "https://job-boards.greenhouse.io/acme/jobs/123")

    def test_same_title_different_requisitions_not_merged(self):
        corpus = dataset.Corpus()
        corpus.add({"employer": "Acme", "title": "Software Engineer", "job_id": "123"}, "a", "1", "test")
        corpus.add({"employer": "Acme", "title": "Software Engineer", "job_id": "456"}, "b", "2", "test")
        corpus.refresh_employers()
        found, method, _ = corpus.lookup({"employer": "Acme", "title": "Software Engineer"})
        self.assertIsNone(found)
        self.assertEqual(method, "ambiguous_employer_title")
        postings, _ = dataset.merge_postings(corpus)
        self.assertEqual(len(postings), 2)

    def test_stale_board_url_does_not_merge_requisitions(self):
        corpus = dataset.Corpus()
        base = {"employer": "Example Health", "title": "Software Engineer",
                "url": "https://example.wd5.myworkdayjobs.com/Example_External_Career_Site"}
        corpus.add(base | {"job_id": "R-12345"}, "a", "1", "test")
        corpus.add(base | {"job_id": "R-23456"}, "a", "2", "test")
        corpus.refresh_employers()
        self.assertEqual(len(dataset.merge_postings(corpus)[0]), 2)
        found, _, _ = corpus.lookup(base | {"job_id": "R-34567"})
        self.assertIsNone(found)

    def test_proven_employer_alias_applies_across_url_changes(self):
        corpus = dataset.Corpus()
        rows = [
            {"employer": "HPE", "job_id": "12345", "url": "https://example.org/jobs/12345"},
            {"employer": "Hewlett Packard Enterprise", "job_id": "12345", "url": "https://example.org/jobs/12345"},
            {"employer": "HPE", "job_id": "23456", "url": "https://example.org/jobs/old/23456"},
            {"employer": "Hewlett Packard Enterprise", "job_id": "23456", "url": "https://example.org/jobs/new/23456"},
        ]
        for i, row in enumerate(rows):
            corpus.add(row, "a", i, "test")
        postings, _ = dataset.merge_postings(corpus)
        self.assertEqual(len(postings), 2)
        self.assertEqual(len({p["normalized_employer"] for p in postings}), 1)


class TextTests(unittest.TestCase):
    def test_contact_redaction_preserves_dates_and_job_urls(self):
        text = "Contact jane.doe+jobs@example.org, +1 (202) 555-0147 or 2025550147. Posted 2026-09-30. https://example.org/jobs/4114318009"
        result = dataset.redact(text)
        self.assertNotIn("example.org,", result)
        self.assertNotIn("555", result)
        self.assertEqual(result.count("[PHONE]"), 2)
        self.assertIn("[EMAIL]", result)
        self.assertIn("2026-09-30", result)
        self.assertIn("https://example.org/jobs/4114318009", result)

    def test_encoded_html_and_hidden_content(self):
        text = "&lt;p&gt;Build &amp;amp; test&lt;/p&gt;&lt;script&gt;secret()&lt;/script&gt;&lt;p&gt;Python&lt;/p&gt;"
        result = dataset.clean_text(text)
        self.assertEqual(result, "Build & test\n\nPython")


class LabelTests(unittest.TestCase):
    def test_outcomes_are_not_decisions(self):
        for outcome in ("rejected", "interview", "assessment", "closed"):
            label = dataset.user_signal({"status": outcome})
            self.assertIsNone(label["decision"])
            self.assertIn(outcome, label["outcomes"])
        label = dataset.user_signal({"status": "rejected", "status_history": [{"previous_status": "applied", "status": "rejected"}]})
        self.assertEqual(label["decision"], "pursue")
        self.assertEqual(label["outcomes"], ["rejected"])

    def test_policy_boundaries(self):
        self.assertEqual(dataset.policy_era("2026-09-20T23:59:59-04:00"), "pre-0921")
        self.assertEqual(dataset.policy_era("2026-09-21"), "0921-0929")
        self.assertEqual(dataset.policy_era("2026-09-29"), "0921-0929")
        self.assertEqual(dataset.policy_era("2026-09-30"), "post-0930")


if __name__ == "__main__":
    unittest.main()
