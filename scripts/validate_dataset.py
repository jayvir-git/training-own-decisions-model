"""Validate generated dataset relationships without reading any live sources."""
import hashlib
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed"


def rows(name):
    with (OUT / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main():
    postings = rows("postings.jsonl")
    agents = rows("labels_agent.jsonl")
    users = rows("labels_user.jsonl")
    splits = json.loads((OUT / "splits.json").read_text(encoding="utf-8"))
    by_id = {p["posting_id"]: p for p in postings}
    assert len(by_id) == len(postings), "Duplicate posting IDs"
    assert len({p["dedupe_key"] for p in postings}) == len(postings), "Duplicate dedupe keys"
    employer_ids = {}
    for posting in postings:
        if not posting["employer"]:
            continue
        for job in posting["job_ids"]:
            job = re.sub(r"^(jr|r)[ -]+(?=\d)", r"\1", str(job).strip().casefold())
            key = (posting["normalized_employer"], job)
            assert key not in employer_ids or employer_ids[key] == posting["posting_id"], "Duplicate normalized employer/job ID"
            employer_ids[key] = posting["posting_id"]
    assert len({a["receipt_fingerprint"] for a in agents}) == len(agents), "Lost receipt uniqueness"
    assert all(a["posting_id"] is None or a["posting_id"] in by_id for a in agents)
    assert all(u["posting_id"] in by_id for u in users)
    assert all(a["verdict"] in {"strong", "stretch", "reject", "needs-information"} for a in agents)
    assert all(u["decision"] in {None, "pursue", "skip", "undecided"} for u in users)
    assert all(not u["decision"] or u["decision_basis"] for u in users)
    assert all(not p["description"] or p["description_source"] in {s["source"] for s in p["source_ids"]} for p in postings)
    agent_ids = {a["posting_id"] for a in agents if a["posting_id"]}
    user_ids = {u["posting_id"] for u in users if u["decision"]}
    eligible = {pid for pid in agent_ids | user_ids if by_id[pid]["description"]}
    assigned, employers = set(), set()
    counts = {}
    for split in ("train", "val", "test"):
        ids = set(splits[split])
        groups = {by_id[pid]["normalized_employer"] for pid in ids}
        assert len(ids) == len(splits[split]), f"Duplicates in {split}"
        assert not assigned & ids, "Posting overlap across splits"
        assert not employers & groups, "Employer leakage across splits"
        assert all(splits["employer_assignment"][group] == split for group in groups)
        assigned.update(ids)
        employers.update(groups)
        counts[split] = {"postings": len(ids), "employers": len(groups)}
    assert assigned == eligible, "Missing or extraneous supervised posting"
    expected_pool = {p["posting_id"] for p in postings if len(p["description"]) >= 300 and p["posting_id"] not in agent_ids}
    assert expected_pool == set(splits["unlabeled_pool"])
    assert all(splits["employer_assignment"].get(by_id[pid]["normalized_employer"]) not in {"val", "test"}
               for pid in splits["unlabeled_train_pool"])
    email = re.compile(r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}")
    assert not any(email.search(p["description"]) for p in postings), "Email leaked into description"
    assert not any(email.search(a["reason"]) for a in agents), "Email leaked into reason"
    assert not any(re.search(r"</?(?:script|div|p|span)(?:\s[^>]*|)>", p["description"], re.I) for p in postings), "HTML remained in descriptions"
    hashes = {}
    for name in ("postings.jsonl", "labels_agent.jsonl", "labels_user.jsonl", "splits.json", "REPORT.md", "summary.json", "quality_issues.json"):
        with (OUT / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {"passed": True, "unique_postings": len(postings), "agent_receipts": len(agents),
              "receipt_to_text": sum(bool(a["posting_id"] and by_id[a["posting_id"]]["description"]) for a in agents),
              "user_records": len(users), "splits": counts, "employer_leakage": 0,
              "unlabeled_pool": len(expected_pool), "artifact_sha256": hashes}
    (OUT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
