"""Build an auditable, untrained dataset using ONLY the latest raw import.

Python standard library only. No network calls and no source-project imports.
The raw snapshot (data/raw/YYYYMMDD-HHMM/, gitignored) is produced by a private
importer that is not published. Its layout is neutral here: agent_a/ holds one
reviewing agent's outputs, ledgers and receipts, agent_b/ another agent's finder
snapshots and tracker, existing/ dashboard exports. See docs/DATA.md.
Agent receipts, user decisions, and application outcomes remain separate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import random
import re
import sqlite3
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit, unquote

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "processed"
SEED = 20261024
VERDICTS = ("strong", "stretch", "reject", "needs-information")
EMAIL = re.compile(r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://[^\s<>\"'`\]\)]+")
PHONE = re.compile(
    r"(?<![\w])(?:\+?1[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}"
    r"(?:\s*(?:ext\.?|x)\s*\d{1,6})?(?!\w)|"
    r"(?<![\w])\+\d{1,3}(?:[ .-]\d{2,5}){2,4}(?!\w)|"
    r"(?<![\w])(?:1)?\d{10}(?![\w])", re.I)
TRACKING = {"source", "ref", "referrer", "referer", "referral", "src", "gh_src",
            "lever-source", "lever-origin", "sourceid", "trid", "trk", "trackingid",
            "fbclid", "gclid", "dclid", "msclkid", "iis", "iisn", "cmpid"}
JOB_PARAMS = {"gh_jid", "jobid", "job_id", "job", "reqid", "req_id", "requisitionid",
              "requisition_id", "requisition", "postingid", "posting_id"}


def normalize_employer(value):
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    suffixes = {"inc", "incorporated", "llc", "llp", "ltd", "limited", "corp", "corporation", "co", "company", "plc"}
    while words and words[-1] in suffixes:
        words.pop()
    return " ".join(words)


def normalize_job_id(value):
    value = str(value or "").strip().casefold()
    if value in {"none", "null", "n/a", "unknown", "-"}:
        return ""
    value = re.sub(r"^(?:job|req(?:uisition)?)[\s:#]+", "", value)
    return re.sub(r"^(jr|r)[ -]+(?=\d)", r"\1", value)


def canonical_url(value):
    value = html.unescape(str(value or "")).strip().rstrip(".,;>")
    if not value:
        return ""
    try:
        p = urlsplit(value)
        if p.scheme not in {"http", "https"} or not p.hostname:
            return ""
        host = p.hostname.casefold().removeprefix("www.")
        host = {"boards.greenhouse.io": "job-boards.greenhouse.io"}.get(host, host)
        path = re.sub(r"/{2,}", "/", p.path).rstrip("/")
        if any(h in host for h in ("myworkdayjobs.com", "myworkdaysite.com")):
            path = re.sub(r"^/[a-z]{2}-[a-z]{2}(?=/)", "", path, flags=re.I)
        if any(h in host for h in ("ashbyhq.com", "lever.co", "greenhouse.io")):
            path = re.sub(r"/(?:apply|application)$", "", path, flags=re.I)
        query = [(k.casefold(), v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                 if k.casefold() in JOB_PARAMS or
                 (k.casefold() not in TRACKING and not k.casefold().startswith(("utm_", "mc_")))]
        port = f":{p.port}" if p.port and p.port not in {80, 443} else ""
        return urlunsplit(("https", host + port, path, urlencode(sorted(query)), ""))
    except ValueError:
        return ""


def url_job_id(value):
    url = canonical_url(value)
    if not url:
        return ""
    p = urlsplit(url)
    for key, val in parse_qsl(p.query):
        if key in JOB_PARAMS and val:
            return normalize_job_id(val)
    path = unquote(p.path)
    if "myworkday" in p.netloc and "/job/" in path:
        m = re.search(r"_([^/]+)$", path)
        if m:
            return normalize_job_id(re.sub(r"-\d{1,2}$", "", m[1]))
    patterns = [r"/(?:jobs?|postings|details|requisition)/(\d+(?:-\d+)?)(?:[-/]|$)"]
    if any(host in p.netloc for host in ("ashbyhq.com", "lever.co")) or re.search(r"/(?:jobs?|postings)/", path, re.I):
        patterns.insert(0, r"/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:/|$)")
    if "smartrecruiters.com" in p.netloc:
        patterns.insert(0, r"/([0-9]{10,})(?:-|$)")
    if "interfolio.com" in p.netloc:
        patterns.insert(0, r"/(\d+)$")
    for pattern in patterns:
        m = re.search(pattern, path, re.I)
        if m:
            return normalize_job_id(m[1])
    return ""


def dedupe_key(row):
    employer = normalize_employer(row.get("employer"))
    url = row.get("url") or row.get("official_url") or ""
    job = normalize_job_id(row.get("job_id") or row.get("req_id") or row.get("requisition_id")) or url_job_id(url)
    if employer and job:
        return "job:" + employer + ":" + job
    if canonical_url(url):
        return "url:" + canonical_url(url)
    return "source:" + str(row.get("source", "")) + ":" + str(row.get("source_id", ""))


def redact(value):
    """Preserve job IDs inside URLs; redact contacts in all narrative text."""
    text = EMAIL.sub("[EMAIL]", str(value or ""))
    pieces, last = [], 0
    for m in URL_RE.finditer(text):
        pieces.extend([PHONE.sub("[PHONE]", text[last:m.start()]), m[0]])
        last = m.end()
    pieces.append(PHONE.sub("[PHONE]", text[last:]))
    return "".join(pieces)


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        if not self.hidden and tag in {"p", "div", "br", "li", "h1", "h2", "h3", "section", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in {"p", "div", "li", "h1", "h2", "h3", "section", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value):
    text = str(value or "")
    # Repair only reversible UTF-8-as-Windows-1252 mojibake, never guess bytes.
    for _ in range(2):
        if not re.search(r"Ã|Â|â€|ðŸ", text):
            break
        try:
            repaired = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if repaired == text:
            break
        text = repaired
    for _ in range(2):
        text = html.unescape(text)
    if re.search(r"</?(?:p|div|br|li|ul|ol|h[1-6]|span|strong|b|script|html|a)(?:\s|>)", text, re.I):
        parser = TextParser()
        parser.feed(text)
        text = "".join(parser.parts)
    text = unicodedata.normalize("NFKC", text).replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return redact(text.strip())


def title_key(value):
    text = normalize_employer(value)
    text = re.sub(r"\bswe\b", "software engineer", text)
    text = re.sub(r"\bsde\b", "software development engineer", text)
    return re.sub(r"\s+", " ", text).strip()


def policy_era(date):
    date = str(date or "")[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return "unknown"
    return "pre-0921" if date < "2026-09-21" else "0921-0929" if date < "2026-09-30" else "post-0930"


def json_load(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


class Corpus:
    def __init__(self):
        self.rows = []
        self.parent = []
        self.index = {}
        self.counts = Counter()
        self.issues = []

    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def keys(self, row):
        keys = {dedupe_key(row)}
        for url in row["urls"]:
            canonical = canonical_url(url)
            # A homepage or bare board is not a posting identity.
            if canonical and (url_job_id(url) or len(urlsplit(canonical).path.strip("/").split("/")) >= 2):
                keys.add("url:" + canonical)
        employer = normalize_employer(row["employer"])
        if employer:
            for job in row["job_ids"]:
                keys.add("job:" + employer + ":" + normalize_job_id(job))
        for fp in row["fingerprints"]:
            keys.add("fp:" + fp)
        return keys

    def add(self, row, source, source_id, category, description_kind="structured_posting"):
        row = dict(row)
        row["source"], row["source_id"], row["category"] = source, str(source_id), category
        for key in ("employer", "title", "location"):
            value = row.get(key) or ""
            if isinstance(value, dict):
                value = value.get("name") or value.get("descriptor") or ""
            row[key] = clean_text(value)
        row["url"] = str(row.get("url") or row.get("official_url") or "")
        row["urls"] = sorted(set(filter(None, [row["url"], *row.get("urls", [])])))
        job = str(row.get("job_id") or "")
        row["job_id"] = job or url_job_id(row["url"])
        row["job_ids"] = sorted(set(str(x) for x in [row["job_id"], *row.get("job_ids", []),
                    *[url_job_id(u) for u in row["urls"]]] if x and normalize_job_id(x)))
        row["fingerprints"] = sorted(set(filter(None, [row.get("fingerprint"), *row.get("fingerprints", [])])))
        row["description"] = clean_text(row.get("description", ""))
        row["description_kind"] = description_kind
        if not row["employer"] and row["url"]:
            # This is explicitly marked inferred, and only used when no named source is available.
            p = urlsplit(row["url"])
            segments = p.path.strip("/").split("/")
            if any(h in p.netloc for h in ("greenhouse.io", "ashbyhq.com", "lever.co", "smartrecruiters.com")):
                row["employer"] = segments[0] if segments else ""
                row["employer_inferred"] = True
        i = len(self.rows)
        self.rows.append(row)
        self.parent.append(i)
        keys = self.keys(row)
        matches = {self.root(self.index[k]) for k in keys if k in self.index}
        for previous in matches:
            self.parent[self.root(i)] = self.root(previous)
        for key in keys:
            self.index[key] = self.root(i)
        self.counts[category] += 1
        return i

    def lookup(self, row, fuzzy=True):
        fp = row.get("fingerprint")
        if fp and "fp:" + fp in self.index:
            return self.root(self.index["fp:" + fp]), "fingerprint", 1.0
        for url in [row.get("official_url"), row.get("url"), *row.get("urls", [])]:
            key = "url:" + canonical_url(url)
            if key in self.index:
                return self.root(self.index[key]), "official_url", 1.0
        employer = normalize_employer(row.get("employer"))
        ids = [row.get("job_id"), *row.get("job_ids", [])]
        for value in ids:
            key = "job:" + employer + ":" + normalize_job_id(value)
            if employer and value and key in self.index:
                return self.root(self.index[key]), "employer_job_id", 1.0
        if not fuzzy or not employer or not row.get("title"):
            return None, "unmatched", 0.0
        title = title_key(row["title"])
        candidates = defaultdict(float)
        requested_ids = {normalize_job_id(value) for value in ids if value}
        for i in self.employer_index.get(employer, []):
            candidate = self.rows[i]
            candidate_ids = {normalize_job_id(value) for value in candidate["job_ids"]}
            if requested_ids and candidate_ids and not requested_ids.intersection(candidate_ids):
                continue
            root = self.root(i)
            score = SequenceMatcher(None, title, title_key(candidate["title"])).ratio()
            candidates[root] = max(candidates[root], score)
        ranking = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
        if ranking and ranking[0][1] >= 0.93 and (len(ranking) == 1 or ranking[0][1] - ranking[1][1] >= 0.08):
            return ranking[0][0], "employer_title_fuzzy", round(ranking[0][1], 5)
        return None, "ambiguous_employer_title" if ranking and ranking[0][1] >= 0.93 else "unmatched", 0.0

    def refresh_employers(self):
        self.employer_index = defaultdict(list)
        for i, row in enumerate(self.rows):
            self.employer_index[normalize_employer(row["employer"])].append(i)


def normalize_capture(obj):
    """Read specific posting fields, never whole JSON or agent reasoning as text."""
    if isinstance(obj, list):
        return [r for item in obj for r in normalize_capture(item)]
    if not isinstance(obj, dict):
        return []
    if isinstance(obj.get("job"), (dict, list)):
        records = normalize_capture(obj["job"])
        for record in records:
            record["urls"] = [*record.get("urls", []), *[obj[k] for k in ("source_url", "official_url") if obj.get(k)]]
        return records
    if isinstance(obj.get("data"), dict) and "jobPosting" in obj["data"]:
        return normalize_capture(obj["data"]["jobPosting"])
    if isinstance(obj.get("items"), list):
        return normalize_capture(obj["items"])
    parent = obj
    obj = obj.get("jobPostingInfo") or obj.get("info") or obj
    if not isinstance(obj, dict):
        return []
    title = obj.get("title") or obj.get("Title") or obj.get("text") or ""
    if not title:
        return []
    org = parent.get("hiringOrganization") or obj.get("hiringOrganization") or {}
    if not isinstance(org, dict):
        org = {}
    identifier = obj.get("identifier") or {}
    identifier = identifier.get("value", "") if isinstance(identifier, dict) else identifier
    urls = [obj[k] for k in ("url", "official_url", "absolute_url", "jobUrl", "hostedUrl", "externalUrl", "applyUrl") if obj.get(k)]
    url = next(iter(urls), "")
    url = url or parent.get("official_url") or org.get("url") or ""
    description = max((str(obj.get(k) or "") for k in ("description", "descriptionPlain", "descriptionHtml", "jobDescription", "content", "ExternalDescriptionStr")), key=lambda x: len(clean_text(x)))
    if "lists" in obj:  # Lever keeps requirements in separate list sections.
        parts = [obj.get("openingPlain") or obj.get("opening") or "", description]
        for section in obj.get("lists", []):
            parts.extend([section.get("text", ""), section.get("content", "")])
        parts.append(obj.get("additionalPlain") or obj.get("additional") or "")
        description = "\n\n".join(str(x) for x in parts if x)
    job = obj.get("jobReqId") or obj.get("job_id") or identifier or obj.get("id") or obj.get("Id") or ""
    return [{"employer": obj.get("employer") or obj.get("company_name") or org.get("name") or "",
             "title": title, "url": url, "urls": urls, "job_id": job, "description": description,
             "source_identifiers": {k: obj[k] for k in ("id", "Id", "job_id", "jobReqId", "requisition_id", "internal_job_id", "identifier") if obj.get(k)},
             "location": obj.get("location") or obj.get("locationName") or obj.get("PrimaryLocation") or "",
             "fingerprint": obj.get("fingerprint"), "ats": obj.get("ats") or obj.get("board") or "",
             "questions": obj.get("questions") or [], "posted": obj.get("posted") or obj.get("datePosted") or ""}]


def extract_urls(text):
    return list(dict.fromkeys(m[0].rstrip(".,;*") for m in URL_RE.finditer(text)))


def job_urls(text):
    explicit = re.findall(r"(?im)^(?:Source|URL|Official (?:URL|posting|page)|Posting)\s*:\s*(https?://[^\s<>]+)", text)
    return list(dict.fromkeys([*[u.rstrip(".,;*") for u in explicit],
            *[u for u in extract_urls(text) if url_job_id(u) or
              re.search(r"/(?:job|jobs|posting|postings|details|requisition)/[^/?]+", u, re.I)]]))


def capture_text(text):
    if len(text) < 300 or re.search(r"urlopen error|<Error>|Access Denied|Just a moment", text[:400], re.I):
        return "", "failed_or_short_capture"
    if "Minimum requirement mapping" in text:
        return "", "annotated_evaluation_excluded"
    # Remove reviewer interpretation appended after the captured job text.
    text = re.split(r"(?im)^#{1,4}\s+(?:Candidate|Recommendation|Fit assessment|Fit /|Screening|Assessment|Application answers|Our verdict|Decision)\b", text)[0]
    # Playwright accessibility snapshots are text captures, not HTML.
    text = re.sub(r'(?m)^\s*-\s*(?:paragraph|text|heading|generic)(?::|\s+)\s*', '', text)
    text = re.sub(r"\s*\[level=\d+\]", "", text)
    return text, "application_capture_text"


def source_category(path):
    parts = path.split("/")
    return parts[0] + "_application_captures"


def read_tracker(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not re.match(r"^\|\s*\d+[a-z]?\s*\|", line):
            continue
        cells = [re.sub(r"\*\*|~~|`", "", s.strip()) for s in re.split(r"(?<!\\)\|", line)[1:-1]]
        if len(cells) < 11:
            continue
        rows.append({"id": cells[0], "employer": cells[1], "title": cells[2],
                     "url": next(iter(job_urls(cells[3])), ""), "applied": cells[6],
                     "status": cells[8], "notes": cells[10]})
    return rows


def decision_from_status(raw):
    status = re.sub(r"[*~`]", "", str(raw or "")).strip().casefold()
    if re.match(r"^(?:applied|queued|prepared|submitted|ready to submit)\b", status):
        return "pursue"
    if re.match(r"^(?:skip|skipped|dropped)\b", status):
        return "skip"
    if re.match(r"^(?:considering|looking)\b", status):
        return "undecided"
    return None


def user_signal(row):
    raw_status = str(row.get("status") or "")
    decision = decision_from_status(raw_status)
    basis = "current_status" if decision else None
    history = row.get("status_history") or []
    if not decision:
        for event in reversed(history):
            for key in ("status", "previous_status"):
                decision = decision_from_status(event.get(key))
                if decision:
                    basis = "status_history." + key
                    break
            if decision:
                break
    applied = str(row.get("applied_date") or row.get("submitted_date") or row.get("applied") or "")
    if not decision and (re.search(r"\b20\d\d-\d\d-\d\d\b", applied) or
                         re.match(r"^submitted\b", applied, re.I)):
        decision, basis = "pursue", "explicit_submission_evidence"
    outcomes = set()
    status_fields = {k: v for k, v in row.items() if v and (k in {
        "stage", "application_stage", "application_status", "outcome", "decision"}
        or k.startswith(("assessment_", "interview_", "rejection_", "rejected_")))}
    if any(row.get(k) for k in ("assessment_completed_at", "assessment_submitted_date", "assessment_invited_date")):
        outcomes.add("assessment")
    if row.get("assessment_status") and str(row["assessment_status"]).casefold() in {"completed", "submitted", "invited", "pending", "passed", "failed"}:
        outcomes.add("assessment")
    texts = [raw_status, *[str(v) for v in status_fields.values()]]
    for text in texts:
        if re.match(r"^\s*(?:unresolved|unclear|unknown)", text, re.I):
            continue
        for outcome, pattern in [("rejected", r"\breject(?:ed|ion)?\b"), ("assessment", r"\bassessment\b"),
                                 ("interview", r"\binterview\b"), ("closed", r"\b(?:closed|dead|expired)\b"),
                                 ("offer", r"\boffer\b"), ("ghosted", r"\bghosted\b"), ("screen", r"\bscreen\b")]:
            if re.search(pattern, text, re.I):
                outcomes.add(outcome)
    return {"decision": decision, "decision_basis": basis, "raw_status": clean_text(raw_status),
            "outcomes": sorted(outcomes), "outcome_fields": scrub(status_fields),
            "status_history": scrub(history), "applied_date": clean_text(applied),
            "decision_date": row.get("decision_date"),
            "rejection_fields": scrub({k: v for k, v in row.items() if k.startswith(("rejection_", "rejected_"))}),
            "attribution": "recorded_candidate_disposition; not independently verified"}


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return clean_text(value) if isinstance(value, str) else value


def infer_title_ids(title):
    matches = re.findall(r"\b(?:JR[ -]?\d{4,}|R[ -]?\d{5,}|\d{6,9}(?:-\d{3,4})?)\b", title, re.I)
    return matches


def load_sources(root, corpus):
    def rel(path):
        return path.relative_to(root).as_posix()

    latest_path = root / "agent_a/output/latest.json"
    latest = json_load(latest_path)["jobs"]
    for i, row in enumerate(latest):
        corpus.add(row, rel(latest_path), row.get("fingerprint") or i, "agent_a_latest")
    db = root / "agent_a/state/jobs.sqlite"
    con = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        for row in con.execute("SELECT * FROM jobs ORDER BY fingerprint"):
            corpus.add(dict(row), rel(db), row["fingerprint"], "agent_a_sqlite")
    finally:
        con.close()
    for path in sorted(root.glob("agent_b/finder/snapshots/**/*.json")):
        row = json_load(path)
        corpus.add(row | {"job_id": row.get("id")}, rel(path), row.get("id", path.stem), "agent_b_snapshots")
        # Previous descriptions are separately identified, not silently mixed with current text.
        if row.get("previous_description"):
            corpus.add(row | {"job_id": row.get("id"), "description": row["previous_description"]},
                       rel(path), str(row.get("id", path.stem)) + ":previous", "agent_b_previous_descriptions", "historical_previous_description")
    path = root / "agent_b/finder/state/postings.json"
    for key, row in json_load(path).items():
        corpus.add(row | {"job_id": row.get("id")}, rel(path), key, "agent_b_postings_state")
    for path in sorted(root.glob("agent_b/finder/out/sweep-*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                corpus.add(row, rel(path), i + 2, "agent_b_sweeps")
    users = []
    for relative, key, category in [("agent_a/state/opportunities.json", "opportunities", "opportunities"),
                                    ("agent_a/known_applications.json", "applications", "known_applications")]:
        path = root / relative
        for i, row in enumerate(json_load(path)[key]):
            users.append((row, rel(path), str(row.get("key") or i), category))
    path = root / "agent_b/tracker.md"
    for row in read_tracker(path):
        users.append((row, rel(path), row["id"], "agent_b_tracker"))
    for path in sorted(root.glob("existing/desk-export-*.json")):
        desk = json_load(path)
        for feed, section in sorted(desk.get("feed", {}).items()):
            for i, row in enumerate(section.get("items", [])):
                if feed == "apps":
                    users.append((row, rel(path), str(row.get("id") or i), "desk_apps"))
                else:
                    corpus.add(row, rel(path), row.get("id") or f"{feed}:{i}", "desk_openings")
        corpus.counts["desk_marks"] += len(desk.get("marks", {}))
    # Add authoritative application identities before parsing captures; title-only
    # records are matched conservatively and cannot collapse distinct requisitions.
    corpus.refresh_employers()
    user_refs = []
    folder_identity = defaultdict(set)
    for row, source, source_id, category in users:
        row = dict(row)
        row["job_ids"] = [*row.get("job_id_aliases", []), *infer_title_ids(row.get("title", ""))]
        found, method, confidence = corpus.lookup(row)
        if found is not None:
            row["fingerprints"] = corpus.rows[found]["fingerprints"]
            # Preserve aliases even when the matched representative has no fingerprint.
            row["urls"] = list(set([*row.get("urls", []), *corpus.rows[found]["urls"]]))
            if not row.get("employer"):
                row["employer"] = corpus.rows[found]["employer"]
        if not row.get("job_id") and row["job_ids"]:
            row["job_id"] = row["job_ids"][0]
        obs = corpus.add(row, source, source_id, category)
        if found is not None:
            corpus.parent[corpus.root(obs)] = corpus.root(found)
        user_refs.append((obs, row, source, source_id, category, method, confidence))
        corpus.employer_index[normalize_employer(row.get("employer"))].append(obs)
        for key, value in row.items():
            if isinstance(value, str) and any(word in key for word in ("path", "folder", "worksheet", "material", "resume")):
                match = re.search(r"applications/([^\s`]+)", value.replace("\\", "/"))
                if match:
                    subpath = match[1].rstrip(".,;")
                    if Path(subpath).suffix:
                        subpath = str(Path(subpath).parent).replace("\\", "/")
                    if subpath not in {"", "."}:
                        folder_identity[subpath].add(obs)

    # A unique ID fragment in a capture's directory is auditable identity
    # evidence (common eight-character Ashby/Lever UUID prefixes).
    id_hints = defaultdict(set)
    for i, row in enumerate(corpus.rows):
        for job in row["job_ids"]:
            normalized = normalize_job_id(job)
            if len(normalized) >= 4:
                id_hints[normalized].add(corpus.root(i))
            if re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]+){4}", normalized):
                id_hints[normalized[:8]].add(corpus.root(i))

    # Prep sheets supply only identity hints. Their recommendations and personal
    # application answers are never used as descriptions or decision labels.
    app_groups = defaultdict(list)
    for path in sorted(root.glob("*/applications/**/*")):
        if path.is_file():
            app_groups[path.parent].append(path)
    for directory, paths in sorted(app_groups.items()):
        prep_urls = []
        prep_title = ""
        prep_paths = []
        search_dirs = [directory]
        parent = directory.parent
        while parent.name != "applications" and parent.is_relative_to(root) and parent != root:
            search_dirs.append(parent)
            parent = parent.parent
        for parent in search_dirs:
            for prep in sorted(parent.glob("*-application.md")):
                text = prep.read_text(encoding="utf-8-sig")
                prep_urls.extend(job_urls(text[:8000]))
                prep_title = prep_title or text.splitlines()[0].lstrip("# ")
                prep_paths.append(rel(prep))
        parsed = []
        for path in paths:
            if "posting" not in path.name.lower() and "job-description" not in path.name.lower():
                continue
            try:
                if path.suffix == ".json":
                    records = normalize_capture(json_load(path))
                    for i, row in enumerate(records):
                        parsed.append((path, i, row, "structured_posting"))
                    if not records:
                        corpus.issues.append({"source": rel(path), "issue": "unrecognized_capture_json"})
                else:
                    raw_text = path.read_text(encoding="utf-8-sig")
                    text, kind = capture_text(raw_text)
                    heading = re.search(r'heading "([^"]+)" \[level=1\]', raw_text)
                    parsed.append((path, 0, {"description": text, "title": heading[1] if heading else "",
                                             "urls": job_urls(raw_text[:12000])}, kind))
                    if not text:
                        corpus.issues.append({"source": rel(path), "issue": kind})
            except (ValueError, UnicodeError) as exc:
                corpus.issues.append({"source": rel(path), "issue": str(exc)})
        structured = [row for _, _, row, kind in parsed if kind == "structured_posting"]
        for path, i, row, kind in parsed:
            # A text capture borrows identity only from an unambiguous sibling
            # posting. Multiple-posting captures do not receive a guessed identity.
            own_urls = [u for u in [row.get("url", ""), *row.get("urls", [])] if u]
            hints = list(dict.fromkeys(own_urls if own_urls else prep_urls))
            found = None
            for url in hints:
                if url:
                    found, _, _ = corpus.lookup({"url": url}, fuzzy=False)
                    if found is not None:
                        break
            hint_method = "capture_url"
            if found is None:
                relative_dir = directory.relative_to(root).as_posix().split("applications/", 1)[1]
                candidates = {corpus.root(i) for i in folder_identity.get(relative_dir, [])}
                if len(candidates) == 1:
                    found = next(iter(candidates))
                    hint_method = "application_path_metadata"
            if found is None:
                folder = directory.name.casefold()
                candidates = set()
                for token in re.findall(r"[0-9a-f]{8}|(?:jr|r)?\d{4,}", folder):
                    candidates.update(corpus.root(i) for i in id_hints.get(token, []))
                if len(candidates) == 1:
                    candidate = next(iter(candidates))
                    # If the capture states a title, require substantial agreement.
                    reference_title = corpus.rows[candidate]["title"]
                    if not row.get("title") or SequenceMatcher(None, title_key(row["title"]), title_key(reference_title)).ratio() >= 0.8:
                        found = candidate
                        hint_method = "unique_directory_job_id"
            sibling = structured[0] if len(structured) == 1 else {}
            if not row.get("url"):
                row["url"] = next((u for u in hints if u), sibling.get("url", ""))
            if found is not None:
                ref = corpus.rows[found]
                for key in ("employer", "title", "job_id"):
                    row[key] = row.get(key) or ref.get(key) or sibling.get(key, "")
                row["url"] = row.get("url") or ref.get("url", "")
                row["capture_identity_method"] = hint_method
            else:
                for key in ("employer", "title", "job_id"):
                    row[key] = row.get(key) or sibling.get(key, "")
            if not row.get("url") and not row.get("job_id"):
                corpus.issues.append({"source": rel(path), "issue": "capture_identity_unresolved"})
            row["prep_identity_sources"] = prep_paths
            obs = corpus.add(row, rel(path), i, source_category(rel(path)), kind)
            if found is not None:
                corpus.parent[corpus.root(obs)] = corpus.root(found)
    corpus.refresh_employers()
    return user_refs


def merge_postings(corpus):
    groups = defaultdict(list)
    for i, row in enumerate(corpus.rows):
        groups[corpus.root(i)].append(row)
    postings, root_to_id = [], {}
    # Unify employer aliases only when exact posting identity provides evidence.
    employer_parent = {}
    def eroot(e):
        employer_parent.setdefault(e, e)
        if employer_parent[e] != e:
            employer_parent[e] = eroot(employer_parent[e])
        return employer_parent[e]
    for rows in groups.values():
        names = sorted({normalize_employer(r["employer"]) for r in rows if r["employer"] and not r.get("employer_inferred")})
        for name in names[1:]:
            a, b = eroot(names[0]), eroot(name)
            employer_parent[max(a, b)] = min(a, b)
    # Apply proven employer aliases corpus-wide, including records whose URL
    # changed. This closes employer+ID duplicates across Workday URL variants.
    employer_job_index = {}
    for i, row in enumerate(corpus.rows):
        employer = normalize_employer(row["employer"])
        if not employer:
            continue
        for job in row["job_ids"]:
            key = (eroot(employer), normalize_job_id(job))
            if key in employer_job_index:
                corpus.parent[corpus.root(i)] = corpus.root(employer_job_index[key])
            else:
                employer_job_index[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(corpus.rows):
        groups[corpus.root(i)].append(row)
    for root, rows in groups.items():
        employer_rows = sorted(rows, key=lambda r: (bool(r.get("employer_inferred")), not bool(r["employer"]), r["source"], r["source_id"]))
        employer = next((r["employer"] for r in employer_rows if r["employer"]), "")
        titles = sorted(set(r["title"] for r in rows if r["title"]))
        keys = sorted(set(dedupe_key(r | {"employer": eroot(normalize_employer(r["employer"]))}) for r in rows))
        key = next((k for k in keys if k.startswith("job:")), keys[0])
        posting_id = "p_" + hashlib.sha256(key.encode()).hexdigest()[:20]
        root_to_id[root] = posting_id
        best = min(rows, key=lambda r: (-len(r["description"]), r["source"], r["source_id"]))
        urls = sorted({u for r in rows for u in r["urls"]})
        normalized = eroot(normalize_employer(employer)) if employer else "unknown:" + posting_id
        postings.append({"posting_id": posting_id, "dedupe_key": key, "employer": employer,
                         "normalized_employer": normalized,
                         "employer_aliases": sorted({r["employer"] for r in rows if r["employer"]}),
                         "title": next((r["title"] for r in rows if r["title"]), ""), "titles": titles,
                         "locations": sorted({r["location"] for r in rows if r["location"]}),
                         "job_ids": sorted({job for r in rows for job in r["job_ids"]}),
                         "urls": urls, "canonical_urls": sorted({canonical_url(u) for u in urls if canonical_url(u)}),
                         "fingerprints": sorted({fp for r in rows for fp in r["fingerprints"]}),
                         "description": best["description"],
                         "description_source": best["source"] if best["description"] else None,
                         "description_source_id": best["source_id"] if best["description"] else None,
                         "description_kind": best["description_kind"] if best["description"] else None,
                         "description_variants": len({r["description"] for r in rows if r["description"]}),
                         "description_sources": sorted({r["source"] for r in rows if r["description"]}),
                         "questions": sorted({clean_text(q if isinstance(q, str) else q.get("label", ""))
                                              for r in rows for q in r.get("questions", [])
                                              if isinstance(q, (str, dict))}),
                         "source_ids": [{"source": r["source"], "id": r["source_id"], "category": r["category"],
                                         **({"source_identifiers": r["source_identifiers"]} if r.get("source_identifiers") else {}),
                                         **({"capture_identity_method": r["capture_identity_method"]} if r.get("capture_identity_method") else {})} for r in rows]})
    return sorted(postings, key=lambda p: p["posting_id"]), root_to_id


def build_splits(postings, agents, users):
    by_id = {p["posting_id"]: p for p in postings}
    agent_ids = {a["posting_id"] for a in agents if a["posting_id"]}
    user_ids = {u["posting_id"] for u in users if u["decision"]}
    latest = {}
    for row in sorted(agents, key=lambda r: (r["reviewed_at"], r["receipt_fingerprint"])):
        if row["posting_id"]:
            latest[row["posting_id"]] = row["verdict"]
    eligible = sorted(pid for pid in agent_ids | user_ids if by_id[pid]["description"])
    groups = defaultdict(list)
    for pid in eligible:
        groups[by_id[pid]["normalized_employer"]].append(pid)
    strata = {pid: latest.get(pid, "user-only") for pid in eligible}
    totals = Counter(strata.values())
    weights = {"train": 0.8, "val": 0.1, "test": 0.1}
    names = list(weights)
    rng = random.Random(SEED)
    order = list(sorted(groups))
    rng.shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))
    counts = {s: Counter() for s in names}
    assignments = {}

    def loss():
        result = 0.0
        for s, ratio in weights.items():
            result += 2 * (sum(counts[s].values()) - len(eligible) * ratio) ** 2 / max(len(eligible) * ratio, 1)
            for verdict, total in totals.items():
                result += (counts[s][verdict] - total * ratio) ** 2 / max(total * ratio, 1)
        return result

    for group in order:
        delta = Counter(strata[pid] for pid in groups[group])
        options = []
        for split in names:
            counts[split].update(delta)
            options.append((loss(), names.index(split), split))
            counts[split].subtract(delta)
        chosen = min(options)[2]
        assignments[group] = chosen
        counts[chosen].update(delta)
    # Deterministic local improvement of whole-employer assignments.
    for _ in range(20):
        improved = False
        for group in order:
            old = assignments[group]
            delta = Counter(strata[pid] for pid in groups[group])
            baseline, chosen = loss(), old
            for new in names:
                if old == new:
                    continue
                counts[old].subtract(delta)
                counts[new].update(delta)
                score = loss()
                counts[new].subtract(delta)
                counts[old].update(delta)
                if score < baseline - 1e-9:
                    baseline, chosen = score, new
            if chosen != old:
                assignments[group] = chosen
                counts[old].subtract(delta)
                counts[chosen].update(delta)
                improved = True
        if not improved:
            break
    splits = {s: sorted(pid for g, ids in groups.items() if assignments[g] == s for pid in ids) for s in names}
    pool = sorted(p["posting_id"] for p in postings if len(p["description"]) >= 300 and p["posting_id"] not in agent_ids)
    return {"seed": SEED, "target_ratios": weights,
            "unit": "posting_id; grouped by normalized_employer including exact-identity aliases",
            "eligibility": "nonempty description and at least one agent verdict or non-null user decision",
            "stratification": "latest agent verdict per posting; user-only stratum; greedy assignment plus whole-employer local improvement",
            **splits, "employer_assignment": dict(sorted(assignments.items())),
            "unlabeled_pool": pool,
            "unlabeled_pool_note": "No agent label, >=300 characters. This is a candidate pool, not a fourth disjoint split; can include user-only labeled postings.",
            "unlabeled_train_pool": [pid for pid in pool if assignments.get(by_id[pid]["normalized_employer"]) not in {"val", "test"}]}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def report(root, manifest, corpus, postings, agents, users, splits):
    by_id = {p["posting_id"]: p for p in postings}
    text_agents = [a for a in agents if a["posting_id"] and by_id[a["posting_id"]]["description"]]
    latest_text_fps = {fp for r in corpus.rows if r["category"] == "agent_a_latest" and r["description"] for fp in r["fingerprints"]}
    latest_text = sum(a["receipt_fingerprint"] in latest_text_fps for a in agents)
    summary = {"import": root.name, "imported_files": len(manifest["files"]),
               "posting_observations": len(corpus.rows), "unique_postings": len(postings),
               "duplicates_merged": len(corpus.rows) - len(postings),
               "postings_with_description": sum(bool(p["description"]) for p in postings),
               "postings_with_300_char_description": sum(len(p["description"]) >= 300 for p in postings),
               "agent_receipts": len(agents), "receipts_joined_to_posting": sum(bool(a["posting_id"]) for a in agents),
               "receipts_joined_to_text": len(text_agents), "receipt_latest_fingerprint_text_baseline": latest_text,
               "unique_agent_postings_with_text": len({a["posting_id"] for a in text_agents}),
               "user_records": len(users), "user_decisions": dict(sorted(Counter(u["decision"] or "no_decision_evidence" for u in users).items())),
               "unlabeled_pool": len(splits["unlabeled_pool"]),
               "changed_during_copy": [e["source_path"] for e in manifest["files"] if e.get("changed_during_copy")],
               "source_counts": dict(sorted(corpus.counts.items())), "parse_issues": len(corpus.issues)}
    lines = ["# Job triage dataset import report", "", f"Snapshot: `{root.name}`. Seed: `{SEED}`. Standard library only; no training or model downloads.", "",
             "## Headline counts", ""]
    for key, value in summary.items():
        if not isinstance(value, (dict, list)):
            lines.append(f"- {key}: **{value:,}**" if isinstance(value, int) else f"- {key}: `{value}`")
    lines += ["", "## Counts per source", "", "| Source | Posting observations |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in sorted(corpus.counts.items())]
    file_counts = Counter(e["dest_path"].split("/")[0] for e in manifest["files"])
    lines += ["", "Imported files by tree: " + ", ".join(f"{k}={v}" for k, v in sorted(file_counts.items())) + ".",
              "", "## Receipt joins", "",
              f"**{len(text_agents):,}/{len(agents):,} receipts ({len(text_agents)/len(agents):.1%}) join to nonempty posting text.** "
              f"The latest.json fingerprint-only baseline is {latest_text:,}; other sources/identities recover {len(text_agents)-latest_text:,} more.",
              "", "| Match method | Receipts | With text |", "|---|---:|---:|"]
    methods = Counter(a["match_method"] for a in agents)
    text_methods = Counter(a["match_method"] for a in text_agents)
    lines += [f"| {k} | {v} | {text_methods[k]} |" for k, v in sorted(methods.items())]
    lines += ["", "## Employer-disjoint splits", "", "| Split | Postings | Employers | Strong | Stretch | Reject | Needs information | User-only |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    latest_agent = {}
    for a in sorted(agents, key=lambda a: (a["reviewed_at"], a["receipt_fingerprint"])):
        if a["posting_id"]:
            latest_agent[a["posting_id"]] = a["verdict"]
    for split in ("train", "val", "test"):
        ids = splits[split]
        counter = Counter(latest_agent.get(pid, "user-only") for pid in ids)
        lines.append(f"| {split} | {len(ids)} | {len({by_id[pid]['normalized_employer'] for pid in ids})} | " + " | ".join(str(counter[v]) for v in (*VERDICTS, "user-only")) + " |")
    lines += ["", "Counts above use the latest receipt per posting; all receipt observations remain in labels_agent.jsonl. "
              "User-only means a recorded decision without an agent receipt. No normalized employer crosses train/val/test.",
              "", "## Verdicts by policy era (all receipts)", "", "| Policy era | Strong | Stretch | Reject | Needs information |", "|---|---:|---:|---:|---:|"]
    for era in ("pre-0921", "0921-0929", "post-0930", "unknown"):
        counter = Counter(a["verdict"] for a in agents if a["policy_era"] == era)
        lines.append(f"| {era} | " + " | ".join(str(counter[v]) for v in VERDICTS) + " |")
    lines += ["", "## User dispositions and outcomes", "", "| Source | Pursue | Skip | Undecided | No decision evidence |", "|---|---:|---:|---:|---:|"]
    for source in sorted({u["source_category"] for u in users}):
        counter = Counter(u["decision"] or "no_decision_evidence" for u in users if u["source_category"] == source)
        lines.append(f"| {source} | " + " | ".join(str(counter[d]) for d in ("pursue", "skip", "undecided", "no_decision_evidence")) + " |")
    lines += ["", "Totals (source observations, not independent votes): `" + json.dumps(summary["user_decisions"], sort_keys=True) + "`.",
              "Outcomes: `" + json.dumps(dict(sorted(Counter(o for u in users for o in u["outcomes"]).items()))) + "`.",
              "Outcome-only records have decision=null unless an explicit historical status or submission field establishes a decision. "
              "Rejection, interview, assessment, and closure never directly map to pursue/skip/undecided.",
              "", "## Agent verdict × user decision (unique postings)", "",
              "| Latest agent verdict | Pursue | Skip | Undecided | Conflicting decisions |", "|---|---:|---:|---:|---:|"]
    decisions = defaultdict(set)
    for u in users:
        if u["decision"]:
            decisions[u["posting_id"]].add(u["decision"])
    for verdict in VERDICTS:
        counter = Counter(next(iter(ds)) if len(ds) == 1 else "conflicting" for pid, ds in decisions.items() if latest_agent.get(pid) == verdict)
        lines.append(f"| {verdict} | " + " | ".join(str(counter[d]) for d in ("pursue", "skip", "undecided", "conflicting")) + " |")
    conflicts = {pid: sorted(ds) for pid, ds in decisions.items() if len(ds) > 1}
    summary["user_decision_conflicts"] = len(conflicts)
    lines += ["", f"{len(conflicts)} postings have different recorded decisions across sources/history. These are retained as evidence, not silently resolved or majority-voted.",
              "", "## Example rows (description at most 300 characters)", ""]
    examples = []
    for verdict in ("strong", "stretch", "reject"):
        selected = next((a for a in text_agents if a["verdict"] == verdict), None)
        if selected:
            p = by_id[selected["posting_id"]]
            examples.append({"posting_id": p["posting_id"], "employer": p["employer"], "title": p["title"],
                             "description": p["description"][:300], "verdict": verdict,
                             "description_source": p["description_source"], "policy_era": selected["policy_era"]})
    for example in examples:
        lines += ["```json", json.dumps(example, ensure_ascii=False, indent=2), "```", ""]
    lines += ["## Data quality and interpretation", "",
              "- Agent receipts are agent judgments, never user decisions. Finder regex scores, desk tiers, prep-sheet recommendations, and collector decisions are not promoted into labels.",
              "- User labels are source observations from candidate ledgers/trackers. Considering/queued can reflect agent-maintained workflow rather than a directly observed candidate choice. Attribution is not independently verified.",
              "- Exact normalized employer + job/requisition ID is preferred; canonical URLs and fingerprints connect aliases. Title-only records remain separate unless employer matches and title similarity >=0.93 with >=0.08 margin. No global title matching.",
              "- Employer aliases are unified for splitting when exact posting identity connects their names. This may conservatively group parent/subsidiary brands; unrelated names without such evidence can still evade grouping.",
              "- Receipts currently lack employer/title fields, so fuzzy fallback cannot recover an entirely orphan receipt without identity evidence. Unmatched receipts remain with posting_id=null and appear in quality_issues.json.",
              "- Longest cleaned nonempty description wins; every candidate source is retained. Older/previous descriptions can win, and text is not guaranteed to be the version reviewed. Receipt content_hash is retained but cannot validate the original collector hash algorithm.",
              "- Capture text can include navigation, transcriber notes, or application form text. Explicit requirement-mapping evaluations, failed fetches, prep sheets and submission confirmations are excluded as description material. Structured ATS fields are extracted without surrounding review reasoning.",
              "- Split eligibility requires nonempty text, not full-text completeness. The unlabeled pool requires >=300 characters and no agent label; it is not a fourth evaluation split. Use unlabeled_train_pool to avoid validation/test employer leakage.",
              "- Source observations and user labels can duplicate desk/tracker/ledger evidence. Cross-tabs collapse to unique postings; conflicts are flagged, and no temporal reconciliation is invented.",
              "- Policy eras use the receipt date as recorded (before Sep 21, Sep 21–29, Sep 30 onward, 2026). The copied current policy includes later edits and is not a historical policy snapshot.",
              "- Emails and recognizable phone numbers are redacted in processed narrative text. Structured job IDs and URLs are preserved; raw copies intentionally retain original content and stay under gitignored data/. This is contact redaction, not full anonymization.",
              "- Sequential file copies are not a single point-in-time snapshot. Each file has before/after size and mtime, destination hash, and a mid-copy change flag. SQLite uses a read-only source connection and a consistent backup; WAL/SHM/journal metadata is also checked.",
              f"- Capture parse/identity notes: {len(corpus.issues)}; see quality_issues.json. Imported files changed during copy: {len(summary['changed_during_copy'])}.",
              "- Source projects are read only through the immutable snapshot; no source-project scripts are run.",
              "", "## Reproduce", "", "```powershell", "python scripts/build_dataset.py", "python -B -m unittest discover -s tests -v", "```", "",
              "The (private) importer never overwrites previous import directories. The builder selects the latest complete YYYYMMDD-HHMM import and reads no live source files. Rebuilding that snapshot is deterministic."]
    return "\n".join(lines) + "\n", summary, conflicts


def main():
    imports = sorted(p for p in RAW.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}-\d{4}", p.name))
    if not imports:
        raise SystemExit("No raw import found under data/raw/ (see docs/DATA.md).")
    root = imports[-1]
    manifest = json_load(root / "manifest.json")
    if not manifest.get("complete"):
        raise SystemExit("Latest import is incomplete; refusing to use an older snapshot silently.")
    for entry in manifest["files"]:
        if entry["status"] != "copied":
            raise SystemExit("Missing import source: " + entry["source_path"])
        path = root / entry["dest_path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes snapshot")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != entry["sha256"]:
            raise ValueError("Raw snapshot hash mismatch: " + entry["dest_path"])
    corpus = Corpus()
    print("Snapshot hashes verified; reading posting sources.", flush=True)
    user_refs = load_sources(root, corpus)
    print(f"Loaded {len(corpus.rows):,} observations; merging identities.", flush=True)
    postings, root_to_id = merge_postings(corpus)
    print(f"Merged {len(postings):,} postings; joining labels and creating splits.", flush=True)
    agents = []
    receipts = json_load(root / "agent_a/state/review-receipts.json")
    for fp, row in sorted(receipts.items()):
        found, method, confidence = corpus.lookup(row | {"fingerprint": fp})
        agents.append({"posting_id": root_to_id[found] if found is not None else None,
                       "receipt_fingerprint": fp, "verdict": row["verdict"], "reason": clean_text(row["reason"]),
                       "reviewed_at": row["reviewed_at"], "policy_era": policy_era(row["reviewed_at"]),
                       "match_method": method, "match_confidence": confidence,
                       "official_url": row.get("official_url"), "content_hash": row.get("content_hash"),
                       "source": "agent_a/state/review-receipts.json", "label_author": "agent"})
    users = []
    for obs, row, source, source_id, category, method, confidence in user_refs:
        users.append({"posting_id": root_to_id[corpus.root(obs)], "source": source,
                      "source_id": source_id, "source_category": category,
                      "match_method": method if method != "unmatched" else "new_source_identity",
                      "match_confidence": confidence, **user_signal(row)})
    splits = build_splits(postings, agents, users)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "postings.jsonl", postings)
    write_jsonl(OUT / "labels_agent.jsonl", agents)
    write_jsonl(OUT / "labels_user.jsonl", users)
    text, summary, conflicts = report(root, manifest, corpus, postings, agents, users, splits)
    (OUT / "REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    diagnostics = {"capture_issues": corpus.issues, "conflicting_user_decisions": conflicts,
                   "unmatched_receipts": [a for a in agents if not a["posting_id"]]}
    for name, value in [("splits.json", splits), ("summary.json", summary), ("quality_issues.json", diagnostics)]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
