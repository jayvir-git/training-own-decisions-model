"""Conservative receipt policy audit; never changes receipts or labeler inputs.

Human-adjudicated entries live in v2/audit-confirmed-{trainval,final}.json.
The other matches are explicitly unconfirmed candidates, not measured label errors.
Final/test source access requires the frozen context choice.
"""
import argparse
import json
import re

from label_agreement_v2 import BASE, OUT, CLASSES, entries_for, posting_inputs, read_json, write_json, now


LOW_YEARS = r'(?i)(?:\b[12]\s*(?:\+|[-–\ufffd]\s*[234])?\s*(?:years?|yrs?)|\b(?:one|two)[ -](?:to[ -]three[ -])?years?)'
STUDENT_TITLE = r'(?i)\b(intern(?:ship)?|co[- ]?op|apprentice(?:ship)?)\b'
FOREIGN = r'(?i)\b(canada|india|united kingdom|ireland|germany|austria|poland|romania|france|taiwan|japan|singapore|malaysia|costa rica|sweden|vietnam|new zealand|australia|spain|brazil|mexico)\b'
US = r'(?i)\b(united states|u\.?s\.?a?|america|california|new york|massachusetts|texas|new jersey)\b'


def audit(final=False):
    if final and not (OUT / 'choice.json').exists():
        raise RuntimeError('No test audit before context selection.')
    entries = entries_for(final=final)
    posts = posting_inputs(entries)
    adjudicated = {}
    for name in ['trainval', 'final'] if final else ['trainval']:
        path = OUT / f'audit-confirmed-{name}.json'
        if path.exists():
            adjudicated.update(read_json(path))
    confirmed, candidates = [], []
    scanned = 0
    for e in entries:
        pid = e['posting_id']
        receipt = e['receipt']
        if receipt['verdict'] not in CLASSES:
            continue
        scanned += 1
        p = posts[pid]
        title, location = p['title'], ' | '.join(p['locations'])
        reasons = []
        if receipt['verdict'] == 'reject' and re.search(LOW_YEARS, receipt['reason']):
            reasons.append('Reject receipt mentions <=2 years; verify relevant evidence and any independent gate/depth requirement.')
        if receipt['verdict'] in ('strong', 'stretch'):
            if re.search(STUDENT_TITLE, title):
                reasons.append('Actionable receipt with a student-role title; check title is this role, not past experience.')
            if re.search(FOREIGN, location) and not re.search(US, location):
                reasons.append('Actionable receipt with foreign location; inspect all locations and description for a US option.')
            if re.search(r'(?i)\b[3-9]\s*(?:\+|[-–\ufffd]\s*\d+)?\s*(?:years?|yrs?)\b', receipt['reason']):
                reasons.append('Actionable receipt mentions >=3 years; verify lower bound, preferred status and accepted equivalencies.')
        if receipt['verdict'] == 'strong' and re.search(LOW_YEARS, p['description'][:12000]):
            reasons.append('Strong receipt with a possible <=2-year requirement; check mandatory status and degree pathways.')
        common = {'posting_id': pid, 'split': e['split'], 'receipt_verdict': receipt['verdict'],
                  'title': title, 'location': p['locations'], 'receipt_reason': receipt['reason']}
        if pid in adjudicated:
            confirmed.append({**common, **adjudicated[pid]})
        elif reasons:
            candidates.append({**common, 'reasons_to_review': reasons})
    unknown = set(adjudicated) - {e['posting_id'] for e in entries}
    if unknown:
        raise RuntimeError(f'Adjudication IDs outside allowed scope: {sorted(unknown)}')
    result = {'updated_at': now(), 'status': 'completed conservative audit' if final else 'train/val only; test locked',
              'scope': 'all 702 frozen receipt-labeled postings' if final else 'train and validation only',
              'postings_scanned': len(entries), 'non_NI_receipts_scanned': scanned,
              'method': 'Deterministic screen of titles, locations and year-language in receipts/postings, with manual inspection '
                        'of selected source requirements and candidate evidence. Confirmed flags are policy conflicts; '
                        'other matches remain unconfirmed, often because an independent specialty/eligibility gate, preferred '
                        'years, a range upper bound, or an accepted equivalency can explain the receipt. The confirmed count '
                        'is a conservative lower bound, not an exhaustive semantic error count. Includes the literal v2 '
                        '<=2-year-to-stretch rule for zero-lower-bound roles previously rated strong. No model votes '
                        'or agreement outcome are used as proof of a policy conflict. Receipts are never rewritten.',
              'confirmed': sorted(confirmed, key=lambda x: (x['split'], x['posting_id'])),
              'candidates': candidates}
    write_json(OUT / ('rule-conflicts.json' if final else 'rule-conflicts-trainval.json'), result)
    return {k: result[k] for k in ('status', 'postings_scanned', 'non_NI_receipts_scanned')} | {
        'confirmed_count': len(confirmed), 'unconfirmed_candidates': len(candidates)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.final), indent=2))
