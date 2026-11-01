"""CPU-only, fixed five-fold threshold assessment from saved validation predictions.

Never reads test. Thresholds maximize calibration F1, breaking ties toward higher
thresholds. All held-out rows receive exactly one out-of-fold decision.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'full-e3': 'export-baseline', 'exp1': 'export-more-e3', 'exp2': 'export-needs-e3'}
GRID = [0, .01, .025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, 1]

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def cohort(val, weak):
    rows = [dict(r, source='original_val') for r in val if 'needs_verification' in r['gold']]
    rows += [dict(r, source='weak_holdout') for r in weak]
    ids = [r['posting_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate posting identity')
    if any(type(r['gold']['needs_verification']) is not bool for r in rows):
        raise ValueError('non-boolean target')
    if any(not r['gold']['needs_verification'] for r in weak):
        raise ValueError('weak holdout must contain positives')
    # Stratify source/target; sorting by a salted hash is independent of predictions.
    for source, label in [('original_val', False), ('original_val', True), ('weak_holdout', True)]:
        group = [r for r in rows if r['source'] == source and r['gold']['needs_verification'] == label]
        group.sort(key=lambda r: hashlib.sha256(('needs-v1:'+r['posting_id']).encode()).hexdigest())
        for i, r in enumerate(group):
            r['fold'] = i % 5
    return rows

def metrics(y, pred):
    tp = sum(a and b for a,b in zip(y,pred)); fp = sum(not a and b for a,b in zip(y,pred))
    fn = sum(a and not b for a,b in zip(y,pred)); tn = len(y)-tp-fp-fn
    return dict(n=len(y), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)

def at(rows, threshold):
    return metrics([r['gold']['needs_verification'] for r in rows], [r['p'] >= threshold for r in rows])

def select(rows):
    return max(GRID, key=lambda t: (at(rows,t)['f1'],t))

def ranking(rows):
    pos = [r['p'] for r in rows if r['gold']['needs_verification']]
    neg = [r['p'] for r in rows if not r['gold']['needs_verification']]
    auc = sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
    # Noninterpolated average precision; ties enter together.
    ap = 0; previous_recall = 0
    for t in sorted(set(pos+neg),reverse=True):
        m = at(rows,t); ap += (m['recall']-previous_recall)*m['precision']; previous_recall=m['recall']
    return dict(roc_auc=auc, pr_auc_average_precision=ap)

def main():
    out = ROOT/'outputs/verification-thresholds'; out.mkdir(exist_ok=True)
    data = ROOT/'data/experiments/export-v3'
    rows = cohort(read(data/'val.jsonl'), read(data/'verification.jsonl'))
    # Persist split/protocol before accessing probabilities.
    manifest = [{k:r[k] for k in ('posting_id','source','fold')} for r in rows]
    (out/'split.json').write_text(json.dumps({'seed':'needs-v1','grid':GRID,'rows':manifest},indent=2))
    result = {}
    for name, directory in MODELS.items():
        path = ROOT/'outputs'/directory/'evaluation'
        saved = read(path/'val-predictions.jsonl') + read(path/'verification-predictions.jsonl')
        pred = {r['posting_id']:r for r in saved}
        if len(pred) != len(saved): raise ValueError('duplicate predictions')
        scored=[]
        for r in rows:
            p=pred[r['posting_id']]
            if p['gold']['needs_verification'] != r['gold']['needs_verification']: raise ValueError('gold mismatch')
            score=p['probabilities']['needs_verification']['true']
            if not 0 <= score <= 1: raise ValueError('invalid probability')
            scored.append(dict(r,p=score))
        decisions=[]; thresholds=[]
        for fold in range(5):
            calibration=[r for r in scored if r['fold'] != fold]
            evaluation=[r for r in scored if r['fold'] == fold]
            t=select(calibration); thresholds.append(t)
            decisions.extend(dict(r,decision=r['p']>=t,threshold=t) for r in evaluation)
        verdict=json.loads((path/'metrics.json').read_text())
        result[name] = dict(thresholds=thresholds, ranking=ranking(scored),
            crossfit=metrics([r['gold']['needs_verification'] for r in decisions],[r['decision'] for r in decisions]),
            by_source={s:metrics([r['gold']['needs_verification'] for r in decisions if r['source']==s], [r['decision'] for r in decisions if r['source']==s]) for s in ('original_val','weak_holdout')},
            curve=[dict(threshold=t,**at(scored,t)) for t in GRID],
            verdict={s:verdict[s]['model']['macro_f1'] for s in ('val','val2')})
        (out/(name+'-decisions.jsonl')).write_text(''.join(json.dumps({k:r[k] for k in ('posting_id','fold','p','decision','threshold','source')})+'\n' for r in decisions))
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
