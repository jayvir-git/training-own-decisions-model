"""Exp 4 data and metrics; CPU-only, no test rows or protected-output writes."""
import copy, hashlib, json, re
from pathlib import Path
from collections import Counter
import build_decision_dataset as b
from build_dataset import canonical_url
from analyze_verification_thresholds import read, at, ranking, select
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/experiments/export-v3'
DATA=ROOT/'data/experiments/fair-verification'
OUT=ROOT/'outputs/fair-verification'

def write(p, rows):
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')

def export_index():
    # Reproduce export-v3 identity components, including old rows whose provenance
    # was retained when their verdict was corrected by the export.
    posts=read(ROOT/'data/processed/postings.jsonl')
    exports=read(ROOT/'data/raw/agent-export-20261028-2111/postings.jsonl')
    nodes=posts+exports; parent=list(range(len(nodes))); index={}
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    for i,p in enumerate(nodes):
        keys=['id:'+p['posting_id']] if p.get('posting_id') else []
        keys+=['fp:'+str(x) for x in [p.get('source_id'),p.get('fingerprint'),*(p.get('fingerprints') or [])] if x]
        keys+=['url:'+canonical_url(u) for u in [p.get('url'),*(p.get('urls') or []),*(p.get('canonical_urls') or [])] if u and canonical_url(u)]
        desc=(p.get('description') or '').strip()
        if desc: keys+=['text:'+hashlib.sha256(desc.encode()).hexdigest()]
        for k in keys:
            if k in index: parent[find(i)]=find(index[k])
            else: index[k]=i
    latest={}
    for i,p in sorted(enumerate(exports,len(posts)),key=lambda x:x[1].get('reviewed_at') or ''):
        if p.get('verdict_normalized') in (*b.VERDICTS,'needs_information'): latest[find(i)]=p
    byid={p['posting_id']:latest[find(i)] for i,p in enumerate(nodes) if p.get('posting_id') and find(i) in latest}
    bysource={p.get('source_id'):p for p in exports if p.get('source_id')}
    return byid,bysource

def relabel(r, review=None):
    r=copy.deepcopy(r)
    if r['label_source']=='export-weak-needs':
        r['gold']={'needs_verification':True};r['verification_stratum']='unresolved'
    elif r['label_source']=='export-review' or (review and review.get('verdict_normalized') in b.VERDICTS):
        r['gold']['needs_verification']=False
        r['verification_stratum']=r['gold']['verdict']
        r['resolved_export']=True
    r['questions']={k:b.QUESTIONS[k] for k in r['gold']}
    return r

def fourth_schema():
    questions=copy.deepcopy(b.QUESTIONS)
    questions.pop('needs_verification')
    questions['verdict']['criteria']['stretch']='credible adjacent evidence with bounded, learnable gaps'
    questions['verdict']['criteria']['needs_information']='a decisive gate in the posting text remains unresolved, with no independent rejection established'
    return questions

def fourth_rows(rows):
    questions=fourth_schema();result=[]
    for original in rows:
        r=copy.deepcopy(original)
        positive=r['gold'].pop('needs_verification',None) is True
        if positive:r['gold']['verdict']='needs_information'
        r['questions']={k:questions[k] for k in r['gold']}
        result.append(r)
    return result

def fourth_answer(probabilities, threshold):
    if probabilities['needs_information'] >= threshold:return 'needs_information'
    return max(b.VERDICTS,key=lambda c:probabilities[c])

def disjoint(a,brows):
    if {r['posting_id'] for r in a}&{r['posting_id'] for r in brows}: raise ValueError('posting overlap')
    if {e for r in a for e in r.get('employer_keys',[])}&{e for r in brows for e in r.get('employer_keys',[])}: raise ValueError('employer overlap')

def prepare():
    DATA.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=True)
    ids,sources=export_index();audit=[]
    def convert(rows):
        result=[]
        for r in rows:
            review=ids.get(r['posting_id']) or sources.get(r.get('source_id'))
            x=relabel(r,review);result.append(x)
            if x.get('resolved_export') and review and re.search(r'ambigu|unclear|uncertain|contradict|truncat',review.get('reasons') or '',re.I):
                audit.append({'posting_id':r['posting_id'],'reasons':review.get('reasons'),'rule':'resolved verdict takes precedence; remains false'})
        return result
    fit=convert(read(SOURCE/'exp2/fit.jsonl'));cal=convert(read(SOURCE/'exp2/calibration.jsonl'))
    fair=convert(read(SOURCE/'val2.jsonl')+read(SOURCE/'verification.jsonl'))
    disjoint(fit,cal);disjoint(fit+cal,fair)
    splits=json.loads((ROOT/'data/processed/splits.json').read_text(encoding='utf-8'))
    held=set(splits['val'])|set(splits['test'])
    heldemp={e for e,s in splits['employer_assignment'].items() if s in ('val','test')}
    assert not held&{r['posting_id'] for r in fit+cal}
    assert not heldemp&{e for r in fit+cal for e in r['employer_keys']}
    threshold_cal=[r for r in cal if r['label_source'] in ('export-review','export-weak-needs')]
    # Fixed 8x weak-positive repetition; no repeated calibration/evaluation rows.
    repeated=[r for r in fit for _ in range(8 if r['label_source']=='export-weak-needs' else 1)]
    for name,rows in [('fit',repeated),('calibration',threshold_cal),('fair',fair),('val',read(SOURCE/'val.jsonl')),('val2',read(SOURCE/'val2.jsonl'))]:write(DATA/(name+'.jsonl'),rows)
    manifest={'counts':{n:dict(n=len(rs),positive=sum(r['gold'].get('needs_verification') is True for r in rs),strata=dict(Counter(r.get('verification_stratum','old') for r in rs))) for n,rs in [('fit_unique',fit),('fit_repeated',repeated),('calibration',threshold_cal),('fair',fair)]},'ambiguity_audit':audit,'rule':'All resolved export verdicts false, including ambiguous wording; weak positives true without verdict. No reason-based exceptions.','acceptance':'precision >= .5, recall >= .5, matched non-reject ROC-AUC > .5, verdict mean >= .630','grid':'analyze_verification_thresholds.GRID; max calibration F1, higher threshold tie','oversample':8}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest['counts']))

def scored(rows,preds):
    index={r['posting_id']:r for r in preds}
    if len(index)!=len(preds):raise ValueError('duplicate predictions')
    out=[]
    for r in rows:
        p=index[r['posting_id']]['probabilities']['needs_verification']['true']
        if not 0<=p<=1:raise ValueError('bad probability')
        out.append(dict(r,p=p))
    return out

def summary(rows,t):
    m=at(rows,t)
    m.update(ranking(rows) if len({r['gold']['needs_verification'] for r in rows})==2 else {'roc_auc':None,'pr_auc_average_precision':None})
    m['false_positive_rate']=m['fp']/(m['fp']+m['tn']) if m['fp']+m['tn'] else None
    return m

def analyze(name, pred_dir):
    cal=scored(read(DATA/'calibration.jsonl'),read(pred_dir/'calibration-predictions.jsonl'))
    fair=scored(read(DATA/'fair.jsonl'),read(pred_dir/'fair-predictions.jsonl'));t=select(cal)
    groups={'overall':fair,'matched_nonreject':[r for r in fair if r['verification_stratum']!='reject']}
    groups.update({s:[r for r in fair if r['verification_stratum']==s] for s in ['strong','stretch','reject','unresolved']})
    groups.update({'positive_vs_'+s:[r for r in fair if r['verification_stratum'] in (s,'unresolved')] for s in ['strong','stretch','reject']})
    result={'threshold':t,'calibration':summary(cal,t),'groups':{s:summary(rs,t) for s,rs in groups.items()}}
    # Pure CPU verdict metrics, receipt-only original val.
    def macro(rs):
        f=[]
        for c in b.VERDICTS:
            tp=sum(r['gold']['verdict']==c and r['pred']['verdict']==c for r in rs)
            fp=sum(r['gold']['verdict']!=c and r['pred']['verdict']==c for r in rs)
            fn=sum(r['gold']['verdict']==c and r['pred']['verdict']!=c for r in rs)
            f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        return sum(f)/3
    verdict_rows={s:[r for r in read(pred_dir/(s+'-predictions.jsonl')) if s!='val' or r['label_source']=='receipt'] for s in ['val','val2']}
    result['raw_verdict']={s:macro(rs) for s,rs in verdict_rows.items()}
    if name=='variant-c':
        for rs in verdict_rows.values():
            for r in rs:r['pred']['verdict']=fourth_answer(r['probabilities']['verdict'],t)
    result['verdict']={s:macro(rs) for s,rs in verdict_rows.items()}
    result['verdict_mean']=sum(result['verdict'].values())/2
    m=result['groups']['overall'];result['passes']=m['precision']>=.5 and m['recall']>=.5 and result['groups']['matched_nonreject']['roc_auc']>.5 and result['verdict_mean']>=.630
    (pred_dir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(name,json.dumps(result),flush=True)
    return result

if __name__=='__main__':prepare()
