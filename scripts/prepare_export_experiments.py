"""Prepare immutable experiment splits without importing GPU libraries."""
import json, hashlib, random, re
from collections import Counter, defaultdict
from pathlib import Path
import build_decision_dataset as b
from build_dataset import normalize_employer, canonical_url
ROOT=b.REPO
OUT=ROOT/'data/experiments/export-v3'
NON_TEXT=re.compile(r'filled|not (currently )?(open|active)|no longer|closed|could not be found|cannot find|not found|does not exist|employer-of-record|third-party payroll|unpaid|existing candidate rejection|already applied|duplicate|stale|expired', re.I)
# Only text-observable uncertainty; source availability, candidate evidence and policy uncertainty are not positive labels.
UNCERTAIN=re.compile(r'sponsor|visa|authoriz|qualifications|requirements|contradict|truncat|duties',re.I)
EXTERNAL=re.compile(r'404|410|timed? out|redirect|shell|not verified|not recovered|no exact|live status|current individual|candidate.confirmed|payroll|client placement|direct.hire|staffing|recruiter',re.I)
def read(p): return b.read_jsonl(p)
def write(p,rs): p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')
def main():
    assert not OUT.exists(), 'Immutable dataset already exists'
    base={s:read(b.OUT/(s+'.jsonl')) for s in ('train','bulk','val','test')}
    posts=read(b.PROCESSED/'postings.jsonl'); exports=read(ROOT/'data/raw/agent-export-20261028-2111/postings.jsonl')
    nodes=posts+exports; parent=list(range(len(nodes)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j): parent[find(i)]=find(j)
    index={}
    for i,p in enumerate(nodes):
        keys=['id:'+p['posting_id']] if p.get('posting_id') else []
        keys+=['fp:'+str(x) for x in [p.get('source_id'),p.get('fingerprint'),*(p.get('fingerprints') or [])] if x]
        keys+=['url:'+canonical_url(u) for u in [p.get('url'),*(p.get('urls') or []),*(p.get('canonical_urls') or [])] if u and canonical_url(u)]
        desc=(p.get('description') or '').strip()
        if desc: keys+=['text:'+hashlib.sha256(desc.encode()).hexdigest()]
        for k in keys:
            if k in index: union(i,index[k])
            else: index[k]=i
    components=defaultdict(list)
    for i,p in enumerate(nodes): components[find(i)].append(p)
    ids={}; emps={}
    for c,ps in components.items():
        ids[c]={p['posting_id'] for p in ps if p.get('posting_id')}
        emps[c]=set().union(*(b.employer_keys({**p,'employer':p.get('employer') or p.get('company')},normalize_employer) for p in ps))
    split=json.loads((b.PROCESSED/'splits.json').read_text())
    held_ids=set(split['val'])|set(split['test']); held_emp={e for e,s in split['employer_assignment'].items() if s in ('val','test')}
    used={r['posting_id'] for s in base.values() for r in s}
    used_emp=set().union(*(emps[c] for c in components if ids[c]&used))
    counts=Counter(); candidates=[]; blocked=set()
    for i,p in enumerate(exports,len(posts)):
        c=find(i)
        if p.get('verdict_normalized')=='reject' and NON_TEXT.search(p.get('reasons') or ''): blocked.add(c)
    # Remove known non-text rejects even when they already had an old train label.
    eligible={}
    for i,p in sorted(enumerate(exports,len(posts)),key=lambda x:x[1].get('reviewed_at') or ''):
        c=find(i)
        if c in blocked: counts['non_text_reject_rows']+=1; continue
        if ids[c]&held_ids or emps[c]&held_emp: counts['held_original_rows']+=1; continue
        if not p.get('description') or not emps[c]: counts['missing_text_or_employer']+=1; continue
        if p['verdict_normalized'] not in (*b.VERDICTS,'needs_information'): continue
        eligible[c]=p
    # Reserve all eligible unseen employers used by either holdout. No val2 or verification evaluation employer trains in either experiment.
    unseen=[c for c in eligible if not ids[c]&used]
    rng=random.Random(3407); rng.shuffle(unseen)
    # Prefer genuinely new employers; fill to 175 from employers costing the fewest old training rows.
    old_train=[r for s in ('train','bulk') for r in base[s]]
    pid_emp={pid:emps[c] for c in ids for pid in ids[c]}
    unseen.sort(key=lambda c:sum(bool(emps[c]&pid_emp[r['posting_id']]) for r in old_train))
    val2=[]; verify=[]; reserved=set()
    for c in unseen:
        p=eligible[c]; reason=p.get('reasons') or ''
        if p['verdict_normalized'] in b.VERDICTS and len(val2)<175:
            val2.append(c); reserved|=emps[c]
        elif p['verdict_normalized']=='needs_information' and UNCERTAIN.search(reason) and not EXTERNAL.search(reason) and not NON_TEXT.search(reason) and not emps[c]&used_emp and len(verify)<20:
            verify.append(c); reserved|=emps[c]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(b.DEFAULT_TOKENIZER)
    count=lambda t:len(tok(t,add_special_tokens=False)['input_ids'])
    profile=b.PROFILE.read_text(encoding='utf-8')
    def row(c,p,needs=False):
        posting={'title':p.get('title'),'employer':p.get('company'),'locations':[p.get('location') or ''],'description':p['description']}
        state,tr=b.make_state(profile,posting,950,count)
        gold={'verdict':'stretch' if needs else p['verdict_normalized']}
        if needs: gold['needs_verification']=True
        return {'posting_id':sorted(ids[c])[0] if ids[c] else 'export:'+hashlib.sha256(str(c).encode()).hexdigest()[:16], 'state':state,'questions':{q:b.QUESTIONS[q] for q in gold},'gold':gold,'employer_keys':sorted(emps[c]),'label_source':'export-weak-needs' if needs else 'export-review','source_id':p.get('source_id'),'labeller_gold':None}
    pid_component={pid:c for c in ids for pid in ids[c]}
    trains={r['posting_id']:dict(r,employer_keys=sorted(emps[pid_component[r['posting_id']]])) for s in ('train','bulk') for r in base[s] if pid_component[r['posting_id']] not in blocked and not emps[pid_component[r['posting_id']]]&reserved}
    extra=[]; positives=[]; conflicts=[]
    for c,p in eligible.items():
        if emps[c]&reserved: continue
        old=[pid for pid in ids[c] if pid in trains]
        if p['verdict_normalized']=='needs_information':
            reason=p.get('reasons') or ''
            if UNCERTAIN.search(reason) and not EXTERNAL.search(reason) and not NON_TEXT.search(reason):
                # Conservative exclusion of explicit hard gates in existing labelled evidence.
                if any(trains[pid]['gold'].get('sponsorship_blocked') or trains[pid]['gold'].get('clearance_or_citizenship_required') or trains[pid]['gold'].get('min_years',0)>=2 for pid in old): continue
                positives.append(row(c,p,True))
            continue
        if old:
            # Keep multi-question state and targets, replace verdict with latest detailed review authority.
            for pid in old:
                if trains[pid]['gold']['verdict']!=p['verdict_normalized']: conflicts.append({'posting_id':pid,'old':trains[pid]['gold']['verdict'],'new':p['verdict_normalized']})
                trains[pid]['gold']=dict(trains[pid]['gold'],verdict=p['verdict_normalized'])
        else: extra.append(row(c,p))
    train1=list(trains.values())+extra
    positive_ids={r['posting_id'] for r in positives}
    train2=[r for r in train1 if r['posting_id'] not in positive_ids]+positives
    heldrows=[row(c,eligible[c]) for c in val2]; verifyrows=[row(c,eligible[c],True) for c in verify]
    for train in (train1,train2):
        assert not {r['posting_id'] for r in train}&held_ids
        assert not set().union(*(set(r['employer_keys']) for r in train))&(reserved|held_emp)
    OUT.mkdir(parents=True)
    # Shared calibration employers chosen without observing validation outcomes.
    cal_emp=set()
    available=sorted({e for r in train2 for e in r['employer_keys']})
    random.Random(919).shuffle(available)
    cal_emp.update(available[:max(1,len(available)//10)])
    while True:
        expanded=cal_emp|{e for r in train2 if set(r['employer_keys'])&cal_emp for e in r['employer_keys']}
        if expanded==cal_emp: break
        cal_emp=expanded
    blocked_ids=set().union(*(ids[c] for c in blocked))
    for split_name in ('val','test'):
        write(OUT/(split_name+'.jsonl'),[r for r in base[split_name] if r['posting_id'] not in blocked_ids])
    for n,rs in [('val2',heldrows),('verification',verifyrows),('conflicts',conflicts)]: write(OUT/(n+'.jsonl'),rs)
    for n,rs in [('exp1',train1),('exp2',train2)]:
        d=OUT/n; d.mkdir(); write(d/'train.jsonl',rs)
        fit=[r for r in rs if not set(r['employer_keys'])&cal_emp]
        cal=[r for r in rs if set(r['employer_keys'])&cal_emp]
        assert not {e for r in fit for e in r['employer_keys']}&{e for r in cal for e in r['employer_keys']}
        write(d/'fit.jsonl',fit); write(d/'calibration.jsonl',cal)
        for s in ('val','test'): write(d/(s+'.jsonl'),base[s])
    manifest={'seed':3407,'filter_counts':dict(counts),'identity_components':len(components),'unseen_eligible_components':len(unseen),'conflicts':len(conflicts),'positive_training':len(positives),'datasets':{n:{'rows':len(rs),'verdict':dict(Counter(r['gold']['verdict'] for r in rs))} for n,rs in [('exp1',train1),('exp2',train2),('val2',heldrows),('verification',verifyrows)]}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
