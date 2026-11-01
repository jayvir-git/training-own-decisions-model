"""Detached sequential runner. Every GPU subprocess is protected by gpu_lock.py."""
import json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'.venv/Scripts/python.exe'
DATA=ROOT/'data/experiments/export-v3'
OUT=ROOT/'outputs/export-experiments'
RUNS={'full-e3':'outputs/export-baseline','exp1':'outputs/export-more-e3','exp2':'outputs/export-needs-e3'}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def report(status):
    m=read(DATA/'manifest.json');lines=['# Export training experiments','',f'Status: {status}','','## Protocol','',
    'Learning objective: typed-decision modelling, held-out macro-F1, per-class precision/recall and calibration. No job-search utility claim.',
    'Unsloth 2026.10.9 / Qwen3.5-0.8B; unchanged five-question schema; rank 16, batch 1, accumulation 8, 3 epochs, backbone LR 2e-4, head LR 1e-4, GPU checkpointing. Both runs start from the same base model.',
    'All GPU processes run through scripts/gpu_lock.py. Each training process has a 3.9-hour watchdog. Full-e3 and full-e3-os are read-only. No commits.',
    '', '## Data and leakage controls','',
    'Final data: data/experiments/export-v3. v1/v2 are preparation-only drafts; neither was trained or used for model selection. Identity components join fingerprint, canonical URL (preserving job query parameters), posting ID and exact description hashes. All matching IDs and employer aliases are checked against original val/test assignments.',
    'Val2 contains never-trained posting identities. Selection prefers employers unseen by full-e3, then employers with minimal old-training overlap to reach 175; all val2 and verification employers are excluded from BOTH new training sets. Thus full-e3 may have seen other postings at some val2 employers, but no val2 posting. This favors the reference baseline. Employer-disjoint calibration subsets are drawn from each training pool, independent of original val/val2. No test outcomes enter selection.',
    'The requested non-text-reject reason regex excludes whole matched identity components, including matching old training rows and evaluation rows. Conservative identity joins can overexclude reused descriptions/URLs.',
    'Conflict policy: latest export detailed review verdict wins over old labeller verdict, consistent with receipt-based evaluation; retain the other four existing targets and existing state. This mixes data expansion and label correction, so causal improvement cannot be attributed to row count alone.',
    'Partial labels are supported by installed unsloth/models/decision.py, _clef_items lines 365-430: missing gold questions are skipped; kept questions alone are encoded. New verdict-only rows carry only that question. Missing labels are never fabricated as negatives. Context7 lacked version-specific partial-label documentation; installed source is authoritative.',
    '', '| Dataset | Rows | Verdict counts |','|---|---:|---|']
    for n,x in m['datasets'].items():lines.append(f"| {n} | {x['rows']} | {x['verdict']} |")
    for n in ('exp1','exp2'):
        for split in ('fit','calibration'):
            rs=[json.loads(x) for x in (DATA/n/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
            lines.append(f"| {n} {split} | {len(rs)} | verification positives: {sum(r['gold'].get('needs_verification') is True for r in rs)} |")
    lines+=['',f"Resolved verdict conflicts: {m['conflicts']}; weak positive training rows before calibration: {m['positive_training']}. Filter event counts: {m['filter_counts']}.", '',
    'Weak needs_verification supervision maps needs_information to stretch + true only when reasons mention text requirements, qualifications, sponsorship/authorization or contradictions, excluding source availability and non-text status reasons. Existing explicit sponsorship/citizenship/3+year hard-gate labels veto these positives. This is a conservative lexical proxy, NOT audited ground truth. Remaining candidate-evidence or policy ambiguity may still be noise. No negative verification labels are inferred from export verdicts.',
    'The verification holdout is positive-only; its recall measures proxy recovery, and precision on that subset alone is uninformative. Inspect original-val positive precision/recall and false positives as well. Original val has very few genuine positives. No class weighting in these first controlled runs.',
    '', '## Commands','', 'Run orchestration: `.venv/Scripts/python.exe -B scripts/run_export_experiments.py` (launch detached with PowerShell Start-Process; logs under outputs/export-experiments).',
    'Each train command: `.venv/Scripts/python.exe -B scripts/gpu_lock.py -- .venv/Scripts/python.exe -B scripts/train_export_experiment.py --dataset data/experiments/export-v3/expN --epochs 3 --batch-size 1 --grad-accum 8 --lr 2e-4 --head-lr 1e-4 --rank 16 --grad-ckpt gpu --out outputs/<run>`.',
    'Scoring uses evaluate_export_experiment.py --adapter <adapter> --out <distinct evaluation directory>. Only the final chosen new model gets --test; the test-started.json sentinel prevents accidental reruns. If full-e3 wins, its existing test result is reused, not rescored.',
    '', '## Results','', '| Model | Split | N | Accuracy | Macro-F1 | NLL | Brier |','|---|---|---:|---:|---:|---:|---:|']
    for n,d in RUNS.items():
        path=ROOT/d/'evaluation/metrics.json'
        if not path.exists():continue
        for split,x in read(path).items():
            z=x['model'];cal=x.get('calibration',{})
            lines.append(f"| {n} | {split} | {z['n']} | {z['accuracy']} | {z['macro_f1']} | {cal.get('nll',0):.3f} | {cal.get('brier',0):.3f} |")
            z=x['always_reject'];lines.append(f"| always reject | {split} | {z['n']} | {z['accuracy']} | {z['macro_f1']} | - | - |")
        lines+=['',f'### {n}: per-class precision / recall','', '| Split | Class | Precision | Recall | Support |','|---|---|---:|---:|---:|']
        for split,x in read(path).items():
            for c,z in x['model']['per_class'].items():lines.append(f"| {split} | {c} | {z['precision']} | {z['recall']} | {z['support']} |")
            z=x['needs_verification']['per_class']['true'];lines.append(f"| {split} | needs_verification=true | {z['precision']} | {z['recall']} | {z['support']} |")
        lines+=['','| Model | Split | N | Accuracy | Macro-F1 | NLL | Brier |','|---|---|---:|---:|---:|---:|---:|']
    lines+=['','## Selection and limitations','',
    'Predeclared verdict score = mean macro-F1 on original receipt val and val2. Experiment 1 competes against full-e3 on this score. Experiment 2 is preferred only if its score is within 0.02 of that winner and verification-holdout positive recall improves. Original-val false positives are reported, not hidden. Test is scored once after final selection; never used to choose. Export labels are agent judgements, not independent human truth; policy-era drift and tiny minority-class supports limit conclusions.',
    'Calibration reports held-out multiclass NLL and Brier (sum over three classes), using training-employer holdout temperature calibration. Full-e3 was calibrated on original val, so its original-val calibration numbers are optimistic; val2 provides the independent comparison.',
    'Existing regression suites: 21 tests in tests/ and 47 scripts/test_*.py tests passed; legacy resource warnings did not fail tests. No existing trainer or dataset builder was edited.']
    selection=OUT/'selection.json'
    if selection.exists():lines+=['', 'Selection: `'+json.dumps(read(selection))+'`']
    test=OUT/'final-test/metrics.json'
    if test.exists():lines+=['','Final held-out test: `'+json.dumps(read(test))+'`']
    (ROOT/'training/EXPERIMENTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run(script,args,d,timeout=14040):
    d.mkdir(parents=True,exist_ok=True)
    cmd=[str(PY),'-B',str(ROOT/'scripts/gpu_lock.py'),'--',str(PY),'-B',str(ROOT/'scripts'/script),*args]
    (d/'command.json').write_text(json.dumps(cmd,indent=2))
    with (d/'stdout.log').open('w') as out,(d/'stderr.log').open('w') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err)
        (d/'process.json').write_text(json.dumps({'pid':p.pid,'started':time.time()}))
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],check=False)
            raise RuntimeError(f'{script} exceeded watchdog; process tree stopped')
        if code:raise RuntimeError(f'{script} exit {code}; see {d}')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'started.json').open('x') as f:json.dump({'started':time.time()},f)
    try:
        report('baseline evaluation running; no new model results yet')
        run('evaluate_export_experiment.py',['--adapter','outputs/full-e3/adapter','--out',RUNS['full-e3']+'/evaluation'],ROOT/RUNS['full-e3']/'evaluation')
        for n in ('exp1','exp2'):
            report(n+' training running')
            run('train_export_experiment.py',['--dataset','data/experiments/export-v3/'+n,'--epochs','3','--batch-size','1','--grad-accum','8','--lr','2e-4','--head-lr','1e-4','--rank','16','--grad-ckpt','gpu','--out',RUNS[n]],ROOT/RUNS[n])
            run('evaluate_export_experiment.py',['--adapter',RUNS[n]+'/adapter','--out',RUNS[n]+'/evaluation'],ROOT/RUNS[n]/'evaluation')
            report(n+' completed')
        metrics={n:read(ROOT/d/'evaluation/metrics.json') for n,d in RUNS.items()}
        scores={n:sum(x[s]['model']['macro_f1'] for s in ('val','val2'))/2 for n,x in metrics.items()}
        winner=max(('full-e3','exp1'),key=lambda n:scores[n])
        rec=lambda n:metrics[n]['verification']['needs_verification']['per_class']['true']['recall']
        if scores['exp2']>=scores[winner]-.02 and rec('exp2')>rec(winner):winner='exp2'
        adapter='outputs/full-e3/adapter' if winner=='full-e3' else RUNS[winner]+'/adapter'
        (OUT/'selection.json').write_text(json.dumps({'winner':winner,'adapter':adapter,'validation_scores':scores,'test_used_for_selection':False},indent=2))
        report('selection frozen; final test pending')
        if winner!='full-e3':run('evaluate_export_experiment.py',['--adapter',adapter,'--out','outputs/export-experiments/final-test','--test'],OUT/'final-test')
        report('complete; full-e3 existing test reused if reference won')
        (OUT/'complete.json').write_text(json.dumps({'finished':time.time(),'adapter':adapter}))
    except Exception:
        (OUT/'failure.txt').write_text(traceback.format_exc());report('FAILED: see outputs/export-experiments/failure.txt');raise
if __name__=='__main__':main()
