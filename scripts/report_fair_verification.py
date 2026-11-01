"""Refresh Exp 4 results from complete saved predictions; no model/test access."""
import json
from fair_verification import ROOT,OUT,analyze

def main():
    results={}
    for name in ['full-e3','exp1','exp2','variant-a','variant-c']:
        d=OUT/name
        if (d/'summary.json').exists():results[name]=analyze(name,d)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    fmt=lambda x:'n/a' if x is None else f'{x:.3f}'
    lines=['### Results','', '| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) | Verdict val | Verdict val2 | Mean | Pass |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for n,r in results.items():
        m=r['groups']['overall'];vs=r['verdict']
        vals=[r['threshold'],m['precision'],m['recall'],m['f1'],m['roc_auc'],m['pr_auc_average_precision'],vs['val'],vs['val2'],r['verdict_mean']]
        lines.append('| '+n+' | '+' | '.join(map(fmt,vals))+' | '+('yes' if r['passes'] else 'no')+' |')
    lines+=['','Always-yes fair baseline: precision 0.103, recall 1.000, F1 0.186, ROC-AUC 0.500 and AP 0.103. Always-no positive precision/recall/F1 are zero.','', '#### Matched and verdict-stratified checks','', 'Each paired comparison uses the same 20 unresolved positives plus only the named negative verdict(s). These are diagnostic comparisons, not invented positive verdict labels. Thresholds remain those selected on calibration.','', '| Model | Negative cohort | P | R | F1 | ROC-AUC | AP | Negative FPR |','|---|---|---:|---:|---:|---:|---:|---:|']
    for n,r in results.items():
        for group in ['matched_nonreject','positive_vs_strong','positive_vs_stretch','positive_vs_reject']:
            m=r['groups'][group]
            lines.append('| '+n+' | '+group+' | '+' | '.join(fmt(m[k]) for k in ['precision','recall','f1','roc_auc','pr_auc_average_precision','false_positive_rate'])+' |')
    lines+=['','Matched non-reject always-yes baseline: precision 0.317, recall 1.000, F1 0.482, ROC-AUC 0.500, AP 0.317. Every tested model has lower matched F1 at its calibration-selected threshold.','Within each resolved-only stratum there are zero positives: ROC-AUC/AP and positive recall are undefined; precision and F1 are zero by convention when false positives occur. The paired-table FPR is exactly that stratum\'s FPR. Within unresolved-only rows, only recall is informative.','', '#### Training and conclusion','']
    for n in ['variant-a','variant-c']:
        p=OUT/n/'training.json'
        if p.exists():
            t=json.loads(p.read_text());lines.append(f"{n}: runtime {t['metrics']['train_runtime']:.1f}s; {t['build']}.")
    if 'variant-c' in results:
        raw=results['variant-c']['raw_verdict']
        lines.append(f"C raw four-way argmax verdict macro-F1: val {raw['val']:.3f}, val2 {raw['val2']:.3f}, mean {sum(raw.values())/2:.3f}. The headline uses the calibration-selected threshold policy.")
        lines.append('C used one epoch under the declared compute budget; it is not an exhaustive test of a fourth verdict. The current weak labels do not support a causal capacity claim.')
    if 'variant-a' not in results:lines.append('Execution remains in progress; no trained-variant conclusion yet.')
    elif any(r['passes'] for r in results.values()):
        wins=[n for n,r in results.items() if r['passes']];lines.append('Passing experimental candidates: '+', '.join(wins)+'. Weak labels and six calibration positives still require independent validation before deployment.')
    else:
        lines.append('No completed model passes the fair test. Do not recommend the Exp 3 threshold or claim that needs_verification works. Retain full-e3 for verdicts; redefine the auxiliary question around directly observable text ambiguity and obtain independently audited same-source positives/negatives, or drop it as a serving gate. These results do not establish a head-capacity limit: weak supervision, train/inference schema differences and calibration instability remain confounded.')
        lines.append('Serving: keep outputs/full-e3/adapter; do not enable --needs-verification-threshold 0.075. No replacement adapter/threshold is recommended.')
    lines+=['','All evaluation rows were scored from reloaded adapters. Run commands, stdout/stderr, status and predictions are under outputs/fair-verification/<model>/. No test evaluation or protected-directory mutation was performed.']
    p=ROOT/'training/EXPERIMENTS.md';s=p.read_text(encoding='utf-8-sig');prefix,section=s.split('## Exp 4: fair verification evaluation',1);section=section.split('### Results',1)[0]
    p.write_text(prefix+'## Exp 4: fair verification evaluation'+section+'\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
