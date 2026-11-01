"""Score locked experiment splits. Never overwrites reference model directories."""
import argparse,json,math
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--adapter',required=True);ap.add_argument('--out',required=True);ap.add_argument('--test',action='store_true');a=ap.parse_args()
    import train_decision as td
    from unsloth import FastDecisionModel as F
    import build_decision_dataset as b
    out=td.REPO/a.out;out.mkdir(parents=True,exist_ok=True)
    marker=out/'test-started.json'
    if a.test:
        with marker.open('x') as f: json.dump({'adapter':a.adapter},f)
    model,tok=F.from_pretrained(str(td.REPO/a.adapter),max_seq_length=2048,load_in_4bit=True);F.for_inference(model)
    root=td.REPO/'data/experiments/export-v3'
    splits={'test':root/'test.jsonl'} if a.test else {'val':root/'val.jsonl','val2':root/'val2.jsonl','verification':root/'verification.jsonl'}
    allmetrics={}
    for name,path in splits.items():
        preds=[]
        with (out/(name+'-predictions.jsonl')).open('x',encoding='utf-8') as f:
            for r in td.read_jsonl(path):
                ans=F.predict(model,tok,r['state'],b.QUESTIONS)
                p={'posting_id':r['posting_id'],'gold':r['gold'],'label_source':r['label_source'],'pred':{k:v['answer'] for k,v in ans.items()},'probabilities':{k:v['probabilities'] for k,v in ans.items()}}
                f.write(json.dumps(p)+'\n');f.flush();preds.append(p)
        sub=[p for p in preds if name not in ('val','test') or p['label_source']=='receipt']
        g=[p['gold']['verdict'] for p in sub];pr=[p['pred']['verdict'] for p in sub]
        metrics={'model':td.prf(g,pr,td.VERDICTS),'always_reject':td.prf(g,['reject']*len(g),td.VERDICTS)}
        if sub:
            probs=[p['probabilities']['verdict'] for p in sub]
            metrics['calibration']={'nll':sum(-math.log(max(p.get(y,0),1e-8)) for p,y in zip(probs,g))/len(g),'brier':sum(sum((p.get(c,0)-(y==c))**2 for c in td.VERDICTS) for p,y in zip(probs,g))/len(g)}
        v=[p for p in preds if 'needs_verification' in p['gold']]
        metrics['needs_verification']=td.prf([p['gold']['needs_verification'] for p in v],[p['pred']['needs_verification'] for p in v],[True,False])
        allmetrics[name]=metrics
        (out/'metrics.json').write_text(json.dumps(allmetrics,indent=2),encoding='utf-8')
        print(name,json.dumps(metrics),flush=True)
if __name__=='__main__':main()
