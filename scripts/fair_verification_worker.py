"""Locked GPU worker for Exp 4: score or head-only train. Never reads test."""
import argparse,json,time
from pathlib import Path
from fair_verification import ROOT,DATA,read,write,fourth_schema,fourth_rows

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    ap=argparse.ArgumentParser();ap.add_argument('--adapter',required=True);ap.add_argument('--out',required=True);ap.add_argument('--train',action='store_true');ap.add_argument('--fourth-verdict',action='store_true');ap.add_argument('--epochs',type=int,default=3);a=ap.parse_args()
    import unsloth
    from unsloth import FastDecisionModel as F, DecisionTrainer, is_bfloat16_supported
    from transformers import TrainingArguments
    import build_decision_dataset as b
    import copy
    questions=copy.deepcopy(b.QUESTIONS)
    if a.fourth_verdict:questions=fourth_schema()
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    model,tok=F.from_pretrained(str(ROOT/a.adapter),max_seq_length=2048,load_in_4bit=True,use_gradient_checkpointing=True)
    if a.train:
        F.for_training(model,use_gradient_checkpointing=True);F.freeze_backbone(model)
        model.head.requires_grad_(True)
        assert not any(p.requires_grad for p in model.encoder.parameters())
        params={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
        assert params and all(n.startswith('head.') for n in params)
        (out/'trainable.json').write_text(json.dumps(params,indent=2))
        rows=read(DATA/'fit.jsonl')
        if a.fourth_verdict:rows=fourth_rows(rows)
        items,report=F.build_dataset([{'state':r['state'],'questions':r['questions'],'gold':r['gold']} for r in rows],tok,model)
        print('build',report,flush=True)
        trainer=DecisionTrainer(model=model,processing_class=tok,train_dataset=items,head_learning_rate=1e-4,args=TrainingArguments(per_device_train_batch_size=1,gradient_accumulation_steps=8,num_train_epochs=a.epochs,learning_rate=1e-4,warmup_steps=10,lr_scheduler_type='cosine',weight_decay=.01,bf16=is_bfloat16_supported(),fp16=not is_bfloat16_supported(),logging_steps=5,save_strategy='no',output_dir=str(out/'trainer'),report_to='none',seed=3407,dataloader_num_workers=0))
        stats=trainer.train()
        (out/'training.json').write_text(json.dumps({'metrics':stats.metrics,'build':report},indent=2))
        model.save_pretrained(str(out/'adapter'))
        # Evaluate the saved serving artifact, not only the in-memory head.
        del trainer,model,items
        import gc,torch
        gc.collect();torch.cuda.empty_cache()
        model,tok=F.from_pretrained(str(out/'adapter'),max_seq_length=2048,load_in_4bit=True,use_gradient_checkpointing=True)
    F.for_inference(model)
    for split in ['calibration','fair','val','val2']:
        # Exp2's fair/val scores already exist for the identical five-question schema.
        # Only the new calibration subset requires GPU inference.
        if not a.train and a.adapter=='outputs/export-needs-e3/adapter' and split!='calibration':
            old=ROOT/'outputs/export-needs-e3/evaluation'
            saved=(read(old/'val2-predictions.jsonl')+read(old/'verification-predictions.jsonl')) if split=='fair' else read(old/(split+'-predictions.jsonl'))
            write(out/(split+'-predictions.jsonl'),saved)
            continue
        with (out/(split+'-predictions.jsonl')).open('x',encoding='utf-8') as f:
            for r in read(DATA/(split+'.jsonl')):
                ans=F.predict(model,tok,r['state'],questions)
                p={k:r[k] for k in ['posting_id','gold','label_source']}
                p.update(pred={k:v['answer'] for k,v in ans.items()},probabilities={k:v['probabilities'] for k,v in ans.items()})
                if a.fourth_verdict:
                    prob=p['probabilities']['verdict']['needs_information']
                    p['probabilities']['needs_verification']={'true':prob,'false':1-prob}
                f.write(json.dumps(p)+'\n');f.flush()
        print('scored',split,flush=True)
if __name__=='__main__':main()
