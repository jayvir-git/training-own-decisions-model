"""Detached sequential GPU jobs with process-tree watchdog and durable status."""
import json,subprocess,time,traceback,sys
from fair_verification import ROOT,OUT,analyze
PY=ROOT/'.venv/Scripts/python.exe'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    only_c='--only-c' in sys.argv
    results=json.loads((OUT/'results.json').read_text()) if only_c else {}
    if only_c:
        assert 'variant-a' in results and not results['variant-a']['passes'], 'C requires failed A'
    jobs=[('variant-c','outputs/full-e3/adapter',True)] if only_c else [('full-e3','outputs/full-e3/adapter',False),('exp1','outputs/export-more-e3/adapter',False),('exp2','outputs/export-needs-e3/adapter',False),('variant-a','outputs/full-e3/adapter',True)]
    try:
        for name,adapter,train in jobs:
            d=OUT/name;d.mkdir(exist_ok=False)
            cmd=[str(PY),'-B','scripts/gpu_lock.py','--',str(PY),'-B','scripts/fair_verification_worker.py','--adapter',adapter,'--out',str(d)]+(['--train'] if train else [])+(['--fourth-verdict','--epochs','1'] if only_c else [])
            (d/'command.json').write_text(json.dumps(cmd))
            with (d/'stdout.log').open('w') as stdout,(d/'stderr.log').open('w') as stderr:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
                (OUT/'status.json').write_text(json.dumps({'stage':name,'pid':p.pid,'started':time.time()}))
                try:rc=p.wait(timeout=14040)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F']);raise RuntimeError('3.9h watchdog expired')
                if rc:raise RuntimeError(f'{name} exit {rc}')
            results[name]=analyze(name,d)
            (OUT/'results.json').write_text(json.dumps(results,indent=2))
        (OUT/'status.json').write_text(json.dumps({'stage':'complete','finished':time.time()}))
    except Exception:
        (OUT/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
