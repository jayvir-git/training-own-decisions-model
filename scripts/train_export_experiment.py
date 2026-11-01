"""Run existing trainer on an isolated dataset; no changes to legacy defaults."""
import argparse, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True)
    args,rest=ap.parse_known_args()
    import train_decision as td
    td.DATA=td.REPO/args.dataset
    original=td.read_jsonl
    def read(path):
        # A dedicated training-employer holdout calibrates; original val never calibrates these runs.
        if Path(path).name=='val.jsonl': return original(td.DATA/'calibration.jsonl')
        if Path(path).name=='train.jsonl': return original(td.DATA/'fit.jsonl')
        return original(path)
    td.read_jsonl=read
    sys.argv=[sys.argv[0],*rest]
    td.main()
if __name__=='__main__': main()
