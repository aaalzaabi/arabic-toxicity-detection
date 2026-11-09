import argparse
from pathlib import Path
import pandas as pd
from aratox_nlp.config import ROOT
from aratox_nlp.constants import TOXIC_LABELS

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); a=ap.parse_args(); p=Path(a.predictions); p=p if p.is_absolute() else ROOT/p; df=pd.read_csv(p); prefix=next((x for x in ['qarib','svm','jais'] if all(f'{x}_pred_{l}' in df.columns for l in TOXIC_LABELS)),None)
    if prefix is None: raise ValueError('Could not detect prediction prefix')
    pcs=[f'{prefix}_pred_{l}' for l in TOXIC_LABELS]; valid=df[pcs].notna().all(axis=1); d=df[valid].copy(); mismatch=(d[TOXIC_LABELS].astype(int).values!=d[pcs].astype(int).values).any(axis=1); err=d[mismatch].copy(); err['error_type']=''; err['notes']=''; out=ROOT/f'outputs/errors/{prefix}_validation_errors.csv'; err.to_csv(out,index=False); print(f'Saved {len(err)} errors to {out}')
if __name__=='__main__': main()
