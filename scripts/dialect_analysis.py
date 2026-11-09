import argparse
from pathlib import Path
import pandas as pd
from aratox_nlp.config import ROOT,load_yaml
from aratox_nlp.evaluation import metrics_by_group

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); ap.add_argument('--prefix',choices=['svm','qarib','jais'],required=True); a=ap.parse_args(); c=load_yaml('configs/data.yaml'); col=c['dataset'].get('dialect_column')
    if not col: raise SystemExit('No official dialect_column configured. Do not invent dialect labels.')
    p=Path(a.predictions); p=p if p.is_absolute() else ROOT/p; df=pd.read_csv(p)
    if col not in df.columns: raise SystemExit(f'Configured dialect column {col!r} is absent.')
    out=metrics_by_group(df,a.prefix,col); dst=ROOT/f'outputs/metrics/{a.prefix}_by_dialect.csv'; out.to_csv(dst,index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
