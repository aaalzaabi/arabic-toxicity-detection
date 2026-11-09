import argparse,json,numpy as np,pandas as pd
from sklearn.metrics import f1_score
from aratox_nlp.config import ROOT,load_yaml
from aratox_nlp.constants import TOXIC_LABELS

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); a=ap.parse_args(); p=ROOT/a.predictions if not str(a.predictions).startswith('/') else a.predictions; df=pd.read_csv(p); g=load_yaml('configs/experiment.yaml')['threshold_grid']; grid=np.arange(float(g['start']),float(g['stop'])+1e-9,float(g['step'])); out={}
    for l in TOXIC_LABELS:
        scores=df[f'qarib_score_{l}'].values; y=df[l].astype(int).values; best=(.5,-1)
        for t in grid:
            f=f1_score(y,(scores>=t).astype(int),zero_division=0)
            if f>best[1]: best=(float(t),float(f))
        out[l]={'threshold':best[0],'validation_f1':best[1]}
    dst=ROOT/'outputs/metrics/thresholds.json'; dst.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
