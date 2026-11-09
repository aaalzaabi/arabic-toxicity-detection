import argparse,pandas as pd,matplotlib.pyplot as plt
from aratox_nlp.config import ROOT,ensure_output_dirs
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.metrics import multilabel_metrics,binary_metrics
from aratox_nlp.evaluation import save_json

def eval_file(path,prefix):
    df=pd.read_csv(path); pcs=[f'{prefix}_pred_{l}' for l in TOXIC_LABELS]; valid=df[pcs].notna().all(axis=1); df=df[valid].copy(); yt=df[TOXIC_LABELS].astype(int).values; yp=df[pcs].astype(int).values; bt=df.binary_toxic.astype(int).values; bp=df[f'{prefix}_pred_binary'].astype(int).values if f'{prefix}_pred_binary' in df.columns else (yp.sum(axis=1)>0).astype(int); return df,{'multilabel':multilabel_metrics(yt,yp,TOXIC_LABELS),'binary':binary_metrics(bt,bp)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split',choices=['validation','test'],default='validation'); a=ap.parse_args(); ensure_output_dirs(); cand=[('SVM',ROOT/f'outputs/predictions/svm_{a.split}.csv','svm'),('QARiB',ROOT/f'outputs/predictions/qarib_{a.split}.csv','qarib'),('Jais-2 zero-shot',ROOT/f'outputs/predictions/jais_zeroshot_{a.split}.csv','jais'),('Jais-2 few-shot',ROOT/f'outputs/predictions/jais_fewshot_{a.split}.csv','jais')]; rows=[]; details={}
    for name,p,prefix in cand:
        if not p.exists(): continue
        df,m=eval_file(p,prefix); details[name]=m; rows.append({'model':name,'macro_f1':m['multilabel']['macro_f1'],'micro_f1':m['multilabel']['micro_f1'],'hamming_loss':m['multilabel']['hamming_loss'],'subset_accuracy':m['multilabel']['subset_accuracy'],'binary_macro_f1':m['binary']['macro_f1'],'binary_toxic_f1':m['binary']['toxic_f1'],'n':len(df)})
    table=pd.DataFrame(rows); table.to_csv(ROOT/f'outputs/metrics/model_comparison_{a.split}.csv',index=False); save_json(details,ROOT/f'outputs/metrics/model_comparison_{a.split}.json')
    if not table.empty:
        fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(table.model,table.macro_f1); ax.set_ylabel('Macro-F1'); ax.set_title(f'Multi-label Macro-F1 — {a.split}'); ax.tick_params(axis='x',rotation=20); fig.tight_layout(); fig.savefig(ROOT/f'outputs/figures/model_comparison_{a.split}.png',dpi=180); plt.close(fig)
    print(table.to_string(index=False))
if __name__=='__main__': main()
