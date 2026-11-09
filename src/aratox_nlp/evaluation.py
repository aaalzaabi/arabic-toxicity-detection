import json
from pathlib import Path
import pandas as pd
from .constants import TOXIC_LABELS
from .metrics import multilabel_metrics,binary_metrics

def prediction_frame(source_df,scores,preds,prefix):
    out=source_df[['text_raw','text_norm','binary_toxic',*TOXIC_LABELS]].copy()
    for i,l in enumerate(TOXIC_LABELS): out[f'{prefix}_score_{l}']=scores[:,i]; out[f'{prefix}_pred_{l}']=preds[:,i]
    out[f'{prefix}_pred_binary']=(preds.sum(axis=1)>0).astype(int)
    return out

def full_metrics(df,prefix):
    yt=df[TOXIC_LABELS].astype(int).values; yp=df[[f'{prefix}_pred_{l}' for l in TOXIC_LABELS]].astype(int).values
    bt=df.binary_toxic.astype(int).values; bp=df[f'{prefix}_pred_binary'].astype(int).values
    return {'multilabel':multilabel_metrics(yt,yp,TOXIC_LABELS),'binary':binary_metrics(bt,bp)}

def metrics_by_group(df,prefix,group_col):
    if group_col not in df.columns: return pd.DataFrame()
    rows=[]
    for value,g in df.groupby(group_col,dropna=False):
        yt=g[TOXIC_LABELS].astype(int).values; yp=g[[f'{prefix}_pred_{l}' for l in TOXIC_LABELS]].astype(int).values; m=multilabel_metrics(yt,yp,TOXIC_LABELS)
        rows.append({group_col:str(value),'n':len(g),'macro_f1':m['macro_f1'],'micro_f1':m['micro_f1'],'hamming_loss':m['hamming_loss'],'subset_accuracy':m['subset_accuracy']})
    return pd.DataFrame(rows).sort_values('n',ascending=False)

def save_json(obj,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
