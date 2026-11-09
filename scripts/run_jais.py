import argparse,time,pandas as pd
from aratox_nlp.config import ROOT,load_yaml,ensure_output_dirs
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.data import load_processed
from aratox_nlp.jais import TransformersJais,OpenAICompatibleJais

def pick_examples(train,n):
    out=[]
    for l in TOXIC_LABELS:
        s=train[train[l]==1]
        if len(s):
            r=s.iloc[0]; out.append({'text':str(r.text_raw),'output':{'binary_toxic':int(r.binary_toxic),'labels':{x:int(r[x]) for x in TOXIC_LABELS}}})
        if len(out)>=n: break
    return out[:n]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--backend',choices=['transformers','openai'],default='transformers'); ap.add_argument('--base-url',default='http://localhost:8000/v1'); ap.add_argument('--model',default=None); ap.add_argument('--split',choices=['validation','test'],default='validation'); ap.add_argument('--max-items',type=int); ap.add_argument('--few-shot',action='store_true'); a=ap.parse_args()
    ensure_output_dirs(); c=load_yaml('configs/models.yaml')['jais']; model=a.model or c['model_name']; clf=TransformersJais(model,int(c['max_new_tokens'])) if a.backend=='transformers' else OpenAICompatibleJais(a.base_url,model)
    df=load_processed(a.split,ROOT/'data/processed');
    if a.max_items: df=df.head(a.max_items).copy()
    examples=pick_examples(load_processed('train',ROOT/'data/processed'),int(c['few_shot_examples'])) if a.few_shot else None
    rows=[]
    for _,r in df.iterrows():
        t=time.perf_counter(); parsed=None; err=''
        try: parsed=clf.classify(str(r.text_norm),examples)
        except Exception as e: err=repr(e)
        rec={'text_raw':r.text_raw,'text_norm':r.text_norm,'binary_toxic':int(r.binary_toxic),'latency_seconds':time.perf_counter()-t,'parse_error':err}
        for l in TOXIC_LABELS:
            rec[l]=int(r[l]); rec[f'jais_pred_{l}']=None if parsed is None else int(parsed['labels'][l])
        rec['jais_pred_binary']=None if parsed is None else int(parsed['binary_toxic']); rows.append(rec)
        if len(rows)%25==0: print(f'{len(rows)}/{len(df)} done; parse failures so far={sum(1 for x in rows if x["parse_error"])}; last latency={rec["latency_seconds"]:.1f}s',flush=True)
    out=pd.DataFrame(rows); mode='fewshot' if a.few_shot else 'zeroshot'; dst=ROOT/f'outputs/predictions/jais_{mode}_{a.split}.csv'; out.to_csv(dst,index=False); print(f'Saved {len(out)} rows to {dst}; parse failures={(out.parse_error.astype(str)!="").sum()}')
if __name__=='__main__': main()
