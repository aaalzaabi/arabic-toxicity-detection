from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .constants import TOXIC_LABELS, NOT_LABEL
from .preprocessing import normalize_arabic, config_from_dict

def _hash(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()

_COLUMN_ALIASES = {
    'appearance': 'Appearance',
    'cussing': 'Cussing',
    'hatred': 'Hatred',
    'not': 'NOT',
    'racial': 'Racial',
    'sexual': 'Sexual',
    'violence': 'Violence',
    'text': 'Text',
    'wordcount': 'WordCount',
    'wordcountgroup': 'WordCountGroup',
}

def canonicalize_columns(df):
    renamed = {}
    for col in df.columns:
        key = str(col).replace('\ufeff', '').strip()
        renamed[col] = _COLUMN_ALIASES.get(key.lower(), key)
    out = df.rename(columns=renamed)
    return out.loc[:, ~out.columns.duplicated()]

def _read_hub_csv(hf_id, filename):
    # Train and test CSVs on the Hub do not share a schema, so load_dataset()
    # cannot cast them together. Read each file, then normalize column names.
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=hf_id, filename=filename, repo_type='dataset')
    return canonicalize_columns(pd.read_csv(path, encoding='utf-8-sig'))

def validate_columns(df, text_col):
    required = [text_col, *TOXIC_LABELS, NOT_LABEL]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f'Missing required AraTox columns: {missing}')
    for c in [*TOXIC_LABELS, NOT_LABEL]:
        vals = set(pd.Series(df[c]).dropna().astype(int).unique().tolist())
        if not vals.issubset({0,1}): raise ValueError(f'{c} must be binary; found {sorted(vals)}')

def add_derived_columns(df, text_col, preprocess_cfg):
    out = df.copy()
    cfg = config_from_dict(preprocess_cfg)
    out['text_raw'] = out[text_col].astype(str)
    out['text_norm'] = out['text_raw'].map(lambda x: normalize_arabic(x, cfg))
    out['text_hash'] = out['text_norm'].map(_hash)
    out['binary_toxic'] = (out[TOXIC_LABELS].sum(axis=1) > 0).astype(int)
    out['not_consistent'] = (((out['binary_toxic']==0)&(out[NOT_LABEL].astype(int)!=1)) | ((out['binary_toxic']==1)&(out[NOT_LABEL].astype(int)!=0)))
    return out

def create_validation_split(train_df, validation_size, seed):
    sig = train_df[TOXIC_LABELS].astype(int).astype(str).agg(''.join, axis=1)
    counts = sig.value_counts()
    strat = sig if len(counts)>1 and (counts>=2).all() else train_df['binary_toxic']
    a,b = train_test_split(np.arange(len(train_df)), test_size=validation_size, random_state=seed, stratify=strat)
    return train_df.iloc[a].reset_index(drop=True), train_df.iloc[b].reset_index(drop=True)

def build_report(train, val, test, dialect_col=None):
    report = {'splits':{}, 'overlap':{}, 'dialect_column': dialect_col}
    for name, df in [('train',train),('validation',val),('test',test)]:
        row = {'rows':len(df), 'binary_toxic':int(df.binary_toxic.sum()), 'binary_non_toxic':int((1-df.binary_toxic).sum()), 'normalized_duplicate_rows':int(df.text_hash.duplicated().sum()), 'label_counts':{c:int(df[c].sum()) for c in TOXIC_LABELS}, 'not_consistency_flags':int(df.not_consistent.sum())}
        if dialect_col and dialect_col in df.columns: row['dialect_counts'] = df[dialect_col].fillna('Unknown').astype(str).value_counts().to_dict()
        report['splits'][name]=row
    for name,a,b in [('train_validation',train,val),('train_test',train,test),('validation_test',val,test)]:
        report['overlap'][name] = len(set(a.text_hash)&set(b.text_hash))
    return report

def load_and_prepare(cfg):
    ds_cfg, split_cfg = cfg['dataset'], cfg['split']
    pub_train = _read_hub_csv(ds_cfg['hf_id'], 'aratox_train.csv')
    test = _read_hub_csv(ds_cfg['hf_id'], 'aratox_test.csv')
    validate_columns(pub_train, ds_cfg['text_column']); validate_columns(test, ds_cfg['text_column'])
    pub_train = add_derived_columns(pub_train, ds_cfg['text_column'], cfg['preprocessing'])
    test = add_derived_columns(test, ds_cfg['text_column'], cfg['preprocessing'])
    train,val = create_validation_split(pub_train, float(split_cfg['validation_size']), int(split_cfg['seed']))
    return train,val,test,build_report(train,val,test,ds_cfg.get('dialect_column'))

def save_processed(train,val,test,report,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    train.to_csv(out/'train.csv',index=False); val.to_csv(out/'validation.csv',index=False); test.to_csv(out/'test.csv',index=False)
    (out/'data_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

def load_processed(split, root):
    p=Path(root)/f'{split}.csv'
    if not p.exists(): raise FileNotFoundError(f'{p} missing. Run scripts/prepare_data.py first.')
    return pd.read_csv(p)
