import pandas as pd
from aratox_nlp.data import add_derived_columns, canonicalize_columns
from aratox_nlp.constants import TOXIC_LABELS

def test_binary_derivation():
    rows=[]
    for toxic in [0,1]:
        r={'Text':'اختبار','NOT':1-toxic,**{l:0 for l in TOXIC_LABELS}}
        if toxic: r['Hatred']=1
        rows.append(r)
    out=add_derived_columns(pd.DataFrame(rows),'Text',{})
    assert out.binary_toxic.tolist()==[0,1]

def test_canonicalize_lowercase_and_bom_headers():
    df=pd.DataFrame({'\ufeffappearance':[0],'cussing':[0],'hatred':[1],'not':[0],'racial':[0],'sexual':[0],'violence':[0],'text':['نص']})
    out=canonicalize_columns(df)
    assert list(out.columns)==['Appearance','Cussing','Hatred','NOT','Racial','Sexual','Violence','Text']
