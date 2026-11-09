import json,pytest
from aratox_nlp.jais import extract_json
from aratox_nlp.constants import TOXIC_LABELS

def obj(): return {'binary_toxic':1,'labels':{l:int(l=='Hatred') for l in TOXIC_LABELS}}
def test_valid(): assert extract_json(json.dumps(obj()))['labels']['Hatred']==1
def test_invalid_missing():
    x=obj(); del x['labels']['Violence']
    with pytest.raises(ValueError): extract_json(json.dumps(x))
