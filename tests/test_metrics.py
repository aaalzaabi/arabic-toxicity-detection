import numpy as np
from aratox_nlp.metrics import multilabel_metrics,binary_metrics

def test_multilabel_metrics():
    y=np.array([[1,0],[0,1],[1,1]]); m=multilabel_metrics(y,y,['a','b']); assert m['macro_f1']==1.0 and m['subset_accuracy']==1.0

def test_binary_metrics():
    y=np.array([0,1,1,0]); m=binary_metrics(y,y); assert m['accuracy']==1.0 and m['toxic_f1']==1.0
