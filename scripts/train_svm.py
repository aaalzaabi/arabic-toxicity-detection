import json,time,joblib,numpy as np
from aratox_nlp.config import ROOT,load_yaml,ensure_output_dirs
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.data import load_processed
from aratox_nlp.classical import ClassicalConfig,build_features,build_multilabel_svm,build_binary_svm
from aratox_nlp.metrics import multilabel_metrics,binary_metrics
from aratox_nlp.evaluation import prediction_frame,save_json

def main():
    ensure_output_dirs(); c=ClassicalConfig(**load_yaml('configs/models.yaml')['svm']); train=load_processed('train',ROOT/'data/processed'); val=load_processed('validation',ROOT/'data/processed')
    feats=build_features(c); X=feats.fit_transform(train.text_norm.astype(str)); Xv=feats.transform(val.text_norm.astype(str)); y=train[TOXIC_LABELS].astype(int).values; yv=val[TOXIC_LABELS].astype(int).values
    ml=build_multilabel_svm(c); t=time.perf_counter(); ml.fit(X,y); secs=time.perf_counter()-t; scores=np.asarray(ml.decision_function(Xv)); pred=(scores>=0).astype(int)
    binary=build_binary_svm(c); binary.fit(X,train.binary_toxic.astype(int).values); bpred=binary.predict(Xv)
    joblib.dump({'features':feats,'model':ml},ROOT/'outputs/models/tfidf_svm_multilabel.joblib'); joblib.dump({'features':feats,'model':binary},ROOT/'outputs/models/tfidf_svm_binary.joblib')
    pf=prediction_frame(val,scores,pred,'svm'); pf.to_csv(ROOT/'outputs/predictions/svm_validation.csv',index=False)
    result={'model':'TF-IDF + One-vs-Rest Linear SVM','train_seconds':secs,'multilabel':multilabel_metrics(yv,pred,TOXIC_LABELS),'binary':binary_metrics(val.binary_toxic.astype(int).values,bpred)}; save_json(result,ROOT/'outputs/metrics/svm_validation.json'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
