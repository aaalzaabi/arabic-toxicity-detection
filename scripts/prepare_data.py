import json
from aratox_nlp.config import ROOT,load_yaml,ensure_output_dirs
from aratox_nlp.data import load_and_prepare,save_processed

def main():
    ensure_output_dirs(); cfg=load_yaml('configs/data.yaml'); train,val,test,report=load_and_prepare(cfg); save_processed(train,val,test,report,ROOT/'data/processed')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['overlap']['train_test'] or report['overlap']['validation_test']: print('WARNING: normalized overlap with held-out test. Review before training.')
if __name__=='__main__': main()
