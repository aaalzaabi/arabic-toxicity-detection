import json
from aratox_nlp.config import ROOT,load_yaml,ensure_output_dirs
from aratox_nlp.data import load_processed
from aratox_nlp.qarib import train_qarib

def main():
    ensure_output_dirs(); c=load_yaml('configs/models.yaml')['qarib']; train=load_processed('train',ROOT/'data/processed'); val=load_processed('validation',ROOT/'data/processed')
    r=train_qarib(train,val,c['model_name'],int(c['max_length']),int(c['batch_size']),int(c['epochs']),float(c['learning_rate']),float(c['weight_decay']),float(c['threshold']),int(c['seed']),ROOT/'outputs/models/qarib'); print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
