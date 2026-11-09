import argparse
from torch.utils.data import DataLoader
from aratox_nlp.config import ROOT,load_yaml,ensure_output_dirs
from aratox_nlp.data import load_processed
from aratox_nlp.qarib import load_trained,TextMultiLabelDataset,predict,pick_device
from aratox_nlp.metrics import threshold_predictions
from aratox_nlp.evaluation import prediction_frame,full_metrics,save_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split',choices=['validation','test'],default='validation'); ap.add_argument('--threshold',type=float,default=0.5); a=ap.parse_args()
    ensure_output_dirs(); c=load_yaml('configs/models.yaml')['qarib']; df=load_processed(a.split,ROOT/'data/processed'); tok,model=load_trained(ROOT/'outputs/models/qarib'); device=pick_device(); model.to(device)
    loader=DataLoader(TextMultiLabelDataset(df,tok,int(c['max_length'])),batch_size=int(c['batch_size']),shuffle=False); probs,_=predict(model,loader,device); pred=threshold_predictions(probs,a.threshold)
    out=prediction_frame(df,probs,pred,'qarib'); out.to_csv(ROOT/f'outputs/predictions/qarib_{a.split}.csv',index=False); save_json(full_metrics(out,'qarib'),ROOT/f'outputs/metrics/qarib_{a.split}.json'); print(f'saved {len(out)} predictions')
if __name__=='__main__': main()
