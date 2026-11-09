from pathlib import Path
import json, random, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from .constants import TOXIC_LABELS
from .metrics import multilabel_metrics, threshold_predictions

class TextMultiLabelDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts=df['text_norm'].astype(str).tolist(); self.labels=df[TOXIC_LABELS].astype(np.float32).values
        self.tokenizer=tokenizer; self.max_length=max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc=self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item={k:v.squeeze(0) for k,v in enc.items()}; item['labels']=torch.tensor(self.labels[idx],dtype=torch.float32); return item

def pick_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_model(model_name):
    tok=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(TOXIC_LABELS), problem_type='multi_label_classification', id2label={i:l for i,l in enumerate(TOXIC_LABELS)}, label2id={l:i for i,l in enumerate(TOXIC_LABELS)})
    return tok,model

@torch.no_grad()
def predict(model, loader, device):
    model.eval(); probs=[]; labels=[]
    for batch in loader:
        y=batch.pop('labels').numpy(); batch={k:v.to(device) for k,v in batch.items()}
        logits=model(**batch).logits; probs.append(torch.sigmoid(logits).cpu().numpy()); labels.append(y)
    return np.vstack(probs),np.vstack(labels)

def train_qarib(train_df,val_df,model_name,max_length,batch_size,epochs,learning_rate,weight_decay,threshold,seed,out_dir):
    seed_everything(seed); device=pick_device(); tok,model=make_model(model_name); model.to(device)
    train_loader=DataLoader(TextMultiLabelDataset(train_df,tok,max_length),batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(TextMultiLabelDataset(val_df,tok,max_length),batch_size=batch_size,shuffle=False)
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    total=max(1,len(train_loader)*epochs); sched=get_linear_schedule_with_warmup(opt,max(1,total//10),total)
    best=-1; hist=[]; out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train(); losses=[]; t0=time.perf_counter()
        n_steps=len(train_loader)
        for step, batch in enumerate(train_loader, 1):
            labels=batch.pop('labels').to(device); batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad(set_to_none=True)
            outputs=model(**batch,labels=labels); outputs.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step(); losses.append(float(outputs.loss.detach().cpu()))
            if step == 1 or step % 200 == 0 or step == n_steps:
                print(f'epoch {epoch} step {step}/{n_steps} loss {losses[-1]:.4f}', flush=True)
        probs,y=predict(model,val_loader,device); pred=threshold_predictions(probs,threshold); m=multilabel_metrics(y.astype(int),pred,TOXIC_LABELS)
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'seconds':time.perf_counter()-t0,'macro_f1':m['macro_f1'],'micro_f1':m['micro_f1']}; hist.append(row); print(json.dumps(row), flush=True)
        if m['macro_f1']>best:
            best=m['macro_f1']; model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    (out_dir/'training_history.json').write_text(json.dumps(hist,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'best_validation_macro_f1':best,'device':str(device),'history':hist}

def load_trained(path): return AutoTokenizer.from_pretrained(path),AutoModelForSequenceClassification.from_pretrained(path)
