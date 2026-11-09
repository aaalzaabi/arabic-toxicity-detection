import json,re
from .constants import TOXIC_LABELS

SYSTEM_PROMPT='''You are an Arabic toxicity classification model. Return JSON only. Do not explain. Allowed labels: Appearance, Cussing, Hatred, Racial, Sexual, Violence. Each label must be 0 or 1. Also return binary_toxic as 0 or 1.'''

def build_prompt(text,examples=None):
    ex=''
    if examples:
        ex='\n\n'.join('Example text: '+e['text']+'\nExample output: '+json.dumps(e['output'],ensure_ascii=False) for e in examples)+'\n\n'
    schema={'binary_toxic':0,'labels':{l:0 for l in TOXIC_LABELS}}
    user=ex+'Classify the following Arabic text.\nText: '+text+'\nOutput exactly one JSON object matching this schema:\n'+json.dumps(schema,ensure_ascii=False)
    return [{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':user}]

def validate_output(obj):
    if not isinstance(obj,dict) or 'binary_toxic' not in obj or 'labels' not in obj: raise ValueError('Missing binary_toxic or labels')
    b=int(obj['binary_toxic']);
    if b not in (0,1): raise ValueError('binary_toxic must be 0/1')
    labels=obj['labels']; clean={}
    for l in TOXIC_LABELS:
        if l not in labels: raise ValueError(f'Missing label {l}')
        v=int(labels[l]);
        if v not in (0,1): raise ValueError(f'{l} must be 0/1')
        clean[l]=v
    return {'binary_toxic':b,'labels':clean}

def extract_json(text):
    candidates=[text.strip()]; m=re.search(r'\{.*\}',text,re.S)
    if m: candidates.append(m.group(0))
    last=None
    for c in candidates:
        try: return validate_output(json.loads(c))
        except Exception as e: last=e
    raise ValueError(f'Invalid structured output: {last}')

class TransformersJais:
    def __init__(self,model_name,max_new_tokens=180):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.torch=torch; self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        kwargs={'device_map':'auto'}
        # Half precision on any GPU: fp32 weights for an 8B model (~32 GB) do not fit in Apple unified memory.
        if torch.cuda.is_available(): kwargs['dtype']=torch.float16
        elif getattr(torch.backends,'mps',None) and torch.backends.mps.is_available(): kwargs['dtype']=torch.bfloat16
        self.model=AutoModelForCausalLM.from_pretrained(model_name,**kwargs); self.max_new_tokens=max_new_tokens
    def classify(self,text,examples=None):
        msgs=build_prompt(text,examples); inputs=self.tokenizer.apply_chat_template(msgs,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to(self.model.device); inputs.pop('token_type_ids',None)
        with self.torch.no_grad(): out=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
        generated=out[0][inputs['input_ids'].shape[-1]:]; return extract_json(self.tokenizer.decode(generated,skip_special_tokens=True))

class OpenAICompatibleJais:
    def __init__(self,base_url,model_name,api_key='local'):
        from openai import OpenAI
        self.client=OpenAI(base_url=base_url,api_key=api_key); self.model_name=model_name
    def classify(self,text,examples=None):
        r=self.client.chat.completions.create(model=self.model_name,messages=build_prompt(text,examples),temperature=0)
        return extract_json(r.choices[0].message.content)
