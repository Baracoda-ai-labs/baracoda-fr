import json, unicodedata, numpy as np
from tokenizers import Tokenizer
ND='/root/NTREX/NTREX-128'
EU={'bul':'Bulgarian','hrv':'Croatian','ces':'Czech','dan':'Danish','nld':'Dutch','est':'Estonian','fin':'Finnish','fra':'French','deu':'German','ell':'Greek','hun':'Hungarian','gle':'Irish','ita':'Italian','lav':'Latvian','lit':'Lithuanian','mlt':'Maltese','pol':'Polish','por':'Portuguese','ron':'Romanian','slk':'Slovak','slv':'Slovenian','spa':'Spanish','swe':'Swedish'}
rd=lambda p:[unicodedata.normalize('NFC',l.strip()) for l in open(p,encoding='utf-8').read().splitlines()]
eng=rd(f'{ND}/newstest2019-src.eng.txt'); L={c:rd(f'{ND}/newstest2019-ref.{c}.txt') for c in EU}
doc=open('/root/NTREX/DOCUMENT_IDS.tsv').read().split('\n')[:1997]
ids={d:i for i,d in enumerate(dict.fromkeys(doc))}; g=np.array([ids[d] for d in doc]); D=len(ids)
M=np.zeros((D,1997)); M[g,np.arange(1997)]=1; dr=np.random.default_rng(0).integers(0,D,size=(2000,D))
T={'OpenAI o200k':'/root/tok/node_modules/@lenml/tokenizer-gptoss/models/tokenizer.json',
   'Meta Llama 3':'/root/tok/node_modules/@lenml/tokenizer-llama3_1/models/tokenizer.json',
   'Alibaba Qwen3':'/root/tok/node_modules/@lenml/tokenizer-qwen3/models/tokenizer.json',
   'DeepSeek V3':'/root/tok/node_modules/@lenml/tokenizer-deepseek_v3/models/tokenizer.json',
   'Google Gemma 3':'/root/tok/node_modules/@lenml/tokenizer-gemma3/models/tokenizer.json',
   'Mistral Tekken':'/root/tok/node_modules/@lenml/tokenizer-mistral_nemo/models/tokenizer.json',
   'Baracoda FR v1.2':'/home/claude/pack/baracoda-fr/tokenizers/baracoda-fr-v1_2.json'}
def gini(x):
    x=np.sort(np.asarray(x)); n=len(x); return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum()))
out={'languages':EU,'tokenizers':{}}
for name,p in T.items():
    t=Tokenizer.from_file(p); c=lambda X:np.array([len(e.ids) for e in t.encode_batch(X,add_special_tokens=False)])
    e=c(eng); E=M@e; res={'english_tokens':int(e.sum()),'premium':{}}
    for k,X in L.items():
        x=c(X); Xd=M@x; bs=Xd[dr].sum(1)/E[dr].sum(1)
        res['premium'][k]=[float(x.sum()/e.sum()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)),int(x.sum())]
    v=[res['premium'][k][0] for k in EU]
    res.update(mean=float(np.mean(v)),max=float(max(v)),argmax=max(EU,key=lambda k:res['premium'][k][0]),min=float(min(v)),gini=gini(v),total_eu_tokens=int(sum(res['premium'][k][3] for k in EU)))
    out['tokenizers'][name]=res; print(f"{name:18s} mean {res['mean']:.2f} max {res['max']:.2f} ({EU[res['argmax']]}) gini {res['gini']:.3f} totalEU {res['total_eu_tokens']:,}")
out['chars_ratio']={k:sum(map(len,X))/sum(map(len,eng)) for k,X in L.items()}
ro=''.join(L['ron']); out['romanian_diacritics']={'s_comma(ș)':ro.count('ș')+ro.count('Ș'),'s_cedilla(ş)':ro.count('ş')+ro.count('Ş'),'t_comma(ț)':ro.count('ț')+ro.count('Ț'),'t_cedilla(ţ)':ro.count('ţ')+ro.count('Ţ')}
print(out['romanian_diacritics'])
json.dump(out,open('eu24_ntrex.json','w'),indent=1,ensure_ascii=False)
