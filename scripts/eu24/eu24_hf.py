"""EU-23 token premiums on NTREX-128 for tokenizers available on Hugging Face (EuroLLM, Salamandra, ...).
Run from ~/baracoda-fr/eu24 with the pack's environment. Downloads NTREX at the pinned commit."""
import json, unicodedata, urllib.request, numpy as np
from transformers import AutoTokenizer
C='468c6b69c7f6a75d31d4743d9daba2af566cc18d'; R=f'https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{C}/'
EU=['bul','hrv','ces','dan','nld','est','fin','fra','deu','ell','hun','gle','ita','lav','lit','mlt','pol','por','ron','slk','slv','spa','swe']
get=lambda p:urllib.request.urlopen(R+p,timeout=120).read().decode('utf-8')
rd=lambda p:[unicodedata.normalize('NFC',l.strip()) for l in get(p).splitlines()]
eng=rd('NTREX-128/newstest2019-src.eng.txt'); L={c:rd(f'NTREX-128/newstest2019-ref.{c}.txt') for c in EU}
doc=get('DOCUMENT_IDS.tsv').split('\n')[:1997]; assert len(eng)==1997 and all(len(v)==1997 for v in L.values())
ids={d:i for i,d in enumerate(dict.fromkeys(doc))}; g=np.array([ids[d] for d in doc]); D=len(ids)
M=np.zeros((D,1997)); M[g,np.arange(1997)]=1; dr=np.random.default_rng(0).integers(0,D,size=(2000,D))
TOK={'EuroLLM-9B':'utter-project/EuroLLM-9B','Salamandra-7B':'BSC-LT/salamandra-7b','Mistral (HF, check)':'mistralai/Mistral-Nemo-Instruct-2407'}
out={}
for name,repo in TOK.items():
    try: t=AutoTokenizer.from_pretrained(repo)
    except Exception as ex: out[name]={'error':repr(ex)[:300]}; print(name,'FAILED',repr(ex)[:200]); continue
    c=lambda X:np.array([len(x) for x in t(X,add_special_tokens=False)['input_ids']])
    e=c(eng); E=M@e; res={'repo':repo,'vocab':len(t),'english_tokens':int(e.sum()),'premium':{}}
    for k,X in L.items():
        x=c(X); bs=(M@x)[dr].sum(1)/E[dr].sum(1)
        res['premium'][k]=[float(x.sum()/e.sum()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)),int(x.sum())]
    v=[res['premium'][k][0] for k in EU]; s=np.sort(v); n=len(s)
    res.update(mean=float(np.mean(v)),max=float(max(v)),argmax=max(EU,key=lambda k:res['premium'][k][0]),gini=float((2*np.arange(1,n+1)-n-1).dot(s)/(n*s.sum())))
    out[name]=res; print(f"{name:20s} vocab {len(t):,} mean {res['mean']:.3f} max {res['max']:.2f} ({res['argmax']}) gini {res['gini']:.3f}")
json.dump(out,open('eu24_ntrex_hf.json','w'),indent=1)
