import json, glob, unicodedata, numpy as np
from tokenizers import Tokenizer
ND='/root/NTREX'
doc=open(f'{ND}/DOCUMENT_IDS.tsv').read().split('\n')[:1997]
ids={d:i for i,d in enumerate(dict.fromkeys(doc))}; g=np.array([ids[d] for d in doc]); D=len(ids)
rng=np.random.default_rng(0); B=2000
draws=rng.integers(0,D,size=(B,D))
M=np.zeros((D,len(g))); M[g,np.arange(len(g))]=1   # doc x sentence
def dsum(c): return M@np.asarray(c,float)
def ratio_ci(a,b):  # premium sum(a)/sum(b), doc bootstrap
    A,Bb=dsum(a),dsum(b); r=A[draws].sum(1)/Bb[draws].sum(1)
    return float(np.sum(a)/np.sum(b)), float(np.percentile(r,2.5)), float(np.percentile(r,97.5))
def rd(p): return [unicodedata.normalize('NFC',l.strip()) for l in open(p,encoding='utf-8').read().splitlines()]
eng=rd(f'{ND}/NTREX-128/newstest2019-src.eng.txt'); fra=rd(f'{ND}/NTREX-128/newstest2019-ref.fra.txt'); zho=rd(f'{ND}/NTREX-128/newstest2019-ref.zho-CN.txt')
T={'o200k':'gptoss','Llama 3':'llama3_1','Qwen3':'qwen3','DeepSeek':'deepseek_v3','Gemma 3':'gemma3','Tekken':'mistral_nemo'}
out={'n_documents':D,'draws':B,'premiums':{}}
# also sentence-level for comparison
sidx=np.random.default_rng(0).integers(0,1997,size=(1000,1997))
def sent_ci(a,b):
    a,b=np.asarray(a),np.asarray(b); r=a[sidx].sum(1)/b[sidx].sum(1); return [float(np.percentile(r,2.5)),float(np.percentile(r,97.5))]
for n,p in T.items():
    t=Tokenizer.from_file(f'/root/tok/node_modules/@lenml/tokenizer-{p}/models/tokenizer.json')
    c=lambda L:[len(e.ids) for e in t.encode_batch(L,add_special_tokens=False)]
    e,f,z=c(eng),c(fra),c(zho)
    out['premiums'][n]={'fr':ratio_ci(f,e),'zh':ratio_ci(z,e),'fr_sentence_ci':sent_ci(f,e),'zh_sentence_ci':sent_ci(z,e)}
cl=json.load(open('/home/claude/pack/baracoda-fr/results/archived/claude_ntrex_pud_per_sentence.json'))
out['premiums']['Claude 5']={'fr':ratio_ci(cl['ntrex_fr'],cl['ntrex_en']),'zh':ratio_ci(cl['ntrex_zh'],cl['ntrex_en']),'fr_sentence_ci':sent_ci(cl['ntrex_fr'],cl['ntrex_en'])}
C=json.load(open('/mnt/user-data/uploads/baracoda-fr/pack/results/counts.json'))['counts']
for n in ['baracoda-fr-v1_2','croissant','o200k','tekken']:
    out['premiums'].setdefault('pack:'+n,{})['fr']=ratio_ci(C[n]['NTREX-128 fr'],C[n]['NTREX-128 en'])
# paired v1.2 vs Tekken, doc bootstrap: ratio sum(v12)/sum(tekken)-1
for lang in ['fr','en']:
    a=C['baracoda-fr-v1_2'][f'NTREX-128 {lang}']; b=C['tekken'][f'NTREX-128 {lang}']
    r,lo,hi=ratio_ci(a,b); out[f'v1_2_vs_tekken_{lang}']=[r-1,lo-1,hi-1]
    a=C['baracoda-fr-v1_2'][f'NTREX-128 {lang}']; b=C['croissant'][f'NTREX-128 {lang}']
    r,lo,hi=ratio_ci(a,b); out[f'v1_2_vs_croissant_{lang}']=[r-1,lo-1,hi-1]
hw=[(n,k,(v[k][2]-v[k][1])/2) for n,v in out['premiums'].items() for k in ('fr','zh') if k in v]
out['max_halfwidth']=max(hw,key=lambda x:x[2])
json.dump(out,open('ntrex_document_bootstrap.json','w'),indent=1)
for n,v in out['premiums'].items(): print(n,{k:[round(x,3) for x in (w if isinstance(w,list) else w)] for k,w in v.items()})
for k in out:
    if k.startswith('v1_2') or k=='max_halfwidth': print(k,out[k])
