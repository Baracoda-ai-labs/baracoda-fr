"""Mistral Large 4 tokenizer vs earlier Mistral tokenizers and references, on NTREX-128 (same protocol as the
language-tax studies: NFC, no special tokens, 1,997 sentences, bootstrap over the 123 documents).
Also reports German, French and Chinese for the German edition of study 1. Written by Claude, 29 Oct 2026."""
import hashlib, json, os, time, unicodedata, urllib.request
import numpy as np
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
OUT = os.environ.get("OUT_DIR", "results/openllm"); os.makedirs(OUT, exist_ok=True); CACHE = os.environ.get("HF_CACHE")
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
C = '468c6b69c7f6a75d31d4743d9daba2af566cc18d'; R = f'https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{C}/'
EU = ['bul','hrv','ces','dan','nld','est','fin','fra','deu','ell','hun','gle','ita','lav','lit','mlt','pol','por','ron','slk','slv','spa','swe']
get = lambda p: urllib.request.urlopen(R + p, timeout=120).read().decode('utf-8')
rd = lambda p: [unicodedata.normalize('NFC', l.strip()) for l in get(p).splitlines()][:1997]
T = {'eng': rd('NTREX-128/newstest2019-src.eng.txt'), **{c: rd(f'NTREX-128/newstest2019-ref.{c}.txt') for c in EU + ['zho-CN']}}
doc = get('DOCUMENT_IDS.tsv').split('\n')[:1997]
ids = {d: i for i, d in enumerate(dict.fromkeys(doc))}; g = np.array([ids[d] for d in doc]); D = len(ids)
M = np.zeros((D, 1997)); M[g, np.arange(1997)] = 1; dr = np.random.default_rng(0).integers(0, D, size=(2000, D))
REPOS = {'Mistral Large 4': 'mistralai/Mistral-Large-4-1T-A52B', 'Mistral Large 3': 'mistralai/Mistral-Large-3-675B-Instruct-2512',
         'Mistral Medium 3.5': 'mistralai/Mistral-Medium-3.5-128B', 'Mistral Small 4': 'mistralai/Mistral-Small-4-119B-2603',
         'Mistral Tekken (Nemo)': 'mistralai/Mistral-Nemo-Base-2407', 'Alibaba Qwen3': 'Qwen/Qwen3-0.6B-Base', 'DeepSeek V3': 'deepseek-ai/DeepSeek-V3',
         'BSC Salamandra': 'BSC-LT/salamandra-7b'}
tok, info = {}, {}
for n, r in REPOS.items():
    try:
        p = hf_hub_download(r, 'tokenizer.json', cache_dir=CACHE); t = Tokenizer.from_file(p)
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
        tok[n] = lambda X, t=t: np.array([len(e.ids) for e in t.encode_batch(X, add_special_tokens=False)])
        info[n] = {'repo': r, 'vocab': t.get_vocab_size(), 'sha256_16': h, 'vocab_hash': hashlib.md5(json.dumps(sorted(t.get_vocab().items())).encode()).hexdigest()[:12]}
        log('OK', n, info[n])
    except Exception as e: log('IGNORÉ', n, repr(e)[:200])
try:
    import tiktoken; o = tiktoken.get_encoding('o200k_base')
    tok['OpenAI o200k'] = lambda X: np.array([len(x) for x in o.encode_ordinary_batch(X)]); info['OpenAI o200k'] = {'vocab': o.n_vocab}
except Exception as e: log('IGNORÉ o200k', e)
res = {'date': time.strftime('%Y-%m-%d %H:%M'), 'tokenizers': info, 'ratios': {}}
for n, f in tok.items():
    c = {L: M @ f(X) for L, X in T.items()}; E = c['eng']
    r = {'english_tokens': int(E.sum()), 'ratio': {}}
    for L in T:
        if L == 'eng': continue
        bs = c[L][dr].sum(1) / E[dr].sum(1)
        r['ratio'][L] = [float(c[L].sum() / E.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(c[L].sum())]
    v = [r['ratio'][L][0] for L in EU]
    r.update(mean_ratio_23=float(np.mean(v)), worst=max(EU, key=lambda L: r['ratio'][L][0]), total_23=int(sum(r['ratio'][L][3] for L in EU)))
    res['ratios'][n] = r
    log(f"{n:24s} vocab {info[n].get('vocab')} EN {r['english_tokens']:,} | de {r['ratio']['deu'][0]:.3f} fr {r['ratio']['fra'][0]:.3f} zh {r['ratio']['zho-CN'][0]:.3f} | mean23 {r['mean_ratio_23']:.3f} worst {r['worst']} {r['ratio'][r['worst']][0]:.2f} | total23 {r['total_23']:,}")
json.dump(res, open(os.path.join(OUT, 'mistral_large4.json'), 'w'), indent=1)
md = ['# Mistral Large 4 tokenizer vs earlier tokenizers - NTREX-128', '', f"Run {res['date']}.", '',
      '| Tokenizer | Vocab | English tokens | German / EN | French / EN | Chinese / EN | Mean, 23 EU languages | Worst | Total, 23 languages |', '|---|---:|---:|---:|---:|---:|---:|---|---:|']
for n, r in res['ratios'].items():
    q = r['ratio']; md.append(f"| {n} | {info[n].get('vocab', '')} | {r['english_tokens']:,} | {q['deu'][0]:.3f} | {q['fra'][0]:.3f} | {q['zho-CN'][0]:.3f} | {r['mean_ratio_23']:.3f} | {r['worst']} {q[r['worst']][0]:.2f} | {r['total_23']:,} |")
open(os.path.join(OUT, 'MISTRAL_LARGE4.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md)); log('TERMINÉ')
