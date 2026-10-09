"""EU-23 token premiums on FLORES+ devtest (second corpus) for 9 tokenizers, with a sanity check on NTREX.
Run from ~/baracoda-fr/eu24 with ../pack/.venv-audit. FLORES+ is gated on Hugging Face: accept its terms first."""
import json, unicodedata, urllib.request, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import tiktoken

EU = {'bul':'bul_Cyrl','hrv':'hrv_Latn','ces':'ces_Latn','dan':'dan_Latn','nld':'nld_Latn','est':'ekk_Latn','fin':'fin_Latn',
      'fra':'fra_Latn','deu':'deu_Latn','ell':'ell_Grek','hun':'hun_Latn','gle':'gle_Latn','ita':'ita_Latn','lav':'lvs_Latn',
      'lit':'lit_Latn','mlt':'mlt_Latn','pol':'pol_Latn','por':'por_Latn','ron':'ron_Latn','slk':'slk_Latn','slv':'slv_Latn',
      'spa':'spa_Latn','swe':'swe_Latn'}
nfc = lambda s: unicodedata.normalize('NFC', s.strip())

def flores(cfg):
    ds = load_dataset('openlanguagedata/flores_plus', cfg, split='devtest')
    ds = sorted(ds, key=lambda r: r['id'])
    return [nfc(r['text']) for r in ds], [r['url'] for r in ds]

eng, urls = flores('eng_Latn')
L = {}
for k, cfg in EU.items():
    try:
        L[k], u = flores(cfg); assert len(L[k]) == len(eng) and u == urls, cfg
    except Exception as ex:
        if k == 'lav':  # some FLORES+ versions use lav_Latn
            L[k], u = flores('lav_Latn'); assert len(L[k]) == len(eng)
        else:
            raise
docs = {d: i for i, d in enumerate(dict.fromkeys(urls))}; g = np.array([docs[d] for d in urls]); D = len(docs); N = len(eng)
M = np.zeros((D, N)); M[g, np.arange(N)] = 1; dr = np.random.default_rng(0).integers(0, D, size=(2000, D))

def hf(repo):
    t = AutoTokenizer.from_pretrained(repo)
    return lambda X: np.array([len(x) for x in t(X, add_special_tokens=False)['input_ids']])
def tk():
    e = tiktoken.get_encoding('o200k_base'); return lambda X: np.array([len(x) for x in e.encode_ordinary_batch(X)])
def local(p):
    t = Tokenizer.from_file(p); return lambda X: np.array([len(e.ids) for e in t.encode_batch(X, add_special_tokens=False)])
TOK = {'BSC Salamandra': lambda: hf('BSC-LT/salamandra-7b'), 'EuroLLM': lambda: hf('utter-project/EuroLLM-9B'),
       'OpenAI o200k': tk, 'Google Gemma 3': lambda: hf('unsloth/gemma-3-4b-it'),
       'Mistral Tekken': lambda: hf('mistralai/Mistral-Nemo-Instruct-2407'), 'DeepSeek V3': lambda: hf('deepseek-ai/DeepSeek-V3'),
       'Meta Llama 3': lambda: hf('unsloth/Meta-Llama-3.1-8B'), 'Alibaba Qwen3': lambda: hf('Qwen/Qwen3-8B'),
       'Baracoda FR v1.2': lambda: local('../pack/tokenizers/baracoda-fr-v1_2.json')}

# sanity check on NTREX French (expected premiums from the paper)
C = '468c6b69c7f6a75d31d4743d9daba2af566cc18d'; R = f'https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{C}/NTREX-128/'
get = lambda f: [nfc(l) for l in urllib.request.urlopen(R + f, timeout=120).read().decode('utf-8').splitlines()]
n_en, n_fr = get('newstest2019-src.eng.txt'), get('newstest2019-ref.fra.txt')
EXPECT = {'BSC Salamandra': 1.31, 'EuroLLM': 1.35, 'OpenAI o200k': 1.36, 'Google Gemma 3': 1.41, 'Mistral Tekken': 1.31,
          'DeepSeek V3': 1.55, 'Meta Llama 3': 1.58, 'Alibaba Qwen3': 1.56, 'Baracoda FR v1.2': 1.20}
out = {'corpus': 'FLORES+ devtest', 'sentences': N, 'documents': D, 'tokenizers': {}}
for name, make in TOK.items():
    try: f = make()
    except Exception as ex: out['tokenizers'][name] = {'error': repr(ex)[:300]}; print(name, 'FAILED', repr(ex)[:200]); continue
    chk = f(n_fr).sum() / f(n_en).sum()
    e = f(eng); E = M @ e; res = {'ntrex_fr_check': [float(chk), EXPECT[name]], 'english_tokens': int(e.sum()), 'premium': {}}
    for k, X in L.items():
        x = f(X); bs = (M @ x)[dr].sum(1) / E[dr].sum(1)
        res['premium'][k] = [float(x.sum() / e.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(x.sum())]
    v = np.sort([p[0] for p in res['premium'].values()]); n = len(v)
    res.update(mean=float(v.mean()), max=float(v[-1]), gini=float((2*np.arange(1, n+1)-n-1).dot(v)/(n*v.sum())),
               total_eu_tokens=int(sum(p[3] for p in res['premium'].values())))
    out['tokenizers'][name] = res
    flag = 'OK' if abs(chk - EXPECT[name]) < 0.006 else 'MISMATCH'
    print(f"{name:18s} NTREX-fr {chk:.3f} (expected {EXPECT[name]}) {flag} | FLORES mean {res['mean']:.3f} max {res['max']:.2f} gini {res['gini']:.3f} total {res['total_eu_tokens']:,}")
json.dump(out, open('eu24_flores.json', 'w'), indent=1)
