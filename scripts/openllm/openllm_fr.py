"""OpenLLM-France tokenizers (Lucie-7B, Luciole 1B/8B/23B) and CroissantLLM against our references, on NTREX-128.
Written by Claude, 15 Oct 2026. Same protocol as the EU study: NFC, no special tokens, 1,997 sentences,
2,000 bootstrap draws over the 123 documents (paired across tokenizers for totals).
Outputs (aggregates only): tokenizer_eu/openllm/openllm_ntrex.json and OPENLLM.md. Run from ~/baracoda-fr/tokenizer_eu."""
import glob, hashlib, json, os, time, unicodedata, urllib.request
import numpy as np
from transformers import AutoTokenizer
from tokenizers import Tokenizer
ICI = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("OUT_DIR", os.path.join(ICI, "openllm")); os.makedirs(OUT, exist_ok=True)
CACHE = os.environ.get("HF_CACHE", os.path.join(os.path.dirname(ICI), "audio", "cache")); T0 = time.time()
def log(*a): print(time.strftime("%H:%M:%S"), f"[{(time.time()-T0)/60:.0f} min]", *a, flush=True)
C = '468c6b69c7f6a75d31d4743d9daba2af566cc18d'; R = f'https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{C}/'
EU = ['bul','hrv','ces','dan','nld','est','fin','fra','deu','ell','hun','gle','ita','lav','lit','mlt','pol','por','ron','slk','slv','spa','swe']
get = lambda p: urllib.request.urlopen(R + p, timeout=120).read().decode('utf-8')
rd = lambda p: [unicodedata.normalize('NFC', l.strip()) for l in get(p).splitlines()]
eng = rd('NTREX-128/newstest2019-src.eng.txt'); L = {c: rd(f'NTREX-128/newstest2019-ref.{c}.txt') for c in EU}
try: L['zho'] = rd('NTREX-128/newstest2019-ref.zho-CN.txt')
except Exception as e: log('chinois indisponible', e)
doc = get('DOCUMENT_IDS.tsv').split('\n')[:1997]; assert len(eng) == 1997 and all(len(v) == 1997 for v in L.values())
ids = {d: i for i, d in enumerate(dict.fromkeys(doc))}; g = np.array([ids[d] for d in doc]); D = len(ids)
M = np.zeros((D, 1997)); M[g, np.arange(1997)] = 1; dr = np.random.default_rng(0).integers(0, D, size=(2000, D))
nchar = lambda s: sum(ch.isalpha() for ch in s)
chars = {'eng': sum(map(nchar, eng)), **{k: sum(map(nchar, v)) for k, v in L.items()}}

tok = {}
def hf(name, repo):
    try:
        t = AutoTokenizer.from_pretrained(repo, cache_dir=CACHE)
        tok[name] = (lambda X, t=t: [len(x) for x in t(X, add_special_tokens=False)['input_ids']], len(t),
                     hashlib.md5(json.dumps(sorted(t.get_vocab().items())).encode()).hexdigest()[:10], repo)
    except Exception as e:
        try:  # fallback: raw tokenizer.json (e.g. architectures unknown to this transformers version)
            from huggingface_hub import hf_hub_download
            local(name, hf_hub_download(repo, 'tokenizer.json', cache_dir=CACHE), src=repo)
        except Exception as e2: log('IGNORÉ', name, repr(e)[:120], '|', repr(e2)[:120])
def local(name, path, src=None):
    if not path or not os.path.exists(path): log('IGNORÉ', name, 'fichier absent', path); return
    t = Tokenizer.from_file(path)
    tok[name] = (lambda X, t=t: [len(e.ids) for e in t.encode_batch(X, add_special_tokens=False)], t.get_vocab_size(),
                 hashlib.md5(json.dumps(sorted(t.get_vocab().items())).encode()).hexdigest()[:10], src or os.path.basename(path))
# OpenLLM-France and French-centric
for n, r in [('OpenLLM Lucie-7B', 'OpenLLM-France/Lucie-7B'), ('OpenLLM Luciole-1B', 'OpenLLM-France/Luciole-1B-Instruct-1.1'),
             ('OpenLLM Luciole-8B', 'OpenLLM-France/Luciole-8B-Instruct-1.1'), ('OpenLLM Luciole-23B', 'OpenLLM-France/Luciole-23B-Instruct-1.1'),
             ('CroissantLLM', 'croissantllm/CroissantLLMBase'),
             # references from our studies
             ('Mistral Tekken', 'mistralai/Mistral-Nemo-Base-2407'), ('BSC Salamandra', 'BSC-LT/salamandra-7b'),
             ('EuroLLM', 'utter-project/EuroLLM-9B'), ('Google Gemma 3', 'google/gemma-3-1b-pt'), ('Meta Llama 3.1', 'meta-llama/Llama-3.1-8B')]:
    hf(n, r)
try:
    import tiktoken; o = tiktoken.get_encoding('o200k_base')
    tok['OpenAI o200k'] = (lambda X: [len(o.encode(x, disallowed_special=())) for x in X], o.n_vocab, 'o200k', 'tiktoken')
except Exception as e: log('IGNORÉ o200k', e)
SWEEP = os.path.join(ICI, 'out_sweep')
local('Baracoda EU V4 parity 256k', os.path.join(SWEEP, 'baracoda-eu-v4-parity-256k.json'))
local('Baracoda EU V4 conventional 256k', os.path.join(SWEEP, 'baracoda-eu-v4-classical-256k.json'))
fr = sorted(glob.glob(os.path.expanduser('~/baracoda-fr/**/baracoda-fr-v1_2.json'), recursive=True) + glob.glob(os.path.join(os.path.dirname(os.path.dirname(ICI)), 'tokenizers', 'baracoda-fr-v1_2.json')))
local('Baracoda FR v1.2 (French only)', fr[0] if fr else None)
# identical tokenizers (e.g. Luciole sizes sharing one vocabulary) are reported once
seen = {}; dup = {}
for n in list(tok):
    h = tok[n][2]
    if h in seen: dup[n] = seen[h]; del tok[n]
    else: seen[h] = n
log('tokenizers :', ', '.join(tok), '| doublons :', dup)

res = {'date': time.strftime('%Y-%m-%d %H:%M'), 'corpus': 'NTREX-128 (1,997 sentences, 123 documents)', 'identical_tokenizers': dup,
       'chars_ratio_to_english': {k: chars[k] / chars['eng'] for k in L}, 'tokenizers': {}}
docs = {}
for n, (f, V, h, src) in tok.items():
    e = np.array(f(eng)); E = M @ e; r = {'source': src, 'vocab': V, 'english_tokens': int(e.sum()), 'ratio': {}}
    docs[n] = {'eng': E}
    for k, X in L.items():
        x = np.array(f(X)); Xd = M @ x; docs[n][k] = Xd; bs = Xd[dr].sum(1) / E[dr].sum(1)
        r['ratio'][k] = [float(x.sum() / e.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(x.sum())]
    v = [r['ratio'][k][0] for k in EU if k in r['ratio']]
    ab = np.array([e.sum()] + [r['ratio'][k][3] for k in EU]); s = np.sort(ab); m = len(s)
    r.update(mean_ratio_23=float(np.mean(v)), worst=max(EU, key=lambda k: r['ratio'][k][0]), worst_ratio=float(max(v)),
             total_23=int(sum(r['ratio'][k][3] for k in EU)), gini_24_absolute=float((2*np.arange(1, m+1)-m-1).dot(s)/(m*s.sum())),
             max_over_min_24=float(s[-1] / s[0]), french_tokens_per_char_vs_english=r['ratio']['fra'][0] / (chars['fra'] / chars['eng']))
    res['tokenizers'][n] = r
    log(f"{n:34s} vocab {V:>7,} | fr {r['ratio']['fra'][0]:.3f} | zh {r['ratio'].get('zho', [float('nan')])[0]:.3f} | "
        f"mean23 {r['mean_ratio_23']:.3f} | worst {r['worst']} {r['worst_ratio']:.2f} | gini {r['gini_24_absolute']:.3f}")
# paired comparisons of total tokens over the 23 languages (same document draws for all tokenizers)
ref = [n for n in ['Baracoda EU V4 parity 256k', 'BSC Salamandra', 'OpenAI o200k'] if n in docs]
tot = {n: sum(docs[n][k] for k in EU) for n in docs}
res['paired_total_23'] = {}
for a in docs:
    for b in ref:
        if a == b: continue
        bs = tot[a][dr].sum(1) / tot[b][dr].sum(1) - 1
        res['paired_total_23'][f'{a} vs {b}'] = [float(tot[a].sum() / tot[b].sum() - 1), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
json.dump(res, open(os.path.join(OUT, 'openllm_ntrex.json'), 'w'), ensure_ascii=False, indent=1)
T = sorted(res['tokenizers'].items(), key=lambda kv: kv[1]['ratio']['fra'][0])
md = ['# OpenLLM-France tokenizers vs our references — NTREX-128', '', f"Run {res['date']}. Identical tokenizers: {dup or 'none'}.", '',
      '| Tokenizer | Vocab | French / English (95% CI) | Chinese / English | Mean, 23 EU languages | Worst | Gini (24, absolute) | Total 23 vs V4 parity |',
      '|---|---:|---:|---:|---:|---|---:|---:|']
for n, r in T:
    fr_ = r['ratio']['fra']; zh = r['ratio'].get('zho'); p = res['paired_total_23'].get(f'{n} vs Baracoda EU V4 parity 256k')
    md.append(f"| {n} | {r['vocab']:,} | {fr_[0]:.3f} ({fr_[1]:.3f}–{fr_[2]:.3f}) | {zh[0]:.3f} | {r['mean_ratio_23']:.3f} | "
              f"{r['worst']} {r['worst_ratio']:.2f} | {r['gini_24_absolute']:.3f} | {'%+.1f%%' % (100*p[0]) if p else '—'} |" if zh else
              f"| {n} | {r['vocab']:,} | {fr_[0]:.3f} | — | {r['mean_ratio_23']:.3f} | {r['worst']} {r['worst_ratio']:.2f} | {r['gini_24_absolute']:.3f} | — |")
open(os.path.join(OUT, 'OPENLLM.md'), 'w').write('\n'.join(md) + '\n')
log('TERMINÉ')
