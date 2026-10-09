"""Recompte phrase par phrase (NTREX-128 et FLORES+ devtest) pour les tokenizers exécutés localement, avec les mêmes
fichiers, la même normalisation (NFC + strip) et le même appel de comptage que les scripts publiés :
- NTREX, six tokenizers d'entreprise : eu24_container_script.py (fichiers tokenizer.json des paquets npm
  @lenml/tokenizer-* , installés dans _npm/) ; Baracoda FR v1.2 : pack/tokenizers ;
- NTREX, Salamandra et EuroLLM : eu24_hf.py (AutoTokenizer, add_special_tokens=False) ;
- FLORES+ : eu24_flores.py (dépôts HF et tiktoken o200k_base, configuration estonienne ekk_Latn).
Vérifie que chaque total (anglais + 23 langues) reproduit EXACTEMENT les JSON publiés ; s'arrête sinon.
Sorties : counts_ntrex.json, counts_flores.json (comptes par phrase), recount_check.json.
Environnement : ~/baracoda-fr/pack/.venv-audit (tokenizers 0.23.2, transformers, tiktoken, datasets)."""
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

ICI = Path(__file__).resolve().parent
EU24 = ICI.parent.parent / "eu24"
PACK = ICI.parent.parent / "pack"
NPM = ICI / "_npm" / "node_modules" / "@lenml"
C = "468c6b69c7f6a75d31d4743d9daba2af566cc18d"
R = f"https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{C}/"
EU = ['bul', 'hrv', 'ces', 'dan', 'nld', 'est', 'fin', 'fra', 'deu', 'ell', 'hun', 'gle', 'ita', 'lav', 'lit', 'mlt', 'pol',
      'por', 'ron', 'slk', 'slv', 'spa', 'swe']
FLORES_CFG = {'bul': 'bul_Cyrl', 'hrv': 'hrv_Latn', 'ces': 'ces_Latn', 'dan': 'dan_Latn', 'nld': 'nld_Latn', 'est': 'ekk_Latn',
              'fin': 'fin_Latn', 'fra': 'fra_Latn', 'deu': 'deu_Latn', 'ell': 'ell_Grek', 'hun': 'hun_Latn', 'gle': 'gle_Latn',
              'ita': 'ita_Latn', 'lav': 'lvs_Latn', 'lit': 'lit_Latn', 'mlt': 'mlt_Latn', 'pol': 'pol_Latn', 'por': 'por_Latn',
              'ron': 'ron_Latn', 'slk': 'slk_Latn', 'slv': 'slv_Latn', 'spa': 'spa_Latn', 'swe': 'swe_Latn'}
nfc = lambda s: unicodedata.normalize('NFC', s.strip())


def ntrex_texts():
    get = lambda p: urllib.request.urlopen(R + p, timeout=120).read().decode('utf-8')
    rd = lambda p: [nfc(l) for l in get(p).splitlines()]
    T = {'eng': rd('NTREX-128/newstest2019-src.eng.txt')}
    T.update({k: rd(f'NTREX-128/newstest2019-ref.{k}.txt') for k in EU})
    doc = get('DOCUMENT_IDS.tsv').split('\n')[:1997]
    assert all(len(v) == 1997 for v in T.values()) and len(doc) == 1997
    return T, doc


def flores_texts():
    from datasets import load_dataset

    def one(cfg):
        ds = sorted(load_dataset('openlanguagedata/flores_plus', cfg, split='devtest'), key=lambda r: r['id'])
        return [nfc(r['text']) for r in ds], [r['url'] for r in ds]
    eng, urls = one('eng_Latn')
    T = {'eng': eng}
    for k, cfg in FLORES_CFG.items():
        T[k], u = one(cfg)
        assert u == urls and len(T[k]) == len(eng), cfg
    return T, urls


def file_tok(path):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(path))
    return lambda X: [len(e.ids) for e in t.encode_batch(X, add_special_tokens=False)]


def hf_tok(repo):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(repo)
    return lambda X: [len(x) for x in t(X, add_special_tokens=False)['input_ids']]


def o200k():
    import tiktoken
    e = tiktoken.get_encoding('o200k_base')
    return lambda X: [len(x) for x in e.encode_ordinary_batch(X)]


NTREX_TOK = {  # nom publié -> (fabrique, fichier JSON publié)
    'OpenAI o200k': (lambda: file_tok(NPM / 'tokenizer-gptoss/models/tokenizer.json'), 'ntrex'),
    'Meta Llama 3': (lambda: file_tok(NPM / 'tokenizer-llama3_1/models/tokenizer.json'), 'ntrex'),
    'Alibaba Qwen3': (lambda: file_tok(NPM / 'tokenizer-qwen3/models/tokenizer.json'), 'ntrex'),
    'DeepSeek V3': (lambda: file_tok(NPM / 'tokenizer-deepseek_v3/models/tokenizer.json'), 'ntrex'),
    'Google Gemma 3': (lambda: file_tok(NPM / 'tokenizer-gemma3/models/tokenizer.json'), 'ntrex'),
    'Mistral Tekken': (lambda: file_tok(NPM / 'tokenizer-mistral_nemo/models/tokenizer.json'), 'ntrex'),
    'Baracoda FR v1.2': (lambda: file_tok(PACK / 'tokenizers/baracoda-fr-v1_2.json'), 'ntrex'),
    'Salamandra-7B': (lambda: hf_tok('BSC-LT/salamandra-7b'), 'hf'),
    'EuroLLM-9B': (lambda: hf_tok('utter-project/EuroLLM-9B'), 'hf'),
}
FLORES_TOK = {
    'BSC Salamandra': lambda: hf_tok('BSC-LT/salamandra-7b'), 'EuroLLM': lambda: hf_tok('utter-project/EuroLLM-9B'),
    'OpenAI o200k': o200k, 'Google Gemma 3': lambda: hf_tok('unsloth/gemma-3-4b-it'),
    'Mistral Tekken': lambda: hf_tok('mistralai/Mistral-Nemo-Instruct-2407'), 'DeepSeek V3': lambda: hf_tok('deepseek-ai/DeepSeek-V3'),
    'Meta Llama 3': lambda: hf_tok('unsloth/Meta-Llama-3.1-8B'), 'Alibaba Qwen3': lambda: hf_tok('Qwen/Qwen3-8B'),
    'Baracoda FR v1.2': lambda: file_tok(PACK / 'tokenizers/baracoda-fr-v1_2.json'),
}


def published(corpus, name, where):
    if corpus == 'ntrex':
        j = json.load(open(EU24 / ('eu24_ntrex.json' if where == 'ntrex' else 'eu24_ntrex_hf.json')))
        r = j['tokenizers'][name] if where == 'ntrex' else j[name]
    else:
        r = json.load(open(EU24 / 'eu24_flores.json'))['tokenizers'][name]
    return {'eng': r['english_tokens'], **{k: r['premium'][k][3] for k in EU}}


def run(corpus):
    T, groups = ntrex_texts() if corpus == 'ntrex' else flores_texts()
    toks = NTREX_TOK if corpus == 'ntrex' else {k: (v, None) for k, v in FLORES_TOK.items()}
    out, check, bad = {'groups': groups, 'counts': {}}, {}, []
    for name, (make, where) in toks.items():
        f = make()
        c = {k: f(X) for k, X in T.items()}
        pub = published(corpus, name, where)
        diff = {k: sum(c[k]) - pub[k] for k in pub if sum(c[k]) != pub[k]}
        check[name] = {'identical': not diff, 'diff': diff, 'total_eu': sum(sum(c[k]) for k in EU)}
        print(f"{corpus} {name:18s} {'IDENTIQUE' if not diff else 'ÉCART ' + str(diff)}", flush=True)
        if diff:
            bad.append(name)
        out['counts'][name] = c
    if corpus == 'ntrex':  # Claude : comptes par phrase publiés, aucun appel à l'API
        cl = json.load(open(EU24 / 'eu24_claude_counts.json'))['languages']
        out['counts']['Anthropic Claude'] = {k: cl[k]['counts'] for k in ['eng'] + EU}
        pub = json.load(open(EU24 / 'eu24_claude.json'))['Claude (Opus 5.5)']
        ok = sum(cl['eng']['counts']) == pub['english_tokens'] and all(sum(cl[k]['counts']) == pub['premium'][k][3] for k in EU)
        check['Anthropic Claude'] = {'identical': ok, 'source': 'eu24_claude_counts.json (aucun nouvel appel)'}
        if not ok:
            bad.append('Anthropic Claude')
    json.dump(out, open(ICI / f'counts_{corpus}.json', 'w'))
    return check, bad


if __name__ == '__main__':
    allc = json.load(open(ICI / 'recount_check.json')) if (ICI / 'recount_check.json').exists() else {}
    allbad = []
    for corpus in sys.argv[1:] or ['ntrex', 'flores']:
        allc[corpus], bad = run(corpus)
        allbad += [f'{corpus}:{b}' for b in bad]
        json.dump(allc, open(ICI / 'recount_check.json', 'w'), indent=1, ensure_ascii=False)
        if bad:
            sys.exit(f"ARRÊT : totaux non reproduits pour {bad}")
    print("TOUS LES TOTAUX REPRODUITS")
