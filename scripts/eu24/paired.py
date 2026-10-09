"""Bootstrap apparié entre tokenizers (NTREX-128 et FLORES+), à partir des comptes par phrase de recount.py.

Rééchantillonnage des DOCUMENTS (NTREX : 123 articles de DOCUMENT_IDS.tsv ; FLORES+ : 281 articles par URL source),
10 000 tirages, graine 0, les mêmes tirages pour tous les tokenizers et toutes les langues. IC 95 % = percentiles 2,5/97,5.
a) écart relatif de tokens totaux (23 langues, anglais exclu) entre tokenizers adjacents du classement observé
   (ordre croissant du total), et entre Salamandra et chacun des autres : Δ = total(X) / total(Y) − 1 ;
b) par langue, Salamandra (et EuroLLM) contre le tokenizer d'entreprise le moins coûteux pour cette langue parmi les six
   exécutés localement (choisi sur les comptes observés, comme le tableau 3 / la figure 1 ; identité fixée dans les tirages),
   Δ = tokens(Salamandra) / tokens(entreprise) − 1 ; en complément, le même écart contre le minimum recalculé à chaque tirage ;
c) probabilité, sur les tirages, que l'ordre complet soit l'ordre observé (par total, et par prime moyenne), et pour chaque
   paire adjacente que son ordre soit conservé.
Classement principal : les 8 tokenizers exécutés localement (Baracoda FR v1.2 exclu : référence). NTREX : aussi avec Claude.
Usage : python paired.py  ->  paired_ntrex.json, paired_flores.json, PAIRED.md"""
import json
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
EU = ['bul', 'hrv', 'ces', 'dan', 'nld', 'est', 'fin', 'fra', 'deu', 'ell', 'hun', 'gle', 'ita', 'lav', 'lit', 'mlt', 'pol',
      'por', 'ron', 'slk', 'slv', 'spa', 'swe']
LANG = {'bul': 'Bulgarian', 'hrv': 'Croatian', 'ces': 'Czech', 'dan': 'Danish', 'nld': 'Dutch', 'est': 'Estonian',
        'fin': 'Finnish', 'fra': 'French', 'deu': 'German', 'ell': 'Greek', 'hun': 'Hungarian', 'gle': 'Irish',
        'ita': 'Italian', 'lav': 'Latvian', 'lit': 'Lithuanian', 'mlt': 'Maltese', 'pol': 'Polish', 'por': 'Portuguese',
        'ron': 'Romanian', 'slk': 'Slovak', 'slv': 'Slovenian', 'spa': 'Spanish', 'swe': 'Swedish'}
RENAME = {'Salamandra-7B': 'BSC Salamandra', 'EuroLLM-9B': 'EuroLLM'}
COMPANY = ['OpenAI o200k', 'Meta Llama 3', 'Alibaba Qwen3', 'DeepSeek V3', 'Google Gemma 3', 'Mistral Tekken']
LOCAL8 = ['BSC Salamandra', 'EuroLLM'] + COMPANY
N_BOOT, SEED = 10000, 0


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def analyse(corpus, with_claude):
    d = json.load(open(ICI / f'counts_{corpus}.json'))
    counts = {RENAME.get(k, k): v for k, v in d['counts'].items()}
    groups = d['groups']
    ids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    g = np.array([ids[x] for x in groups]); D = len(ids)
    draws = np.random.default_rng(SEED).integers(0, D, size=(N_BOOT, D))
    names = LOCAL8 + (['Anthropic Claude'] if with_claude else [])
    per_doc = {t: {k: np.bincount(g, weights=counts[t][k], minlength=D) for k in ['eng'] + EU} for t in names}
    obs = {t: {k: float(sum(counts[t][k])) for k in ['eng'] + EU} for t in names}
    boot = {t: {k: per_doc[t][k][draws].sum(1) for k in ['eng'] + EU} for t in names}
    tot_obs = {t: sum(obs[t][k] for k in EU) for t in names}
    tot_b = {t: sum(boot[t][k] for k in EU) for t in names}
    mean_obs = {t: float(np.mean([obs[t][k] / obs[t]['eng'] for k in EU])) for t in names}
    mean_b = {t: np.mean([boot[t][k] / boot[t]['eng'] for k in EU], axis=0) for t in names}
    out = {'corpus': corpus, 'documents': D, 'sentences': len(groups), 'n_boot': N_BOOT, 'seed': SEED, 'tokenizers': names,
           'total_tokens_23': {t: int(tot_obs[t]) for t in names}, 'mean_ratio': mean_obs}

    def order_block(sel, metric_obs, metric_b):
        order = sorted(sel, key=lambda t: metric_obs[t])
        mat = np.stack([metric_b[t] for t in order], 1)               # tirages × tokenizers, dans l'ordre observé
        same = np.all(np.diff(mat, axis=1) > 0, axis=1)
        adj = []
        for a, b in zip(order, order[1:]):
            r = metric_b[b] / metric_b[a] - 1
            adj.append({'lower': a, 'higher': b, 'delta': float(metric_obs[b] / metric_obs[a] - 1), 'ci95': ci(r),
                        'p_order_kept': float(np.mean(metric_b[a] < metric_b[b]))})
        return {'order': order, 'p_full_order': float(same.mean()), 'adjacent': adj}

    # a) + c) classements : 8 locaux, et (NTREX) 9 avec Claude
    out['by_total'] = {'local8': order_block(LOCAL8, tot_obs, tot_b)}
    out['by_mean_ratio'] = {'local8': order_block(LOCAL8, mean_obs, mean_b)}
    if with_claude:
        out['by_total']['with_claude9'] = order_block(names, tot_obs, tot_b)
        out['by_mean_ratio']['with_claude9'] = order_block(names, mean_obs, mean_b)
    out['salamandra_vs'] = {t: {'delta': float(tot_obs['BSC Salamandra'] / tot_obs[t] - 1),
                                'ci95': ci(tot_b['BSC Salamandra'] / tot_b[t] - 1)} for t in names if t != 'BSC Salamandra'}
    # b) par langue, contre le tokenizer d'entreprise le moins coûteux (6 locaux)
    per_lang = {}
    for k in EU:
        low = min(COMPANY, key=lambda t: obs[t][k])
        mins = np.min(np.stack([boot[t][k] for t in COMPANY]), 0)
        row = {'lowest_company': low, 'lowest_tokens': int(obs[low][k])}
        for euro in ('BSC Salamandra', 'EuroLLM'):
            row[euro] = {'tokens': int(obs[euro][k]), 'delta': float(obs[euro][k] / obs[low][k] - 1),
                         'ci95': ci(boot[euro][k] / boot[low][k] - 1),
                         'p_fewer': float(np.mean(boot[euro][k] < boot[low][k])),
                         'ci95_vs_min_redrawn': ci(boot[euro][k] / mins - 1)}
        per_lang[k] = row
    out['per_language_vs_lowest_company'] = per_lang
    return out


def pct(x):
    return f"{100 * x:+.1f} %".replace(".", ",")


def pci(c):
    return f"[{100 * c[0]:+.1f} ; {100 * c[1]:+.1f}]".replace(".", ",")


def md(res):
    L = ["# Comparaisons appariées entre tokenizers (bootstrap sur les documents)", "",
         f"Généré par `paired.py` à partir des comptes par phrase de `recount.py` (totaux identiques aux JSON publiés, "
         f"voir `recount_check.json`). {N_BOOT:,} tirages".replace(",", " ") + f", graine {SEED} ; mêmes documents tirés pour "
         "tous les tokenizers. Δ = tokens(X) / tokens(Y) − 1 ; IC 95 % par percentiles.", ""]
    for r in res:
        name = {'ntrex': 'NTREX-128', 'flores': 'FLORES+ devtest'}[r['corpus']]
        L += [f"## {name} ({r['documents']} documents, {r['sentences']} phrases)", ""]
        for key, title in (('local8', '8 tokenizers exécutés localement'), ('with_claude9', '9 tokenizers, avec Claude')):
            if key not in r['by_total']:
                continue
            bt, bm = r['by_total'][key], r['by_mean_ratio'][key]
            L += [f"### a) et c) Classement par total de tokens (23 langues) — {title}", "",
                  "| Rang | Tokenizer | Tokens (23 langues) | Écart avec le suivant | IC 95 % | P(ordre conservé) |",
                  "|---:|---|---:|---:|---|---:|"]
            for i, t in enumerate(bt['order']):
                a = bt['adjacent'][i] if i < len(bt['adjacent']) else None
                L.append(f"| {i + 1} | {t} | {r['total_tokens_23'][t]:,} | ".replace(",", " ")
                         + (f"{pct(a['delta'])} | {pci(a['ci95'])} | {a['p_order_kept']:.4f} |" if a else "— | — | — |"))
            L += ["", f"Probabilité que l'ordre complet (total) soit l'ordre observé : **{bt['p_full_order']:.4f}**.",
                  f"Ordre par prime moyenne : {' < '.join(bm['order'])} ; probabilité de l'ordre complet : "
                  f"**{bm['p_full_order']:.4f}** ; paire adjacente la moins sûre : "
                  + "{} < {} (P = {:.4f})".format(*min(((a['lower'], a['higher'], a['p_order_kept']) for a in bm['adjacent']),
                                                       key=lambda x: x[2])) + ".", ""]
        L += ["### a) Salamandra contre chacun des autres (total, 23 langues)", "", "| Contre | Δ Salamandra | IC 95 % |", "|---|---:|---|"]
        for t, v in r['salamandra_vs'].items():
            L.append(f"| {t} | {pct(v['delta'])} | {pci(v['ci95'])} |")
        L += ["", "### b) Par langue : tokenizer européen contre le tokenizer d'entreprise le moins coûteux", "",
              "Identité du tokenizer d'entreprise fixée sur les comptes observés (comme le tableau 3) ; dernière colonne : "
              "IC contre le minimum des six recalculé à chaque tirage.", "",
              "| Langue | Entreprise | Tokens | Δ Salamandra | IC 95 % | Δ EuroLLM | IC 95 % | IC Salamandra (min. retiré) |",
              "|---|---|---:|---:|---|---:|---|---|"]
        rows = sorted(r['per_language_vs_lowest_company'].items(), key=lambda kv: -kv[1]['lowest_tokens'])
        for k, v in rows:
            s, e = v['BSC Salamandra'], v['EuroLLM']
            L.append(f"| {LANG[k]} | {v['lowest_company']} | {v['lowest_tokens']:,} | ".replace(",", " ")
                     + f"{pct(s['delta'])} | {pci(s['ci95'])} | {pct(e['delta'])} | {pci(e['ci95'])} | {pci(s['ci95_vs_min_redrawn'])} |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == '__main__':
    res = []
    for corpus, claude in (('ntrex', True), ('flores', False)):
        r = analyse(corpus, claude)
        json.dump(r, open(ICI / f'paired_{corpus}.json', 'w'), indent=1, ensure_ascii=False)
        res.append(r)
        print(corpus, 'ordre (total) :', ' < '.join(r['by_total']['local8']['order']),
              '| P(ordre complet) total', r['by_total']['local8']['p_full_order'], 'moyenne', r['by_mean_ratio']['local8']['p_full_order'])
    (ICI / 'PAIRED.md').write_text(md(res), encoding='utf-8')
