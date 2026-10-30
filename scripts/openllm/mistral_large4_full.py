"""Mistral Large 4 tokenizer on the full protocol of the language-tax studies, so that it can replace Tekken as
Mistral's current tokenizer in the synthesis paper. Written by Claude, 30 Oct 2026.
A. NTREX-128, all 127 non-English references (NFC, no special tokens): ratio to English, French rank among the
   124 non-English references (English variants excluded), median ratio. Tekken and o200k are recomputed as
   positive controls against the archived values.
B. FLEURS dev+test text for 26 African and Asian languages, on the sentence ids shared with English used in the
   paper (ids only, in data/fleurs_afr_keys.json): ratio of sums, 2,000 bootstrap draws over sentences. Tekken and
   o200k recomputed as positive controls against the archived values."""
import csv, json, os, time, unicodedata, urllib.request
import numpy as np
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import tiktoken
OUT = os.environ.get("OUT_DIR", "results/openllm"); os.makedirs(OUT, exist_ok=True); CACHE = os.environ.get("HF_CACHE")
HERE = os.path.dirname(os.path.abspath(__file__))
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def hf_tok(repo):
    t = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", cache_dir=CACHE))
    return lambda X: np.array([len(e.ids) for e in t.encode_batch(X, add_special_tokens=False)])
o = tiktoken.get_encoding("o200k_base")
TOK = {"Mistral Large 4": hf_tok("mistralai/Mistral-Large-4-1T-A52B"),
       "Mistral Tekken (Nemo)": hf_tok("mistralai/Mistral-Nemo-Base-2407"),
       "OpenAI o200k": lambda X: np.array([len(x) for x in o.encode_ordinary_batch(X)])}
CTRL_NAME = {"Mistral Tekken (Nemo)": "Mistral Tekken (Nemo)", "OpenAI o200k": "OpenAI o200k (GPT-4o / GPT-5 / gpt-oss)"}
res = {"date": time.strftime("%Y-%m-%d %H:%M")}

# ---------- A. NTREX-128, all languages ----------
C = "468c6b69c7f6a75d31d4743d9daba2af566cc18d"; R = f"https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{C}/"
get = lambda p: urllib.request.urlopen(R + p, timeout=120).read().decode("utf-8")
ctrl = json.load(open(os.path.join(HERE, "data", "ntrex_control.json")))
langs = sorted(ctrl["Mistral Tekken (Nemo)"])
T = {}
for L in langs:
    p = "NTREX-128/newstest2019-src.eng.txt" if L == "eng" else f"NTREX-128/newstest2019-ref.{L}.txt"
    try: T[L] = [unicodedata.normalize("NFC", l.strip()) for l in get(p).splitlines()][:1997]
    except Exception as e: log("NTREX missing", L, e)
log("NTREX languages read:", len(T))
A = {}
for n, f in TOK.items():
    E = f(T["eng"]).sum(); r = {L: float(f(X).sum() / E) for L, X in T.items() if L != "eng"}
    nonen = sorted((v, L) for L, v in r.items() if not L.startswith("eng"))
    rank = [i + 1 for i, (_, L) in enumerate(nonen) if L == "fra"][0]
    A[n] = {"english_tokens": int(E), "ratio": r, "french_rank": rank, "n_non_english": len(nonen),
            "median_non_english": float(np.median([v for v, _ in nonen])), "max": nonen[-1]}
    if n in CTRL_NAME:
        c = ctrl[CTRL_NAME[n]]; d = max(abs(r[L] - c[L]) for L in r if L in c)
        A[n]["control_max_abs_diff_vs_archived"] = d
        log(f"CONTROL {n}: max |diff| vs archived = {d:.4f}")
    log(f"{n}: fr {r['fra']:.3f} rank {rank}/{len(nonen)} median {A[n]['median_non_english']:.3f} zh {r['zho-CN']:.3f} max {nonen[-1]}")
res["ntrex_all"] = A

# ---------- B. FLEURS, African and Asian languages ----------
K = json.load(open(os.path.join(HERE, "data", "fleurs_afr_keys.json")))
def texts(code):
    out = {}
    for s in ["dev", "test"]:
        p = hf_hub_download("google/fleurs", f"data/{code}/{s}.tsv", repo_type="dataset", cache_dir=CACHE)
        for r in csv.reader(open(p, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            out.setdefault(int(r[0]), r[2])
    return out
EN = texts("en_us"); B = {}
rng = np.random.default_rng(0)
for L, code in K["codes"].items():
    keys = K["keys"][L]; tx = texts(code)
    keys = [k for k in keys if k in tx and k in EN]
    S = rng.integers(0, len(keys), (2000, len(keys))); B[L] = {"n": len(keys)}
    for n, f in TOK.items():
        a = f([tx[k] for k in keys]).astype(float); e = f([EN[k] for k in keys]).astype(float)
        br = a[S].sum(1) / e[S].sum(1)
        B[L][n] = [float(a.sum() / e.sum()), float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]
    B[L]["control_tekken_archived"] = K["control_tekken"].get(L); B[L]["control_o200k_archived"] = K["control_o200k"].get(L)
    log(f"{L} n={len(keys)} L4 {B[L]['Mistral Large 4'][0]:.3f} | Tekken {B[L]['Mistral Tekken (Nemo)'][0]:.3f} (archived {B[L]['control_tekken_archived']:.3f}) | o200k {B[L]['OpenAI o200k'][0]:.3f} (archived {B[L]['control_o200k_archived']:.3f})")
res["fleurs_afr_asia"] = B
lat = ["sw", "ha", "yo", "ig", "wo", "ff", "ln", "so", "om", "zu", "xh", "sn", "lg", "ny", "nso", "luo", "kam", "umb"]
res["summary"] = {n: {"latin_african_median": float(np.median([B[L][n][0] for L in lat])),
                      "latin_african_range": [min(B[L][n][0] for L in lat), max(B[L][n][0] for L in lat)],
                      "amharic": B["am"][n][0], "asian_max": max(B[L][n][0] for L in ["hi", "bn", "ta", "km", "lo", "my"]),
                      "arabic_eg": B["ar"][n][0]} for n in TOK}
log("SUMMARY", json.dumps(res["summary"], indent=1))
json.dump(res, open(os.path.join(OUT, "mistral_large4_full.json"), "w"), indent=1)
log("TERMINÉ")
