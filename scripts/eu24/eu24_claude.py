"""EU-23 token premiums of Claude on NTREX-128 (same corpus, list and bootstrap as eu24_hf.py).
Method of ../pack/scripts/claude_counts.py: free /v1/messages/count_tokens endpoint, one request per sentence,
per-message overhead calibrated by difference (count(x), count(xx), count(xxx)) and subtracted.
Requires ANTHROPIC_API_KEY in the environment (never written to disk, never printed).
Usage (from ~/baracoda-fr/eu24, pack environment):
  python eu24_claude.py count     # calibration + counts, saved after each language (resumable)
  python eu24_claude.py models    # 50 random sentences (seed 0) x eng/fra/ell/mlt on three Claude models
  python eu24_claude.py results   # checks + eu24_claude.json + summary
Anthropic describes the endpoint as an estimate; these are not invoices."""
import json, os, sys, time, unicodedata, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np

MODEL = "claude-opus-5-5"
CHECK_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"]
KEY = os.environ.get("ANTHROPIC_API_KEY")
COUNTS, MODELS_OUT, RESULT = "eu24_claude_counts.json", "eu24_claude_models.json", "eu24_claude.json"
EXPECTED_FRA = 1.45  # premium of French on NTREX in the paper (tolerance 0.01)

# Corpus: identical to eu24_hf.py
C = '468c6b69c7f6a75d31d4743d9daba2af566cc18d'; R = f'https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{C}/'
EU = ['bul','hrv','ces','dan','nld','est','fin','fra','deu','ell','hun','gle','ita','lav','lit','mlt','pol','por','ron','slk','slv','spa','swe']
get = lambda p: urllib.request.urlopen(R + p, timeout=120).read().decode('utf-8')
rd = lambda p: [unicodedata.normalize('NFC', l.strip()) for l in get(p).splitlines()]


def corpus():
    L = {'eng': rd('NTREX-128/newstest2019-src.eng.txt')}
    L.update({c: rd(f'NTREX-128/newstest2019-ref.{c}.txt') for c in EU})
    doc = get('DOCUMENT_IDS.tsv').split('\n')[:1997]
    assert all(len(v) == 1997 for v in L.values())
    return L, doc


def count(text, model=MODEL):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}]}).encode()
    for a in range(12):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages/count_tokens", data=body,
                                     headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))["input_tokens"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(min(60, 2 ** a)); continue
            raise RuntimeError(f"HTTP {e.code} for model {model}: {e.read()[:300].decode('utf-8', 'replace')}") from None
    raise RuntimeError("too many retries")


def calibrate(model=MODEL):
    probes = ["Bonjour.\n", "The cat sat on the mat.\n", "Le chat est sur le tapis.\n", "1234\n", "Hello world\n", "a\n",
              "Tous les êtres humains naissent libres.\n"]
    cal = [{"text": x, "c1": count(x, model), "c2": count(x * 2, model), "c3": count(x * 3, model)} for x in probes]
    for c in cal:
        c["overhead_2"] = 2 * c["c1"] - c["c2"]; c["overhead_3"] = (3 * c["c1"] - c["c3"]) / 2
    overheads = {c["overhead_2"] for c in cal} | {c["overhead_3"] for c in cal}
    if len(overheads) != 1:
        raise SystemExit(f"Inconsistent overhead estimates for {model}: {overheads}")
    return int(overheads.pop()), cal


def now():
    return datetime.now(timezone.utc).isoformat()


def do_count():
    L, _ = corpus()
    out = json.load(open(COUNTS)) if os.path.exists(COUNTS) else None
    if out is None:
        oh, cal = calibrate()
        out = {"model": MODEL, "overhead": oh, "calibration": cal, "calibrated_at": now(), "ntrex_commit": C,
               "requests": 3 * len(cal), "languages": {}}
        json.dump(out, open(COUNTS, 'w'))
        print(f"overhead {oh}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for k in ['eng', 'fra'] + [c for c in EU if c != 'fra']:  # French early: the paper check can stop the run
            if k in out["languages"]:
                print(k, "already counted", flush=True)
            else:
                t0 = now()
                raw = list(ex.map(count, L[k]))
                out["languages"][k] = {"counts": [c - out["overhead"] for c in raw], "started": t0, "finished": now()}
                out["requests"] += len(raw)
                json.dump(out, open(COUNTS, 'w'))
                print(k, sum(out["languages"][k]["counts"]), flush=True)
            if k == 'fra':
                p = sum(out["languages"]["fra"]["counts"]) / sum(out["languages"]["eng"]["counts"])
                print(f"French premium {p:.4f} (paper {EXPECTED_FRA})", flush=True)
                if abs(p - EXPECTED_FRA) > 0.01:
                    raise SystemExit(f"STOP: French premium {p:.4f} differs from {EXPECTED_FRA} by more than 0.01")


def do_models():
    L, _ = corpus()
    idx = sorted(np.random.default_rng(0).choice(1997, 50, replace=False).tolist())
    res = {"sentence_indices": idx, "languages": ["eng", "fra", "ell", "mlt"], "models": {}, "at": now()}
    with ThreadPoolExecutor(8) as ex:
        for m in CHECK_MODELS:
            try:
                res["models"][m] = {k: list(ex.map(lambda s: count(s, m), [L[k][i] for i in idx])) for k in res["languages"]}
            except RuntimeError as e:
                res["models"][m] = {"error": str(e)}
            print(m, "error" if "error" in res["models"][m] else {k: sum(v) for k, v in res["models"][m].items()}, flush=True)
    json.dump(res, open(MODELS_OUT, 'w'), indent=1)


def do_results():
    _, doc = corpus()
    d = json.load(open(COUNTS))
    assert set(d["languages"]) == {'eng', *EU}, "counting not finished"
    ids = {x: i for i, x in enumerate(dict.fromkeys(doc))}; g = np.array([ids[x] for x in doc]); D = len(ids)
    M = np.zeros((D, 1997)); M[g, np.arange(1997)] = 1; dr = np.random.default_rng(0).integers(0, D, size=(2000, D))
    e = np.array(d["languages"]["eng"]["counts"]); E = M @ e
    res = {"model": MODEL, "overhead": d["overhead"], "english_tokens": int(e.sum()), "premium": {}}
    for k in EU:
        x = np.array(d["languages"][k]["counts"]); bs = (M @ x)[dr].sum(1) / E[dr].sum(1)
        res["premium"][k] = [float(x.sum() / e.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(x.sum())]
    fra = res["premium"]["fra"][0]
    print(f"Contrôle français : {fra:.4f} (attendu {EXPECTED_FRA} ± 0,01)")
    if abs(fra - EXPECTED_FRA) > 0.01:
        raise SystemExit(f"STOP: French premium {fra:.4f} differs from {EXPECTED_FRA} by more than 0.01")
    v = [res["premium"][k][0] for k in EU]; s = np.sort(v); n = len(s)
    res.update(mean=float(np.mean(v)), max=float(max(v)), argmax=max(EU, key=lambda k: res["premium"][k][0]),
               gini=float((2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum())),
               total_eu_tokens=int(sum(res["premium"][k][3] for k in EU)), requests=d["requests"])
    json.dump({"Claude (Opus 5.5)": res}, open(RESULT, 'w'), indent=1)


if __name__ == "__main__":
    if not KEY and sys.argv[1] != "results":
        raise SystemExit("Set ANTHROPIC_API_KEY")
    {"count": do_count, "models": do_models, "results": do_results}[sys.argv[1]]()
