# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baracoda Group (Baracoda AI Labs)
"""Contamination check on the final test (protocol criterion). A final-test sentence is contaminated if it is
longer than 40 characters and appears verbatim in any training document in data/*.jsonl (NFC text).
Writes results/contamination.json: per-corpus denominators, contaminated sentence ids and hashes, and
v1.2-vs-Tekken results with and without the contaminated sentences. Requires results/counts.json."""
import hashlib, json
import ahocorasick
from common import DATA, RESULTS, config, load
from evaluate import block_ci


def main():
    final = load("final")
    counts = json.loads((RESULTS / "counts.json").read_text())["counts"]
    files = sorted(DATA.glob("*.jsonl"))
    if not files:
        raise SystemExit("No training data in data/: run scripts/collect_training.py first")
    A = ahocorasick.Automaton()
    for s in {x for L in final.values() for x in L if len(x) > 40}:
        A.add_word(s, s)
    A.make_automaton()
    hits, per_file = set(), {}
    for f in files:
        n = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                for _, s in A.iter(json.loads(line)["text"]):
                    hits.add(s); n += 1
        per_file[f.name] = n
    cfg, rep = config(), {"match_occurrences_per_training_file": per_file, "corpora": {}, "contaminated": []}
    pooled = {}
    for name, L in final.items():
        v, t = counts["baracoda-fr-v1_2"][name], counts["tekken"][name]
        keep = [i for i, s in enumerate(L) if s not in hits]
        bad = [i for i, s in enumerate(L) if s in hits]
        rep["contaminated"] += [{"corpus": name, "index": i, "sha256": hashlib.sha256(L[i].encode()).hexdigest()} for i in bad]
        elig = sum(len(s) > 40 for s in L)
        rep["corpora"][name] = {"sentences": len(L), "eligible_over_40_chars": elig, "contaminated": len(bad),
                                "share_of_eligible": len(bad) / elig, "full": block_ci(v, t),
                                "clean": block_ci([v[i] for i in keep], [t[i] for i in keep])}
        p = pooled.setdefault(cfg[name]["lang"], {"full": ([], []), "clean": ([], [])})
        p["full"][0].extend(v); p["full"][1].extend(t)
        p["clean"][0].extend(v[i] for i in keep); p["clean"][1].extend(t[i] for i in keep)
    for lang, p in pooled.items():
        rep[f"pooled_{lang}"] = {m: block_ci(*p[m]) for m in p}
    rep["distinct_contaminated_sentences"] = len(hits)
    (RESULTS / "contamination.json").write_text(json.dumps(rep, indent=1))
    for n, c in rep["corpora"].items():
        print(f"{n:16s} {c['contaminated']:4d}/{c['eligible_over_40_chars']:5d}  full {c['full'][0]:+.2%}  clean {c['clean'][0]:+.2%}")
    for lang in pooled:
        print(lang, {m: f"{x[0]:+.2%}" for m, x in rep[f'pooled_{lang}'].items()})


if __name__ == "__main__":
    main()
