# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baracoda Group (Baracoda AI Labs)
"""Check whether current Mistral tokenizer files give the same counts as Tekken 240911 (Mistral Nemo, v3).
Downloads tekken.json from each model repository on Hugging Face and counts every corpus.
Writes results/mistral_versions.json (file SHA-256, Tekken version, per-corpus totals)."""
import hashlib, json
from huggingface_hub import hf_hub_download
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from common import RESULTS, load

MODELS = ["mistralai/Mistral-Nemo-Instruct-2407", "mistralai/Ministral-3-8B-Instruct-2512",
          "mistralai/Mistral-Large-3-675B-Instruct-2512", "mistralai/Mistral-Medium-3.5-128B",
          "mistralai/Mistral-Small-4-119B-2603"]


def main():
    corpora, out = {**load("dev"), **load("final")}, {}
    for m in MODELS:
        p = hf_hub_download(m, "tekken.json")
        raw = open(p, "rb").read()
        t = Tekkenizer.from_file(p)
        out[m] = {"sha256": hashlib.sha256(raw).hexdigest(), "version": json.loads(raw)["config"].get("version"),
                  "totals": {c: sum(len(t.encode(s, bos=False, eos=False)) for s in L) for c, L in corpora.items()}}
        print(m, out[m]["version"], out[m]["totals"]["UD French-PUD"], flush=True)
    ref = out[MODELS[0]]["totals"]
    out["identical_to_nemo"] = {m: out[m]["totals"] == ref for m in MODELS}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "mistral_versions.json").write_text(json.dumps(out, indent=1))
    print(out["identical_to_nemo"])


if __name__ == "__main__":
    main()
