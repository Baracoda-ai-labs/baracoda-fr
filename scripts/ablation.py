# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baracoda Group (Baracoda AI Labs)
"""Controlled experiment: effect of the French share (in UTF-8 BYTES) of tokenizer training data.
Data: Europarl v7 fr-en, one line = one sentence, whole sentences only. For each seed, sentences are shuffled
(random.Random(seed) for French, seed+100 for English) and taken in order until the byte budget is reached.
Total budget fixed at --total bytes; French share in {0, 0.25, 0.5, 0.75, 1}. Pre-tokenizer: Qwen3's.
Records bytes actually retained and the effective vocabulary size reached (which can be below --vocab
when the data are too small for min_frequency=2)."""
import argparse, json, random, urllib.request
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from common import RESULTS, ROOT, load

EUROPARL = "https://media.githubusercontent.com/media/dksifoua/NMT/master/data/europarl-v7.fr-en."


def europarl(lang):
    p = ROOT / "data" / f"europarl-v7.fr-en.{lang}"
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(EUROPARL + lang, p)
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def qwen3_pretokenizer():
    from huggingface_hub import hf_hub_download
    return Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-8B", "tokenizer.json")).pre_tokenizer


def take(lines, order, budget, path):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for i in order:
            b = len(lines[i].encode("utf-8")) + 1
            if n + b > budget:
                break
            f.write(lines[i] + "\n"); n += b
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=50000)
    ap.add_argument("--total", type=int, default=48_000_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    dev = load("dev"); PF, PE = dev["UD French-PUD"], dev["UD English-PUD"]
    FR, EN, pre, out = europarl("fr"), europarl("en"), qwen3_pretokenizer(), []
    tmp = ROOT / "data"
    for seed in a.seeds:
        rf, re_ = list(range(len(FR))), list(range(len(EN)))
        random.Random(seed).shuffle(rf); random.Random(seed + 100).shuffle(re_)
        for share in [0, 0.25, 0.5, 0.75, 1.0]:
            files, bf, be = [], 0, 0
            if share > 0:
                bf = take(FR, rf, int(a.total * share), tmp / "ab.fr"); files.append(str(tmp / "ab.fr"))
            if share < 1:
                be = take(EN, re_, int(a.total * (1 - share)), tmp / "ab.en"); files.append(str(tmp / "ab.en"))
            t = Tokenizer(models.BPE()); t.pre_tokenizer = pre; t.decoder = decoders.ByteLevel()
            t.train(files, trainers.BpeTrainer(vocab_size=a.vocab, min_frequency=2, show_progress=False,
                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
            c = lambda L: sum(len(e.ids) for e in t.encode_batch(L, add_special_tokens=False))
            r = {"seed": seed, "share_target": share, "bytes_fr": bf, "bytes_en": be, "share_actual": bf / (bf + be),
                 "vocab_target": a.vocab, "vocab_effective": t.get_vocab_size(), "pud_fr": c(PF), "pud_en": c(PE)}
            r["premium"] = r["pud_fr"] / r["pud_en"]; out.append(r); print(r, flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"ablation_bytes_v{a.vocab}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
