# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baracoda Group (Baracoda AI Labs)
"""Train a Baracoda FR tokenizer with the v1.1/v1.2 settings on data/*.jsonl.
Usage: python scripts/train.py [--vocab 131072] [--out tokenizers/baracoda-fr-v1_2.json]
--vocab is the TOTAL size including the 4 special tokens (v1.2: 131,072 = 131,068 ordinary + 4 special;
Tekken: 131,072 = 130,072 ordinary + 1,000 special). --vocab 130076 gives Tekken's ordinary budget."""
import argparse, json, time
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from common import DATA, ROOT

SOURCES = ["fr_web", "en_web", "europarl_fr", "europarl_en", "python"]
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+['’](?=\p{L})|[^\r\n\p{L}\p{N}]?\p{L}+"
         r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def build():
    t = Tokenizer(models.BPE())
    t.normalizer = normalizers.NFC()
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(REGEX), behavior="isolated"),
                                               pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    return t


def documents():
    for s in SOURCES:
        with open(DATA / f"{s}.jsonl", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=131072)
    ap.add_argument("--out", default=str(ROOT / "tokenizers" / "baracoda-fr-v1_2.json"))
    a = ap.parse_args()
    t, t0 = build(), time.time()
    t.train_from_iterator(documents(), trainers.BpeTrainer(vocab_size=a.vocab, min_frequency=2, special_tokens=SPECIAL,
                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    t.save(a.out)
    print(f"vocab {t.get_vocab_size()} ; {time.time() - t0:.0f} s -> {a.out}")
