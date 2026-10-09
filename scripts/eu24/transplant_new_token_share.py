"""Part pondérée (par occurrences) des nouveaux tokens : pour Baracoda FR v1.2 et v2-t100k-bytes, segmente le split de
validation du banc eval-metier en passages (step2.passages, ≤ 2 000 octets, identiques à la mesure bits/octet) et compte
la part des occurrences de tokens SANS correspondance exacte dans le vocabulaire de Qwen3-0.6B-Base, avec la règle de
transfer_report.json : « exact » = même chaîne ByteLevel dans le vocabulaire du modèle Qwen (hors tokens ajoutés),
sinon « décomposé » ; tokens spéciaux comptés à part. L'<|endoftext|> de contexte placé avant chaque passage n'est pas
compté (texte seulement). Tous les textes, puis textes métier seuls (banque, droit_contrats, audit_compta).
Sortie : new_token_share.json. Environnement : ~/baracoda-fr/eval-metier/.venv"""
import json
import sys
from collections import Counter
from pathlib import Path

ICI = Path(__file__).resolve().parent
EM = ICI.parent.parent / "eval-metier"
sys.path.insert(0, str(EM))
from baracoda_eval import config, step2  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

METIER = {"banque", "droit_contrats", "audit_compta"}

if __name__ == "__main__":
    cfg = config.load(EM / "configs" / "qwen_06b.yaml")
    items = step2.passages(step2.documents(cfg, "val"), cfg["eval"]["max_passage_bytes"])
    qwen = Tokenizer.from_file(str(step2.base_dir(cfg) / "tokenizer.json"))
    qv = qwen.get_vocab(with_added_tokens=False)
    out = {"split": "val", "passages": len(items), "bytes": sum(len(i["text"].encode()) for i in items),
           "rule": "exact = même chaîne dans le vocabulaire du modèle Qwen3 (hors tokens ajoutés) ; sinon décomposé",
           "tokenizers": {}}
    for arm, label in (("Cprime", "Baracoda FR v1.2"), ("C", "Baracoda FR v2-t100k-bytes")):
        tok, eos, path = step2.load_tokenizer(cfg, arm)
        specials = {a.content for a in tok.get_added_tokens_decoder().values()}
        vocab = tok.get_vocab(with_added_tokens=True)
        types = Counter("special" if t in specials else "exact" if t in qv else "new" for t in vocab)
        res = {"sha256": config.sha256_file(path), "vocabulary": {**types, "size": len(vocab),
               "new_share_of_types": types["new"] / (types["new"] + types["exact"])}}
        for scope, sel in (("all", items), ("metier", [i for i in items if i["domain"] in METIER])):
            c = Counter()
            for e in tok.encode_batch([i["text"] for i in sel], add_special_tokens=False):
                for t in e.tokens:
                    c["special" if t in specials else "exact" if t in qv else "new"] += 1
            n = c["exact"] + c["new"]
            res[scope] = {"passages": len(sel), "bytes": sum(len(i["text"].encode()) for i in sel), "occurrences": n + c["special"],
                          "exact": c["exact"], "new": c["new"], "special": c["special"], "new_share_weighted": c["new"] / n}
        out["tokenizers"][label] = res
        print(f"{label:28s} types nouveaux {res['vocabulary']['new_share_of_types']:.1%} | occurrences nouvelles : "
              f"tout {res['all']['new_share_weighted']:.1%} ({res['all']['new']:,}/{res['all']['exact'] + res['all']['new']:,}), "
              f"métier {res['metier']['new_share_weighted']:.1%}")
    json.dump(out, open(ICI / "new_token_share.json", "w"), indent=1, ensure_ascii=False)
