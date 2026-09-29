# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baracoda Group (Baracoda AI Labs)
"""Collecte des données d'entraînement avec décontamination.

Usage : python scripts/collect_training.py <source>   (fr_web | en_web | europarl_fr | europarl_en | python)

Chaque document est normalisé NFC. Un document est écarté s'il contient, telle quelle,
une phrase de plus de 40 caractères issue d'un banc de test. La lecture s'arrête dès que
le volume CONSERVÉ atteint la cible (1 Mo = 1 000 000 octets UTF-8).
Sortie : data/<source>.jsonl (documents conservés) + data/<source>.stats.json
"""
import json
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import ahocorasick

from common import DATA, load

MO = 1_000_000

CIBLES = {"fr_web": 300, "en_web": 300, "europarl_fr": 50, "europarl_en": 50, "python": 60}
EUROPARL = "https://media.githubusercontent.com/media/dksifoua/NMT/master/data/europarl-v7.fr-en."


def automate():
    a = ahocorasick.Automaton()
    # Reproduces v1.2 exactly: only the development benchmarks were known when v1.2 was trained.
    # The final-test corpora did not exist yet; their overlap is measured afterwards by contamination.py.
    phrases = {p for liste in load("dev").values() for p in liste if len(p) > 40}
    for p in phrases:
        a.add_word(p, p)
    a.make_automaton()
    return a, len(phrases)


def docs_hf(depot, config):
    from datasets import load_dataset

    for ligne in load_dataset(depot, name=config, split="train", streaming=True):
        yield ligne["text"]


def docs_europarl(langue):
    with urllib.request.urlopen(EUROPARL + langue) as r:
        for ligne in r:
            yield ligne.decode("utf-8").rstrip("\n")


def docs_python():
    import os
    stdlib = Path(os.environ.get("PYTHON_STDLIB", "/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/lib/python3.14"))
    files = sorted(p for p in stdlib.rglob("*.py") if "site-packages" not in p.parts)
    if not files:
        raise SystemExit(f"No .py files under {stdlib}: set PYTHON_STDLIB")
    for chemin in files:
        try:
            yield chemin.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # quelques fichiers de test volontairement non UTF-8
            print(f"ignoré (non UTF-8) : {chemin}", flush=True)
            yield ""


def sources(nom):
    return {
        "fr_web": lambda: docs_hf("HuggingFaceFW/fineweb-2", "fra_Latn"),
        "en_web": lambda: docs_hf("HuggingFaceFW/fineweb", "sample-10BT"),
        "europarl_fr": lambda: docs_europarl("fr"),
        "europarl_en": lambda: docs_europarl("en"),
        "python": docs_python,
    }[nom]()


def main(nom):
    DATA.mkdir(exist_ok=True)
    a, n_phrases = automate()
    cible = CIBLES[nom] * MO
    lus = gardes = ecartes = octets = octets_ecartes = vides = 0
    t0 = time.time()
    with open(DATA / f"{nom}.jsonl", "w", encoding="utf-8") as f:
        for doc in sources(nom):
            lus += 1
            doc = unicodedata.normalize("NFC", doc)
            if not doc.strip():
                vides += 1
                continue
            taille = len(doc.encode("utf-8"))
            if next(a.iter(doc), None) is not None:
                ecartes += 1
                octets_ecartes += taille
                continue
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            gardes += 1
            octets += taille
            if lus % 20000 == 0:
                print(f"{nom}: {octets / MO:.1f} Mo, {gardes} gardés, {ecartes} écartés", flush=True)
            if octets >= cible:
                break
    stats = {
        "source": nom, "cible_mo": CIBLES[nom], "octets_conserves": octets,
        "docs_lus": lus, "docs_conserves": gardes, "docs_vides_ignores": vides,
        "docs_ecartes_decontamination": ecartes, "octets_ecartes": octets_ecartes,
        "cible_atteinte": octets >= cible, "phrases_test_gt40": n_phrases,
        "secondes": round(time.time() - t0),
    }
    (DATA / f"{nom}.stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
