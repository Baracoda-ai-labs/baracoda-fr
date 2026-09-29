# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baracoda Group (Baracoda AI Labs)
"""Shared helpers: paths, normalisation, corpus loading. All paths are relative to the repository root."""
import json, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "corpora.json"
CORPORA = ROOT / "corpora"
RESULTS = ROOT / "results"
TOKENIZERS = ROOT / "tokenizers"
DATA = ROOT / "data"


def norm(s: str) -> str:
    """NFC normalisation and removal of leading/trailing whitespace (the protocol's text unit)."""
    return unicodedata.normalize("NFC", s).strip()


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def load(role=None) -> dict:
    """Load prepared corpora (name -> list of sentences). Fails loudly if a corpus is missing or empty."""
    out = {}
    for name, c in config().items():
        if role and c["role"] != role:
            continue
        p = CORPORA / f"{safe(name)}.json"
        if not p.exists():
            raise SystemExit(f"Missing corpus {name}: run `python scripts/prepare_corpora.py` first")
        sents = json.loads(p.read_text(encoding="utf-8"))
        if not sents:
            raise SystemExit(f"Corpus {name} is empty")
        out[name] = sents
    return out


def safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")
