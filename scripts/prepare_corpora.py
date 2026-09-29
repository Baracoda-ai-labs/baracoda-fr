# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baracoda Group (Baracoda AI Labs)
"""Download every evaluation corpus at a pinned commit, extract sentences, normalise, write corpora/*.json
and manifests/corpora.sha256.json (plus corpora/*.groups.json
document ids where the corpus provides them). Fails if a download is empty or a corpus has no sentence."""
import hashlib, json, urllib.request
from common import CORPORA, ROOT, config, norm, safe

RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{file}"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    if not data:
        raise SystemExit(f"Empty download: {url}")
    return data


def sentences(text: str, fmt: str) -> list:
    lines = text.splitlines()
    if fmt == "conllu":
        lines = [l[len("# text = "):] for l in lines if l.startswith("# text = ")]
    return [s for s in (norm(l) for l in lines) if s]


def main():
    CORPORA.mkdir(exist_ok=True)
    manifest = {}
    for name, c in config().items():
        sents = []
        for f in sorted(c["files"]):
            raw = fetch(RAW.format(repo=c["repo"], commit=c["commit"], file=f))
            manifest[f"{c['repo']}@{c['commit']}/{f}"] = hashlib.sha256(raw).hexdigest()
            sents += sentences(raw.decode("utf-8"), c["format"])
        if not sents:
            raise SystemExit(f"No sentence extracted for {name}")
        (CORPORA / f"{safe(name)}.json").write_text(json.dumps(sents, ensure_ascii=False), encoding="utf-8")
        if "groups" in c:  # document id of each sentence (NTREX: DOCUMENT_IDS.tsv, 123 documents)
            raw = fetch(RAW.format(repo=c["repo"], commit=c["commit"], file=c["groups"]))
            manifest[f"{c['repo']}@{c['commit']}/{c['groups']}"] = hashlib.sha256(raw).hexdigest()
            groups = [l.strip() for l in raw.decode("utf-8").splitlines() if l.strip()]
            if len(groups) != len(sents):
                raise SystemExit(f"{name}: {len(groups)} document ids for {len(sents)} sentences")
            (CORPORA / f"{safe(name)}.groups.json").write_text(json.dumps(groups), encoding="utf-8")
        print(f"{name:22s} {len(sents):6d} sentences ({c['role']})")
    (ROOT / "manifests").mkdir(exist_ok=True)
    (ROOT / "manifests" / "corpora.sha256.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
