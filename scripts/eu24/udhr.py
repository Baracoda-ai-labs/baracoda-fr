"""Reproduit eu24/eu24_udhr.json : Déclaration universelle des droits de l'homme, textes udhr2 (corpus NLTK, textes
Unicode UDHR, publics), fichiers listés dans la clé `files` du JSON publié ; anglais : eng.txt.
Comptage : chaque ligne du fichier est normalisée NFC et débarrassée de ses espaces de bord, comptée sans tokens spéciaux ;
le compte d'une langue est la somme des comptes de ses lignes (même convention que les scripts NTREX/FLORES+ ; c'est la
seule variante qui retrouve les comptes publiés, voir udhr_check.json). Tokenizers et chargement : ceux de eu24_flores.py.
Source : https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr2.zip (téléchargé dans _data/).
Sorties : udhr_recount.json (même structure que le JSON publié) et udhr_check.json (comparaison exacte).
Environnement : ~/baracoda-fr/pack/.venv-audit."""
import io
import json
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

from recount import EU, FLORES_TOK

ICI = Path(__file__).resolve().parent
PUB = ICI.parent.parent / "eu24" / "eu24_udhr.json"
ZIP = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr2.zip"


def texts(files):
    d = ICI / "_data" / "udhr2"
    if not d.exists():
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(ZIP, timeout=120).read())).extractall(ICI / "_data")
    rd = lambda f: [unicodedata.normalize("NFC", l.strip()) for l in (d / f"{f}.txt").read_text(encoding="utf-8").splitlines()]
    return {"eng": rd("eng"), **{k: rd(f) for k, f in files.items()}}


if __name__ == "__main__":
    pub = json.load(open(PUB))
    T = texts(pub["files"])
    out = {"corpus": pub["corpus"], "languages": pub["languages"], "files": pub["files"], "tokenizers": {}}
    check, all_ok = {}, True
    for name, make in FLORES_TOK.items():
        f = make()
        cnt = {k: sum(f([l for l in X if l]) or [0]) for k, X in T.items()}
        r = {"english_tokens": cnt["eng"], "premium": {k: [cnt[k] / cnt["eng"], cnt[k]] for k in EU}}
        v = [r["premium"][k][0] for k in EU]
        r.update(mean=sum(v) / len(v), max=max(v), total=sum(cnt[k] for k in EU))
        out["tokenizers"][name] = r
        p = pub["tokenizers"][name]
        diff = {k: cnt[k] - (p["english_tokens"] if k == "eng" else p["premium"][k][1]) for k in cnt
                if cnt[k] != (p["english_tokens"] if k == "eng" else p["premium"][k][1])}
        same_floats = all(abs(r[x] - p[x]) < 1e-12 for x in ("mean", "max")) and r["total"] == p["total"]
        check[name] = {"counts_identical": not diff, "diff": diff, "mean_max_total_identical": same_floats}
        all_ok &= (not diff) and same_floats
        print(f"{name:18s} {'IDENTIQUE' if not diff and same_floats else 'ÉCART ' + str(diff)}", flush=True)
    check["_all_identical"] = all_ok
    json.dump(out, open(ICI / "udhr_recount.json", "w"), indent=1, ensure_ascii=False)
    json.dump(check, open(ICI / "udhr_check.json", "w"), indent=1, ensure_ascii=False)
    print("DUDH :", "tous les comptes identiques au JSON publié" if all_ok else "ÉCARTS (voir udhr_check.json)")
