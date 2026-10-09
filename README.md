# Baracoda FR v1.2 — a French-optimised tokenizer prototype and its evaluation

**Baracoda AI Labs · 29 September 2026 · research prototype · code and tokenizer: Apache 2.0 · results and documentation: CC BY 4.0**

Paper: T. Serval, The Invisible Language Tax: Token Premiums of French and Regional Languages in 2026 LLM Tokenizers, and a French-Optimized Prototype, Baracoda AI Labs, 2026. [arXiv:2609.39001](https://arxiv.org/abs/2609.39001)

Baracoda FR v1.2 is a byte-level BPE tokenizer with 131,072 entries (131,068 ordinary tokens + 4 special tokens), trained on 736 MB of French and English web text, Europarl and Python code. This repository contains the tokenizer, the evaluation corpora definitions (pinned commits), every script, and the results of the study *The Invisible Language Tax* (Baracoda AI Labs, 2026).

## Headline result (segmentation only)

On a final test of six Universal Dependencies corpora, defined in a protocol declared fixed before the test (`protocol/`):

| | Baracoda FR v1.2 | Mistral Tekken (Nemo) | Difference (95% block-bootstrap CI) |
|---|---:|---:|---|
| French (ParTUT, ParisStories, FQB) | 96,793 | 109,355 | **−11.49 %** [−11.86, −11.15] |
| English (ParTUT, LinES, GENTLE) | 184,163 | 191,165 | **−3.66 %** [−3.86, −3.48] |

These counts were reproduced independently (external audit, 29 September 2026) and regenerated from scratch on a fresh Python environment with `python run.py corpora eval mistral contamination` (identical tables, identical corpora manifest). Defensible statement:

> Baracoda FR v1.2 produces 11.49 % fewer tokens than Tekken-Nemo on the three French corpora evaluated and 3.66 % fewer on the three English corpora. The final test overlaps partly with the training data (below); excluding the overlapping sentences gives −11.44 % and −3.67 %. These are segmentation results: no gain in model quality, latency or cost has been demonstrated.

What this does **not** show: better model quality, lower latency, lower invoices, or general superiority. CroissantLLM (32k vocabulary) uses fewer tokens than v1.2 on spontaneous spoken French (ParisStories: 42,376 vs 43,281), and OpenAI o200k slightly fewer on English GENTLE (20,268 vs 20,317). v1.2 is also much worse than Tekken on other languages (Spanish +30 %, German +38 %, Chinese ×2.1 on NTREX).

## Answers to the external audit (29 September 2026)

| Audit point | Status in this version |
|---|---|
| Contamination check not reproducible (`run_final.py` contained a placeholder loop) | Replaced by `scripts/contamination.py`; `results/contamination.json` gives per-corpus denominators, the id and SHA-256 of each overlapping sentence, and results with and without them. 503 sentence occurrences (501 distinct) of 11,198 eligible (> 40 characters); French ParTUT 175/948, English ParTUT 314/1,928, all others ≤ 8. Clean results: French −11.44 %, English −3.67 %. The training data were decontaminated only against the development benchmarks, because the final-test corpora were chosen afterwards. |
| Controlled experiment measured characters, not bytes; interleaved subsamples; effective vocabulary not recorded | Redone in `scripts/ablation.py`: UTF-8 byte budgets, whole sentences, three random shuffles (seeds 1–3), bytes and effective vocabulary recorded. With a 131,072 target the effective vocabulary only reached 56.8k–87.1k (data too small), so the main run uses 50,000, reached in every condition. Results below. |
| Missing modules, absolute paths, no environment | All code is in `scripts/` with paths relative to the repository; `requirements.txt` (minimal, pinned) and `requirements.lock.txt` (full lock of the original run); `python run.py corpora eval` regenerates `results/TABLES.md`; scripts fail on empty or missing corpora. |
| "Pre-registered" | Wording changed to **protocol declared fixed before the test**: the SHA-256 values prove file identity, not the date. Future protocols will have their hashes deposited publicly before computation. |
| Vocabulary budget (Tekken reserves 1,000 special tokens, v1.2 only 4) | Control `tokenizers/controls/baracoda-fr-v1_2-equal-ordinary-budget.json` (130,072 ordinary tokens, as Tekken): French −11.48 %, English −3.64 % on the final test. The 996 extra slots account for 0.01–0.03 points. |
| Lossless claim | Exact restitution holds for NFC-normalised text. An NFD input (e + combining accent) is returned in NFC form. A literal `<|im_start|>` in input is recognised as a special token and dropped by `decode()` unless `skip_special_tokens=False`. See `results/lossless_behaviour.json`. |
| Claude overhead | Measured by difference (7 tokens, seven probes, `results/archived/claude_calibration_2026-09-29.json`); counts corrected. `scripts/claude_counts.py` now stores calibration and timestamps and reads the key from `ANTHROPIC_API_KEY`. |
| Mistral v13/v15 identity not reproducible | `scripts/mistral_versions.py` (Python, fresh environment, 29 September 2026): Tekken files of Ministral 3 (v13), Mistral Large 3 (v13), Mistral Medium 3.5 (v15) and Mistral Small 4 (v15, same file as Medium 3.5) give counts identical to Nemo on all 15 corpora (`results/mistral_versions.json`). |
| CroissantLLM on GENTLE: 22,676 (audit) vs 22,784 (our browser count) | Our value came from transformers.js in a browser. Re-run in Python on a fresh environment: 22,676, as the audit; all tables use this value. |
| NTREX intervals used blocks of 10 sentences, although document boundaries are published (`DOCUMENT_IDS.tsv`, 123 documents) | `prepare_corpora.py` now downloads `DOCUMENT_IDS.tsv` at the pinned commit and `evaluate.py` resamples NTREX by document. v1.2 vs Tekken on NTREX: French −12.88 % [−13.31, −12.42], English −4.67 % [−5.04, −4.31]. Document-level premium intervals for the seven tokenizers of the main measurement: `results/ntrex_document_bootstrap.json` (widest ±2.3 points). |
| NTREX scope | 129 files: English source + 128 references, of which 3 English variants; 124 of the 125 non-English references are used (the second Spanish reference `ref-2.spa` is excluded). They are language varieties, not 124 distinct languages. |
| README and licence | This README. Code and tokenizer files under Apache 2.0, results and documentation under CC BY 4.0 (see Licence). |

## Controlled experiment (French share of training data, bytes)

Europarl v7 fr-en, 48,000,000 bytes, BPE with Qwen3's pre-tokenizer, UD PUD French/English, three random subsamples. Premium = French tokens / English tokens.

| French share | Premium, vocabulary 50,000 (reached in all runs) | Premium, target 131,072 (effective 56.8k–87.1k) |
|---:|---|---|
| 0 % | 1.814–1.830 | 1.819–1.829 |
| 25 % | 1.167 | 1.154–1.159 |
| 50 % | 1.131–1.133 | 1.133–1.136 |
| 75 % | 1.097–1.099 | 1.110–1.115 |
| 100 % | 0.743–0.747 | 0.767–0.776 |

A modest share of French captures most of the gain; beyond it, French improves little while English degrades. This is a descriptive result on parliamentary text at one data size.

## October 2026 follow-up: the 23 official EU languages

Token premiums of the 23 non-English official EU languages on nine tokenizers (NTREX-128, checked on FLORES+ and the UDHR), and a test of transplanting Baracoda FR onto an existing model: [`results/eu24/README.md`](results/eu24/README.md).

## Repository layout

```
config/corpora.json          evaluation corpora: repository, pinned commit, files, language, role (dev / final)
protocol/                    final-test protocol and its SHA-256 (declared fixed before the test)
tokenizers/                  baracoda-fr-v1.json, -v1_1.json, -v1_2.json (+ controls/)
scripts/                     prepare_corpora, collect_training, train, evaluate, contamination, ablation,
                             mistral_versions, claude_counts (all relative paths)
manifests/                   SHA-256 of tokenizers, corpora downloads and v1.2 training data; collection statistics
results/                     TABLES.md (generated), counts.json, contamination.json, ablations, controls;
                             archived/ = earlier or browser-based results kept for traceability
                             (archived/ablation*.json are the superseded character-based runs)
results/eu24/                October 2026 follow-up: token premiums of the 23 non-English official EU
                             languages on 9 tokenizers (README.md there); scripts in scripts/eu24/
results/transplant_qwen3_0.6b/  Baracoda FR transplanted onto Qwen3-0.6B-Base (see results/eu24/README.md)
```

## Reproduce

```
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python run.py corpora eval   # corpora at pinned commits + all tokenizer counts + hypotheses + results/TABLES.md
python run.py mistral        # current Mistral tokenizer files vs Tekken-Nemo
python run.py data train     # optional: re-collect the 736 MB of training data and retrain v1.2
python run.py contamination  # needs data/
python run.py controls       # overlap-removed and equal-budget controls -> results/CONTROLS.md (needs eval + results/contamination.json)
```

Training data are not redistributed (FineWeb, FineWeb-2 and Europarl have their own licences); `manifests/training_data_v1_2.sha256` identifies the exact files used. The code part is the Python 3.14 standard library (set `PYTHON_STDLIB`).

## Technical settings

Byte-level BPE (Hugging Face `tokenizers` 0.23.2), NFC normaliser, `min_frequency=2`, special tokens `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|pad|>`. Pre-tokenisation: Qwen3-style regular expression with French elisions detached (`l'`, `qu'`) and digits grouped by up to three.

## History

v1: Europarl + code. v1.1: same data, elisions and digit grouping. v1.2: web data added (FineWeb-2 French, FineWeb English). Settings for v1 and v1.1 were chosen by looking at NTREX and UD PUD; those and the other UD sets in `config/corpora.json` with role `dev` are development and monitoring benchmarks. The tokenizer v1.2 is frozen (SHA-256 `8d9fdb7b…464d`) and will not be tuned on the final-test results.

## Licence

| Part | Licence |
|---|---|
| Code (`scripts/`, `run.py`) and tokenizer files (`tokenizers/`) | Apache License 2.0 (`LICENSE`, `NOTICE`) |
| Results, protocol, manifests and documentation (`results/`, `protocol/`, `manifests/`, `config/`, this README) | Creative Commons Attribution 4.0 International (`LICENSE-CC-BY-4.0.txt`) |

Suggested attribution: *T. Serval, The Invisible Language Tax: Token Premiums of French and Regional Languages in 2026 LLM Tokenizers, and a French-Optimized Prototype, Baracoda AI Labs, 2026, arXiv:2609.39001.*

Third-party material is **not** included and keeps its own licence: training data (FineWeb, FineWeb-2, Europarl, Python standard library), evaluation corpora (NTREX-128, Universal Dependencies treebanks, some of which are non-commercial), the Mistral Tekken and CroissantLLM tokenizer files. The scripts download them from their sources at pinned versions; only derived counts, digests and statistics are distributed here.

The Apache licence does not grant permission to use the Baracoda names or logos (section 6). This repository is provided "as is", without warranty.

## Citation

```bibtex
@misc{serval2026languagetax,
  title         = {The Invisible Language Tax: Token Premiums of French and Regional Languages in 2026 LLM Tokenizers, and a French-Optimized Prototype},
  author        = {Serval, Thomas},
  year          = {2026},
  eprint        = {2609.39001},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.39001},
  note          = {Baracoda AI Labs. Code and data: https://github.com/Baracoda-ai-labs/baracoda-fr}
}
```
