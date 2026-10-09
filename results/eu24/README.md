# The EU language tax: token premiums of the 23 non-English official EU languages

**Baracoda AI Labs · October 2026 · results and documentation: CC BY 4.0 · scripts: Apache 2.0**

Data behind the research note *Europe pays a language tax on AI* (October 2026), a follow-up to T. Serval, *The Invisible Language Tax*, [arXiv:2609.39001](https://arxiv.org/abs/2609.39001).

## What is measured

The **token premium** of a language is its token count divided by the English token count for the same content, within the same tokenizer (1.50 = 50 % more tokens than English). Because a tokenizer can improve parity by being worse at English, absolute counts are also given.

- **Main corpus:** NTREX-128 (1,997 English news sentences from 123 documents, professionally translated), pinned commit `468c6b69c7f6a75d31d4743d9daba2af566cc18d`. All 23 non-English official EU languages, including Irish and Maltese.
- **Robustness corpora:** FLORES+ devtest (1,012 sentences, 281 articles) and the Universal Declaration of Human Rights (udhr2 full texts).
- **Tokenizers:** OpenAI o200k, Meta Llama 3.1, Alibaba Qwen3, DeepSeek V3, Google Gemma 3, Mistral Tekken (Nemo), BSC Salamandra, EuroLLM, Anthropic Claude (NTREX only), plus Baracoda FR v1.2 as a reference.
- **Uncertainty:** 95 % intervals by bootstrap over documents (2,000 draws, seed 0). Paired intervals for differences between tokenizers are not computed yet.
- Texts are NFC-normalised and counted without special tokens.

## Files

| File | Content |
|---|---|
| `eu24_ntrex.json` | NTREX: six commercial tokenizers + Baracoda FR v1.2. Per language: `[premium, CI low, CI high, token count]`; English count; mean, max, Gini; character ratio to English |
| `eu24_ntrex_hf.json` | NTREX: Salamandra-7B, EuroLLM-9B, and Mistral Nemo loaded a second way (check, identical counts) |
| `eu24_flores.json` | FLORES+ devtest, same structure |
| `eu24_udhr.json` | UDHR, same structure (a single document per language: orders of magnitude only) |
| `eu24_claude.json` | NTREX counts for Claude, via Anthropic's token-counting endpoint, 7 October 2026 |
| `eu24_claude_counts.json` | Per-sentence Claude counts, overhead calibration (7 tokens per message), request count (47,949) |
| `eu24_claude_models.json` | Check sample (200 sentences, English, French, Greek, Maltese): Opus 5.5, Sonnet 5.5 and Fable 5.1 give identical counts |

Scripts are in `scripts/eu24/`:

- `eu24_container_script.py` (commercial tokenizers, NTREX) loads the tokenizer files distributed in the `@lenml/tokenizer-*` npm packages. Its paths are those of the original run: edit `ND` and `T` before rerunning. For OpenAI it uses the o200k vocabulary as packaged for gpt-oss, which encodes ordinary text exactly as o200k_base.
- `eu24_hf.py` covers Salamandra and EuroLLM.
- `eu24_flores.py` covers FLORES+, which is gated on Hugging Face: accept its terms first.
- `eu24_claude.py` covers Claude and reads `ANTHROPIC_API_KEY` from the environment.
- The UDHR run used the same counting code in a temporary environment. Its script is not included yet.

## Headline numbers (NTREX, 23 languages)

| Tokenizer | Mean premium | English tokens | Tokens, 23 languages |
|---|---:|---:|---:|
| BSC Salamandra | 1.18 | 57,514 | 1,556,121 |
| EuroLLM | 1.35 | 56,708 | 1,760,957 |
| OpenAI o200k | 1.57 | 51,860 | 1,873,043 |
| Google Gemma 3 | 1.58 | 53,048 | 1,922,837 |
| Anthropic Claude | 1.59 | 83,988 | 3,075,686 |
| Mistral Tekken | 1.61 | 53,975 | 1,992,630 |
| DeepSeek V3 | 1.82 | 52,498 | 2,193,453 |
| Meta Llama 3 | 1.84 | 52,235 | 2,215,551 |
| Alibaba Qwen3 | 1.98 | 53,052 | 2,415,479 |

## Limits

- **Tokens, not invoices:** per-token prices differ between providers.
- **Tokenizers, not models:** model quality per language is not measured.
- **Translations, not native text.**
- **Snapshot:** versions available on 1 October 2026 (7 October for Claude). GPT-6 Astra is not covered because its tokenizer is not published. Anthropic describes its counting endpoint as an estimate.

---

# Transplant test: Baracoda FR on Qwen3-0.6B-Base (`results/transplant_qwen3_0.6b/`)

**Question:** can a more compact French tokenizer be fitted onto an existing open model?

**Model:** Qwen/Qwen3-0.6B-Base, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.

**Arms:**

| Arm | Description |
|---|---|
| A | Original model, no training |
| B | Original tokenizer, continued training |
| C′ | Baracoda FR v1.2 transplanted, same training |
| C | Baracoda FR v2 (SuperBPE, 131,073 entries) transplanted, same training |

**Initialisation:** tokens shared with Qwen are copied exactly. The others are set to the mean of their Qwen decomposition, a simple baseline rather than TokAlign. This reconstructs 57 % of rows for v1.2 and 61 % for v2. Input and output embeddings are tied.

**Training:** one pass over 11.1 MB of French banking, legal and audit text (2,625 documents, the same documents in the same order for B, C and C′). Settings:

- sequence length 512, 16 sequences per step;
- learning rate 3e-5;
- for C and C′, the first ~1 MB trains the embeddings only.

Full settings are in `config.yaml`.

**Metric:** bits per byte on the validation split, which does not depend on the tokenizer. Professional texts = banking, legal and audit.

| Arm | Bits per byte, all (95 % CI) | Bits per byte, professional texts (95 % CI) | Validation tokens (all) |
|---|---|---|---:|
| A | 1.211 [1.006, 1.431] | 0.864 [0.832, 0.902] | 1,027,312 |
| B, final | **0.771** [0.685, 0.865] | **0.598** [0.562, 0.636] | 1,027,312 |
| C′ before training | 2.153 | 1.702 | 799,254 |
| C′, final | 0.880 [0.790, 0.980] | 0.682 [0.638, 0.729] | 799,254 |
| C before training | 2.337 | 1.828 | 703,484 |
| C, final | 0.963 [0.880, 1.057] | 0.786 [0.738, 0.835] | 703,484 |

Tokens on the professional texts only (1,216,426 bytes): A and B 341,883; C′ 250,493; C 186,724.

Bits per byte: validation split into passages of at most 2,000 UTF-8 bytes at line ends, identical for all arms; each passage is scored alone after an `<|endoftext|>` that is context only; bits per byte = Σ token NLL ÷ ln 2 ÷ UTF-8 bytes, aggregated per domain. Special tokens: 4 (v1.2) and 1 (v2), handled separately from the copied and reconstructed rows.

**Reading:**

- Both transplanted models end better than the untrained original (A) on this measure, but behind the control trained with its own tokenizer (B).
- Of the two variants, the more compact one shows the larger gap; this experiment cannot isolate why.

Because the compact tokenizers cut the same text into fewer tokens, they received fewer training steps:

| Arm | Training tokens | Steps | Embedding-only steps |
|---|---:|---:|---:|
| B | 3,819,050 | 551 | 0 |
| C′ | 3,001,556 | 454 | 42 |
| C | 2,607,440 | 410 | 38 |

**Limits:** one small model, one pass, one run, a simple initialisation, no downstream tasks, and no frequency-weighted share of new tokens. The training code (`eval-metier`) is not released yet. `eval_val.json` holds all arms and checkpoints, by domain, with intervals.
