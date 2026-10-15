# OpenLLM-France tokenizers vs our references — NTREX-128

Run 2026-10-15 04:14. Identical tokenizers: {'OpenLLM Luciole-8B': 'OpenLLM Luciole-1B', 'OpenLLM Luciole-23B': 'OpenLLM Luciole-1B'}.

| Tokenizer | Vocab | French / English (95% CI) | Chinese / English | Mean, 23 EU languages | Worst | Gini (24, absolute) | Total 23 vs V4 parity |
|---|---:|---:|---:|---:|---|---:|---:|
| CroissantLLM | 32,000 | 1.163 (1.153–1.173) | 0.942 | 2.071 | ell 4.47 | 0.138 | — |
| Baracoda FR v1.2 (French only) | 131,072 | 1.198 (1.186–1.208) | 3.025 | 2.224 | ell 5.03 | 0.166 | — |
| OpenLLM Luciole-1B | 128,000 | 1.215 (1.202–1.227) | 4.028 | 1.921 | ell 4.70 | 0.186 | — |
| OpenLLM Lucie-7B | 65,024 | 1.249 (1.237–1.262) | 3.952 | 1.921 | ell 3.75 | 0.152 | — |
| BSC Salamandra | 256,000 | 1.309 (1.295–1.323) | 3.858 | 1.176 | mlt 1.35 | 0.051 | — |
| Mistral Tekken | 131,072 | 1.310 (1.295–1.324) | 1.396 | 1.605 | mlt 2.23 | 0.102 | — |
| OpenAI o200k | 200,019 | 1.357 (1.342–1.371) | 1.259 | 1.570 | ell 2.14 | 0.094 | — |
