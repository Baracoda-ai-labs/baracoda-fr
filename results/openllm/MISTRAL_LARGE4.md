# Mistral Large 4 tokenizer vs earlier tokenizers - NTREX-128

Run 2026-10-29 17:48.

| Tokenizer | Vocab | English tokens | German / EN | French / EN | Chinese / EN | Mean, 23 EU languages | Worst | Total, 23 languages |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Mistral Large 4 | 204800 | 54,188 | 1.239 | 1.299 | 1.239 | 1.585 | ell 2.31 | 1,975,324 |
| Mistral Large 3 | 131072 | 53,975 | 1.293 | 1.310 | 1.396 | 1.605 | mlt 2.23 | 1,992,630 |
| Mistral Medium 3.5 | 131072 | 53,975 | 1.293 | 1.310 | 1.396 | 1.605 | mlt 2.23 | 1,992,630 |
| Mistral Small 4 | 131072 | 53,975 | 1.293 | 1.310 | 1.396 | 1.605 | mlt 2.23 | 1,992,630 |
| Mistral Tekken (Nemo) | 131072 | 53,975 | 1.293 | 1.310 | 1.396 | 1.605 | mlt 2.23 | 1,992,630 |
| Alibaba Qwen3 | 151665 | 53,052 | 1.542 | 1.559 | 1.025 | 1.980 | ell 4.81 | 2,415,479 |
| DeepSeek V3 | 128815 | 52,498 | 1.500 | 1.548 | 0.952 | 1.817 | ell 2.62 | 2,193,453 |
| BSC Salamandra | 256000 | 57,514 | 1.173 | 1.309 | 3.858 | 1.176 | mlt 1.35 | 1,556,121 |
| OpenAI o200k | 200019 | 51,860 | 1.303 | 1.357 | 1.259 | 1.570 | ell 2.14 | 1,873,043 |
