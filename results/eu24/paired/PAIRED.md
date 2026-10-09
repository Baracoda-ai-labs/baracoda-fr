# Comparaisons appariées entre tokenizers (bootstrap sur les documents)

Généré par `paired.py` à partir des comptes par phrase de `recount.py` (totaux identiques aux JSON publiés  voir `recount_check.json`). 10 000 tirages, graine 0 ; mêmes documents tirés pour tous les tokenizers. Δ = tokens(X) / tokens(Y) − 1 ; IC 95 % par percentiles.

## NTREX-128 (123 documents, 1997 phrases)

### a) et c) Classement par total de tokens (23 langues) — 8 tokenizers exécutés localement

| Rang | Tokenizer | Tokens (23 langues) | Écart avec le suivant | IC 95 % | P(ordre conservé) |
|---:|---|---:|---:|---|---:|
| 1 | BSC Salamandra | 1 556 121 | +13,2 % | [+12,8 ; +13,5] | 1.0000 |
| 2 | EuroLLM | 1 760 957 | +6,4 % | [+6,0 ; +6,8] | 1.0000 |
| 3 | OpenAI o200k | 1 873 043 | +2,7 % | [+2,5 ; +2,9] | 1.0000 |
| 4 | Google Gemma 3 | 1 922 837 | +3,6 % | [+3,4 ; +3,9] | 1.0000 |
| 5 | Mistral Tekken | 1 992 630 | +10,1 % | [+9,8 ; +10,3] | 1.0000 |
| 6 | DeepSeek V3 | 2 193 453 | +1,0 % | [+0,9 ; +1,2] | 1.0000 |
| 7 | Meta Llama 3 | 2 215 551 | +9,0 % | [+8,9 ; +9,2] | 1.0000 |
| 8 | Alibaba Qwen3 | 2 415 479 | — | — | — |

Probabilité que l'ordre complet (total) soit l'ordre observé : **1.0000**.
Ordre par prime moyenne : BSC Salamandra < EuroLLM < OpenAI o200k < Google Gemma 3 < Mistral Tekken < DeepSeek V3 < Meta Llama 3 < Alibaba Qwen3 ; probabilité de l'ordre complet : **0.9778** ; paire adjacente la moins sûre : OpenAI o200k < Google Gemma 3 (P = 0.9778).

### a) et c) Classement par total de tokens (23 langues) — 9 tokenizers, avec Claude

| Rang | Tokenizer | Tokens (23 langues) | Écart avec le suivant | IC 95 % | P(ordre conservé) |
|---:|---|---:|---:|---|---:|
| 1 | BSC Salamandra | 1 556 121 | +13,2 % | [+12,8 ; +13,5] | 1.0000 |
| 2 | EuroLLM | 1 760 957 | +6,4 % | [+6,0 ; +6,8] | 1.0000 |
| 3 | OpenAI o200k | 1 873 043 | +2,7 % | [+2,5 ; +2,9] | 1.0000 |
| 4 | Google Gemma 3 | 1 922 837 | +3,6 % | [+3,4 ; +3,9] | 1.0000 |
| 5 | Mistral Tekken | 1 992 630 | +10,1 % | [+9,8 ; +10,3] | 1.0000 |
| 6 | DeepSeek V3 | 2 193 453 | +1,0 % | [+0,9 ; +1,2] | 1.0000 |
| 7 | Meta Llama 3 | 2 215 551 | +9,0 % | [+8,9 ; +9,2] | 1.0000 |
| 8 | Alibaba Qwen3 | 2 415 479 | +27,3 % | [+26,7 ; +28,0] | 1.0000 |
| 9 | Anthropic Claude | 3 075 686 | — | — | — |

Probabilité que l'ordre complet (total) soit l'ordre observé : **1.0000**.
Ordre par prime moyenne : BSC Salamandra < EuroLLM < OpenAI o200k < Google Gemma 3 < Anthropic Claude < Mistral Tekken < DeepSeek V3 < Meta Llama 3 < Alibaba Qwen3 ; probabilité de l'ordre complet : **0.9122** ; paire adjacente la moins sûre : Google Gemma 3 < Anthropic Claude (P = 0.9649).

### a) Salamandra contre chacun des autres (total, 23 langues)

| Contre | Δ Salamandra | IC 95 % |
|---|---:|---|
| EuroLLM | -11,6 % | [-11,9 ; -11,3] |
| OpenAI o200k | -16,9 % | [-17,4 ; -16,4] |
| Meta Llama 3 | -29,8 % | [-30,3 ; -29,2] |
| Alibaba Qwen3 | -35,6 % | [-36,1 ; -35,0] |
| DeepSeek V3 | -29,1 % | [-29,6 ; -28,5] |
| Google Gemma 3 | -19,1 % | [-19,6 ; -18,6] |
| Mistral Tekken | -21,9 % | [-22,4 ; -21,4] |
| Anthropic Claude | -49,4 % | [-49,8 ; -49,0] |

### b) Par langue : tokenizer européen contre le tokenizer d'entreprise le moins coûteux

Identité du tokenizer d'entreprise fixée sur les comptes observés (comme le tableau 3) ; dernière colonne : IC contre le minimum des six recalculé à chaque tirage.

| Langue | Entreprise | Tokens | Δ Salamandra | IC 95 % | Δ EuroLLM | IC 95 % | IC Salamandra (min. retiré) |
|---|---|---:|---:|---|---:|---|---|
| Greek | OpenAI o200k | 111 025 | -30,0 % | [-30,5 ; -29,5] | -11,2 % | [-11,5 ; -10,8] | [-30,5 ; -29,5] |
| Latvian | OpenAI o200k | 97 997 | -30,0 % | [-30,7 ; -29,3] | -14,5 % | [-15,1 ; -13,8] | [-30,7 ; -29,3] |
| Irish | OpenAI o200k | 94 194 | -17,9 % | [-18,7 ; -17,1] | -6,5 % | [-6,9 ; -6,1] | [-18,7 ; -17,1] |
| Maltese | OpenAI o200k | 93 878 | -17,0 % | [-17,8 ; -16,3] | -3,7 % | [-4,3 ; -3,1] | [-17,8 ; -16,3] |
| Lithuanian | OpenAI o200k | 90 460 | -28,8 % | [-29,7 ; -28,0] | -13,0 % | [-13,5 ; -12,4] | [-29,7 ; -28,0] |
| Hungarian | Mistral Tekken | 88 986 | -13,8 % | [-14,4 ; -13,2] | -6,4 % | [-6,8 ; -6,1] | [-14,4 ; -13,2] |
| Bulgarian | Google Gemma 3 | 87 975 | -21,5 % | [-22,1 ; -21,0] | -7,4 % | [-7,8 ; -7,0] | [-22,1 ; -21,0] |
| Polish | Google Gemma 3 | 85 408 | -16,9 % | [-17,4 ; -16,2] | -10,9 % | [-11,5 ; -10,3] | [-17,4 ; -16,2] |
| Slovak | Google Gemma 3 | 83 321 | -19,8 % | [-20,4 ; -19,2] | -9,0 % | [-9,4 ; -8,5] | [-20,4 ; -19,2] |
| Romanian | Google Gemma 3 | 83 126 | -14,3 % | [-14,7 ; -13,9] | -2,0 % | [-2,3 ; -1,7] | [-14,7 ; -13,9] |
| Finnish | OpenAI o200k | 79 687 | -22,0 % | [-22,7 ; -21,3] | -6,6 % | [-7,1 ; -6,0] | [-22,7 ; -21,3] |
| Slovenian | OpenAI o200k | 79 081 | -19,9 % | [-20,6 ; -19,3] | -5,0 % | [-5,5 ; -4,6] | [-20,6 ; -19,3] |
| Croatian | OpenAI o200k | 78 004 | -23,1 % | [-23,7 ; -22,5] | -5,5 % | [-5,9 ; -5,2] | [-23,7 ; -22,5] |
| Czech | Meta Llama 3 | 77 914 | -18,1 % | [-18,7 ; -17,5] | -1,2 % | [-2,0 ; -0,5] | [-18,7 ; -17,5] |
| Estonian | OpenAI o200k | 76 055 | -20,8 % | [-21,5 ; -20,1] | -5,7 % | [-6,2 ; -5,2] | [-21,5 ; -20,1] |
| French | OpenAI o200k | 70 384 | +7,0 % | [+6,6 ; +7,4] | +8,6 % | [+8,0 ; +9,2] | [+6,6 ; +7,4] |
| Danish | OpenAI o200k | 69 739 | -12,8 % | [-13,4 ; -12,2] | -1,5 % | [-1,9 ; -1,1] | [-13,4 ; -12,2] |
| Italian | Google Gemma 3 | 68 529 | -3,5 % | [-3,9 ; -3,1] | -2,6 % | [-2,9 ; -2,2] | [-3,9 ; -3,1] |
| Swedish | OpenAI o200k | 68 468 | -12,6 % | [-13,2 ; -11,9] | +0,1 % | [-0,3 ; +0,5] | [-13,2 ; -11,9] |
| German | OpenAI o200k | 67 598 | -0,2 % | [-0,6 ; +0,3] | +1,6 % | [+1,1 ; +2,1] | [-0,6 ; +0,3] |
| Dutch | OpenAI o200k | 66 499 | +0,8 % | [+0,5 ; +1,1] | +3,4 % | [+3,0 ; +3,7] | [+0,5 ; +1,1] |
| Spanish | OpenAI o200k | 65 103 | +0,7 % | [+0,3 ; +1,1] | +4,1 % | [+3,7 ; +4,5] | [+0,3 ; +1,1] |
| Portuguese | OpenAI o200k | 63 814 | -0,0 % | [-0,5 ; +0,4] | +2,9 % | [+2,5 ; +3,3] | [-0,5 ; +0,4] |

## FLORES+ devtest (281 documents, 1012 phrases)

### a) et c) Classement par total de tokens (23 langues) — 8 tokenizers exécutés localement

| Rang | Tokenizer | Tokens (23 langues) | Écart avec le suivant | IC 95 % | P(ordre conservé) |
|---:|---|---:|---:|---|---:|
| 1 | BSC Salamandra | 796 512 | +12,6 % | [+12,2 ; +13,0] | 1.0000 |
| 2 | EuroLLM | 896 868 | +8,2 % | [+7,8 ; +8,5] | 1.0000 |
| 3 | OpenAI o200k | 970 010 | +2,4 % | [+2,1 ; +2,6] | 1.0000 |
| 4 | Google Gemma 3 | 992 995 | +3,9 % | [+3,6 ; +4,1] | 1.0000 |
| 5 | Mistral Tekken | 1 031 280 | +10,4 % | [+10,1 ; +10,7] | 1.0000 |
| 6 | DeepSeek V3 | 1 138 273 | +1,7 % | [+1,6 ; +1,9] | 1.0000 |
| 7 | Meta Llama 3 | 1 158 155 | +9,3 % | [+9,1 ; +9,5] | 1.0000 |
| 8 | Alibaba Qwen3 | 1 266 095 | — | — | — |

Probabilité que l'ordre complet (total) soit l'ordre observé : **1.0000**.
Ordre par prime moyenne : BSC Salamandra < EuroLLM < OpenAI o200k < Google Gemma 3 < Mistral Tekken < DeepSeek V3 < Meta Llama 3 < Alibaba Qwen3 ; probabilité de l'ordre complet : **1.0000** ; paire adjacente la moins sûre : BSC Salamandra < EuroLLM (P = 1.0000).

### a) Salamandra contre chacun des autres (total, 23 langues)

| Contre | Δ Salamandra | IC 95 % |
|---|---:|---|
| EuroLLM | -11,2 % | [-11,5 ; -10,9] |
| OpenAI o200k | -17,9 % | [-18,4 ; -17,4] |
| Meta Llama 3 | -31,2 % | [-31,8 ; -30,7] |
| Alibaba Qwen3 | -37,1 % | [-37,6 ; -36,6] |
| DeepSeek V3 | -30,0 % | [-30,6 ; -29,5] |
| Google Gemma 3 | -19,8 % | [-20,2 ; -19,4] |
| Mistral Tekken | -22,8 % | [-23,2 ; -22,3] |

### b) Par langue : tokenizer européen contre le tokenizer d'entreprise le moins coûteux

Identité du tokenizer d'entreprise fixée sur les comptes observés (comme le tableau 3) ; dernière colonne : IC contre le minimum des six recalculé à chaque tirage.

| Langue | Entreprise | Tokens | Δ Salamandra | IC 95 % | Δ EuroLLM | IC 95 % | IC Salamandra (min. retiré) |
|---|---|---:|---:|---|---:|---|---|
| Greek | OpenAI o200k | 57 277 | -29,8 % | [-30,4 ; -29,3] | -11,3 % | [-11,7 ; -10,9] | [-30,4 ; -29,3] |
| Maltese | OpenAI o200k | 50 208 | -18,6 % | [-19,4 ; -17,8] | -4,2 % | [-4,8 ; -3,6] | [-19,4 ; -17,8] |
| Irish | OpenAI o200k | 49 671 | -19,4 % | [-20,0 ; -18,7] | -6,7 % | [-7,0 ; -6,3] | [-20,0 ; -18,7] |
| Latvian | OpenAI o200k | 48 725 | -32,5 % | [-33,1 ; -31,8] | -18,2 % | [-18,8 ; -17,7] | [-33,1 ; -31,8] |
| Lithuanian | OpenAI o200k | 45 014 | -30,3 % | [-31,0 ; -29,5] | -15,3 % | [-15,8 ; -14,8] | [-31,0 ; -29,5] |
| Hungarian | Mistral Tekken | 44 818 | -14,7 % | [-15,3 ; -14,0] | -7,4 % | [-7,8 ; -6,9] | [-15,3 ; -14,0] |
| Bulgarian | Google Gemma 3 | 43 921 | -20,7 % | [-21,2 ; -20,2] | -7,6 % | [-8,0 ; -7,3] | [-21,2 ; -20,2] |
| Slovak | Google Gemma 3 | 43 833 | -20,9 % | [-21,4 ; -20,3] | -11,0 % | [-11,4 ; -10,6] | [-21,4 ; -20,3] |
| Romanian | Google Gemma 3 | 42 702 | -14,5 % | [-15,0 ; -14,1] | -2,8 % | [-3,1 ; -2,5] | [-15,0 ; -14,1] |
| Finnish | OpenAI o200k | 41 945 | -22,6 % | [-23,3 ; -22,0] | -7,2 % | [-7,7 ; -6,7] | [-23,3 ; -22,0] |
| Czech | Meta Llama 3 | 40 864 | -20,7 % | [-21,4 ; -20,1] | -5,4 % | [-5,9 ; -4,8] | [-21,4 ; -20,1] |
| Polish | Google Gemma 3 | 40 850 | -16,8 % | [-17,3 ; -16,3] | -13,8 % | [-14,3 ; -13,4] | [-17,3 ; -16,3] |
| Slovenian | OpenAI o200k | 40 617 | -21,1 % | [-21,7 ; -20,4] | -7,1 % | [-7,5 ; -6,7] | [-21,7 ; -20,4] |
| Estonian | OpenAI o200k | 40 184 | -21,5 % | [-22,2 ; -20,8] | -7,3 % | [-7,8 ; -6,8] | [-22,2 ; -20,8] |
| Croatian | OpenAI o200k | 39 643 | -24,7 % | [-25,4 ; -24,0] | -6,3 % | [-6,8 ; -5,8] | [-25,4 ; -24,0] |
| Italian | Google Gemma 3 | 37 313 | -4,2 % | [-4,5 ; -3,8] | -3,5 % | [-3,8 ; -3,1] | [-4,5 ; -3,8] |
| Danish | OpenAI o200k | 37 169 | -14,0 % | [-14,6 ; -13,4] | -2,1 % | [-2,6 ; -1,7] | [-14,6 ; -13,4] |
| French | Mistral Tekken | 36 698 | +6,8 % | [+6,4 ; +7,3] | +6,5 % | [+6,1 ; +6,9] | [+6,4 ; +7,3] |
| Swedish | OpenAI o200k | 36 287 | -14,1 % | [-14,7 ; -13,4] | -1,3 % | [-1,8 ; -0,9] | [-14,7 ; -13,4] |
| German | OpenAI o200k | 35 224 | -0,9 % | [-1,4 ; -0,5] | +0,3 % | [-0,2 ; +0,8] | [-1,4 ; -0,5] |
| Spanish | Google Gemma 3 | 35 131 | +0,5 % | [+0,2 ; +0,8] | +2,7 % | [+2,3 ; +3,0] | [+0,2 ; +0,8] |
| Dutch | OpenAI o200k | 33 495 | +0,2 % | [-0,2 ; +0,7] | +2,2 % | [+1,8 ; +2,7] | [-0,2 ; +0,7] |
| Portuguese | OpenAI o200k | 32 852 | -0,5 % | [-0,9 ; -0,1] | +1,0 % | [+0,6 ; +1,4] | [-0,9 ; -0,1] |

