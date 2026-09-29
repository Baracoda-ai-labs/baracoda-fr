# Protocole pré-enregistré — test final de Baracoda FR v1.2

Rédigé et horodaté AVANT tout calcul sur les corpus ci-dessous (29 septembre 2026). Aucun de ces corpus n'a été consulté pendant la conception de v1, v1.1 ou v1.2.

## Tokenizer évalué (figé)
- baracoda-fr-v1_2.json (131 072 tokens), fichier tel que produit le 28 septembre 2026. SHA-256 indiqué ci-dessous.

## Comparateurs
- Mistral Tekken (mistral-common 1.12, tekken_240911.json), comparateur principal.
- CroissantLLM (croissantllm/CroissantLLMBase), OpenAI o200k (gpt-oss), comparateurs secondaires.

## Corpus du test final (Universal Dependencies, toutes les partitions, lignes « # text = »)
- Français : UD_French-ParTUT (juridique, web, Wikipédia), UD_French-ParisStories (oral spontané transcrit), UD_French-FQB (questions).
- Anglais : UD_English-ParTUT, UD_English-LinES (littérature, documentation), UD_English-GENTLE (genres variés).
Texte normalisé NFC, espaces de bord retirés, comptage phrase par phrase sans tokens spéciaux.

## Critères, fixés à l'avance
- Critère principal : rapport tokens(v1.2) / tokens(Tekken) sur le français, par corpus et regroupé, avec intervalle de confiance à 95 % par bootstrap par blocs de 10 phrases consécutives (2 000 tirages, graine 0).
- H1 : v1.2 compte moins de tokens que Tekken sur chacun des trois corpus français.
- H1b : sur le français regroupé, la réduction est d'au moins 8 %.
- H2 : sur l'anglais regroupé, v1.2 ne dépasse pas Tekken de plus de 2 %.
- Secondaire : mêmes rapports face à CroissantLLM et o200k, sans hypothèse.
- Contrôle de contamination : proportion de phrases de plus de 40 caractères des corpus finaux présentes telles quelles dans les données d'entraînement de v1.2 ; rapportée quel que soit le résultat.
- Tous les résultats sont publiés, y compris si H1, H1b ou H2 échouent. Aucun réglage du tokenizer n'est modifié après ce test.
8d9fdb7b10f2ab278259e6a08b7601e8405db9f256894518723982cff924464d  baracoda-fr-v1_2.json
2026-09-28T22:29:30Z
