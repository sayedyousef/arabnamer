---
title: "I built an open Arabic name transliteration library (and shrank the model 7×)"
published: false
description: arabnamer — 22K EN-AR name pairs, a pruned XGBoost model, and the hunt for the smallest model that still produces identical predictions.
tags: python, machinelearning, nlp, opensource
cover_image: https://raw.githubusercontent.com/sayedyousef/arabnamer/main/docs/assets/cover.png
---

# I built an open Arabic name transliteration library (and shrank the model 7×)

> `pip install arabnamer` — English → Arabic names, offline, 40 MB, one Python call.

Arabic-name handling is a real gap in Python NLP. Most Arabic libraries
(`PyArabic`, `arabic-reshaper`, CAMeL, AraBERT) focus on **text** — rendering,
tokenisation, stemming. Proper-noun transliteration and fuzzy-matching of
name variants is underserved, yet it's critical for KYC, library cataloguing,
and search relevance across MENA markets.

So I built [**arabnamer**](https://github.com/sayedyousef/arabnamer), an
open-source library extracted from my MSc thesis work.

```python
from arabnamer import translit, similarity

translit("Mohammed Ali").arabic        # → 'محمد علي'
translit("Layla Al Saleh").arabic      # → 'ليلى الصالح'

similarity("أحمد حسن", "احمد حسن")      # → (True, 100)
```

This post covers three things:

1. The data pipeline — merging JRC-Names, Google Translate, and an LLM fill
2. The XGBoost model and its featurisation (34 features per character)
3. The pruning experiment — how I shrank the model 7× with zero accuracy loss

---

## 1. The data

**22,798 English → Arabic name pairs.** Sourced from:

| Source | Contribution |
|---|---|
| [JRC-Names](https://joint-research-centre.ec.europa.eu/) (European Commission) | primary seed |
| Google Translate (via `deep_translator`) | fill-in for unseen names |
| Claude (Anthropic) LLM | further fill for compound/article names |
| Manual audit | phonetic compatibility, hamza repair, outlier drop |

Key decision: **single-token only**, both sides. Multi-token compounds
(`"Abdel Rahman"`) are either merged into single tokens (`"abdelrahman"`)
or handled by the library's token-level article-merger.

---

## 2. The model

A per-character multiclass classifier. For each English character in a token,
predict the Arabic label token emitted at that position.

Features (34 ints per character):

- Padded character IDs (13 chars left-aligned)
- Position + length + boolean flags (is_first, is_last, is_double)
- Phonetic class flags (vowel, sibilant, plosive, nasal, liquid, fricative, glide)
- Bigram IDs (prev-cur, cur-next)
- Trigram IDs (prev-cur-next, prev2-prev-cur, cur-next-next2)

XGBoost, `n_estimators=800`, `max_depth=12`, `tree_method="hist"`, 335 output
classes (the inventory of Arabic label tokens produced by Needleman-Wunsch
alignment of training pairs).

Training takes ~5 minutes on a laptop CPU. Initial model size: **285 MB**.

---

## 3. The pruning experiment — 285 MB → 40 MB, same accuracy

285 MB for a transliteration model felt wrong. Could I cut it down?

**Idea**: XGBoost lets you use only the first `k` rounds at inference time
(`iteration_range=(0, k)`) or permanently slice the booster (`bst[0:k]`).
No retraining needed — just chop off the tail of the ensemble.

**Coarse sweep** across k = 50 … 800:

| k rounds | Size | Avg lenient | Exact matches |
|---|---|---|---|
| 800 | 285 MB | 98.4 | 21/25 |
| 600 | 228 MB | 98.4 | 21/25 |
| 400 | 165 MB | 98.4 | 21/25 |
| 300 | 130 MB | 98.3 | 21/25 |
| 200 | 71 MB | 97.9 | 19/25 |
| 100 | 36 MB | 95.0 | 13/25 |

Below 400 rounds accuracy drops. Above 400 it doesn't change. The last
400 rounds were adding nothing on this benchmark.

**Then a binary search** in (300, 400] to find the **minimum k whose per-name
scores exactly match** the baseline (not just aggregates):

```
k=350 → avg 98.7, exact 22 — better than baseline(!), but different
k=375 → better, different
k=387 → identical
k=381 → better, different
k=384 → better, different
k=385 → better, different
k=386 → IDENTICAL. Minimum.
```

Saved a 160 MB uncompressed model at k=386. Gzipped: **38 MB**.
→ 7.5× smaller than the original. Byte-identical predictions.

The PyPI wheel ends up at **40 MB** (gzipped model + dict + code + metadata).

A curiosity from the binary search: **k=385 actually beats baseline by 0.3
points** on this specific benchmark. Small-sample noise — one late tree
flipping one edge case. Shipped k=386 instead for boring predictability.

---

## Why this library exists

Because "just use Google Translate" is a bad answer:

- Needs an API key
- Doesn't handle offline / edge / on-prem deployments
- Doesn't expose a similarity scorer
- No way to retrain on your domain corpus

arabnamer is all those things, in 40 MB, MIT-licensed (dataset CC-BY-4.0).

## Try it

```bash
pip install arabnamer
```

```python
from arabnamer import translit
translit("Mohammed Ali")
# Result(input='Mohammed Ali', arabic='محمد علي', score=0.0, accepted=True, engine='xgboost')
```

Repo: **https://github.com/sayedyousef/arabnamer**

Feedback welcome on GitHub issues — especially from anyone doing Arabic NLP,
MENA compliance, or cross-language entity resolution. What's missing?
