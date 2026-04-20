# arabnamer — Arabic name transliteration & similarity

[![PyPI version](https://img.shields.io/pypi/v/arabnamer.svg)](https://pypi.org/project/arabnamer/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset License: CC-BY-4.0](https://img.shields.io/badge/dataset-CC--BY--4.0-orange.svg)](LICENSE-DATA)
[![Downloads](https://static.pepy.tech/badge/arabnamer/month)](https://pepy.tech/project/arabnamer)

**Convert English names to Arabic and score Arabic name matches — offline, in one line of Python.**

```python
from arabnamer import translit, similarity

translit("Mohammed Ali").arabic        # → 'محمد علي'
translit("Ayman El Desouky").arabic    # → 'أيمن الدسوقي'

similarity("أحمد حسن", "احمد حسن")      # → (True, 100)
```

`arabnamer` solves the **Arabic name transliteration** problem — turning `Mohammed Ali` into `محمد علي`,
and scoring matches between Arabic name variants (with hamza, tashkeel, taa-marbuta differences).
It ships a 38 MB pruned XGBoost model trained on **22,798 English-Arabic name pairs** (JRC-Names + Google
Translate + manual audit) and reaches **98.4 % lenient accuracy** on an independent MENA-names benchmark.

Built for: KYC / sanctions screening, library cataloguing, Arabic NLP preprocessing, search-relevance
normalization, entity resolution across English and Arabic corpora.

---

## Install

```bash
pip install arabnamer
```

Works offline after install — the model is bundled (gzipped, ~38 MB). No external services, no API keys.

---

## Quick start

### 1. Transliterate English → Arabic

```python
from arabnamer import translit, translit_batch

result = translit("Mohammed Ali")
print(result.arabic)    # 'محمد علي'
print(result.score)     # 0.0 (no reference supplied)
print(result.engine)    # 'xgboost'

# batch
results = translit_batch(["Ahmad Hassan", "Marwa Farag", "Ayman El Desouky"])
for r in results:
    print(f"{r.input:<25} -> {r.arabic}")
```

### 2. Score with a reference (lenient, normalized)

```python
from arabnamer import translit

r = translit("Adham Saouli", reference="أدهم ساولي")
print(r.score)       # 100.0
print(r.accepted)    # True  (>= default threshold 85)
```

### 3. Arabic ↔ Arabic similarity (tashkeel / hamza / taa-marbuta insensitive)

```python
from arabnamer import similarity

similarity("أحمد حسن", "احمد حسن")          # (True, 100)   hamza variant
similarity("مروة فرج", "مروه فرج")           # (True, 100)   taa-marbuta vs haa
similarity("محمد", "أحمد", threshold=75)     # (True, 75) — different names
```

### 4. Configurable: threshold + engine

```python
from arabnamer import Transliterator

t = Transliterator(engine="model", threshold=90)   # stricter pass bar
t.translit("Mohammed Ali", reference="محمد علي")

# Rule-based engine (deterministic, dict-first)
t_rules = Transliterator(engine="rules")
t_rules.translit("Mohammed Ali")

# Hybrid: dict -> model -> rules fallback
t_hybrid = Transliterator(engine="hybrid")
```

---

## How accurate is it?

Benchmarked on 25 MENA-region names (authors, journalists, public figures):

| Metric | Score |
|---|---|
| Average lenient similarity | **98.4** |
| Pass rate (≥ 70) | **25 / 25** |
| Pass rate (≥ 90) | **24 / 25** |
| Exact match (= 100) | **21 / 25** |

Model: XGBoost, 386 boosting rounds × 335 output classes, 34 input features per character
(char IDs + position + phonetic class + bigram/trigram IDs). See [`benchmarks/REPORT.md`](benchmarks/REPORT.md)
for the full breakdown.

---

## What's inside this repo

```
arabnamer/
├── src/arabnamer/           # pip-installable library
│   ├── core/                # Transliterator orchestrator + Result
│   ├── prediction/          # XGBoost model loader + featurize
│   ├── rules/               # deterministic rule-based walker
│   ├── scoring/             # fuzzy match + Arabic normalizer
│   └── utils/               # tokenizer + dict lookup
│
├── model/                   # gzipped model + labels (ships in wheel)
├── dataset/                 # 22,798 EN -> AR name pairs (dict_FINAL.json)
├── training/                # scripts to retrain from scratch
├── tests/                   # smoke + benchmark eval
├── benchmarks/              # eval results + REPORT.md
└── docs/                    # data sources + architecture + API details
```

---

## Data sources

Training data comes from three open sources plus a manual audit pass. Full provenance in
[`docs/data_sources.md`](docs/data_sources.md). In summary:

| Source | License | Contribution |
|---|---|---|
| [JRC-Names](https://joint-research-centre.ec.europa.eu/) (European Commission, multilingual name gazetteer) | EU open-data | primary EN/AR name pairs |
| [Google Translate](https://translate.google.com/) (via `deep_translator`) | generated text | fill-in for names absent from JRC |
| Claude (Anthropic) LLM fill | generated text | supplementary fill for DI-thesis-specific names |
| Manual audit + rule-based cleanup | — | phonetic-compatibility filter, hamza repair, outlier removal |

**Model and dataset are released under CC-BY-4.0** (attribution required). Library code is MIT.

---

## Reproducing the model

```bash
git clone https://github.com/sayedyousef/arabnamer
cd arabnamer
pip install -e ".[dev]"
cd training
python train_xgboost.py      # ~5 min on CPU, produces ~285 MB UBJ
python prune_trees.py        # ~1 min, saves pruned variants
python find_min_k.py         # ~30 sec, finds the smallest-identical model
```

XGBoost with multi-threaded histogram training is approximately (not bit-perfectly) deterministic.
See [`training/README.md`](training/README.md) for the full note.

---

## Why this library exists

Arabic name handling is a real gap in open NLP tooling. Most Arabic libraries
(`PyArabic`, `arabic-reshaper`, AraBERT, CAMeL Tools) focus on **text** — rendering,
tokenisation, stemming. Proper-noun transliteration and fuzzy-matching of name variants
(`أحمد` vs `احمد` vs `Ahmad`) is underserved.

`arabnamer` is the first open library specifically for:
- **Arabic ↔ English name transliteration** at the word level
- **Arabic name similarity** that's insensitive to hamza, tashkeel, taa-marbuta, and alef-maksura variants
- **Offline, pip-installable** — no API keys, no network, no LLM dependency

Typical use cases:
- **KYC & sanctions screening** — match Arabic names against English watchlists
- **Library & archive cataloguing** — normalize author names across MENA languages
- **Search relevance** — expand queries to cover Arabic-script variants
- **Data integration** — reconcile person entities across EN/AR CRMs

---

## Citation

If you use `arabnamer` in a paper, product, or research tool, please cite:

```bibtex
@software{yousef_arabnamer_2026,
  author  = {Yousef, Elsayed},
  title   = {arabnamer: Arabic name transliteration and similarity},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/sayedyousef/arabnamer},
}
```

A `CITATION.cff` file is included so GitHub's "Cite this repository" button works.

---

## Commercial support

For production deployments, custom models trained on your domain corpus, on-premise
integration, or paid support:

**elsayed.yousef@gmail.com**

---

## License

- **Source code** (`src/arabnamer/`, `training/`): [MIT License](LICENSE)
- **Dataset, model weights, test data**: [CC-BY-4.0](LICENSE-DATA) (attribution required)

---

## Acknowledgements

- [JRC-Names](https://joint-research-centre.ec.europa.eu/) — European Commission Joint Research Centre name gazetteer (primary training data)
- [XGBoost](https://xgboost.readthedocs.io/) — gradient-boosted tree library
- [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) — fast fuzzy string matching
- The thesis work this library was extracted from (Doha Institute for Graduate Studies)
