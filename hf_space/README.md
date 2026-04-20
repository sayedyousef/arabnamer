---
title: arabnamer demo
emoji: 🕌
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.7.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
tags:
  - arabic
  - transliteration
  - name-matching
  - arabic-nlp
  - mena
  - kyc
  - entity-resolution
short_description: Offline Arabic name transliteration & fuzzy similarity.
---

# arabnamer — live demo

This Space runs the [arabnamer](https://github.com/sayedyousef/arabnamer) Python
library in an interactive UI.

## Three tabs

1. **Transliterate** — English name → Arabic name, with selectable engine
   (XGBoost model / rule-based / hybrid) and optional reference scoring.
2. **Similarity** — Arabic ↔ Arabic lenient fuzzy matching, insensitive to
   tashkeel, hamza variants, taa-marbuta, and alef-maksura.
3. **Batch** — paste a list of English names, get a table + CSV output.

## Offline by design

No external API calls, no LLM, names never leave this Space container. The
entire pipeline runs on the 38 MB bundled XGBoost model + deterministic
rule-based engine.

## Install locally

```bash
pip install arabnamer
```

Then:

```python
from arabnamer import translit, similarity
print(translit("Mohammed Ali").arabic)        # → 'محمد علي'
print(similarity("أحمد حسن", "احمد حسن"))     # → (True, 100)
```

## Links

- 🔗 [GitHub repo](https://github.com/sayedyousef/arabnamer)
- 🔗 [PyPI package](https://pypi.org/project/arabnamer/)
- 🔗 [Model (this Space loads it transitively via the library)](https://huggingface.co/Sayedyousef/arabnamer-xgboost)
- 🔗 [Training dataset](https://huggingface.co/datasets/Sayedyousef/arabic-name-pairs)

## License

- Code (this app): MIT
- Bundled model weights + training dictionary: CC-BY-4.0
