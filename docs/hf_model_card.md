---
license: cc-by-4.0
language:
  - ar
  - en
tags:
  - transliteration
  - arabic
  - name-matching
  - xgboost
  - sequence-labeling
  - mena
library_name: xgboost
pipeline_tag: translation
---

# arabnamer — XGBoost Arabic name transliteration model

**Pruned to 386 boosting rounds, ~38 MB gzipped, trained on 22,798 English-Arabic name pairs.**

This is the model artifact used by the [`arabnamer` Python library](https://github.com/sayedyousef/arabnamer).
It predicts the Arabic transliteration of a lowercase English name token via per-character
multiclass classification (335 Arabic label tokens).

## Quick use

```bash
pip install arabnamer
```

```python
from arabnamer import translit
r = translit("Mohammed Ali")
print(r.arabic)   # 'محمد علي'
```

The wheel ships with this model bundled — no download from Hugging Face needed at
runtime. This HF page exists for discoverability and the inference widget.

## Model details

| | |
|---|---|
| **Framework** | XGBoost 2.0+ (saved in UBJ binary format, gzipped) |
| **Boosting rounds (trees × 335 classes)** | 386 × 335 = 129,310 trees |
| **Max depth** | 12 |
| **Output classes** | 335 (Arabic label tokens + empty string for silent-char) |
| **Input features** | 34 per character: padded char IDs, position, phonetic class, bigram/trigram IDs |
| **File size** | 160 MB uncompressed `.ubj`, 38 MB gzipped `.ubj.gz` |
| **Inference speed** | ~0.5 ms per 8-char token on CPU |

## Training

- **Dataset**: [arabnamer/arabic-name-pairs](https://huggingface.co/datasets/sayedyousef/arabic-name-pairs) — 22,798 EN→AR name pairs
- **Training script**: [`training/train_xgboost.py`](https://github.com/sayedyousef/arabnamer/blob/main/training/train_xgboost.py)
- **Pruning pipeline**: start at 800 rounds, use `find_min_k.py` to search for the smallest k whose per-name scores exactly match the baseline → result: 386 rounds
- **Hardware**: any CPU with 4 GB RAM — roughly 5 min on a modern laptop
- **Determinism**: approximate (multi-threaded `tree_method="hist"`); use `n_jobs=1` for bit-perfect reproducibility

## Evaluation

| Metric | Value |
|---|---|
| Average lenient similarity | **98.0** |
| Pass rate ≥ 70 | 25 / 25 |
| Pass rate ≥ 90 | 23 / 25 |
| Exact match (= 100) | 20 / 25 |

Benchmark: 25 generic Arab-name pairs (common first + last combinations covering compound articles, hamza variants, feminine endings). Scoring is lenient (tashkeel stripped, hamza/taa-marbuta/alef-maksura unified, then `max(fuzz.ratio, fuzz.partial_ratio)`).

Per-name results: [benchmarks/REPORT.md](https://github.com/sayedyousef/arabnamer/blob/main/benchmarks/REPORT.md)

## Intended uses

- **Arabic name transliteration** for search, KYC, entity resolution, library cataloguing
- **Training-data augmentation** — use this model's predictions as silver labels for larger corpora
- **Research** into trade-offs between classical ML and neural transliteration

## Limitations

- **Single-token input required** — multi-token names are handled by the library wrapper but this model itself expects one English word at a time
- **Max length 13 characters** — tokens longer are returned unchanged
- **MENA focus** — trained primarily on Arab / Persian / Turkic names transliterated into English; less accurate for unusual African or South Asian names with Arabic transliteration traditions
- **Dialect neutrality** — outputs Modern Standard Arabic spelling; colloquial or regional spellings (Egyptian, Maghrebi) are not modeled
- **No Arabic→English direction** — this model only handles English→Arabic

## Citation

```bibtex
@software{yousef_arabnamer_2026,
  author  = {Yousef, Elsayed},
  title   = {arabnamer: Arabic name transliteration and similarity},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/sayedyousef/arabnamer},
}
```

## License

**CC-BY-4.0** — attribution required. Commercial use permitted.
