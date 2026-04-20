---
license: cc-by-4.0
language:
  - ar
  - en
task_categories:
  - translation
  - token-classification
tags:
  - arabic
  - transliteration
  - names
  - name-matching
  - mena
  - arabic-nlp
size_categories:
  - 10K<n<100K
---

# arabic-name-pairs — 22,798 English ↔ Arabic name pairs

Curated training dictionary used to build the
[`arabnamer`](https://github.com/sayedyousef/arabnamer) Arabic name transliteration library.

## Load

```python
from datasets import load_dataset
ds = load_dataset("sayedyousef/arabic-name-pairs")
print(ds["train"][0])
# {'name_en': 'mohammed', 'name_ar': 'محمد'}
```

Or directly as JSON:

```python
import json, urllib.request
url = "https://huggingface.co/datasets/sayedyousef/arabic-name-pairs/resolve/main/dict_FINAL.json"
pairs = json.loads(urllib.request.urlopen(url).read())
print(len(pairs))  # 22798
```

## What's inside

A flat JSON mapping of lowercase English name tokens → Arabic transliterations:

```json
{
  "mohammed":    "محمد",
  "ahmad":       "أحمد",
  "ayman":       "أيمن",
  "aldesouky":   "الدسوقي",
  "abdelrahman": "عبد الرحمن",
  ...
}
```

## Sources

Merged from:

1. **[JRC-Names](https://joint-research-centre.ec.europa.eu/)** — primary seed (European Commission's multilingual name gazetteer)
2. **Google Translate** (via `deep_translator`) — fill-in for names absent from JRC
3. **Claude (Anthropic) LLM** — supplementary fill-in for names the MT step got wrong
4. **Manual audit + rule-based cleanup** — phonetic-compatibility filter, hamza repair, tashkeel stripping

Full provenance: [`docs/data_sources.md`](https://github.com/sayedyousef/arabnamer/blob/main/docs/data_sources.md) in the main repo.

## Intended uses

- **Training Arabic-name transliteration models** (classical ML, neural, LLM fine-tuning)
- **Lookup tables** for KYC / sanctions screening / library cataloguing systems
- **Evaluation** — a reference set for benchmarking new Arabic NER / transliteration systems
- **Research** on name normalization and entity resolution in MENA-language contexts

## Known limitations

- **Single-token entries only** — compound names are merged with no separator (`"alRahman"` not `"al Rahman"`); splitting logic lives in the `arabnamer` library
- **Max length 13 characters per side** — long surnames may not appear
- **Modern Standard Arabic spelling** — regional / dialect variants are not represented
- **Public-figure bias** — JRC-Names skews toward news-reported names (politicians, journalists, authors)

## Attribution

Per CC-BY-4.0, if you use this dataset please credit:

> Uses the arabnamer Arabic-name dictionary (Yousef, E., 2026).
> https://github.com/sayedyousef/arabnamer — licensed under CC-BY-4.0.

## BibTeX

```bibtex
@dataset{yousef_arabic_name_pairs_2026,
  author    = {Yousef, Elsayed},
  title     = {arabic-name-pairs: 22,798 English-Arabic name pairs},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/sayedyousef/arabic-name-pairs},
}
```

## License

**CC-BY-4.0**. Commercial use permitted with attribution.
