# Data sources

The training data (`dataset/dict_FINAL.json`, 22,798 English → Arabic name pairs)
is a merge of the following sources, followed by a manual audit pass.

## Sources used

| Source | Role | License / terms | Notes |
|---|---|---|---|
| **[JRC-Names](https://joint-research-centre.ec.europa.eu/)** | Primary seed | EU Open Data Licence | Multilingual name gazetteer maintained by the European Commission's Joint Research Centre. Original entries include Arabic transliterations of public-figure and news-reported names. |
| **[Google Translate](https://translate.google.com/)** (via `deep_translator`) | Fill-in | Generated text, no explicit TOS claim | Used to generate Arabic forms for English names absent from JRC. A rate-limited thread pool ran roughly 17 K translations in one pass. |
| **Claude (Anthropic)** LLM fill | Fill-in | Generated text | Used for a smaller batch of names the MT step got wrong or never saw (typically Dhl-thesis-specific and compound/article names). Can be regenerated with your own API key via `training/claude_fill.py` (not yet shipped). |
| **Manual audit + rule-based cleanup** | Quality control | — | Includes phonetic compatibility filtering (English-letter ↔ Arabic-letter compatibility), hamza repair, tashkeel stripping, taa-marbuta vs haa normalisation, explicit-space-containing Arabic outputs capped at 5 tokens, and an unresolved-name drop list. |

## Sources explicitly NOT used in `dict_FINAL.json`

The thesis code (`Thesis_Submission_validator/code/`) contains adapters for several
additional corpora that were explored but did NOT contribute entries to the final
shipped dictionary:

- VIAF (Virtual International Authority File)
- ANETAC (EN-AR transliteration academic dataset)
- ANGF (Arabic names gender files scraped from public HTML lists)
- Google AI Studio / Gemini batch output

These are present in the thesis for methodology comparison but were not used to
train the shipped model.

## Attribution

Per the `CC-BY-4.0` license on `dataset/dict_FINAL.json` — if you use this data,
please credit:

> Uses the arabnamer Arabic-name dictionary (Yousef, E., 2026).
> https://github.com/sayedyousef/arabnamer — licensed under CC-BY-4.0.

## Reproducibility

The final dict is a derivative artefact — it cannot be regenerated bit-identically
from the original sources because:

1. **JRC-Names** updates weekly (the snapshot used is from early 2026).
2. **Google Translate** outputs can drift as their models update.
3. **Claude API** outputs are sampled and vary across runs.

For a defensible research artefact, cite the version + commit hash of this repo,
not the upstream sources.

## Safety note

No personally identifying private data is included — all names in the dict are
either:

- Public figures (authors, journalists, politicians) surfaced by JRC-Names
- Common given-names / surnames in the MENA region
- Programmatically generated compounds (e.g., `Abd + <divine-name>`)

If you believe a specific entry should be removed for privacy or accuracy reasons,
open an issue at https://github.com/sayedyousef/arabnamer/issues.
