# LinkedIn launch post

Copy/paste this when you're ready to announce the release. Adjust tone to your
voice — the template below is professional but not corporate.

---

## Version A — technical / product-led

**🚀 Releasing arabnamer — an open-source Arabic name transliteration library for Python.**

Arabic-name handling is a real gap in NLP tooling. Most Arabic libraries focus
on text — tokenization, stemming, embeddings. Proper-noun transliteration and
fuzzy-matching of name variants (أحمد vs احمد vs Ahmad) is underserved, yet
critical for KYC, compliance, library cataloguing, and search relevance across
MENA markets.

arabnamer does three things, offline, in Python:

✅ Transliterate English names to Arabic — `translit("Mohammed Ali") → محمد علي`
✅ Score Arabic name similarity — insensitive to tashkeel, hamza, taa-marbuta
✅ Retrain on your own corpus — training scripts included

Numbers that matter:
• 98.0 average lenient accuracy on a 25-name benchmark
• 22,798 EN-AR name pairs in the training dictionary
• 40 MB pip install, fully offline after install, no API keys

Available now:
🔗 PyPI: `pip install arabnamer`
🔗 GitHub: https://github.com/sayedyousef/arabnamer
🔗 HuggingFace: (coming in a week)

Licensed MIT (code) + CC-BY-4.0 (dataset & model). Built as an extraction from
my MSc thesis work on cross-language name normalization. Grateful to the
JRC-Names team at the European Commission's JRC for the underlying data.

Would love feedback from anyone working on Arabic NLP, MENA compliance tooling,
or multilingual entity resolution — what's missing? What would you add?

#ArabicNLP #MENA #OpenSource #Python #NLP #NER #EntityResolution

---

## Version B — career / story-led (for follow-up posts)

**How a PhD side-project became an open-source library with 22K hand-curated Arabic name pairs.**

A year ago I was building a validator for MENA-region academic forms and ran
into a problem every Arabic-NLP engineer knows: there's no good way to handle
names. Not just transliteration — the harder problem is that أحمد, احمد, and
Ahmad all refer to the same person, and no library makes that obvious.

So I built one. After:
• Merging three open data sources (JRC-Names + Google Translate + manual audit)
• Training a gradient-boosted transliteration model (XGBoost, 386 rounds)
• Pruning the model from 285 MB down to 40 MB with zero accuracy loss
• Wrapping it all in a ~150 line Python API

The result: `pip install arabnamer`, one call, offline.

Full write-up including the model-pruning experiment (binary-searching the
smallest identical XGBoost): link in comments.

#OpenSource #MachineLearning #ArabicNLP #MSc #ResearchToProduct

---

## Version C — one-paragraph teaser (for Twitter crosspost)

arabnamer 0.1 is live on PyPI 🚀

English → Arabic name transliteration + fuzzy similarity, offline, 40 MB, one
Python call. 98.0% accuracy on a benchmark. Training data (22K names) and
XGBoost model included. MIT + CC-BY-4.0.

→ pip install arabnamer
→ https://github.com/sayedyousef/arabnamer

#ArabicNLP #MENA #OpenSource
