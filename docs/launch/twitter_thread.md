# X / Twitter launch thread

Copy tweet-by-tweet. Replace `@user` placeholders if tagging is appropriate.

---

**1/7** 🚀 Releasing arabnamer — an open-source Arabic name transliteration library for Python.

`pip install arabnamer`
`translit("Mohammed Ali")` → `محمد علي`

40 MB, offline, MIT licensed. Built on JRC-Names + a curated 22K-pair dataset.

🔗 https://github.com/sayedyousef/arabnamer

---

**2/7** Why? Arabic-name handling is a real gap.

أحمد / احمد / Ahmad / Ahmed all refer to the same person. No Python library makes that obvious. Critical for:

• KYC / sanctions screening
• Library & archive cataloguing
• Cross-language search relevance
• Arabic NER / entity resolution

---

**3/7** What's in the box:

• XGBoost model pruned to 386 rounds (from 800 — zero accuracy loss) = 40 MB
• 22,798 cleaned EN↔AR name pairs
• Rule-based fallback engine for unknown names
• Lenient Arabic similarity scorer (tashkeel / hamza / taa-marbuta insensitive)
• Training scripts — retrain on your own corpus

---

**4/7** Numbers from a 25-name benchmark (common Arab first + last combinations):

Avg lenient accuracy: **98.0**
Pass rate ≥ 70: **25/25**
Pass rate ≥ 90: **23/25**
Exact match: **20/25**

Benchmark + per-name CSV: https://github.com/sayedyousef/arabnamer/blob/main/benchmarks/REPORT.md

---

**5/7** Interesting model-pruning finding:

Full model = 285 MB (800 boosting rounds).
Pruned = 40 MB gzipped (386 rounds).
**Accuracy: identical** (byte-identical per-name scores).

Binary-searched the smallest k with matching output — turns out 386. Everything past that was noise.

---

**6/7** Licenses, so you don't have to hunt:

📝 Source code: MIT
📊 Dataset + model weights: CC-BY-4.0 (attribution required, commercial OK)
🙏 Underlying seed data: JRC-Names (European Commission, EU open-data licence)

---

**7/7** What's coming next:

• Hugging Face model + dataset pages (inference widget)
• Short arXiv tech report on the pruning experiment
• BiLSTM alternative engine (5 MB, slightly lower accuracy)
• Arabic→English reverse direction

Feedback welcome. Issues on GitHub or DM me.

🔗 https://github.com/sayedyousef/arabnamer
🔗 pip: https://pypi.org/project/arabnamer/
