# arabnamer — benchmark results

**Test set:** tests\test_names.csv (25 names)
**Engine:** model (XGBoost, pruned 386 rounds)
**Threshold:** 85 (lenient scoring)

## Summary

| Metric | Score |
|---|---|
| Names evaluated | **25** |
| Average lenient similarity | **98.0** |
| Pass rate (>= 70) | **25 / 25** (100%) |
| Pass rate (>= 90) | **23 / 25** (92%) |
| Exact match (= 100) | **20 / 25** (80%) |

## Per-name results

| Name (EN) | Predicted (AR) | Reference (AR) | Score | Passed |
|---|---|---|---|---|
| Mohammed Ali | محمد علي | محمد علي | 100.0 | ✅ |
| Ahmad Hassan | أحمد حسن | أحمد حسن | 100.0 | ✅ |
| Omar Khalil | عمر خليل | عمر خليل | 100.0 | ✅ |
| Fatima Mansour | فاطمة منصور | فاطمة منصور | 100.0 | ✅ |
| Samir Ibrahim | سمير إبراهيم | سمير إبراهيم | 100.0 | ✅ |
| Layla Al Saleh | ليلة الصالح | ليلى الصالح | 90.9 | ✅ |
| Khalid El Masri | خالد المصري | خالد المصري | 100.0 | ✅ |
| Noor Rashid | نور راشد | نور راشد | 100.0 | ✅ |
| Yasmin Farouk | ياسمين فاروق | ياسمين فاروق | 100.0 | ✅ |
| Hasan El Amin | حسن الأمين | حسن الأمين | 100.0 | ✅ |
| Karim Shawqi | كريم شوكي | كريم شوقي | 88.9 | ✅ |
| Zaynab Nasser | زيناب ناصر | زينب ناصر | 94.7 | ✅ |
| Bilal Othman | بلال عثمان | بلال عثمان | 100.0 | ✅ |
| Salma Dahlan | سلمى دحلان | سلمى دحلان | 100.0 | ✅ |
| Rania Hakim | رانيا حكيم | رانيا حكيم | 100.0 | ✅ |
| Tamer Abdel Rahim | تامر عبد الرحيم | تامر عبد الرحيم | 100.0 | ✅ |
| Mariam Habib | مريم حبيب | مريم حبيب | 100.0 | ✅ |
| Hassan El Khatib | حسن الخطيب | حسن الخطيب | 100.0 | ✅ |
| Nada Ramzi | ا شيء رمزي | ندى رمزي | 80.0 | ❌ |
| Abdelrahman Saber | عبد الرحمن صابر | عبد الرحمن صابر | 100.0 | ✅ |
| Tariq Salama | طارق سلامة | طارق سلامة | 100.0 | ✅ |
| Reem Abdul Karim | ريم عبد الكريم | ريم عبد الكريم | 100.0 | ✅ |
| Adam Ayoub | آدم أيوب | آدم أيوب | 100.0 | ✅ |
| Ghadeer Anwar | غادير أنور | غدير أنور | 94.7 | ✅ |
| Mostafa Al Khatib | مصطفى الخطيب | مصطفى الخطيب | 100.0 | ✅ |

## Reproduce

```bash
pip install arabnamer
python tests/test_eval.py   # writes benchmarks/eval_results.csv + REPORT.md
```

Scoring is **lenient** — tashkeel stripped, hamza variants unified (أ/إ/آ → ا), taa-marbuta → haa (ة → ه), alef-maksura → yaa (ى → ي), then `max(fuzz.ratio, fuzz.partial_ratio)` via rapidfuzz.