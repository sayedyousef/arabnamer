# arabnamer — benchmark results

**Test set:** tests\test_names.csv (25 names)
**Engine:** model (XGBoost, pruned 386 rounds)
**Threshold:** 85 (lenient scoring)

## Summary

| Metric | Score |
|---|---|
| Names evaluated | **25** |
| Average lenient similarity | **98.4** |
| Pass rate (>= 70) | **25 / 25** (100%) |
| Pass rate (>= 90) | **24 / 25** (96%) |
| Exact match (= 100) | **21 / 25** (84%) |

## Per-name results

| Name (EN) | Predicted (AR) | Reference (AR) | Score | Passed |
|---|---|---|---|---|
| Sayed Ali | سيد علي | سيد علي | 100.0 | ✅ |
| Mohammed Ali | محمد علي | محمد علي | 100.0 | ✅ |
| Muhammad Ali | محمد علي | محمد علي | 100.0 | ✅ |
| Ahmad Hassan | أحمد حسن | أحمد حسن | 100.0 | ✅ |
| Marwa Farag | مروة فرج | مروة فرج | 100.0 | ✅ |
| Bassel Salloukh | باسل صلوخ | باسل صلوخ | 100.0 | ✅ |
| Nael Jebril | نايل جبريل | نائل جبريل | 90.0 | ✅ |
| Ayman El Desouky | أيمن الدسوقي | أيمن الدسوقي | 100.0 | ✅ |
| Issam Nassar | عصام نصار | عصام نصار | 100.0 | ✅ |
| Elizabeth Kassab | إليزابيث قصاب | إليزابيث قصاب | 100.0 | ✅ |
| Natalie Tayim | ناتالي تيم | ناتالي تيم | 100.0 | ✅ |
| Ismail Nashef | إسماعيل ناشف | إسماعيل ناشف | 100.0 | ✅ |
| Ayhab Saad | اهاب سعد | إيهاب سعد | 94.1 | ✅ |
| Julia Barbar | جوليا باربر | جوليا بربر | 95.2 | ✅ |
| Rabia Naguib | ربيعة نجيب | ربيعة نجيب | 100.0 | ✅ |
| Hosam Haffz | حسام حافظ | حسام حافظ | 100.0 | ✅ |
| Diala Hawi | ديالا حاوي | ديالا حاوي | 100.0 | ✅ |
| Mahdi Arar | مهدي عرعر | مهدي عرعر | 100.0 | ✅ |
| Adham Saouli | أدهم ساولي | أدهم ساولي | 100.0 | ✅ |
| Tariq Da'na | طارق دعنا | طارق دعنا | 100.0 | ✅ |
| Ferdoos Alissa | فردوس العيسى | فردوس العيسى | 100.0 | ✅ |
| Moataz El Fegiry | معتز الفجيري | معتز الفجيري | 100.0 | ✅ |
| Elias Khalil | إلياس خليل | الياس خليل | 100.0 | ✅ |
| Basim Tweissi | باسم تويسي | باسم الطويسي | 81.8 | ❌ |
| Abdennour Benantar | عبد النور بن عنتر | عبد النور بن عنتر | 100.0 | ✅ |

## Reproduce

```bash
pip install arabnamer
python tests/test_eval.py   # writes benchmarks/eval_results.csv + REPORT.md
```

Scoring is **lenient** — tashkeel stripped, hamza variants unified (أ/إ/آ → ا), taa-marbuta → haa (ة → ه), alef-maksura → yaa (ى → ي), then `max(fuzz.ratio, fuzz.partial_ratio)` via rapidfuzz.