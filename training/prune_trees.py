"""Prune trees from the existing XGBoost model (no retraining).

Uses XGBoost's Booster slicing: bst[0:k] keeps first k rounds.
Saves a shrunk UBJ and re-evaluates on the 25 DI test names.
Compares size vs accuracy at multiple truncation points.
"""
import csv, json, re, string, sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = _REPO_ROOT / "model" / "xgb_pruned_386r.ubj"  # or any full model to prune
LABELS_PATH = _REPO_ROOT / "model" / "labels.json"

PAD_LEN = 13
ALPHA = list(string.ascii_lowercase) + ["'"]
CHAR_IDX = {c: i + 1 for i, c in enumerate(ALPHA)}
VOWELS = set("aeiouy"); SIBILANTS = set("sz"); PLOSIVES = set("bdgkptq")
NASALS = set("mn"); LIQUIDS = set("lr"); FRICATIVES = set("fvh")
GLIDES = set("wy"); CONSONANTS = set("bcdfghjklmnpqrstvwxz")

MERGE_ARTICLES = {"al", "el", "ul", "ed", "ad", "abd", "abdul", "abdel", "abdal",
                  "abou", "abo", "abdou", "bin", "ben", "ibn", "binn", "benn"}

TESTS = [
    ("Sayed Ali","سيد علي"),("Mohammed Ali","محمد علي"),
    ("Muhammad Ali","محمد علي"),("Ahmad Hassan","أحمد حسن"),
    ("Marwa Farag","مروة فرج"),("Bassel Salloukh","باسل صلوخ"),
    ("Nael Jebril","نائل جبريل"),("Ayman El Desouky","أيمن الدسوقي"),
    ("Issam Nassar","عصام نصار"),("Elizabeth Kassab","إليزابيث قصاب"),
    ("Natalie Tayim","ناتالي تيم"),("Ismail Nashef","إسماعيل ناشف"),
    ("Ayhab Saad","إيهاب سعد"),("Julia Barbar","جوليا بربر"),
    ("Rabia Naguib","ربيعة نجيب"),("Hosam Haffz","حسام حافظ"),
    ("Diala Hawi","ديالا حاوي"),("Mahdi Arar","مهدي عرعر"),
    ("Adham Saouli","أدهم ساولي"),("Tariq Da'na","طارق دعنا"),
    ("Ferdoos Alissa","فردوس العيسى"),("Moataz El Fegiry","معتز الفجيري"),
    ("Elias Khalil","الياس خليل"),("Basim Tweissi","باسم الطويسي"),
    ("Abdennour Benantar","عبد النور بن عنتر"),
]


def cid(c): return CHAR_IDX.get(c, 0)
def bigram_id(a, b): return cid(a)*28 + cid(b)
def trigram_id(a, b, c): return bigram_id(a, b)*28 + cid(c)


def featurize(word, pos):
    n = len(word); w = word[:PAD_LEN]
    pad = [cid(c) for c in w] + [0]*(PAD_LEN-len(w))
    ch = word[pos] if 0 <= pos < n else ""
    prev_c = word[pos-1] if pos-1 >= 0 else ""
    next_c = word[pos+1] if pos+1 < n else ""
    prev2 = word[pos-2] if pos-2 >= 0 else ""
    next2 = word[pos+2] if pos+2 < n else ""
    feats = pad + [pos, n,
                   1 if pos == 0 else 0, 1 if pos == n-1 else 0,
                   1 if 0 < pos < n-1 and word[pos] == word[pos-1] else 0,
                   1 if 0 < pos < n-1 and word[pos] == word[pos+1] else 0,
                   1 if ch in "aeiouy" else 0]
    feats += [
        1 if ch in VOWELS else 0, 1 if ch in CONSONANTS else 0,
        1 if ch in SIBILANTS else 0, 1 if ch in PLOSIVES else 0,
        1 if ch in NASALS else 0, 1 if ch in LIQUIDS else 0,
        1 if ch in FRICATIVES else 0, 1 if ch in GLIDES else 0,
    ]
    feats += [bigram_id(prev_c, ch), bigram_id(ch, next_c)]
    feats += [trigram_id(prev_c, ch, next_c),
              trigram_id(prev2, prev_c, ch),
              trigram_id(ch, next_c, next2)]
    return feats


def strip_tashkeel(s): return re.sub(r"[\u064B-\u0652\u0670\u0653-\u0655]", "", s or "")
def normalize(s):
    s = strip_tashkeel(s)
    return s.translate(str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ة':'ه','ى':'ي'})).strip()
def lenient(c, r):
    a, b = normalize(c), normalize(r)
    if not a or not b: return 0
    return round(max(fuzz.ratio(a, b), fuzz.partial_ratio(a, b)), 1)


def evaluate(bst, labels, k, pad_threshold=PAD_LEN):
    """Predict using first k rounds of the booster."""
    def predict_tok(tok):
        if not tok or len(tok) > pad_threshold: return tok
        Xt = np.array([featurize(tok, i) for i in range(len(tok))], dtype=np.int32)
        dm = xgb.DMatrix(Xt)
        probs = bst.predict(dm, iteration_range=(0, k))
        pred_ids = probs.argmax(axis=1)
        return "".join(labels[int(p)] for p in pred_ids)

    def predict_name(name_en):
        tokens = [t for t in name_en.lower().replace("-", " ").split() if t]
        merged, i = [], 0
        while i < len(tokens):
            t = tokens[i]
            if t in MERGE_ARTICLES and i+1 < len(tokens):
                merged.append(t + tokens[i+1]); i += 2
            else:
                merged.append(t); i += 1
        return " ".join(predict_tok(t) for t in merged)

    rows = []
    for en, ref in TESTS:
        pred = predict_name(en)
        rows.append((en, pred, lenient(pred, ref), ref))
    avg = sum(r[2] for r in rows) / len(rows)
    p70 = sum(1 for r in rows if r[2] >= 70)
    p90 = sum(1 for r in rows if r[2] >= 90)
    p100 = sum(1 for r in rows if r[2] >= 100)
    return avg, p70, p90, p100, rows


def main():
    print(f"Loading: {MODEL_PATH}")
    bst = xgb.Booster()
    bst.load_model(str(MODEL_PATH))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    # total rounds in multi:softprob = total_trees / num_classes
    num_class = len(labels)
    total_trees = bst.num_boosted_rounds()
    orig_size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"Num classes:   {num_class}")
    print(f"Boosting rounds (iterations): {total_trees}")
    print(f"Original size: {orig_size_mb:.1f} MB")
    print()

    # sweep truncation points
    points = [800, 600, 400, 300, 200, 150, 100, 80, 50]
    points = [k for k in points if k <= total_trees]
    if total_trees not in points:
        points.insert(0, total_trees)

    print("Evaluating each truncation (no retraining, using iteration_range):")
    print(f"{'k_rounds':>10}  {'size_est_MB':>12}  {'avg':>6}  {'>=70':>5}  {'>=90':>5}  {'exact':>6}")
    print("-" * 60)

    results = []
    for k in points:
        # evaluate by iteration_range
        avg, p70, p90, p100, rows = evaluate(bst, labels, k)
        # estimate size: proportional to k/total (roughly)
        size_est = orig_size_mb * k / total_trees
        results.append((k, size_est, avg, p70, p90, p100, rows))
        print(f"{k:>10}  {size_est:>11.1f}M  {avg:>6.1f}  {p70:>5}  {p90:>5}  {p100:>6}")
    print()

    # actually save pruned models for the interesting k points (accuracy maintained)
    # user wants to keep original intact -> save NEW files with suffix _pruned_k
    # pick k where avg drop is < 0.5 AND size is meaningfully smaller
    baseline_avg = results[0][2]
    print("Saving pruned models (<= 0.5 pt avg drop from baseline):")
    for k, size_est, avg, p70, p90, p100, rows in results:
        if k == points[0]: continue  # skip baseline (original size)
        if baseline_avg - avg > 0.5: continue
        out_path = MODEL_PATH.with_name(f"xgb_pure_pruned_{k}r.ubj")
        sliced = bst[0:k]
        sliced.save_model(str(out_path))
        actual_size = out_path.stat().st_size / 1024 / 1024
        print(f"  k={k:>4}  saved={actual_size:>6.1f} MB  avg={avg:.1f}  out={out_path.name}")


if __name__ == "__main__":
    main()
