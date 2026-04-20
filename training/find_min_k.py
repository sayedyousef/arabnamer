"""Binary search for the LOWEST k that gives identical metrics to k=800 baseline.

Baseline (k=800): avg=98.4, p70=25, p90=24, exact=21
Known identical at k=400. Known different at k=300 (avg 98.3).
Search between (300, 400] to find minimum identical k.
"""
import csv, json, re, string, sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = _REPO_ROOT / "model" / "xgb_pruned_386r.ubj"
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
    feats += [1 if ch in VOWELS else 0, 1 if ch in CONSONANTS else 0,
              1 if ch in SIBILANTS else 0, 1 if ch in PLOSIVES else 0,
              1 if ch in NASALS else 0, 1 if ch in LIQUIDS else 0,
              1 if ch in FRICATIVES else 0, 1 if ch in GLIDES else 0]
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


def eval_k(bst, labels, k):
    def predict_tok(tok):
        if not tok or len(tok) > PAD_LEN: return tok
        Xt = np.array([featurize(tok, i) for i in range(len(tok))], dtype=np.int32)
        probs = bst.predict(xgb.DMatrix(Xt), iteration_range=(0, k))
        return "".join(labels[int(p)] for p in probs.argmax(axis=1))

    def predict(name_en):
        tokens = [t for t in name_en.lower().replace("-", " ").split() if t]
        merged, i = [], 0
        while i < len(tokens):
            t = tokens[i]
            if t in MERGE_ARTICLES and i+1 < len(tokens):
                merged.append(t + tokens[i+1]); i += 2
            else:
                merged.append(t); i += 1
        return " ".join(predict_tok(t) for t in merged)

    preds = [(en, predict(en), ref) for en, ref in TESTS]
    scores = [lenient(p, r) for _, p, r in preds]
    return {
        "avg": round(sum(scores)/len(scores), 1),
        "p70": sum(1 for s in scores if s >= 70),
        "p90": sum(1 for s in scores if s >= 90),
        "exact": sum(1 for s in scores if s >= 100),
        "preds": preds,
        "scores": scores,
    }


def identical(a, b):
    return (a["avg"] == b["avg"] and a["p70"] == b["p70"]
            and a["p90"] == b["p90"] and a["exact"] == b["exact"]
            and a["scores"] == b["scores"])


def main():
    print(f"Loading {MODEL_PATH}")
    bst = xgb.Booster(); bst.load_model(str(MODEL_PATH))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    total = bst.num_boosted_rounds()
    print(f"Total rounds: {total}")

    baseline = eval_k(bst, labels, total)
    print(f"Baseline k={total}: avg={baseline['avg']} p70={baseline['p70']} p90={baseline['p90']} exact={baseline['exact']}")
    print()

    # binary search between lo=300 (known different) and hi=400 (known identical)
    lo, hi = 300, 400
    tested = {}
    print(f"Binary search in ({lo}, {hi}]:")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        r = eval_k(bst, labels, mid)
        tested[mid] = r
        same = identical(r, baseline)
        print(f"  k={mid:>4} avg={r['avg']} p70={r['p70']} p90={r['p90']} exact={r['exact']}  identical={same}")
        if same:
            hi = mid
        else:
            lo = mid
    min_k = hi
    print()
    print(f"Minimum identical k = {min_k}")

    # save that model
    sliced = bst[0:min_k]
    out_path = MODEL_PATH.with_name(f"xgb_pure_pruned_MIN_{min_k}r.ubj")
    sliced.save_model(str(out_path))
    size = out_path.stat().st_size / 1024 / 1024
    orig = MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"Saved: {out_path.name}  size={size:.1f} MB  (orig {orig:.1f} MB, {100*size/orig:.1f}%)")


if __name__ == "__main__":
    main()
