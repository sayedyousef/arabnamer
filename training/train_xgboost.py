"""New training run on the enriched Google dict (hamza fixes + artificial data).

Dict: _dict_FINAL_clean_arab_with_jrc_hamza_plus_allah_plus_bin_plus_alel.json
  - 13,753 Google-validated Arab names (hamza-fixed)
  - + 3,399 Abd+DivineName compounds (99 names of Allah + Allah)
  - + 4,289 Bin/Ben/Ibn + common-name compounds (space & no-space)
  - + 2,063 Al/El + divine/surname compounds (space & no-space)
  - Total: 23,504 entries

Training filter:
  - English keys must be pure [a-z'] (no spaces) — char-level features expect tokens
  - Arabic values must contain no space (skip multi-token compound Arabic)
  - Both must be <= PAD_LEN chars
  The single-token Al/El entries (e.g. alnour -> النور, aldesouky -> الدسوقي)
  DO make it through and contribute new char-alignment examples.

Config: 800 trees, depth 12, lr 0.1, all 33 features.
"""
import csv
import json
import re
import string
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")

_REPO_ROOT = Path(__file__).resolve().parent.parent
DICT_PATH = _REPO_ROOT / "dataset" / "dict_FINAL.json"
TS = datetime.now().strftime("%Y_%m_%d_%H%M")
MODEL_PATH = _REPO_ROOT / "model" / f"xgb_pure_{TS}.ubj"
LABELS_PATH = _REPO_ROOT / "model" / f"xgb_pure_{TS}_labels.json"
ALIAS_MODEL = _REPO_ROOT / "model" / "xgb_pure_model.ubj"
ALIAS_LABELS = _REPO_ROOT / "model" / "xgb_pure_labels.json"
OUTPUT_CSV = _REPO_ROOT / "training" / f"_xgb_pure_{TS}_output.csv"

PAD_LEN = 13
ALPHA = list(string.ascii_lowercase) + ["'"]
CHAR_IDX = {c: i + 1 for i, c in enumerate(ALPHA)}
VOWELS = set("aeiouy"); SIBILANTS = set("sz"); PLOSIVES = set("bdgkptq")
NASALS = set("mn"); LIQUIDS = set("lr"); FRICATIVES = set("fvh")
GLIDES = set("wy"); CONSONANTS = set("bcdfghjklmnpqrstvwxz")

MERGE_ARTICLES = {"al", "el", "ul", "ed", "ad", "abd", "abdul", "abdel", "abdal",
                  "abou", "abo", "abdou",
                  "bin", "ben", "ibn", "binn", "benn"}

TESTS = [
    ("Mohammed Ali","محمد علي"),("Ahmad Hassan","أحمد حسن"),
    ("Omar Khalil","عمر خليل"),("Fatima Mansour","فاطمة منصور"),
    ("Samir Ibrahim","سمير إبراهيم"),("Layla Al Saleh","ليلى الصالح"),
    ("Khalid El Masri","خالد المصري"),("Noor Rashid","نور راشد"),
    ("Yasmin Farouk","ياسمين فاروق"),("Hasan El Amin","حسن الأمين"),
    ("Karim Shawqi","كريم شوقي"),("Zaynab Nasser","زينب ناصر"),
    ("Bilal Othman","بلال عثمان"),("Salma Dahlan","سلمى دحلان"),
    ("Rania Hakim","رانيا حكيم"),("Tamer Abdel Rahim","تامر عبد الرحيم"),
    ("Mariam Habib","مريم حبيب"),("Hassan El Khatib","حسن الخطيب"),
    ("Nada Ramzi","ندى رمزي"),("Abdelrahman Saber","عبد الرحمن صابر"),
    ("Tariq Salama","طارق سلامة"),("Reem Abdul Karim","ريم عبد الكريم"),
    ("Adam Ayoub","آدم أيوب"),("Ghadeer Anwar","غدير أنور"),
    ("Mostafa Al Khatib","مصطفى الخطيب"),
]

EN_AR_COMPAT = {
    "a":{"ا","أ","إ","آ","ى","ع","ة","ه"}, "b":{"ب","پ"},
    "c":{"ك","س","ص","ق","ش"}, "d":{"د","ض","ذ","ظ"},
    "e":{"ي","ا","إ","ى","ع","ة","ه"}, "f":{"ف","ڤ"},
    "g":{"ج","غ","ق","ك","گ"}, "h":{"ه","ح","خ","ة"},
    "i":{"ي","ى","إ","ا","ع"},
    "j":{"ج","ي","ژ"}, "k":{"ك","ق","خ"}, "l":{"ل"},
    "m":{"م"}, "n":{"ن"},
    "o":{"و","ا","ؤ","ع","أ"}, "p":{"ب","پ","ف"},
    "q":{"ق","ك"}, "r":{"ر"},
    "s":{"س","ص","ش","ز","ث"}, "t":{"ت","ط","ث","ة"},
    "u":{"و","ا","أ","ؤ","ع"},
    "v":{"ف","ڤ","و"}, "w":{"و","ؤ"},
    "x":{"كس","خ"}, "y":{"ي","ى","ج"},
    "z":{"ز","ظ","ذ","ض"},
    "'":{"ع","ء","أ","إ","ؤ","ئ"}, "`":{"ع","ء"},
}


def _cost(e, a): return 0 if a in EN_AR_COMPAT.get(e, set()) else 3


def align(en, ar):
    m, n = len(en), len(ar)
    dp = [[0]*(n+1) for _ in range(m+1)]
    bp = [[None]*(n+1) for _ in range(m+1)]
    for j in range(1, n+1): dp[0][j] = j; bp[0][j] = ("ins", j-1)
    for i in range(1, m+1):
        dp[i][0] = i; bp[i][0] = ("del", None)
        for j in range(1, n+1):
            cm = dp[i-1][j-1] + _cost(en[i-1], ar[j-1])
            cd = dp[i-1][j] + 1
            ci = dp[i][j-1] + 1
            b = min(cm, cd, ci); dp[i][j] = b
            bp[i][j] = ("match", j-1) if b == cm else ("del", None) if b == cd else ("ins", j-1)
    labels = [[] for _ in range(m)]
    i, j = m, n
    while i > 0 or j > 0:
        if bp[i][j] is None: break
        op, idx = bp[i][j]
        if op == "match": labels[i-1].insert(0, ar[idx]); i -= 1; j -= 1
        elif op == "del": i -= 1
        else:
            if i == 0: j -= 1
            else: labels[i-1].insert(0, ar[idx]); j -= 1
    return ["".join(l) for l in labels]


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


def build_training():
    """Include compound entries (AR with spaces) too — the per-EN-char label
    can absorb spaces/prefix tokens via NW alignment. Only skip entries whose
    English side has a space (our training unit is one EN token)."""
    pairs = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    X, y = [], []
    skipped_bad_char = 0
    skipped_too_long = 0
    kept_simple = 0
    kept_compound = 0
    for en, ar in pairs.items():
        en = en.lower().strip(); ar = ar.strip()
        if not en or not ar: continue
        if " " in en:
            skipped_bad_char += 1; continue
        if any(c not in string.ascii_lowercase + "'" for c in en):
            skipped_bad_char += 1; continue
        # allow compound AR (with spaces); adjust length check accordingly
        if len(en) > PAD_LEN or len(ar) > PAD_LEN + 4:  # room for "ال " prefix etc.
            skipped_too_long += 1; continue
        labels = align(en, ar)
        for pos in range(len(en)):
            X.append(featurize(en, pos))
            y.append(labels[pos])
        if " " in ar:
            kept_compound += 1
        else:
            kept_simple += 1
    return np.array(X, dtype=np.int32), y, {
        "kept": kept_simple + kept_compound,
        "kept_simple": kept_simple,
        "kept_compound": kept_compound,
        "too_long": skipped_too_long,
        "bad_char": skipped_bad_char,
    }


def strip_tashkeel(s): return re.sub(r"[\u064B-\u0652\u0670\u0653-\u0655]", "", s or "")


def normalize(s):
    s = strip_tashkeel(s)
    return s.translate(str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ة':'ه','ى':'ي'})).strip()


def lenient(c, r):
    a, b = normalize(c), normalize(r)
    if not a or not b: return 0
    return round(max(fuzz.ratio(a, b), fuzz.partial_ratio(a, b)), 1)


def main():
    print(f"Dict:  {DICT_PATH}")
    print(f"Model: {MODEL_PATH}")
    print()
    X, y_raw, stats = build_training()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Training pairs kept:       {stats['kept']:,}")
    print(f"  Simple (AR single-tok):  {stats['kept_simple']:,}")
    print(f"  Compound (AR multi-tok): {stats['kept_compound']:,}  (NOW INCLUDED in training)")
    print(f"Skipped (en has space):    {stats['bad_char']:,}")
    print(f"Skipped (too long):        {stats['too_long']:,}")
    print(f"Training rows (char-level):{len(X):,}")
    print(f"Classes:                   {len(le.classes_)}")
    print(f"Features per row:          {X.shape[1]}")
    print()

    print("Training XGBoost: 800 trees, depth 12, lr 0.1, n_jobs=-1...")
    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=12, learning_rate=0.1,
        objective="multi:softprob", num_class=len(le.classes_),
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X, y)

    model.get_booster().save_model(str(MODEL_PATH))
    LABELS_PATH.write_text(json.dumps(list(le.classes_), ensure_ascii=False), encoding="utf-8")
    import shutil
    shutil.copy(MODEL_PATH, ALIAS_MODEL)
    shutil.copy(LABELS_PATH, ALIAS_LABELS)
    msize = MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"Saved:    {MODEL_PATH}  ({msize:.1f} MB)")
    print(f"Aliased:  {ALIAS_MODEL}")
    print()

    # ---------- eval on 25 DI names, model-only (no dict) ----------
    def predict(name_en):
        tokens = [t for t in name_en.lower().replace("-", " ").split() if t]
        merged, i = [], 0
        while i < len(tokens):
            t = tokens[i]
            if t in MERGE_ARTICLES and i+1 < len(tokens):
                merged.append(t + tokens[i+1]); i += 2
            else:
                merged.append(t); i += 1
        out = []
        for tok in merged:
            if not tok or len(tok) > PAD_LEN:
                out.append(tok); continue
            Xt = np.array([featurize(tok, k) for k in range(len(tok))], dtype=np.int32)
            preds = model.predict(Xt)
            out.append("".join(le.classes_[p] for p in preds))
        return " ".join(out)

    rows = []
    for en, ref in TESTS:
        ar = predict(en)
        rows.append({
            "name_en": en, "prediction": ar,
            "score_lenient": lenient(ar, ref),
            "reference": ref,
        })
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    avg = sum(r["score_lenient"] for r in rows) / len(rows)
    p70 = sum(1 for r in rows if r["score_lenient"] >= 70)
    p90 = sum(1 for r in rows if r["score_lenient"] >= 90)
    p100 = sum(1 for r in rows if r["score_lenient"] >= 100)
    print(f"=== ENRICHED MODEL EVAL (model-only, lenient scoring) ===")
    print(f"  Avg:          {avg:.1f}")
    print(f"  Pass >=70:    {p70}/25")
    print(f"  Pass >=90:    {p90}/25")
    print(f"  Exact =100:   {p100}/25")
    print()
    for r in rows:
        m = "OK" if r["score_lenient"] >= 70 else "NO"
        print(f"  {m} {r['name_en']:<25} -> {r['prediction']:<32} score={r['score_lenient']}  ref={r['reference']}")


if __name__ == "__main__":
    main()
