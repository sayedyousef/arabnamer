"""Character-level feature extraction for the XGBoost transliterator.

Produces a fixed-width feature vector per (word, position). Identical to the
featurizer used at training time — any drift breaks the model.

Feature layout (34 ints per row):
    [0..12]   padded char IDs (PAD_LEN=13, 0 = pad)
    [13]      position index
    [14]      word length
    [15]      is_first
    [16]      is_last
    [17]      is_double_prev  (char[pos] == char[pos-1])
    [18]      is_double_next  (char[pos] == char[pos+1])
    [19]      is_english_vowel (aeiouy)
    [20..27]  phonetic-class flags (vowel, consonant, sibilant, plosive,
                                    nasal, liquid, fricative, glide)
    [28..29]  bigram IDs (prev-cur, cur-next)
    [30..32]  trigram IDs (prev-cur-next, prev2-prev-cur, cur-next-next2)
"""
from __future__ import annotations

import string

PAD_LEN = 13
_ALPHA = list(string.ascii_lowercase) + ["'"]
_CHAR_IDX = {c: i + 1 for i, c in enumerate(_ALPHA)}  # 0 = pad / unknown

_VOWELS = set("aeiouy"); _SIBILANTS = set("sz"); _PLOSIVES = set("bdgkptq")
_NASALS = set("mn"); _LIQUIDS = set("lr"); _FRICATIVES = set("fvh")
_GLIDES = set("wy"); _CONSONANTS = set("bcdfghjklmnpqrstvwxz")


def _cid(c: str) -> int:
    return _CHAR_IDX.get(c, 0)


def _bigram_id(a: str, b: str) -> int:
    return _cid(a) * 28 + _cid(b)


def _trigram_id(a: str, b: str, c: str) -> int:
    return _bigram_id(a, b) * 28 + _cid(c)


def featurize(word: str, pos: int) -> list[int]:
    """Build the 34-int feature vector for one character position in `word`."""
    n = len(word)
    w = word[:PAD_LEN]
    pad = [_cid(c) for c in w] + [0] * (PAD_LEN - len(w))
    ch = word[pos] if 0 <= pos < n else ""
    prev_c = word[pos - 1] if pos - 1 >= 0 else ""
    next_c = word[pos + 1] if pos + 1 < n else ""
    prev2 = word[pos - 2] if pos - 2 >= 0 else ""
    next2 = word[pos + 2] if pos + 2 < n else ""

    feats = pad + [
        pos, n,
        1 if pos == 0 else 0,
        1 if pos == n - 1 else 0,
        1 if 0 < pos < n - 1 and word[pos] == word[pos - 1] else 0,
        1 if 0 < pos < n - 1 and word[pos] == word[pos + 1] else 0,
        1 if ch in "aeiouy" else 0,
    ]
    feats += [
        1 if ch in _VOWELS else 0,
        1 if ch in _CONSONANTS else 0,
        1 if ch in _SIBILANTS else 0,
        1 if ch in _PLOSIVES else 0,
        1 if ch in _NASALS else 0,
        1 if ch in _LIQUIDS else 0,
        1 if ch in _FRICATIVES else 0,
        1 if ch in _GLIDES else 0,
    ]
    feats += [_bigram_id(prev_c, ch), _bigram_id(ch, next_c)]
    feats += [
        _trigram_id(prev_c, ch, next_c),
        _trigram_id(prev2, prev_c, ch),
        _trigram_id(ch, next_c, next2),
    ]
    return feats


def featurize_word(word: str) -> list[list[int]]:
    """Featurize every character in `word`. Returns an n x 34 list."""
    return [featurize(word, i) for i in range(len(word))]
