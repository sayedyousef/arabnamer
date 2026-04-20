"""Custom English -> Arabic name transliterator.

Pure-Python, offline, zero dependencies. Rule-based with:
  - Multi-letter digraph/trigraph lookup (sh -> ش, kh -> خ, ough -> و)
  - Position-aware rules (start / middle / end of word)
  - Common Arabic-name prefix handling (Abd, Abu, Al/El -> ال, عبد, ابو)
  - Feminine ending (-a, -ah -> ة)
  - Double-consonant collapse
  - Apostrophe -> ع (e.g., Da'na -> دعنا)

Designed to be TUNEABLE. Edit the tables below; every rule is visible.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Rule tables — edit freely. Order matters within each table (longest first).
# ---------------------------------------------------------------------------

# Whole-word overrides for very common Arabic-transliterated prefixes/fragments.
# Applied before char-by-char walk. Matches are case-insensitive.
PREFIX_EXACT = {
    "abd":     "عبد",
    "abu":     "أبو",
    "abou":    "أبو",
    "umm":     "أم",
    "ibn":     "ابن",
    "bin":     "بن",
    "el":      "ال",
    "al":      "ال",
    "ul":      "ال",
    "ed":      "الد",
    "ad":      "الد",
}

# Multi-character sequences checked BEFORE single-letter fallback.
# The loop sorts by length (longest first), so 3-letter trigraphs win over
# 2-letter digraphs, which win over single-letter mappings.
DIGRAPHS = {
    # ---- 4-letter (very rare) ----
    "ough": "و",         # "through" → ثرو
    # ---- 3-letter trigraphs ----
    "tch":  "تش",        # "match" → ماتش
    "dge":  "ج",         # "bridge" → بريج
    "oul":  "ول",
    "aou":  "او",
    "aoo":  "او",
    "eau":  "و",         # French-style "eau" → و
    "eou":  "يو",
    # ---- 2-letter consonant digraphs ----
    "sh":   "ش",
    "ch":   "تش",
    "th":   "ث",
    "kh":   "خ",
    "gh":   "غ",
    "ph":   "ف",
    "dh":   "ذ",
    "ck":   "ك",
    "qu":   "كو",
    "dj":   "ج",
    "ng":   "نج",        # "ng" often stays ن+ج
    "zh":   "ج",         # French "j" sound
    # ---- 2-letter vowel combos (a-series) ----
    "aa":   "ا",         # double a collapses
    "ai":   "اي",
    "ay":   "اي",
    "au":   "او",
    "aw":   "او",
    "ae":   "اي",
    "ao":   "او",
    # ---- 2-letter vowel combos (e-series) ----
    "ee":   "ي",
    "ei":   "ي",
    "ey":   "اي",
    "eu":   "يو",
    "ea":   "يا",
    "eo":   "يو",
    # ---- 2-letter vowel combos (i-series) ----
    "ia":   "يا",
    "ie":   "ي",
    "io":   "يو",
    "iu":   "يو",
    # ---- 2-letter vowel combos (o-series) ----
    "oo":   "و",
    "ou":   "و",
    "ow":   "او",
    "oe":   "و",
    "oi":   "وي",
    "oy":   "وي",
    # ---- 2-letter vowel combos (u-series) ----
    "ua":   "وا",
    "ue":   "و",
    "ui":   "وي",
    "uo":   "وو",
    # ---- 2-letter starting with y (semi-vowel) ----
    "ya":   "يا",
    "ye":   "يي",
    "yo":   "يو",
    "yu":   "يو",
    # ---- 2-letter starting with w (semi-vowel) ----
    "wa":   "وا",
    "we":   "وي",
    "wi":   "وي",
    "wo":   "وو",
    "wu":   "وو",
}

# Single-letter mapping (default, middle-of-word context).
SINGLE = {
    "a": "ا", "b": "ب", "c": "ك", "d": "د",
    "e": "",            # e is usually silent in the middle; vowels ride elsewhere
    "f": "ف", "g": "ج", "h": "ح", "i": "ي",
    "j": "ج", "k": "ك", "l": "ل", "m": "م",
    "n": "ن", "o": "و", "p": "ب", "q": "ق",
    "r": "ر", "s": "س", "t": "ت", "u": "و",
    "v": "ف", "w": "و", "x": "كس", "y": "ي",
    "z": "ز",
    "'": "ع",           # apostrophe often marks ع (Da'na -> دعنا)
    "`": "ع",
}

# Start-of-word DIGRAPHS checked BEFORE single-vowel rules.
# Arab-name convention: vowel at start usually takes a hamza'd alef (أ / إ).
#   Ah- -> أح (Ahmad -> أحمد, Ahmed -> أحمد)
#   Ay- -> أي (Ayman -> أيمن, Ayhab -> إيهاب)
#   Ih- -> إي (Ihsan, Ibrahim-style)
#   Is- -> إس (Ismail -> إسماعيل)  [short-i at start often gets kasra/إ]
#   Om- -> عم (Omar -> عمر; 'O' at start of Arab names often = ع)
START_DIGRAPHS = {
    "ah": "أح",     # hamza-above alef + ح (Arab-name Ah- is usually Ahmad-class)
    "ay": "أي",
    "au": "أو",
    "aa": "آ",
    "ab": "أب",     # standalone; "abd" caught earlier by PREFIX_EXACT
    "eh": "إ",
    "ih": "إي",
    # "is": intentionally NOT here — collides with double-s names like "Issam".
    # Default path via START_VOWEL("i") -> إ handles both Ismail & Issam correctly.
    "ib": "إب",     # Ibrahim -> إبراهيم (usually)
    "oh": "أو",
    "um": "أم",     # Umm -> أم
    "uh": "أو",
    "ou": "أو",     # Oud / Our-
}

# When a single vowel starts a word, default to the MOST COMMON Arab-name
# mapping for that leading vowel. More specific 2/3-letter starts in
# START_DIGRAPHS override these (Ah- -> أح, Ay- -> أي, Ab- matched prefix,
# etc.) so this default only kicks in when nothing more specific matches.
#
# Rationale for defaults:
#   "A" alone at start → ع  (Ali -> علي, Arar -> عرار, Asad -> عساد)
#                            — the "أ" cases are caught by Ah/Ay/Au/Aa/Ab above.
#   "E" alone at start → إ  (hamza-below)
#   "I" alone at start → إ
#   "O" alone at start → أو  (Omar-class mostly ع, but not decidable; safe default)
#   "U" alone at start → أو
START_VOWEL = {
    "a": "ع",
    "e": "إ",
    "i": "إ",
    "o": "أو",
    "u": "أو",
}

# 'al'/'el' + consonant prefix (Alissa, Elias, Alhambra, ...) -> ال + rest.
# Handled inside the loop as a position-0 special case.
AL_PREFIX = ("al", "el", "ul")

# Middle-of-word hamza rules. These are applied inside the main loop after
# digraphs fail. They model cases where a glottal stop or hamza-carrier
# appears in common Arab names.
MIDDLE_HAMZA = {
    "ae": "ائ",    # Nael -> نائل, Mikhael -> ميخائيل, Rafael -> رافائيل
    "ua": "ؤا",    # Muaz-style (some names); safe alternative to وا in middle
}

# When a vowel ends a word, Arabic usually uses a full vowel letter.
END_VOWEL = {
    "a": "ا",
    "e": "ي",           # final -e often sounds like -i (Natalie -> ناتالي)
    "i": "ي",
    "o": "و",
    "u": "و",
    "y": "ي",
}

# Known name-final patterns that override END_VOWEL.
# Feminine ending convention: -a/-ah/-at on feminine names -> ة
FEMININE_HINTS = ("ah",)  # e.g., "Sarah" -> ends in ة
# Add name stems here if you want to force ة for specific first names.
KNOWN_FEMININE_STEMS = {"marwa", "sarah", "diala", "rabia", "ferdoos", "mona", "huda"}

# End-of-word 2-letter patterns. In Arab names, a trailing short-vowel + consonant
# often represents a long vowel in Arabic script:
#   "Nassar"  -> نصار  (-ar  -> ار)
#   "Arar"    -> عرار  (-ar  -> ار)
#   "Barbar"  -> بربر  (NOT every -ar; depends on word, but common default)
# Expand as needed.
# 3-letter END patterns — higher priority than 2-letter (more specific wins).
# These handle cases where a 2-letter rule would over-extend.
END_TRIGRAPHS = {
    # -d family (Mohammed, Muhammad, Ahmad end in just -d — short vowel)
    "med": "مد",      # Mohammed -> محمد
    "mad": "مد",      # Muhammad, Ahmad -> مد
    "had": "حد",      # Rashad, Sahad
    # -long-vowel-d (Arabic writes the long vowel explicitly)
    "eed": "يد",      # Saeed -> سعيد
    "aad": "اد",      # Saad sometimes -> سعاد (feminine) else سعد
    "oud": "ود",      # Dawoud -> داود
    "ood": "ود",
    "ied": "يد",      # Majied, Fried-style
    "aid": "يد",      # Said -> سعيد
    # -r family trigraphs
    "aar": "ار",
    "eer": "ير",
    "oor": "ور",
    "our": "ور",
    # -n trigraphs
    "aan": "ان",
    "een": "ين",
    "iin": "ين",
    "oon": "ون",
    # generic silent-ed (English-style names) — consume the e
    "eds": "دز",
}

# 2-letter END patterns. Applied only when no 3-letter END match.
# Removed "ed" and "ad" from this layer — they caused regressions on names
# like Mohammed/Muhammad/Ahmad where the vowel shouldn't materialize.
END_DIGRAPHS = {
    # -r family: Nassar -> نصار, Arar -> عرار, Badir -> بدير
    "ar": "ار", "ir": "ير", "ur": "ور", "or": "ور", "er": "ير",
    # -n family: Hassan -> حسان, Husein -> حسين
    "an": "ان", "in": "ين", "on": "ون", "un": "ون", "en": "ين",
    # -l family
    "al": "ال", "il": "يل", "ul": "ول", "ol": "ول",
    # -s family
    "as": "اس", "is": "يس", "us": "وس", "os": "وس", "es": "يس",
    # -m family: Hosam -> حسام, Karim -> كريم
    "am": "ام", "im": "يم", "um": "وم", "om": "وم",
    # -t family
    "at": "ات", "it": "يت", "ut": "وت", "ot": "وت",
    # -b family: Kassab -> كساب, Habib -> حبيب
    "ab": "اب", "ib": "يب", "ub": "وب", "ob": "وب", "eb": "يب",
    # -k family: Malik -> مالك
    "ak": "اك", "ik": "يك", "uk": "وك", "ok": "وك",
    # -q family: Tariq -> طارق (but ق is a lexical choice)
    "aq": "اق", "iq": "يق", "uq": "وق", "oq": "وق",
    # -z family: Fawaz -> فواز, Aziz -> عزيز
    "az": "از", "iz": "يز", "uz": "وز", "oz": "وز",
    # -g family
    "ag": "اج", "ig": "يج", "ug": "وج", "og": "وج",
    # -f family: Yusuf -> يوسف, Sharif -> شريف
    "af": "اف", "if": "يف", "uf": "وف", "of": "وف",
    # -p (rare but English names)
    "ap": "اب", "ip": "يب", "up": "وب", "op": "وب",
    # -j family
    "aj": "اج", "ij": "يج", "uj": "وج", "oj": "وج",
}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

_DIGRAPH_KEYS_SORTED = sorted(DIGRAPHS.keys(), key=len, reverse=True)


_DOUBLE_PLUS_VOWEL_RE = re.compile(r"([^aeiouy'\s])\1[aeiou]")
_DOUBLED_CONSONANT_RE = re.compile(r"([^aeiouy'\s])\1")


def _preprocess_doubles(s: str) -> str:
    """User rule: doubled consonant + vowel is preprocessed to a single consonant.
      'ssa' -> 'ss' -> 's'  ('Hassan' -> 'Hasn', 'Mohammed' -> 'Mohamd')
      'mme' -> 'mm' -> 'm'
    Applied once before any other rule. Keeps the rest of the loop simple."""
    s = _DOUBLE_PLUS_VOWEL_RE.sub(r"\1\1", s)
    s = _DOUBLED_CONSONANT_RE.sub(r"\1", s)
    return s


def _translit_word(word: str) -> str:
    """Transliterate a single English name token to Arabic."""
    if not word:
        return ""
    lower = _preprocess_doubles(word.lower())

    # 1. Whole-word prefix overrides (Abd, Al, El, ...)
    if lower in PREFIX_EXACT:
        return PREFIX_EXACT[lower]

    # 2. Multi-word form triggered by leading prefix (e.g., "Abdullah" contains Abd)
    # — skipped on purpose; users should separate with space if compound (Abd Allah).

    result: list[str] = []
    i = 0
    n = len(lower)
    just_doubled = False  # set True immediately after a doubled-consonant collapse

    while i < n:
        ch = lower[i]

        # ----- Position = start of word: 2-letter patterns first -----
        if i == 0:
            two = lower[0:2]
            # 'Al-'/'El-' + consonant -> ال  (Alissa -> العيسى, Elias -> الياس)
            if two in AL_PREFIX and n > 2 and lower[2] not in "aeiouy":
                result.append("ال")
                i += 2
                continue
            if two in START_DIGRAPHS:
                result.append(START_DIGRAPHS[two])
                i += 2
                continue
            if ch in START_VOWEL:
                result.append(START_VOWEL[ch])
                i += 1
                continue

        # ----- Middle-position hamza patterns (Nael -> نائل, etc.) -----
        if 0 < i < n - 1:
            two = lower[i:i + 2]
            if two in MIDDLE_HAMZA and i + 2 < n:
                # Only fire when something follows — end-of-word endings are
                # handled later so the feminine/vowel rules can win.
                result.append(MIDDLE_HAMZA[two])
                i += 2
                continue

        # ----- Try digraph/trigraph lookup -----
        matched = False
        for key in _DIGRAPH_KEYS_SORTED:
            if lower.startswith(key, i):
                result.append(DIGRAPHS[key])
                i += len(key)
                matched = True
                break
        if matched:
            continue

        # ----- Double consonant: collapse (Hassan -> حسن, Bassel -> باسل) -----
        if i + 1 < n and ch == lower[i + 1] and ch.isalpha() and ch not in "aeiouy":
            # Emit once, consume two
            result.append(SINGLE.get(ch, ch))
            i += 2
            continue

        # ----- End-of-word 2-letter patterns (long-vowel convention in Arabic) -----
        # e.g., Nassar's last "ar" -> ار so the output contains a long alef.
        if i == n - 2:
            two = lower[i:i + 2]
            if two in END_DIGRAPHS:
                result.append(END_DIGRAPHS[two])
                i += 2
                continue

        # ----- End-of-word vowel handling -----
        if i == n - 1 and ch in END_VOWEL:
            # Feminine hint: word ends in 'ah' (consumed as single char 'h' loop; handled below)
            # Here we just handle bare trailing vowel.
            result.append(END_VOWEL[ch])
            i += 1
            continue

        # ----- Feminine ending 'ah' -> ة, e.g., Sarah -> سارة -----
        if ch == "a" and i + 1 == n - 1 and lower[i + 1] == "h":
            result.append("ة")
            i += 2  # consume a + h
            continue

        # ----- Silent middle 'e' (e.g., Desouky -> دسوقي) -----
        if ch == "e" and 0 < i < n - 1:
            i += 1
            continue

        # ----- Short vowels between consonants — drop (Arabic is consonant-heavy).
        # Keeps the Arabic-script compact like native spelling:
        #   "Mohammed" = M-o-h-a-m-m-e-d -> drop o/a/e -> محمد (approx)
        #   "Hassan"   = H-a-s-s-a-n     -> drop a's -> حسن (approx)
        # 'i' is NOT in this set — it usually represents the long 'ee' sound in
        # Arabic names ("Jebril" -> جبريل, keep the ي). Digraph 'ee' handled earlier.
        VOWELS_DROP = {"a", "u", "o"}
        if (
            ch in VOWELS_DROP
            and 0 < i < n - 1
            and lower[i - 1].isalpha() and lower[i - 1] not in "aeiouy'`"
            and lower[i + 1].isalpha() and lower[i + 1] not in "aeiouy'`"
        ):
            i += 1
            continue

        # ----- Default single-letter mapping -----
        mapped = SINGLE.get(ch)
        if mapped is not None:
            result.append(mapped)
        # unmapped chars (digits, punctuation) silently dropped
        i += 1

    out = "".join(result)

    # ----- Feminine stem post-rule: force ة if known stem ends in 'a' -----
    stem = lower.rstrip("h")  # Sarah/Sara both qualify
    if stem in KNOWN_FEMININE_STEMS and out.endswith("ا"):
        out = out[:-1] + "ة"

    return out


# Article words that in Arab-name convention attach to the following token:
#   "Ayman El Desouky" -> "Ayman El-Desouky" -> "أيمن الدسوقي"
#   "Basim El Tweissi" -> "Basim El-Tweissi" -> "باسم الطويسي"
ARTICLE_TOKENS = {"el", "al", "ul", "ed", "ad"}


def _merge_articles(parts: list[str]) -> list[str]:
    """Fuse 'El'/'Al' tokens into the following word so the AL_PREFIX rule
    inside _translit_word can emit 'ال' properly. Original spacing ignored."""
    merged: list[str] = []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok.lower() in ARTICLE_TOKENS and i + 1 < len(parts):
            merged.append(tok + parts[i + 1])
            i += 2
        else:
            merged.append(tok)
            i += 1
    return merged


def transliterate_name(name_en: str) -> str:
    """Transliterate an English name phrase to Arabic token-by-token.
    Stand-alone 'El'/'Al' articles are fused into the next word first."""
    if not name_en or not name_en.strip():
        return ""
    parts = [p for p in name_en.replace("-", " ").split() if p]
    parts = _merge_articles(parts)
    return " ".join(_translit_word(p) for p in parts)


# ---------------------------------------------------------------------------
# Quick smoke check (not a formal test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        "Sayed Ali",
        "Mohammed Ali",
        "Ahmad Hassan",
        "Marwa Farag",
        "Ayman El Desouky",
        "Abdennour Benantar",
    ]
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for n in examples:
        print(f"{n:30s} -> {transliterate_name(n)}")
