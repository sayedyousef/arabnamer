"""
Arabic text normalization and fuzzy comparison.

Handles:
- Tashkeel (diacritics/harakat) removal
- Hamza/alef variant normalization (أ إ آ ا -> ا)
- Taa marbuta vs haa (ة -> ه)
- Tatweel/kashida removal (ـ)
- General whitespace normalization
"""
from __future__ import annotations

import re
from rapidfuzz import fuzz

# Arabic diacritics (tashkeel/harakat) Unicode ranges
_TASHKEEL_RE = re.compile(
    r"["
    r"\u0617-\u061A"   # Small high ligatures
    r"\u064B-\u0652"   # Fathatan, Dammatan, Kasratan, Fatha, Damma, Kasra, Shadda, Sukun
    r"\u0653-\u0655"   # Maddah, Hamza above/below
    r"\u0656-\u065F"   # Subscript alef, etc.
    r"\u0670"          # Superscript alef
    r"\uFC5E-\uFC63"   # Ligature presentation forms
    r"\uFE70-\uFE7F"   # Arabic presentation forms (isolated harakat)
    r"]"
)

# Tatweel (kashida) character
_TATWEEL_RE = re.compile(r"\u0640")

# Alef variants -> bare alef (ا)
_ALEF_VARIANTS = {
    "\u0622": "\u0627",  # آ -> ا (Alef with Madda)
    "\u0623": "\u0627",  # أ -> ا (Alef with Hamza Above)
    "\u0625": "\u0627",  # إ -> ا (Alef with Hamza Below)
    "\u0671": "\u0627",  # ٱ -> ا (Alef Wasla)
    "\u0672": "\u0627",  # ٲ -> ا (Alef with Wavy Hamza Above)
    "\u0673": "\u0627",  # ٳ -> ا (Alef with Wavy Hamza Below)
}

# Taa marbuta -> Haa
_TAA_MARBUTA = "\u0629"  # ة
_HAA = "\u0647"          # ه

# Yaa variants
_ALEF_MAQSURA = "\u0649"  # ى -> ي
_YAA = "\u064A"            # ي

# Whitespace normalization
_MULTI_SPACE_RE = re.compile(r"\s+")

# Arabic-Indic digits (٠-٩) and Extended Arabic-Indic digits (۰-۹) -> ASCII 0-9
_ARABIC_DIGITS = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    "\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9",
    "01234567890123456789",
)

# Bilingual season aliases — canonical form is English lowercase.
# Applied AFTER normalize_arabic so keys are already normalized (e.g. no tashkeel).
_SEASON_ALIASES = {
    "ربيع": "spring",
    "خريف": "fall",
    "صيف": "summer",
    "autumn": "fall",
    "او": "or",          # Arabic "أو" -> "او" after alef normalization
    "السنه": "",         # "السنة" -> "السنه" — template placeholder filler "the year"
}


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for comparison purposes.

    - Removes all tashkeel (diacritical marks)
    - Removes tatweel (kashida/elongation)
    - Normalizes alef variants (أ إ آ ا) to bare alef (ا)
    - Normalizes taa marbuta (ة) to haa (ه)
    - Normalizes alef maqsura (ى) to yaa (ي)
    - Collapses whitespace
    - Strips and lowercases (for Latin parts)
    """
    if not text:
        return ""

    # Remove tashkeel
    text = _TASHKEEL_RE.sub("", text)

    # Remove tatweel
    text = _TATWEEL_RE.sub("", text)

    # Normalize alef variants
    for old, new in _ALEF_VARIANTS.items():
        text = text.replace(old, new)

    # Normalize taa marbuta to haa
    text = text.replace(_TAA_MARBUTA, _HAA)

    # Normalize alef maqsura to yaa
    text = text.replace(_ALEF_MAQSURA, _YAA)

    # Normalize Arabic-Indic digits to ASCII
    text = text.translate(_ARABIC_DIGITS)

    # Normalize whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    # Lowercase Latin characters (for mixed Arabic-English text)
    text = text.lower()

    return text


def normalize_semester(text: str) -> str:
    """Normalize semester text for bilingual matching.

    Applies normalize_arabic, then maps season names to a canonical English form
    so 'Spring 2026' and 'ربيع ٢٠٢٦' both become 'spring 2026'.
    """
    text = normalize_arabic(text)
    for src, dst in _SEASON_ALIASES.items():
        text = text.replace(src, dst)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def fuzzy_match_semester(text_a: str, text_b: str, threshold: int = 100) -> tuple[bool, int]:
    """Compare two semester strings after bilingual normalization.

    Requires exact match on the normalized form. Semester is short and structured —
    'Spring 2025' vs 'Spring 2026' (one digit off) scores ~91% on a fuzzy ratio but is
    the wrong year, so typo-tolerance here creates false accepts.
    """
    norm_a = normalize_semester(text_a)
    norm_b = normalize_semester(text_b)
    if norm_a == norm_b:
        return True, 100
    return False, fuzz.ratio(norm_a, norm_b)


def fuzzy_match(text_a: str, text_b: str, threshold: int = 75) -> tuple[bool, int]:
    """
    Compare two texts using Arabic-normalized fuzzy matching.

    Returns:
        (is_match, score) where score is 0-100
    """
    norm_a = normalize_arabic(text_a)
    norm_b = normalize_arabic(text_b)

    # Try exact normalized match first
    if norm_a == norm_b:
        return True, 100

    # Fuzzy ratio
    score = fuzz.ratio(norm_a, norm_b)
    if score >= threshold:
        return True, score

    # Also try partial ratio (for substring matching)
    partial_score = fuzz.partial_ratio(norm_a, norm_b)
    if partial_score >= threshold:
        return True, partial_score

    return False, max(score, partial_score)


def is_placeholder(text: str, placeholder: str) -> bool:
    """Check if text is still a placeholder (exact or near-exact match to placeholder).
    Uses FULL ratio only — not partial — to avoid false positives when student's real
    data contains words that appear in the placeholder template.
    """
    norm_a = normalize_arabic(text)
    norm_b = normalize_arabic(placeholder)
    if norm_a == norm_b:
        return True
    score = fuzz.ratio(norm_a, norm_b)
    return score >= 85


def is_field_filled(text: str, placeholders: list[str]) -> bool:
    """
    Check if a field has been filled in (NOT still a placeholder).
    Returns True if the text appears to be real content, False if still a placeholder.
    """
    if not text or not text.strip():
        return False

    for ph in placeholders:
        if is_placeholder(text.strip(), ph):
            return False  # Still a placeholder

    return True  # Content has been changed
