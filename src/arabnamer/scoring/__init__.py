"""Arabic text normalization and similarity scoring.

Re-exports the same functions used in the thesis — no logic changes.
"""
from .arabic_normalizer import (
    normalize_arabic,
    normalize_semester,
    fuzzy_match,
    fuzzy_match_semester,
    is_placeholder,
    is_field_filled,
)

__all__ = [
    "normalize_arabic",
    "normalize_semester",
    "fuzzy_match",
    "fuzzy_match_semester",
    "is_placeholder",
    "is_field_filled",
]
