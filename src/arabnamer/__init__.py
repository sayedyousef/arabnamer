"""arabnamer - English -> Arabic name transliteration and similarity.

Nothing is reinvented: this package is a repackaging of the thesis work
into an installable library. The scorer, normalizer, rule-based walker,
and the XGBoost model are the same artifacts used in the thesis.

Public API:
    translit(name, reference=None)         -> Result
    translit_batch(names, references=None) -> list[Result]
    similarity(a, b, threshold=85)         -> (passed, score)

    Transliterator(engine, threshold)      -> orchestrator class
    Result                                 -> rich transliteration output

Also re-exported from submodules for direct use:
    scoring.fuzzy_match, scoring.normalize_arabic     (verbatim thesis functions)
    rules.transliterate_name                          (verbatim thesis rule walker)
    prediction.XGBoostTransliterator                  (thin model loader)
"""
from __future__ import annotations

from .core import Transliterator, Result
from .scoring import fuzzy_match, normalize_arabic

__version__ = "0.1.3"
__all__ = [
    "translit", "translit_batch", "similarity",
    "Transliterator", "Result",
    "fuzzy_match", "normalize_arabic",
    "__version__",
]

_default: Transliterator | None = None


def _get_default() -> Transliterator:
    global _default
    if _default is None:
        _default = Transliterator(engine="model", threshold=85)
    return _default


def translit(name: str, reference: str | None = None) -> Result:
    """Transliterate one English name to Arabic (XGBoost, threshold=85).
    For many calls, instantiate `Transliterator` once and reuse."""
    return _get_default().translit(name, reference)


def translit_batch(names: list[str], references: list[str | None] | None = None) -> list[Result]:
    """Batch version of `translit`. Provide `references` (aligned to names) to
    get per-result scores and `accepted` flags."""
    return _get_default().translit_batch(names, references)


def similarity(a: str, b: str, threshold: int = 85) -> tuple[bool, int]:
    """Arabic<->Arabic lenient similarity (thesis scorer, verbatim).

    Returns `(passed, score)` where `passed` is True iff `score >= threshold`.
    Score is 0..100.
    """
    return fuzzy_match(a, b, threshold=threshold)
