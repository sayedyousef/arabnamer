"""Result dataclass returned by the public transliteration API."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Result:
    """Result of a single transliteration call.

    Attributes:
        input:      original English input
        arabic:     predicted Arabic form
        score:      lenient similarity 0.0..100.0 when a reference is known
                    (dict hit or batch with gold), 0.0 otherwise
        accepted:   score >= threshold (always True when score == 0.0 and
                    no reference is available)
        engine:     "dict" | "xgboost" | "rules"
        reference:  the reference Arabic string used to compute score, or None
    """
    input: str
    arabic: str
    score: float
    accepted: bool
    engine: str
    reference: str | None = None
