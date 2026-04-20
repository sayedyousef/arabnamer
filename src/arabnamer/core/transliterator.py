"""Orchestrator — dispatches to prediction (XGBoost) or rules, reuses the
thesis scoring functions verbatim.

Nothing is reinvented here. This class only:
  1. loads the XGBoost model once (via prediction.XGBoostTransliterator)
  2. loads the dict once (via utils.DictLookup)
  3. dispatches per token based on the chosen engine
  4. calls `scoring.fuzzy_match` (the thesis scorer) when a reference is given
"""
from __future__ import annotations

from typing import Literal

from ..prediction import XGBoostTransliterator
from ..rules import transliterate_name as _rule_translit_name
from ..scoring import fuzzy_match
from ..utils import DictLookup, merge_articles, split_name
from .result import Result

Engine = Literal["model", "rules", "hybrid"]


class Transliterator:
    """English -> Arabic name transliteration.

    engine:
        "model"  - XGBoost only (default, highest accuracy)
        "rules"  - rule-based only (verbatim thesis rules + dict-lookup head)
        "hybrid" - dict first, then XGBoost, then rules as final fallback
    threshold: lenient-score pass threshold (0..100), default 85.
               Only applied when a `reference` is supplied to `translit()`.
    """

    def __init__(self, engine: Engine = "model", threshold: int = 85):
        if engine not in ("model", "rules", "hybrid"):
            raise ValueError(f"unknown engine: {engine}")
        if not 0 <= threshold <= 100:
            raise ValueError(f"threshold must be 0..100, got {threshold}")
        self.engine = engine
        self.threshold = threshold
        self._dict = DictLookup()
        self._model: XGBoostTransliterator | None = None
        if engine in ("model", "hybrid"):
            self._model = XGBoostTransliterator()

    def _translit_token(self, token_en: str) -> tuple[str, str]:
        """Return (arabic, engine_used) for a single merged token."""
        if self.engine == "rules":
            hit = self._dict.lookup(token_en)
            if hit is not None and " " not in hit:
                return hit, "dict"
            return _rule_translit_name(token_en), "rules"

        if self.engine == "hybrid":
            hit = self._dict.lookup(token_en)
            if hit is not None:
                return hit, "dict"
            try:
                return self._model.predict_token(token_en), "xgboost"
            except Exception:
                return _rule_translit_name(token_en), "rules"

        # engine == "model"
        return self._model.predict_token(token_en), "xgboost"

    def translit(self, name_en: str, reference: str | None = None) -> Result:
        if not name_en or not name_en.strip():
            return Result(input=name_en or "", arabic="", score=0.0,
                          accepted=False, engine=self.engine, reference=reference)

        tokens = merge_articles(split_name(name_en))
        out_tokens: list[str] = []
        engines: list[str] = []
        for t in tokens:
            ar, eng = self._translit_token(t)
            out_tokens.append(ar); engines.append(eng)
        arabic = " ".join(out_tokens)

        if reference is not None:
            passed, score = fuzzy_match(arabic, reference, threshold=self.threshold)
            return Result(input=name_en, arabic=arabic, score=float(score),
                          accepted=passed, engine=_merge_labels(engines),
                          reference=reference)
        return Result(input=name_en, arabic=arabic, score=0.0, accepted=True,
                      engine=_merge_labels(engines), reference=None)

    def translit_batch(
        self,
        names_en: list[str],
        references: list[str | None] | None = None,
    ) -> list[Result]:
        if references is None:
            references = [None] * len(names_en)
        if len(references) != len(names_en):
            raise ValueError("names_en and references must have the same length")
        return [self.translit(n, r) for n, r in zip(names_en, references)]


def _merge_labels(labels: list[str]) -> str:
    uniq: list[str] = []
    for lab in labels:
        if lab not in uniq:
            uniq.append(lab)
    return uniq[0] if len(uniq) == 1 else "+".join(uniq)
