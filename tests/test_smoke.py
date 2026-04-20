"""Smoke test — verify import + a single prediction round-trips.

Run: pytest tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arabnamer import translit, translit_batch, similarity, Transliterator, Result


def test_translit_basic():
    r = translit("Mohammed Ali")
    assert isinstance(r, Result)
    assert r.arabic == "محمد علي"
    assert r.engine == "xgboost"


def test_translit_with_reference_perfect_match():
    r = translit("Ahmad Hassan", reference="أحمد حسن")
    assert r.score >= 99.0
    assert r.accepted is True


def test_translit_batch():
    results = translit_batch(["Mohammed Ali", "Ahmad Hassan"])
    assert len(results) == 2
    assert results[0].arabic == "محمد علي"
    assert results[1].arabic == "أحمد حسن"


def test_similarity_hamza_variants():
    passed, score = similarity("أحمد حسن", "احمد حسن")
    assert passed is True
    assert score == 100


def test_similarity_taa_marbuta():
    passed, score = similarity("مروة فرج", "مروه فرج")
    assert passed is True
    assert score == 100


def test_transliterator_rules_engine():
    t = Transliterator(engine="rules", threshold=85)
    r = t.translit("Mohammed Ali")
    assert "محمد" in r.arabic


def test_transliterator_hybrid_engine():
    t = Transliterator(engine="hybrid", threshold=85)
    r = t.translit("Mohammed Ali")
    assert r.arabic == "محمد علي"


def test_threshold_customization():
    t = Transliterator(engine="model", threshold=95)
    r = t.translit("Ahmad Hassan", reference="أحمد حسن")
    assert r.score >= 99.0
    assert r.accepted is True


if __name__ == "__main__":
    # Allow `python tests/test_smoke.py` without pytest
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("All smoke tests passed.")
