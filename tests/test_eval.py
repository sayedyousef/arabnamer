"""Benchmark eval — runs the library on tests/test_names.csv and writes
benchmarks/eval_results.csv + benchmarks/REPORT.md.

Run from repo root:
    python tests/test_eval.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = REPO_ROOT / "tests" / "test_names.csv"
RESULTS_CSV = REPO_ROOT / "benchmarks" / "eval_results.csv"
REPORT_MD = REPO_ROOT / "benchmarks" / "REPORT.md"

# Allow running from a source checkout without `pip install`
sys.path.insert(0, str(REPO_ROOT / "src"))

from arabnamer import Transliterator  # noqa: E402


def read_tests(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    tests = read_tests(TEST_CSV)
    t = Transliterator(engine="model", threshold=85)

    rows = []
    for row in tests:
        r = t.translit(row["name_en"], reference=row["reference_ar"])
        rows.append({
            "name_en": row["name_en"],
            "reference_ar": row["reference_ar"],
            "predicted_ar": r.arabic,
            "score": round(r.score, 1),
            "passed": r.accepted,
            "engine": r.engine,
            "source": row.get("source", ""),
            "category": row.get("category", ""),
        })

    # Aggregate
    scores = [r["score"] for r in rows]
    n = len(rows)
    avg = round(mean(scores), 1)
    p70 = sum(1 for s in scores if s >= 70)
    p90 = sum(1 for s in scores if s >= 90)
    exact = sum(1 for s in scores if s >= 100)

    # Write per-row CSV
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Write report
    lines = [
        "# arabnamer — benchmark results",
        "",
        f"**Test set:** {TEST_CSV.relative_to(REPO_ROOT)} ({n} names)",
        f"**Engine:** model (XGBoost, pruned 386 rounds)",
        f"**Threshold:** 85 (lenient scoring)",
        "",
        "## Summary",
        "",
        f"| Metric | Score |",
        f"|---|---|",
        f"| Names evaluated | **{n}** |",
        f"| Average lenient similarity | **{avg}** |",
        f"| Pass rate (>= 70) | **{p70} / {n}** ({100*p70/n:.0f}%) |",
        f"| Pass rate (>= 90) | **{p90} / {n}** ({100*p90/n:.0f}%) |",
        f"| Exact match (= 100) | **{exact} / {n}** ({100*exact/n:.0f}%) |",
        "",
        "## Per-name results",
        "",
        "| Name (EN) | Predicted (AR) | Reference (AR) | Score | Passed |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        mark = "✅" if r["passed"] else "❌"
        lines.append(
            f"| {r['name_en']} | {r['predicted_ar']} | {r['reference_ar']} | {r['score']} | {mark} |"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install arabnamer",
        "python tests/test_eval.py   # writes benchmarks/eval_results.csv + REPORT.md",
        "```",
        "",
        "Scoring is **lenient** — tashkeel stripped, hamza variants unified "
        "(أ/إ/آ → ا), taa-marbuta → haa (ة → ه), alef-maksura → yaa (ى → ي), "
        "then `max(fuzz.ratio, fuzz.partial_ratio)` via rapidfuzz.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_MD.relative_to(REPO_ROOT)}")
    print()
    print(f"Avg: {avg}   Pass>=70: {p70}/{n}   Pass>=90: {p90}/{n}   Exact: {exact}/{n}")


if __name__ == "__main__":
    main()
