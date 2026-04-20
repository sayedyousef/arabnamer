"""Name tokenization and Arabic-article merging."""
from __future__ import annotations

MERGE_ARTICLES = frozenset({
    "al", "el", "ul", "ed", "ad",
    "abd", "abdul", "abdel", "abdal",
    "abou", "abo", "abdou",
    "bin", "ben", "ibn", "binn", "benn",
})


def split_name(name_en: str) -> list[str]:
    """Lowercase, strip, split on whitespace and hyphens; drop empties."""
    if not name_en:
        return []
    return [t for t in name_en.lower().replace("-", " ").split() if t]


def merge_articles(tokens: list[str]) -> list[str]:
    """Merge short Arabic prefix-articles with the following token so the model
    sees a single unit. Examples:
        ["abd", "alrahman"] -> ["abdalrahman"]
        ["al", "desouky"]   -> ["aldesouky"]
        ["abdennour"]       -> ["abdennour"]   (already merged)
        ["abou", "khalil"]  -> ["aboukhalil"]
    """
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in MERGE_ARTICLES and i + 1 < len(tokens):
            merged.append(t + tokens[i + 1])
            i += 2
        else:
            merged.append(t)
            i += 1
    return merged


def tokenize_name(name_en: str) -> list[str]:
    """split_name + merge_articles in one call."""
    return merge_articles(split_name(name_en))
