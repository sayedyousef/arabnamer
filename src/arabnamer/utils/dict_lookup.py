"""Dictionary lookup over dict_FINAL.json.

Resolution order (matches XGBoostTransliterator pattern):
    1. Explicit `path` argument
    2. Installed wheel resources: arabnamer/data/dict_FINAL.json
    3. Dev repo: <repo_root>/dataset/dict_FINAL.json (when running from source)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DICT_FILENAME = "dict_FINAL.json"


def _norm_key(key: str) -> str:
    if not key:
        return ""
    return key.lower().strip().replace("-", "")


def _find_dict_file() -> Path | None:
    # Try installed package resources first (pip-installed case)
    try:
        from importlib import resources
        r = resources.files("arabnamer").joinpath("data", DICT_FILENAME)
        if r.is_file():
            return Path(str(r))
    except Exception:
        pass

    # Dev / editable fallback: walk up looking for `dataset/` folder
    here = Path(__file__).resolve()
    for up in range(2, 6):
        if up >= len(here.parents):
            break
        root = here.parents[up]
        for sub in ("dataset", "src/arabnamer/data", "arabnamer/data"):
            p = root / sub / DICT_FILENAME
            if p.is_file():
                return p
    return None


@lru_cache(maxsize=1)
def _load_dict(path: str | None = None) -> dict[str, str]:
    if path is None:
        found = _find_dict_file()
        if found is None:
            raise FileNotFoundError(
                f"Could not locate {DICT_FILENAME} in installed package or "
                f"repo-relative `dataset/` folder."
            )
        path = str(found)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {_norm_key(k): v for k, v in data.items() if k and v}


class DictLookup:
    """Exact-key lookup over dict_FINAL.json (22,798 EN -> AR name pairs)."""

    def __init__(self, path: str | Path | None = None):
        self._data = _load_dict(str(path) if path else None)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return _norm_key(key) in self._data

    def lookup(self, key: str) -> str | None:
        return self._data.get(_norm_key(key))

    def lookup_tokens(self, tokens_en: list[str]) -> list[str | None]:
        return [self.lookup(t) for t in tokens_en]
