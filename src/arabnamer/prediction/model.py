"""XGBoost model loader and inference.

Resolution order for the model file:
    1. Explicit `model_path` argument (highest priority)
    2. ARABNAMER_MODEL_PATH env var
    3. Local repo layout: <repo_root>/model/xgb_pruned_386r.ubj
       (useful during development before publishing to Hugging Face)
    4. User cache: ~/.cache/arabnamer/xgb_pruned_386r.ubj
    5. Hugging Face Hub: sayedyousef/arabnamer-xgboost (downloaded to cache)

Set `ARABNAMER_OFFLINE=1` to forbid Hugging Face downloads.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import xgboost as xgb

from .featurize import PAD_LEN, featurize_word

HF_REPO_ID = "sayedyousef/arabnamer-xgboost"
MODEL_FILENAME = "xgb_pruned_386r.ubj.gz"          # gzipped, shipped in wheel
MODEL_FILENAME_FALLBACK = "xgb_pruned_386r.ubj"    # uncompressed, dev-only
LABELS_FILENAME = "labels.json"
CACHE_DIR_DEFAULT = Path.home() / ".cache" / "arabnamer"


def _repo_root_candidates() -> list[Path]:
    """Walk up from this file to find the repo root(s). We check several
    levels because the package may live at either `arabnamer/src/arabnamer/`
    (src-layout) or `arabnamer/arabnamer/` (flat layout).
    """
    here = Path(__file__).resolve()
    return [here.parents[i] for i in range(2, 6) if i < len(here.parents)]


def _find_local_file(filename: str, subdir: str = "model") -> Path | None:
    for root in _repo_root_candidates():
        p = root / subdir / filename
        if p.is_file():
            return p
    return None


def _find_model_file() -> Path | None:
    """Find model in local repo — prefer gzipped, fall back to uncompressed."""
    for name in (MODEL_FILENAME, MODEL_FILENAME_FALLBACK):
        p = _find_local_file(name, subdir="model")
        if p is not None:
            return p
    # Also check if installed package bundles it (wheel case)
    try:
        from importlib import resources
        for name in (MODEL_FILENAME, MODEL_FILENAME_FALLBACK):
            r = resources.files("arabnamer").joinpath("data", name)
            if r.is_file():
                return Path(str(r))
    except Exception:
        pass
    return None


def _resolve_file(
    filename: str,
    subdir: str,
    env_var: str | None,
    cache_dir: Path | None,
    hf_repo_id: str | None,
) -> Path:
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val:
            p = Path(env_val)
            if p.is_file():
                return p
            raise FileNotFoundError(f"{env_var}={env_val} not found")

    local = _find_local_file(filename, subdir=subdir)
    if local is not None:
        return local

    cache = (cache_dir or CACHE_DIR_DEFAULT) / filename
    if cache.is_file():
        return cache

    if os.environ.get("ARABNAMER_OFFLINE") == "1":
        raise FileNotFoundError(
            f"{filename} not found locally or in cache; "
            f"ARABNAMER_OFFLINE=1 forbids Hugging Face download"
        )

    if hf_repo_id is None:
        raise FileNotFoundError(
            f"{filename} not found and no HF repo configured"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Model not found locally. Install huggingface-hub to auto-download:\n"
            "    pip install huggingface-hub\n"
            f"Or place {filename} in <repo>/{subdir}/ manually."
        ) from e

    cache.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=hf_repo_id,
        filename=filename,
        cache_dir=str(cache_dir or CACHE_DIR_DEFAULT),
    )
    return Path(downloaded)


def _load_booster(path: Path) -> xgb.Booster:
    """Load an XGBoost Booster from .ubj or .ubj.gz."""
    bst = xgb.Booster()
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            buf = f.read()
        bst.load_model(bytearray(buf))
    else:
        bst.load_model(str(path))
    return bst


class XGBoostTransliterator:
    """Load the XGBoost model + labels and predict Arabic transliteration.

    The model file is resolved in this order (first match wins):
      1. `model_path` argument
      2. ARABNAMER_MODEL_PATH env var
      3. <repo_root>/model/xgb_pruned_386r.ubj   (dev-mode)
      4. ~/.cache/arabnamer/xgb_pruned_386r.ubj  (previously downloaded)
      5. Hugging Face Hub: sayedyousef/arabnamer-xgboost
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ):
        cache = Path(cache_dir) if cache_dir else None

        if model_path is None:
            env_val = os.environ.get("ARABNAMER_MODEL_PATH")
            if env_val:
                p = Path(env_val)
                if not p.is_file():
                    raise FileNotFoundError(f"ARABNAMER_MODEL_PATH={env_val} not found")
                model_path = p
            else:
                local = _find_model_file()
                if local is not None:
                    model_path = local
                else:
                    model_path = _resolve_file(
                        MODEL_FILENAME, subdir="model",
                        env_var=None,
                        cache_dir=cache, hf_repo_id=HF_REPO_ID,
                    )
        if labels_path is None:
            labels_path = _resolve_file(
                LABELS_FILENAME, subdir="model",
                env_var="ARABNAMER_LABELS_PATH",
                cache_dir=cache, hf_repo_id=HF_REPO_ID,
            )

        self._booster = _load_booster(Path(model_path))
        with open(str(labels_path), "r", encoding="utf-8") as f:
            self._labels = json.load(f)

    @property
    def num_classes(self) -> int:
        return len(self._labels)

    @property
    def num_rounds(self) -> int:
        return self._booster.num_boosted_rounds()

    def predict_token(self, token_en: str) -> str:
        """Predict the Arabic transliteration of a single English token."""
        if not token_en:
            return ""
        if len(token_en) > PAD_LEN:
            return token_en
        X = np.array(featurize_word(token_en), dtype=np.int32)
        dm = xgb.DMatrix(X)
        probs = self._booster.predict(dm)
        ids = probs.argmax(axis=1)
        return "".join(self._labels[int(i)] for i in ids)

    def predict_tokens(self, tokens_en: list[str]) -> list[str]:
        return [self.predict_token(t) for t in tokens_en]
