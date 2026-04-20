"""Push arabnamer artefacts to Hugging Face Hub — model, dataset, and Space.

One-shot script. Reads $HF_TOKEN (or --token arg) and pushes:

  1. Model  → huggingface.co/Sayedyousef/arabnamer-xgboost
  2. Dataset → huggingface.co/datasets/Sayedyousef/arabic-name-pairs
  3. Space  → huggingface.co/spaces/Sayedyousef/arabnamer-demo

Usage (PowerShell):
    $env:HF_TOKEN = "hf_your_write_token"
    python scripts/push_to_hf.py

or with explicit token (less safe):
    python scripts/push_to_hf.py --token hf_your_write_token

Idempotent — safe to re-run. Existing repos are updated in place.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HF_USERNAME = "Sayedyousef"
MODEL_REPO = f"{HF_USERNAME}/arabnamer-xgboost"
DATASET_REPO = f"{HF_USERNAME}/arabic-name-pairs"
SPACE_REPO = f"{HF_USERNAME}/arabnamer-demo"


def _require_hf():
    try:
        from huggingface_hub import HfApi, create_repo, upload_file, upload_folder  # noqa: F401
        return True
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("       pip install huggingface-hub")
        return False


def push_model(api, token: str):
    print(f"\n--- Pushing model to {MODEL_REPO} ---")
    model_ubj = REPO_ROOT / "model" / "xgb_pruned_386r.ubj"
    labels_json = REPO_ROOT / "model" / "labels.json"
    card = REPO_ROOT / "docs" / "hf_model_card.md"

    if not model_ubj.is_file():
        print(f"  SKIP: {model_ubj} not found (uncompressed model must be kept locally)")
        return False
    if not labels_json.is_file():
        print(f"  SKIP: {labels_json} not found")
        return False
    if not card.is_file():
        print(f"  SKIP: {card} not found")
        return False

    from huggingface_hub import create_repo, upload_file
    create_repo(MODEL_REPO, repo_type="model", token=token, exist_ok=True)

    print(f"  Uploading {model_ubj.name} ({model_ubj.stat().st_size / 1024 / 1024:.1f} MB)...")
    upload_file(
        path_or_fileobj=str(model_ubj),
        path_in_repo="xgb_pruned_386r.ubj",
        repo_id=MODEL_REPO, repo_type="model", token=token,
        commit_message="Upload pruned XGBoost model (386 rounds)",
    )

    print(f"  Uploading {labels_json.name}...")
    upload_file(
        path_or_fileobj=str(labels_json),
        path_in_repo="labels.json",
        repo_id=MODEL_REPO, repo_type="model", token=token,
        commit_message="Upload labels",
    )

    print(f"  Uploading model card (README.md)...")
    upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=MODEL_REPO, repo_type="model", token=token,
        commit_message="Add model card",
    )
    print(f"  OK -> https://huggingface.co/{MODEL_REPO}")
    return True


def push_dataset(api, token: str):
    print(f"\n--- Pushing dataset to {DATASET_REPO} ---")
    dict_json = REPO_ROOT / "dataset" / "dict_FINAL.json"
    card = REPO_ROOT / "docs" / "hf_dataset_card.md"

    if not dict_json.is_file():
        print(f"  SKIP: {dict_json} not found")
        return False
    if not card.is_file():
        print(f"  SKIP: {card} not found")
        return False

    from huggingface_hub import create_repo, upload_file
    create_repo(DATASET_REPO, repo_type="dataset", token=token, exist_ok=True)

    print(f"  Uploading {dict_json.name} ({dict_json.stat().st_size / 1024:.0f} KB)...")
    upload_file(
        path_or_fileobj=str(dict_json),
        path_in_repo="dict_FINAL.json",
        repo_id=DATASET_REPO, repo_type="dataset", token=token,
        commit_message="Upload EN-AR name-pairs dictionary",
    )

    print(f"  Uploading dataset card (README.md)...")
    upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=DATASET_REPO, repo_type="dataset", token=token,
        commit_message="Add dataset card",
    )
    print(f"  OK -> https://huggingface.co/datasets/{DATASET_REPO}")
    return True


def push_space(api, token: str):
    print(f"\n--- Pushing Space to {SPACE_REPO} ---")
    space_dir = REPO_ROOT / "hf_space"

    if not space_dir.is_dir():
        print(f"  SKIP: {space_dir} not found")
        return False

    from huggingface_hub import create_repo, upload_folder
    create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", token=token, exist_ok=True)

    print(f"  Uploading {space_dir} -> Space root...")
    upload_folder(
        folder_path=str(space_dir),
        repo_id=SPACE_REPO, repo_type="space", token=token,
        commit_message="Deploy Gradio demo",
    )
    print(f"  OK -> https://huggingface.co/spaces/{SPACE_REPO}")
    print(f"       (Space will build for ~2 minutes then be live)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Push arabnamer artefacts to Hugging Face Hub.")
    parser.add_argument("--token", help="HF write token (default: $HF_TOKEN)")
    parser.add_argument("--skip-model", action="store_true", help="Skip model upload")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset upload")
    parser.add_argument("--skip-space", action="store_true", help="Skip Space upload")
    args = parser.parse_args()

    if not _require_hf():
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: no HF token found. Set $env:HF_TOKEN or pass --token.")
        print("       Create one at https://huggingface.co/settings/tokens (type: Write)")
        sys.exit(1)

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        user = api.whoami()
        print(f"Authenticated as: {user['name']}")
    except Exception as e:
        print(f"ERROR: token rejected by HF: {e}")
        sys.exit(1)

    results = []
    if not args.skip_model:
        results.append(("Model", push_model(api, token)))
    if not args.skip_dataset:
        results.append(("Dataset", push_dataset(api, token)))
    if not args.skip_space:
        results.append(("Space", push_space(api, token)))

    print("\n=== Summary ===")
    for name, ok in results:
        print(f"  {'OK ' if ok else 'FAIL'}  {name}")
    print(f"\nVisit your profile: https://huggingface.co/{HF_USERNAME}")


if __name__ == "__main__":
    main()
