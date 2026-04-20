# Training scripts

Reproduce the shipped XGBoost transliteration model from scratch.

## What each script does

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `train_xgboost.py` | Train a fresh model on `dataset/dict_FINAL.json` | `../dataset/dict_FINAL.json` | `../model/xgb_pure_<ts>.ubj` + `<ts>_labels.json` |
| `prune_trees.py` | Coarse sweep: evaluate truncations at k=800/600/400/300/200/150/100/80/50 rounds, save pruned UBJs for any k with ≤0.5 pt accuracy drop | `../model/<any full model>.ubj` | `../model/xgb_pure_pruned_<k>r.ubj` |
| `find_min_k.py` | Binary search for the minimum k whose per-name scores exactly match the baseline | `../model/<model>.ubj` | `../model/xgb_pure_pruned_MIN_<k>r.ubj` |

## Reproduce end-to-end

```bash
cd training
python train_xgboost.py          # ~5 minutes on CPU, produces ~285 MB UBJ
python prune_trees.py            # ~1 minute, saves several pruned sizes
python find_min_k.py             # ~30 seconds, finds the smallest identical model
gzip -9 ../model/xgb_pure_pruned_MIN_*.ubj   # optional: ship gzipped version
```

## Determinism note

XGBoost with `tree_method="hist"` and multi-threading is **approximately**
deterministic — a retrain produces a model of **very close** (usually
identical ±0.1 pt) accuracy on the 25-name eval, but the UBJ bytes will
differ slightly from the shipped artifact. For bit-perfect reproducibility,
edit `train_xgboost.py` to use `n_jobs=1` (about 4× slower).

## Regenerating the Claude-assisted name fill

The `dataset/dict_FINAL.json` includes entries filled by an LLM for names
absent from the primary sources (JRC / Google Translate). To regenerate
the fill yourself you need an Anthropic API key — see `docs/data_sources.md`
for the exact procedure.
