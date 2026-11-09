# National Pokédex classifier (1025 classes)

Self-contained training pipeline for replacing the bundled 151-class Kanto model with one that
recognizes every species in the expanded catalog. It is independent of `scripts/replacement/`
(which stays as the record of the shipped Kanto model) but keeps the same checkpoint schema, so the
Apple validation harness under `scripts/replacement/apple/` still applies to the result.

The app is already label-driven: `PokemonClassifier` and `PokemonCatalog.recognizableIDs` read
`iDex/Recognition/pokemon_labels.json`, so dropping in a new `.mlpackage` plus its label file is all
that is needed on the app side (plus the digest assertions in `SelectedModelTests`).

## Files

| File | Purpose |
| --- | --- |
| `pokemon_labels_national.json` | 1025 labels `{id, name}` in National Dex order; becomes the app's `pokemon_labels.json` after export |
| `acquire.py` | Downloads the pinned `JJMack/pokemon-classification-gen1-9` parquet shards (CC BY-NC-SA 4.0, Bulbapedia artwork) and TorchVision ImageNet weights into `artifacts/national/sources/` with SHA-256 checks |
| `prepare.py` | Decodes the shards to `artifacts/national/data/images/<id>/`, dedupes exact bytes, and writes a grouped 85/15 train/val `manifest.jsonl` (normal/shiny pairs never straddle splits) |
| `labels.py` | Label file loader shared by `prepare.py` (no torch needed) and `model.py` |
| `model.py` | EfficientNet-B0 / MobileNetV3-L / EfficientNetV2-S heads sized from the label file, ImageNet normalization embedded |
| `train.py` | CUDA training (bf16 autocast, AdamW, cosine schedule, label smoothing, class-balanced sampling, resume) and `predict` for per-species validation accuracy |
| `export.py` | Mac-side Core ML export (FP32 + FP16) with trace-parity checks and the app label file |
| `run.sh` | The whole GPU side in one command |

## State of `artifacts/national/` (not in git)

Prepared on the Mac on 2026-11-09: the three verified shards and three TorchVision weights sit under
`sources/`, and `data/` already holds 17,914 decoded 256 px images with `manifest.jsonl`
(15,169 train / 2,745 val, every species ≥ 5 images, manifest SHA-256
`df4d69697962f48271d4b7180b310de68bbb5761541cff101822a8d9abd66f23`). Copy the whole `artifacts/national/`
folder (~1.3 GB) next to the repository on the GPU box and `run.sh` skips acquire/prepare.

## Running on the GX10 (DGX Spark class, aarch64 + Blackwell)

繁體中文逐步說明見 [GX10.zh-TW.md](GX10.zh-TW.md)。

Use NVIDIA's PyTorch container so the aarch64 CUDA wheels are already in place:

```sh
docker run --gpus all -it --rm -v "$PWD":/work -w /work nvcr.io/nvidia/pytorch:25.10-py3
pip install pyarrow pillow numpy
bash scripts/national/run.sh efficientnet_b0 20
```

Without a container, install a CUDA 13 aarch64 wheel from https://pytorch.org/get-started/locally/ into a venv and
point `IDEX_TRAIN_PYTHON` at it. The three shards hold 17,916 rows (train 13,188 / validation 2,198 / test 2,530) covering all 1,025 species with 3–40 images each; 66 Gen VIII–IX species have fewer than 5 training images, so expect those to dominate `worst_species`. EfficientNet-B0 at 224 px, batch 128, runs an epoch
in a few minutes on the GX10, so 20 epochs is well under two hours. `--max-minutes` and `--resume` let you stop and
continue.

Suggested sweep once the first run is in: `efficientnet_v2_s` (stronger, still exports cleanly) and
`--size 256`. Compare `artifacts/national/predictions/*-val.summary.json`; the `worst_species` list shows where
the long tail hurts.

## Bringing the model back into the app (on the Mac)

```sh
python3.11 -m venv ~/idex-export && source ~/idex-export/bin/activate
pip install -r scripts/replacement/requirements-mac.txt
python scripts/national/export.py --checkpoint artifacts/national/runs/efficientnet_b0/best.pt \
    --out artifacts/national/export-effb0 --deployment iOS18
```

1. Validate on device with `scripts/replacement/apple/` (logit parity on Apple-prepared pixels).
2. Copy `NationalClassifier-fp32.mlpackage` over `iDex/Recognition/KantoClassifier.mlpackage` (keep the file
   name the app loads) and `pokemon_labels.json` over `iDex/Recognition/pokemon_labels.json`.
3. Update the two digests in `iDexTests/SelectedModelTests.swift`, the `[1, 151]` logits shape, and the
   `labels.count == 151` / `recognizableIDs.count == 151` expectations in `RecognitionTests` and `CatalogTests`.
4. Re-run the camera protocol in `scripts/replacement/CAMERA_TEST_PROTOCOL.md`; 1025-way accuracy on real
   photos will be lower than the Kanto model's, and the abstention threshold in `PokemonClassifier` may need
   retuning.

## Rights

Training images come from a dataset published under CC BY-NC-SA 4.0 and attributed to Bulbapedia; the
resulting weights inherit the non-commercial, share-alike conditions. Keep `iDex/Recognition/MODEL_NOTICE.md`
in step when the bundled model changes. Nothing under `artifacts/` is committed.
