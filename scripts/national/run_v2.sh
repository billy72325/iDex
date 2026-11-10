#!/usr/bin/env bash
# v2 overnight recipe on one CUDA GPU: camera-robust 1,025-class classifier.
# From the repository root, inside the NVIDIA PyTorch container:
#   bash scripts/national/run_v2.sh [arch] [epochs] [size]
set -euo pipefail
cd "$(dirname "$0")/../.."
ARCH="${1:-efficientnet_v2_s}"
EPOCHS="${2:-40}"
SIZE="${3:-224}"
PY="${IDEX_TRAIN_PYTHON:-python3}"
RUN="artifacts/national/runs/v2-${ARCH}-${SIZE}"

echo "== environment"
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print(torch.__version__, torch.cuda.get_device_name(0))"

echo "== sources (artwork + ImageNet weights from v1, real photos + negatives from the original campaign)"
[ -f artifacts/national/sources/JJMack--pokemon-classification-gen1-9/train.parquet ] || "$PY" scripts/national/acquire.py all
"$PY" scripts/national/acquire.py pretrained
"$PY" scripts/replacement/acquire.py metadata
"$PY" scripts/replacement/acquire.py data --dataset sgtsaughter/pokemon-classification-images-151
"$PY" scripts/replacement/acquire.py data --dataset keremberke/pokemon-classification
"$PY" scripts/replacement/acquire.py negatives

echo "== prepare v2 (RGBA artwork, photos, negatives, backgrounds)"
[ -f artifacts/national/v2/manifest.jsonl ] || "$PY" scripts/national/prepare_v2.py

echo "== train ${ARCH} @ ${SIZE}px for ${EPOCHS} epochs"
"$PY" scripts/national/train_v2.py train --arch "$ARCH" --size "$SIZE" --epochs "$EPOCHS" --out "$RUN" --resume

echo "== validation predictions (art / photo / negatives)"
"$PY" scripts/national/train_v2.py predict --checkpoint "$RUN/best.pt" --out "artifacts/national/predictions/v2-${ARCH}-${SIZE}-val.jsonl"

echo
echo "Done. Bring back ${RUN}/best.pt and artifacts/national/predictions/v2-${ARCH}-${SIZE}-val.summary.json."
