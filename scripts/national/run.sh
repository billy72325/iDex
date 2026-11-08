#!/usr/bin/env bash
# End-to-end National Pokédex training on one CUDA GPU (ASUS Ascent GX10 / DGX Spark or any NVIDIA box).
# Run from the repository root:  bash scripts/national/run.sh [arch] [epochs]
set -euo pipefail
cd "$(dirname "$0")/../.."
ARCH="${1:-efficientnet_b0}"
EPOCHS="${2:-20}"
PY="${IDEX_TRAIN_PYTHON:-python3}"
RUN="artifacts/national/runs/${ARCH}"

echo "== environment"
"$PY" - <<'EOF'
import torch, torchvision, sys
print(sys.version.split()[0], 'torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA not visible: start inside the NVIDIA PyTorch container or install a CUDA wheel'
print('GPU:', torch.cuda.get_device_name(0))
EOF

echo "== acquire (pinned dataset + ImageNet weights)"
"$PY" scripts/national/acquire.py all

echo "== prepare (decode, dedupe, grouped split)"
[ -f artifacts/national/data/manifest.jsonl ] || "$PY" scripts/national/prepare.py

echo "== train ${ARCH} for ${EPOCHS} epochs"
"$PY" scripts/national/train.py train --arch "$ARCH" --epochs "$EPOCHS" --out "$RUN" --resume

echo "== validation predictions"
"$PY" scripts/national/train.py predict --arch "$ARCH" --checkpoint "$RUN/best.pt" --out "artifacts/national/predictions/${ARCH}-val.jsonl"

echo
echo "Done. Copy ${RUN}/best.pt to the Mac and run scripts/national/export.py there (see scripts/national/README.md)."
