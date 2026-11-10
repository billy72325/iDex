#!/usr/bin/env bash
# Overnight chain on one CUDA GPU, from the repository root inside the NVIDIA container:
#   bash scripts/national/run_overnight.sh [arch] [v2_epochs] [v3_epochs] [per_species_crawl]
#
#   1. crawl merchandise photos for all 1,025 species (runs to completion first)
#   2. train v2 (composited artwork + Gen I photos + negatives)
#   3. vet the crawl with the v2 model, build the extra manifest (+ your own photos if any)
#   4. train v3 warm-started from v2 on everything, evaluate, done
set -euo pipefail
cd "$(dirname "$0")/../.."
ARCH="${1:-efficientnet_v2_s}"
V2_EPOCHS="${2:-40}"
V3_EPOCHS="${3:-30}"
PER_SPECIES="${4:-60}"
SIZE=224
PY="${IDEX_TRAIN_PYTHON:-python3}"
V2_RUN="artifacts/national/runs/v2-${ARCH}-${SIZE}"
V3_RUN="artifacts/national/runs/v3-${ARCH}-${SIZE}"
mkdir -p artifacts/national/logs

"$PY" -m pip install -q pyarrow pillow numpy ddgs icrawler 2>&1 | tail -1 || true

echo "== 1/4 crawl (log: artifacts/national/logs/crawl.log)"
"$PY" scripts/national/crawl.py --per-species "$PER_SPECIES" --workers 6 2>&1 | tee artifacts/national/logs/crawl.log

echo "== 2/4 v2 training"
bash scripts/national/run_v2.sh "$ARCH" "$V2_EPOCHS" "$SIZE" 2>&1 | tee artifacts/national/logs/v2.log

echo "== 3/4 vet crawl with v2, build extra manifest"
"$PY" scripts/national/filter_crawl.py --checkpoint "$V2_RUN/best.pt" \
    --crawl artifacts/national/crawl --own artifacts/national/own \
    --out artifacts/national/v2/extra_manifest.jsonl 2>&1 | tee artifacts/national/logs/filter.log

echo "== 4/4 v3 training (warm start from v2, all data)"
"$PY" scripts/national/train_v2.py train --arch "$ARCH" --size "$SIZE" --epochs "$V3_EPOCHS" --lr 4e-4 \
    --init-checkpoint "$V2_RUN/best.pt" --extra-manifest artifacts/national/v2/extra_manifest.jsonl \
    --out "$V3_RUN" --resume 2>&1 | tee artifacts/national/logs/v3.log
"$PY" scripts/national/train_v2.py predict --checkpoint "$V3_RUN/best.pt" \
    --extra-manifest artifacts/national/v2/extra_manifest.jsonl \
    --out "artifacts/national/predictions/v3-${ARCH}-${SIZE}-val.jsonl" 2>&1 | tee artifacts/national/logs/v3-predict.log

echo
echo "Done. Compare:"
echo "  artifacts/national/predictions/v2-${ARCH}-${SIZE}-val.summary.json"
echo "  artifacts/national/predictions/v3-${ARCH}-${SIZE}-val.summary.json"
echo "Bring back the better best.pt (${V2_RUN} or ${V3_RUN}) and both summaries."
