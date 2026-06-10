#!/usr/bin/env bash
# Sequential 4-run hyperparameter grid (checkpoint-optimized).
# Fixed: cosine LR, warmup_ratio 0.03, rank 32, batch 1, grad_accum 32, experts ON.
# Searched: alpha {32,64} x LR {1e-4,2e-4}. Each run = 2 epochs; SaveEpochAdapter
# snapshots adapter_ep1 (mid-run) and adapter_ep2, giving all 8 grid points.
set -u
cd /home/ec2-user/dev/nemotron_training
PY=foundation/.venv/bin/python
mkdir -p logs/grid

for combo in "32 1e-4" "32 2e-4" "64 1e-4" "64 2e-4"; do
  set -- $combo; A=$1; LR=$2
  RUN="grid_a${A}_lr${LR}"
  echo "[grid] === $RUN START $(date -u +%FT%TZ) ==="
  $PY train/train_sft.py --run "$RUN" --epochs 2 --alpha "$A" --lr "$LR" \
      --grad-accum 32 --scheduler cosine --warmup-ratio 0.03 \
      > "logs/grid/${RUN}.log" 2>&1
  rc=$?
  echo "[grid] === $RUN END rc=$rc $(date -u +%FT%TZ) ==="
done
echo "[grid] ALL DONE $(date -u +%FT%TZ)"
