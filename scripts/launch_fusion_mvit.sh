#!/usr/bin/env bash
# Launch fusion_mvit (kin + ins + vo, MViT-style multiscale transformer) on GPU 0.
# Same data and split as fusion_v2 — only the architecture changes (--arch mvit).

set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/env.sh

GPU=${GPU:-0}
RUN_DIR="runs/fusion_mvit"
LOG_FILE="logs/fusion_mvit.log"
PID_FILE="logs/fusion_mvit.pid"

echo "Output dir: ${RUN_DIR}"
echo "Launching on CUDA_VISIBLE_DEVICES=${GPU}"

setsid nohup \
  env CUDA_VISIBLE_DEVICES=${GPU} python -u fusion/train.py \
    --arch mvit \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --epochs 50 \
    --batch-size 128 \
    --num-workers 4 \
    --patience 10 \
    --use-vo \
    --ins-file ins.npz \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "PID $(cat ${PID_FILE}) — tail -f ${LOG_FILE}"
