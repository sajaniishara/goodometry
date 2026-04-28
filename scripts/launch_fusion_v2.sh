#!/usr/bin/env bash
# Launch fusion_v2 (kin + ins + vo) on GPU 0 after DROID-SLAM at-scale finishes.
#
# Pre-flight checks:
#   1. DROID-SLAM process (PID 169781) must no longer be running.
#   2. Buffer-overflow re-runs: any trajectories that errored during the
#      main pass should be re-run via run_vo_at_scale.py with --buffer 2048
#      before calling this script.
#   3. At least 800 vo.npz files must exist (>= 80% coverage for a valid split).
#
# Usage:
#   bash scripts/launch_fusion_v2.sh            # IMU-only INS (default)
#   bash scripts/launch_fusion_v2.sh --with-marg  # MARG INS (drift-free yaw)

set -euo pipefail
cd "$(dirname "$0")/.."

WITH_MARG=0
for arg in "$@"; do
  [[ "$arg" == "--with-marg" ]] && WITH_MARG=1
done

source scripts/env.sh

# --- pre-flight ---
VO_COUNT=$(find /mnt/data/go2_research_dataset_v2 -name vo.npz 2>/dev/null | wc -l)
echo "vo.npz files found: ${VO_COUNT}"
if [[ "${VO_COUNT}" -lt 800 ]]; then
  echo "ERROR: need >= 800 vo.npz files before training fusion_v2 (have ${VO_COUNT})."
  echo "Wait for DROID-SLAM to finish and re-run any buffer-overflow failures."
  exit 1
fi

ERROR_COUNT=$(grep -c '"status": "error"' logs/vo_scale_droid.jsonl 2>/dev/null) || ERROR_COUNT=0
if [[ "${ERROR_COUNT}" -gt 0 ]]; then
  echo "WARNING: ${ERROR_COUNT} DROID errors in logs/vo_scale_droid.jsonl."
  echo "Consider re-running failed trajectories with:"
  echo "  python scripts/run_vo_at_scale.py --method droid --buffer 2048 --force"
  echo "Proceeding anyway (failed trajectories will be skipped by the dataset)."
fi

# --- choose run name and INS file ---
if [[ "${WITH_MARG}" -eq 1 ]]; then
  RUN_DIR="runs/fusion_v2_marg"
  INS_ARG="--ins-file ins_marg.npz"
  echo "INS: MARG stitched (drift-free yaw)"
else
  RUN_DIR="runs/fusion_v2"
  INS_ARG="--ins-file ins.npz"
  echo "INS: IMU-only"
fi

echo "Output dir: ${RUN_DIR}"
echo "Launching on CUDA_VISIBLE_DEVICES=0"
echo ""

LOG_FILE="logs/fusion_v2$([ ${WITH_MARG} -eq 1 ] && echo _marg || echo '').log"
PID_FILE="logs/fusion_v2$([ ${WITH_MARG} -eq 1 ] && echo _marg || echo '').pid"

setsid nohup \
  env CUDA_VISIBLE_DEVICES=0 python -u fusion/train.py \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --epochs 50 \
    --batch-size 128 \
    --num-workers 4 \
    --patience 10 \
    --use-vo \
    ${INS_ARG} \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "PID $(cat ${PID_FILE}) — tail -f ${LOG_FILE}"
