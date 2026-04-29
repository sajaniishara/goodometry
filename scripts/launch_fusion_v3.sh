#!/usr/bin/env bash
# Launch fusion_v3 (kin_v3 + ins + vo) on GPU 0.
#
# kin_v3 uses 35-D kinematics features: replaces v_body_legs (3D linear vel)
# with leg_odom_delta (7D SE(3) per-frame delta: Δt from leg-odom + Δq from gyro).
#
# Pre-flight checks:
#   1. vo.npz must exist (>= 800 trajectories).
#   2. leg_odom_delta must be patched into kin.npz
#      (run: python scripts/patch_kin_leg_odom_delta.py first).
#
# Usage:
#   bash scripts/launch_fusion_v3.sh            # IMU-only (default)
#   bash scripts/launch_fusion_v3.sh --with-marg  # MARG INS

set -euo pipefail
cd "$(dirname "$0")/.."

WITH_MARG=0
for arg in "$@"; do
  [[ "$arg" == "--with-marg" ]] && WITH_MARG=1
done

source scripts/env.sh

# --- pre-flight: vo.npz coverage ---
VO_COUNT=$(ls /mnt/data/go2_research_dataset_v2/trajectory_*/vo.npz 2>/dev/null | wc -l)
echo "vo.npz files found: ${VO_COUNT}"
if [[ "${VO_COUNT}" -lt 800 ]]; then
  echo "ERROR: need >= 800 vo.npz files (have ${VO_COUNT})."
  exit 1
fi

# --- pre-flight: leg_odom_delta patch coverage ---
KIN_PATCHED=$(python -c "
import glob, numpy as np
trajs = glob.glob('/mnt/data/go2_research_dataset_v2/trajectory_*/kin.npz')
n = sum(1 for t in trajs if 'leg_odom_delta' in np.load(t, mmap_mode='r').files)
print(n)
" 2>/dev/null || echo 0)
echo "kin.npz files with leg_odom_delta: ${KIN_PATCHED}"
if [[ "${KIN_PATCHED}" -lt 800 ]]; then
  echo "ERROR: need >= 800 kin.npz files patched with leg_odom_delta (have ${KIN_PATCHED})."
  echo "Run: python scripts/patch_kin_leg_odom_delta.py"
  exit 1
fi

# --- choose run name and INS file ---
if [[ "${WITH_MARG}" -eq 1 ]]; then
  RUN_DIR="runs/fusion_v3_marg"
  INS_ARG="--ins-file ins_marg.npz"
  echo "INS: MARG stitched (drift-free yaw)"
else
  RUN_DIR="runs/fusion_v3"
  INS_ARG="--ins-file ins.npz"
  echo "INS: IMU-only"
fi

echo "Output dir: ${RUN_DIR}"
echo "Launching on CUDA_VISIBLE_DEVICES=0"
echo ""

LOG_FILE="logs/fusion_v3$([ ${WITH_MARG} -eq 1 ] && echo _marg || echo '').log"
PID_FILE="logs/fusion_v3$([ ${WITH_MARG} -eq 1 ] && echo _marg || echo '').pid"

setsid nohup \
  env CUDA_VISIBLE_DEVICES=0 python -u fusion/train.py \
    --output-dir "${RUN_DIR}" \
    --device cuda \
    --epochs 50 \
    --batch-size 128 \
    --num-workers 4 \
    --patience 10 \
    --use-vo \
    --use-kin-v3 \
    ${INS_ARG} \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "PID $(cat ${PID_FILE}) — tail -f ${LOG_FILE}"
