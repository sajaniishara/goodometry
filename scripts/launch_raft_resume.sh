#!/usr/bin/env bash
# Detached resume of RAFT-Stereo disparity precompute on GPU 1.
# - 490 trajectories already in disparity_224.h5 are auto-skipped.
# - --iters 32 to MATCH the existing half (consistency for thesis/ablation —
#   distribution shift from mixing iters is undesirable). ~5 days wall-clock.
# - setsid + nohup + disown → survives logout.
#
# Usage:  bash launch_raft_resume.sh
# Watch:  tail -f /home/anyone/projects/goodometry/logs/raft_resume.log

set -euo pipefail

ROOT_GO2=/home/anyone/projects/go2_research
LOGDIR=/home/anyone/projects/goodometry/logs
LOG="${LOGDIR}/raft_resume.log"
PIDFILE="${LOGDIR}/raft_resume.pid"

if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    echo "already running, pid $(cat "${PIDFILE}")" >&2
    exit 1
fi

mkdir -p "${LOGDIR}"
cd "${ROOT_GO2}"

# shellcheck disable=SC1091
source /home/anyone/projects/goodometry/scripts/env.sh
export CUDA_VISIBLE_DEVICES=1

setsid nohup python -u "${ROOT_GO2}/precompute_disparity_h5.py" \
    --dataset-dir /mnt/data/go2_research_dataset_v2 \
    --iters 32 \
    --batch-size 24 \
    --traj-start 0 --traj-end 1008 \
    >> "${LOG}" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PIDFILE}"
disown "${PID}" 2>/dev/null || true

echo "launched RAFT-Stereo resume:  pid ${PID}"
echo "log:    ${LOG}"
echo "pidfile: ${PIDFILE}"
