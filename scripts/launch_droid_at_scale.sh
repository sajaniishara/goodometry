#!/usr/bin/env bash
# Detached launcher: runs DROID-SLAM stereo over every clean v2 trajectory on GPU 0.
# Uses `setsid nohup` so the process survives logout (no controlling TTY, no SIGHUP).
# Resume-safe: run_vo_at_scale.py skips trajectories that already have vo.npz.
#
# Usage:
#   bash /home/anyone/projects/goodometry/scripts/launch_droid_at_scale.sh
#
# Inspect live:
#   tail -f /home/anyone/projects/goodometry/logs/droid_at_scale.log
#   tail -f /home/anyone/projects/goodometry/logs/vo_scale_droid.jsonl
set -euo pipefail

ROOT=/home/anyone/projects/goodometry
LOG="${ROOT}/logs/droid_at_scale.log"
PIDFILE="${ROOT}/logs/droid_at_scale.pid"

if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    echo "already running, pid $(cat "${PIDFILE}")" >&2
    exit 1
fi

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
export CUDA_VISIBLE_DEVICES=0

# setsid gives a new session (detached from TTY); nohup ignores SIGHUP on logout.
setsid nohup python -u "${ROOT}/scripts/run_vo_at_scale.py" --method droid \
    >> "${LOG}" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PIDFILE}"
disown "${PID}" 2>/dev/null || true

echo "launched DROID at-scale: pid ${PID}"
echo "log:   ${LOG}"
echo "jsonl: ${ROOT}/logs/vo_scale_droid.jsonl"
echo "pid:   ${PIDFILE}"
