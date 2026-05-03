#!/usr/bin/env bash
# Wait for CNN RGB Stage-2 v2 to finish on GPU 0, then run fusion_tcn and
# fusion_mvit sequentially on GPU 0.
#
# CNN Stage-2 v2 process is detected by pgrep (no pidfile was written).
# fusion_tcn starts immediately when CNN finishes; fusion_mvit starts when
# fusion_tcn's pidfile process exits.
#
# Usage:
#   bash scripts/watch_cnn_then_fusion.sh &

SELF_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${SELF_DIR}/logs/watch_cnn_then_fusion.log"
CNN_MATCH="cnn_rgb_stage2_v2"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "${LOG}"; }

mkdir -p "${SELF_DIR}/logs"
log "watcher started — polling for CNN process matching '${CNN_MATCH}'"

# --- Phase 1: wait for CNN RGB Stage-2 v2 to finish ---
while pgrep -f "${CNN_MATCH}" > /dev/null 2>&1; do
    PID=$(pgrep -f "${CNN_MATCH}" | head -1)
    log "CNN PID ${PID} still running — sleeping 60 s"
    sleep 60
done
log "CNN RGB Stage-2 v2 complete"

# --- Phase 2: run fusion_tcn, wait for it to exit ---
log "launching fusion_tcn..."
bash "${SELF_DIR}/scripts/launch_fusion_tcn.sh" 2>&1 | tee -a "${LOG}"
TCN_PID=$(cat "${SELF_DIR}/logs/fusion_tcn.pid")
log "fusion_tcn PID ${TCN_PID}"

while kill -0 "${TCN_PID}" 2>/dev/null; do
    log "fusion_tcn PID ${TCN_PID} still running — sleeping 60 s"
    sleep 60
done
log "fusion_tcn complete"

# --- Phase 3: run fusion_mvit ---
log "launching fusion_mvit..."
bash "${SELF_DIR}/scripts/launch_fusion_mvit.sh" 2>&1 | tee -a "${LOG}"
MVIT_PID=$(cat "${SELF_DIR}/logs/fusion_mvit.pid")
log "fusion_mvit PID ${MVIT_PID}"

while kill -0 "${MVIT_PID}" 2>/dev/null; do
    log "fusion_mvit PID ${MVIT_PID} still running — sleeping 60 s"
    sleep 60
done
log "fusion_mvit complete — all done"
