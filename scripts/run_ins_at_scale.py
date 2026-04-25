"""Run Madgwick INS preprocessing on every clean Go2 v2 trajectory (CPU).

Default is IMU-only (use_mag=False) — the sim's magnetic field is hardcoded to
a fixed mid-latitude approximation that won't transfer to a real Go2 (motor
current + actual geographic location both matter). Yaw drift in filter frame
is recovered downstream by DROID-SLAM.

Per trajectory writes `/mnt/data/go2_research_dataset_v2/<traj>/ins.npz`:

    frame_idx    (N,)     int32    # matches sensors.npz['frame_idx']
    quat_wxyz    (N, 4)   float32  # body-to-filter-world quaternion
    gyro_body    (N, 3)   float32  # pass-through body-frame gyro
    accel_body   (N, 3)   float32  # gravity-compensated, body frame
    accel_world  (N, 3)   float32  # gravity-compensated, filter-world frame
    euler_rpy    (N, 3)   float32  # roll/pitch/yaw from quat
    flags        (N, 4)   uint8    # [init, ang_recov, acc_recov, mag_recov]
    use_mag      ()       bool     # diagnostic

Optional `--with-mag` runs a second pass writing ins_marg.npz next to ins.npz
for pure-sim ablation comparison; that pass uses CONVENTION_NED to match the
stored mag axis order (N, E, D) and should produce drift-free yaw within sim.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
import traceback
import numpy as np

sys.path.insert(0, "/home/anyone/projects/goodometry")
from ins.madgwick import run_madgwick

ROOT = "/mnt/data/go2_research_dataset_v2"
LOG_DIR = "/home/anyone/projects/goodometry/logs"


def process(traj_dir: str, use_mag: bool) -> dict:
    sens = np.load(os.path.join(traj_dir, "sensors.npz"))
    # Body is Z-up → filter must use CONVENTION_ENU. Sim stores mag as (N,E,D);
    # we remap → (E,N,U) via mag_axis_order="NED" so the ENU filter sees a
    # proper Earth field.
    out = run_madgwick(
        sens["imu"],
        sens["mag"] if use_mag else None,
        sample_rate_hz=30.0,
        use_mag=use_mag,
        convention="ENU",
        mag_axis_order=("NED" if use_mag else "ENU"),
    )
    return {
        "frame_idx":   sens["frame_idx"].astype(np.int32),
        "quat_wxyz":   out["quat_wxyz"],
        "gyro_body":   out["gyro_body"],
        "accel_body":  out["accel_body"],
        "accel_world": out["accel_world"],
        "euler_rpy":   out["euler_rpy"],
        "flags":       out["flags"],
        "use_mag":     np.bool_(use_mag),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--with-mag", action="store_true",
                    help="run MARG mode and write ins_marg.npz (in addition to, not instead of, ins.npz)")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "ins_scale.jsonl")

    trajs = sorted(glob.glob(os.path.join(args.root, "trajectory_*")))
    trajs = [t for t in trajs
             if os.path.isdir(t)
             and os.path.exists(os.path.join(t, "sensors.npz"))
             and not os.path.exists(os.path.join(t, "FALL"))]
    trajs = trajs[args.start: args.end]
    print(f"{len(trajs)} trajectories in scope  log={log_path}", flush=True)

    ok = fail = skip = 0
    total_t0 = time.time()
    for i, traj_dir in enumerate(trajs):
        name = os.path.basename(traj_dir)
        out_primary = os.path.join(traj_dir, "ins.npz")
        out_marg = os.path.join(traj_dir, "ins_marg.npz")

        did_any = False
        try:
            if args.force or not os.path.exists(out_primary):
                payload = process(traj_dir, use_mag=False)
                np.savez(out_primary, **payload)
                did_any = True
            if args.with_mag and (args.force or not os.path.exists(out_marg)):
                payload = process(traj_dir, use_mag=True)
                np.savez(out_marg, **payload)
                did_any = True

            if not did_any:
                skip += 1
                continue
            ok += 1
            dt = time.time() - (total_t0 if i == 0 else total_t0)  # wall from start; printed relative
            if ok % 50 == 0 or ok == 1:
                print(f"[{i+1}/{len(trajs)}] {name} ok", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "traj": name, "status": "ok"}) + "\n")
        except Exception as e:
            fail += 1
            err = {"ts": time.time(), "traj": name, "status": "error",
                   "error": repr(e), "tb": traceback.format_exc()}
            print(f"[{i+1}/{len(trajs)}] {name} FAIL  {e!r}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(err) + "\n")

    print(f"\nDone. ok={ok} fail={fail} skip={skip} total_s={time.time()-total_t0:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
