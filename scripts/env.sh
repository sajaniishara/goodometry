#!/usr/bin/env bash
# Source this before any DPVO / DROID-SLAM work:
#   source /home/anyone/projects/goodometry/scripts/env.sh
#
# Activates the Isaac-Sim venv (torch 2.11+cu130, sm_120) and points CUDA_HOME
# at the matching toolkit (CUDA 13.0 was installed at /usr/local/cuda-13.0
# during Session 14 of go2_research; system nvcc is still 12.0).

GOODOMETRY_ROOT="/home/anyone/projects/goodometry"
ISAAC_VENV="/home/anyone/projects/isaac/env_isaaclab"
CUDA_ROOT="/usr/local/cuda-13.0"

# shellcheck disable=SC1091
source "${ISAAC_VENV}/bin/activate"

export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}"

# Target Blackwell (RTX 5060 Ti) directly so extension builds skip unsupported archs.
export TORCH_CUDA_ARCH_LIST="12.0"

export GOODOMETRY_ROOT
export GO2_DATASET="/mnt/data/go2_research_dataset_v2"
