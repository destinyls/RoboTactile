#!/usr/bin/env bash
# Container-side environment for one official Vision/tactile encoder worker.
set -Eeuo pipefail

readonly RUNTIME_PYTHON="/mnt/data/task/n0_twam_track32_franka_20260810/runtime/venv_py310_24ec50d/bin/python"
readonly DIFFUSERS_OVERLAY="/mnt/data/task/n0_twam_track32_franka_20260810/deploy/wa_track3_clearup_s21000_20260817/python-overlay"
readonly LEROBOT_OVERLAY="/mnt/data/task/n0_twam_track32_franka_xyzw_20260817_v1/runtime/lerobot_0_3_3_overlay_v4"
readonly IMAGEIO_OVERLAY="/mnt/data/task/n0_twam_track31_retrain_vision_tactile_20260829_v1/runtime/python-overlay-official-cdd87b6-v1"
readonly WORKER_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/worker.py"
readonly CACHE_ROOT="${ROBOTACTILE_N0_LATENT_CACHE_ROOT:-/mnt/data/task/robotactile-cache/n0-twam-cdd87b6-latent-v1}"

[[ $# -eq 1 ]] || { printf 'latent worker expects one payload\n' >&2; exit 2; }
[[ -f /opt/hyhal/env.sh ]] || { printf 'HCU environment is not mounted\n' >&2; exit 2; }
command -v realpath >/dev/null 2>&1 || {
  printf 'realpath is required for cache path validation\n' >&2
  exit 2
}
readonly RESOLVED_CACHE_ROOT="$(realpath -m -- "$CACHE_ROOT")"
[[ "$RESOLVED_CACHE_ROOT" == /mnt/data/* ]] || {
  printf 'latent cache root must remain below /mnt/data\n' >&2
  exit 2
}
mkdir -p \
  "$RESOLVED_CACHE_ROOT/torchinductor" \
  "$RESOLVED_CACHE_ROOT/triton" \
  "$RESOLVED_CACHE_ROOT/xdg"
# shellcheck disable=SC1091
source /opt/hyhal/env.sh
export PYTHONPATH="$DIFFUSERS_OVERLAY:$LEROBOT_OVERLAY:$IMAGEIO_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export HIP_VISIBLE_DEVICES="0,1,5,4,2,3,7,6"
export CUDA_VISIBLE_DEVICES="0,1,5,4,2,3,7,6"
export LD_LIBRARY_PATH="/opt/n0_twam/rccl_plugin:/opt/shca_device_driver/ucx-1.18.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$RESOLVED_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$RESOLVED_CACHE_ROOT/triton"
export XDG_CACHE_HOME="$RESOLVED_CACHE_ROOT/xdg"

exec "$RUNTIME_PYTHON" "$WORKER_SCRIPT" --payload "$1"
