#!/usr/bin/env bash
# Isolated overlay: shared tensor libraries are read-only; no system installs.
set -euo pipefail
test "$#" -eq 1 || { printf 'Usage: bash install_dream_runtime.sh DEPLOYMENT_ROOT\n' >&2; exit 2; }
deploy_root="$1"
case "$deploy_root" in /*) ;; *) exit 2 ;; esac
dream_runtime="$deploy_root/runtime/dream-tac"
shared_runtime="$deploy_root/runtime/ftp1-policy"
test ! -e "$dream_runtime" || { printf 'Refusing to overwrite Dream-Tac runtime\n' >&2; exit 2; }
"$shared_runtime/bin/python" -m venv --without-pip "$dream_runtime"
dream_site="$dream_runtime/lib/python3.11/site-packages"
shared_site="$shared_runtime/lib/python3.11/site-packages"
export PYTHONPATH="$dream_site:$shared_site"
export CUDA_HOME="$deploy_root/runtime/cuda-toolkit-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
export NVTE_FRAMEWORK=pytorch
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80;90;100;120}"
export CPLUS_INCLUDE_PATH="$shared_site/nvidia/cudnn/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CUDNN_INCLUDE_DIR="$shared_site/nvidia/cudnn/include"
export CUDNN_LIBRARY="$shared_site/nvidia/cudnn/lib/libcudnn.so.9"
export CUDNN_HOME="$shared_site/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$shared_runtime/bin/python" -m pip install --no-deps --no-cache-dir --target "$dream_site" \
  hydra-core==1.3.2 loguru==0.7.3 peft==0.17.1 accelerate==1.6.0 megatron-core==0.12.1 \
  iopath==0.1.10 portalocker==3.2.0 decord==0.6.0 ftfy==6.3.1 \
  webdataset==0.2.111 braceexpand==0.1.7 transformers==4.51.3 \
  ninja==1.11.1.4 pynvml==12.0.0 termcolor==3.1.0 better-profanity==0.7.0 \
  fvcore==0.1.5.post20221221 yacs==0.1.8 tabulate==0.9.0 flask-cors==6.0.1 \
  pytz==2025.2
# PyTorch only honors MAX_JOBS through Ninja. pip --target places the
# executable under the overlay's bin directory, which is not on PATH by default.
export PATH="$dream_site/bin:$PATH"
"$shared_runtime/bin/python" -m pip install --no-deps --no-cache-dir --no-build-isolation \
  --target "$dream_site" 'transformer-engine[torch]==2.2.0' transformer-engine-cu12==2.2.0 transformer-engine-torch==2.2.0 \
  flash-attn==2.7.3
"$dream_runtime/bin/python" -m pip freeze --path "$dream_site"
