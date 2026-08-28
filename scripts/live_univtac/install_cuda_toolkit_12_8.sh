#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBOTACTILE_CUDA_TOOLKIT_PROFILE="12.8"
exec bash "$SCRIPT_DIR/install_cuda_toolkit_12_4.sh" "$@"
