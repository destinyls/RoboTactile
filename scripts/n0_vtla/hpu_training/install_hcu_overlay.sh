#!/usr/bin/env bash
# Install only non-accelerator N0-VTLA dependencies into a repo-local overlay.
# Usage: bash install_hcu_overlay.sh /mnt/data/task/.../runtime/n0_vtla_py310_overlay
set -Eeuo pipefail

readonly IMAGE_REF="docker-registry-sh.tencentcloudcr.com/hg/hg_wm:v3.5"
readonly EXPECTED_IMAGE_ID="sha256:09822e7616a3f9794284341aee19e1dfd531c2d2cb58168a40b14fcf818fdf9a"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REQUIREMENTS="$SCRIPT_DIR/hcu-requirements-py310.txt"
readonly PYPI_INDEX_URL="${N0_VTLA_PYPI_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}"
readonly OFFICIAL_COMMIT="03a0ce4d7091ca2354864796770715aa212601b7"

die() {
  printf 'N0-VTLA HCU overlay error: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 1 ]] || die "expected one absolute overlay destination"
readonly destination="$1"
[[ "$destination" == /mnt/data/* ]] || die "overlay must live under /mnt/data"
[[ ! -e "$destination" ]] || die "refusing to overwrite overlay: $destination"
[[ -f "$REQUIREMENTS" ]] || die "requirements file missing"
readonly project_root="$(dirname "$(dirname "$destination")")"
readonly source_repo="$project_root/source/N0-VTLA"
readonly replacement_root="$source_repo/n0vtla/models_pytorch/transformers_replace"
readonly shim_root="$project_root/tooling/hpu_training/runtime_shims"
[[ -d "$source_repo/.git" ]] || die "official source checkout missing"
[[ "$(git -C "$source_repo" rev-parse HEAD)" == "$OFFICIAL_COMMIT" ]] || \
  die "official source commit mismatch"
[[ -f "$replacement_root/models/siglip/check.py" ]] || die "transformers replacement tree missing"
[[ -f "$shim_root/sitecustomize.py" ]] || die "runtime shim missing"
command -v docker >/dev/null 2>&1 || die "docker is unavailable"
command -v python3 >/dev/null 2>&1 || die "host Python 3 is unavailable"
readonly actual_image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")"
[[ "$actual_image_id" == "$EXPECTED_IMAGE_ID" ]] || \
  die "vendor image identity mismatch: $actual_image_id"

readonly parent="$(dirname "$destination")"
mkdir -p "$parent"
readonly staging="$(mktemp -d "$parent/.n0-vtla-overlay.XXXXXX")"
cleanup() {
  [[ ! -d "$staging" ]] || rm -rf -- "$staging"
}
trap cleanup EXIT

PIP_CONFIG_FILE=/dev/null python3 -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --retries 12 \
  --ignore-installed \
  --index-url "$PYPI_INDEX_URL" \
  --target "$staging" \
  -r "$REQUIREMENTS"
PIP_CONFIG_FILE=/dev/null python3 -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --retries 12 \
  --no-deps \
  --index-url "$PYPI_INDEX_URL" \
  --target "$staging" \
  git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5

cp -a "$replacement_root/." "$staging/transformers/"

readonly container_pythonpath="$shim_root:$staging:$source_repo"
PYTHONPATH="$container_pythonpath" docker run --rm \
  --network none \
  --entrypoint /bin/bash \
  --mount type=bind,src=/mnt/data,dst=/mnt/data \
  -e PYTHONPATH="$container_pythonpath" \
  -e ROBOTACTILE_ENABLE_N0_VTLA_HCU_SHIMS=1 \
  "$IMAGE_REF" \
  -lc 'python - <<'"'"'PY'"'"'
import importlib.metadata
import inspect
import pathlib

for forbidden in ("torch", "torchvision", "triton", "nvidia-cublas-cu12", "jax-cuda12-plugin"):
    distributions = [
        dist
        for dist in importlib.metadata.distributions(path=[pathlib.Path(__import__("os").environ["PYTHONPATH"])])
        if (dist.metadata.get("Name") or "").lower() == forbidden
    ]
    if distributions:
        raise SystemExit(f"forbidden accelerator package in overlay: {forbidden}")

import etils.epath
import flax
import jax
import lerobot
import n0vtla.shared.normalize
import n0vtla.training.config
import transformers
import tyro
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.siglip import check

if "cond" not in inspect.signature(GemmaRMSNorm.forward).parameters:
    raise SystemExit("transformers replacement did not patch GemmaRMSNorm.forward")
if not check.check_whether_transformers_replace_is_installed_correctly():
    raise SystemExit("transformers replacement self-check failed")

print("overlay imports OK", jax.__version__, flax.__version__, transformers.__version__)
PY'

mv "$staging" "$destination"
trap - EXIT
printf 'N0-VTLA HCU overlay ready: %s\n' "$destination"
