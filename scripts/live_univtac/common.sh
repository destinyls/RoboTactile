#!/usr/bin/env bash

# Shared, source-only helpers for the live UniVTAC deployment entry points.

set -Eeuo pipefail
umask 022

ROBOTACTILE_DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-}"
ROBOTACTILE_LOCK_DIR=""
ROBOTACTILE_CUDA_ROOT=""
ROBOTACTILE_CUDA_TOOLKIT_VERSION=""
ROBOTACTILE_CUDA_NVCC_IDENTITY=""
ROBOTACTILE_CUDA_ARCHITECTURE=""
ROBOTACTILE_CUDA_COMPUTE_CAPABILITY=""
ROBOTACTILE_REPOSITORY_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P
)"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

info() {
  printf 'INFO: %s\n' "$*" >&2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

validate_cuda_build_options() {
  local cuda_root="$1" cuda_architecture="$2" gpu_index="$3"
  case "$gpu_index" in ''|*[!0-9]*) die "--gpu must be a non-negative integer" ;; esac
  case "$cuda_architecture" in
    auto|[1-9][0-9]|[1-9][0-9][0-9]) ;;
    *) die "--cuda-architecture must be auto or a two/three-digit compute target such as 80 or 86" ;;
  esac
  case "$cuda_root" in ''|/*) ;; *) die "--cuda-root must be an absolute path" ;; esac
  case "$cuda_root" in *$'\n'*) die "--cuda-root must not contain a newline" ;; esac
}

resolve_cuda_build_target() {
  local deploy_root="$1" isaac_sim_path="$2" requested_root="$3"
  local requested_architecture="$4" gpu_index="$5"
  local cuda_root="$requested_root" deploy_real probe nvcc_output nvcc_build detected major
  if [ -z "$cuda_root" ]; then
    if [ -x "$deploy_root/runtime/cuda-toolkit-12.8/bin/nvcc" ]; then
      cuda_root="$deploy_root/runtime/cuda-toolkit-12.8"
    elif [ -x "$deploy_root/runtime/cuda-toolkit-12.4/bin/nvcc" ]; then
      cuda_root="$deploy_root/runtime/cuda-toolkit-12.4"
    elif [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
      cuda_root="/usr/local/cuda-12.8"
    elif [ -x /usr/local/cuda-12.4/bin/nvcc ]; then
      cuda_root="/usr/local/cuda-12.4"
    else
      die "CUDA 12.8/12.4 toolkit is absent; pass --cuda-root to a toolkit below $deploy_root"
    fi
  fi
  [ -d "$cuda_root" ] || die "CUDA toolkit root is absent: $cuda_root"
  cuda_root="$(cd "$cuda_root" && pwd -P)"
  if [ -n "$requested_root" ]; then
    deploy_real="$(cd "$deploy_root" && pwd -P)"
    case "$cuda_root" in "$deploy_real"/*) ;; *) die "--cuda-root must resolve below the deployment root" ;; esac
  fi
  [ -x "$cuda_root/bin/nvcc" ] || die "CUDA toolkit nvcc is absent: $cuda_root/bin/nvcc"
  nvcc_output="$("$cuda_root/bin/nvcc" --version 2>&1)" || die "CUDA toolkit nvcc probe failed"
  ROBOTACTILE_CUDA_TOOLKIT_VERSION="$(printf '%s\n' "$nvcc_output" | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p' | tail -n 1)"
  nvcc_build="$(printf '%s\n' "$nvcc_output" | sed -n 's/.*release [^,]*, *\(V[0-9][0-9.]*\).*/\1/p' | tail -n 1)"
  case "$ROBOTACTILE_CUDA_TOOLKIT_VERSION" in
    12.4|12.8) ;;
    *) die "CUDA toolkit must be 12.4 or 12.8; detected ${ROBOTACTILE_CUDA_TOOLKIT_VERSION:-unknown} at $cuda_root" ;;
  esac
  [ -n "$nvcc_build" ] || die "CUDA nvcc build identity is unavailable"
  ROBOTACTILE_CUDA_NVCC_IDENTITY="$ROBOTACTILE_CUDA_TOOLKIT_VERSION|$nvcc_build"
  probe="$(env CUDA_VISIBLE_DEVICES="$gpu_index" "$isaac_sim_path/python.sh" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; c = torch.cuda.get_device_capability(0); print(f"ROBOTACTILE_CUDA_ARCH={c[0]}{c[1]}")' 2>&1)" || die "failed to detect compute capability for GPU $gpu_index: $probe"
  detected="$(printf '%s\n' "$probe" | sed -n 's/^ROBOTACTILE_CUDA_ARCH=\([1-9][0-9]*\)$/\1/p' | tail -n 1)"
  case "$detected" in [1-9][0-9]|[1-9][0-9][0-9]) ;; *) die "invalid compute capability probe for GPU $gpu_index" ;; esac
  if [ "$requested_architecture" = "auto" ]; then
    ROBOTACTILE_CUDA_ARCHITECTURE="$detected"
  elif [ "$requested_architecture" = "$detected" ]; then
    ROBOTACTILE_CUDA_ARCHITECTURE="$requested_architecture"
  else
    die "requested CUDA architecture $requested_architecture does not match GPU $gpu_index capability $detected"
  fi
  case "$ROBOTACTILE_CUDA_ARCHITECTURE" in
    100|101|120)
      [ "$ROBOTACTILE_CUDA_TOOLKIT_VERSION" = "12.8" ] || \
        die "CUDA architecture $ROBOTACTILE_CUDA_ARCHITECTURE requires CUDA 12.8; detected $ROBOTACTILE_CUDA_TOOLKIT_VERSION at $cuda_root"
      ;;
  esac
  major="${ROBOTACTILE_CUDA_ARCHITECTURE%?}"
  ROBOTACTILE_CUDA_COMPUTE_CAPABILITY="$major.${ROBOTACTILE_CUDA_ARCHITECTURE#"$major"}"
  ROBOTACTILE_CUDA_ROOT="$cuda_root"
}

default_deployment_root() {
  printf '%s\n' "${ROBOTACTILE_DEPLOY_ROOT:-$ROBOTACTILE_REPOSITORY_ROOT/deployment}"
}

resolve_external_pin() {
  local integration_id="$1"
  local field="$2"
  local python_bin="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
  require_command "$python_bin"
  "$python_bin" \
    "$ROBOTACTILE_REPOSITORY_ROOT/integrations/resolve_pin.py" \
    "$integration_id" \
    "$field"
}

validate_absolute_root() {
  local root="$1"
  case "$root" in
    /*) ;;
    *) die "deployment root must be an absolute path: $root" ;;
  esac
  case "$root" in
    /|/data|/data1|/data2|/mnt|/mnt/data) die "deployment root is too broad: $root" ;;
  esac
  case "$root" in
    *$'\n'*) die "deployment root must not contain a newline" ;;
  esac
}

initialize_layout() {
  local root="$1"
  validate_absolute_root "$root"
  ROBOTACTILE_DEPLOY_ROOT="$root"
  export ROBOTACTILE_DEPLOY_ROOT

  mkdir -p \
    "$root/sources" \
    "$root/artifacts/models/act" \
    "$root/artifacts/models/n0_twam" \
    "$root/artifacts/deployment" \
    "$root/artifacts/preflight" \
    "$root/artifacts/rest-references" \
    "$root/artifacts/live-univtac" \
    "$root/requests/calibration" \
    "$root/requests/four-condition" \
    "$root/requests/primary-matrix" \
    "$root/outputs/matrices" \
    "$root/outputs/reports" \
    "$root/logs" \
    "$root/runtime/cache/cuda" \
    "$root/runtime/cache/pycache" \
    "$root/runtime/cache/torch" \
    "$root/runtime/home" \
    "$root/runtime/locks" \
    "$root/runtime/omni-cache/kit" \
    "$root/runtime/omni-cache/ov" \
    "$root/runtime/omni-cache/user" \
    "$root/runtime/pip-cache" \
    "$root/runtime/tmp"

  # Upstream installers and applications see a deployment-local HOME. This
  # prevents post_install or pip from touching the invoking user's HOME.
  export HOME="$root/runtime/home"
  export XDG_CACHE_HOME="$root/runtime/cache"
  export PIP_CACHE_DIR="$root/runtime/pip-cache"
  export TMPDIR="$root/runtime/tmp"
  export TORCH_HOME="$root/runtime/cache/torch"
  export CUDA_CACHE_PATH="$root/runtime/cache/cuda"
  export PYTHONPYCACHEPREFIX="$root/runtime/cache/pycache"
  export OMNI_KIT_USER_DATA_PATH="$root/runtime/omni-cache/kit"
  export OV_CACHE_DIR="$root/runtime/omni-cache/ov"
  export OMNI_USER_FOLDER="$root/runtime/omni-cache/user"
}

acquire_lock() {
  local component="$1"
  local lock_dir="$ROBOTACTILE_DEPLOY_ROOT/runtime/locks/$component.lock"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    die "another deployment process holds lock: $lock_dir"
  fi
  ROBOTACTILE_LOCK_DIR="$lock_dir"
}

release_lock() {
  if [ -n "$ROBOTACTILE_LOCK_DIR" ] && [ -d "$ROBOTACTILE_LOCK_DIR" ]; then
    rmdir "$ROBOTACTILE_LOCK_DIR" || true
  fi
  ROBOTACTILE_LOCK_DIR=""
}

sha256_file() {
  local path="$1"
  local digest=""
  if command -v openssl >/dev/null 2>&1; then
    digest="$(LC_ALL=C openssl dgst -sha256 "$path" | awk '{print $NF}')"
  elif command -v sha256sum >/dev/null 2>&1; then
    digest="$(LC_ALL=C sha256sum "$path" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    digest="$(LC_ALL=C LANG=C shasum -a 256 "$path" | awk '{print $1}')"
  else
    die "no SHA-256 implementation is available"
  fi
  case "$digest" in
    *[!0-9a-fA-F]*|'') die "invalid SHA-256 output for $path" ;;
  esac
  [ "${#digest}" -eq 64 ] || die "invalid SHA-256 length for $path"
  printf '%s\n' "$(printf '%s' "$digest" | tr 'A-F' 'a-f')"
}

sha512_file() {
  local path="$1"
  local digest=""
  if command -v openssl >/dev/null 2>&1; then
    digest="$(LC_ALL=C openssl dgst -sha512 "$path" | awk '{print $NF}')"
  elif command -v sha512sum >/dev/null 2>&1; then
    digest="$(LC_ALL=C sha512sum "$path" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    digest="$(LC_ALL=C LANG=C shasum -a 512 "$path" | awk '{print $1}')"
  else
    die "no SHA-512 implementation is available"
  fi
  case "$digest" in
    *[!0-9a-fA-F]*|'') die "invalid SHA-512 output for $path" ;;
  esac
  [ "${#digest}" -eq 128 ] || die "invalid SHA-512 length for $path"
  printf '%s\n' "$(printf '%s' "$digest" | tr 'A-F' 'a-f')"
}

validate_sha256() {
  local digest="$1"
  case "$digest" in
    *[!0-9a-fA-F]*|'') die "expected SHA-256 must contain 64 hex characters" ;;
  esac
  [ "${#digest}" -eq 64 ] || die "expected SHA-256 must contain 64 hex characters"
}

new_log_path() {
  local label="$1"
  local timestamp
  local path
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  path="$ROBOTACTILE_DEPLOY_ROOT/logs/${label}-${timestamp}-$$.log"
  if ! (set -C; : > "$path") 2>/dev/null; then
    die "refusing to overwrite deployment log: $path"
  fi
  printf '%s\n' "$path"
}

run_logged() {
  local log_path="$1"
  shift
  {
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } >> "$log_path" 2>&1
}

relative_to_deploy_root() {
  local path="$1"
  case "$path" in
    "$ROBOTACTILE_DEPLOY_ROOT"/*)
      printf '%s\n' "${path#"$ROBOTACTILE_DEPLOY_ROOT"/}"
      ;;
    *) die "path is outside deployment root: $path" ;;
  esac
}

write_receipt() {
  local target="$1"
  shift
  local receipt_dir
  local temporary
  local python_bin="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
  require_command "$python_bin"
  receipt_dir="$(dirname "$target")"
  mkdir -p "$receipt_dir"
  [ ! -e "$target" ] || die "refusing to overwrite deployment receipt: $target"
  temporary="$(mktemp "$receipt_dir/.receipt.XXXXXX")"

  "$python_bin" - "$temporary" "$@" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
document: dict[str, str] = {
    "schema_version": "robotactile.live_univtac.deployment_receipt.v1",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
}
for item in sys.argv[2:]:
    key, separator, value = item.partition("=")
    if not separator or not key or key in document:
        raise SystemExit(f"invalid receipt field: {item!r}")
    document[key] = value
with output.open("w", encoding="utf-8") as stream:
    json.dump(document, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

  if ! ln "$temporary" "$target" 2>/dev/null; then
    rm -f "$temporary"
    die "refusing to overwrite deployment receipt: $target"
  fi
  rm -f "$temporary"
}

receipt_matches() {
  local receipt="$1"
  shift
  local python_bin="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
  [ -f "$receipt" ] || return 1
  "$python_bin" - "$receipt" "$@" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in sys.argv[2:]:
    key, separator, expected = item.partition("=")
    if not separator or document.get(key) != expected:
        raise SystemExit(1)
PY
}

receipt_field() {
  local receipt="$1"
  local field="$2"
  local python_bin="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
  "$python_bin" - "$receipt" "$field" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = document.get(sys.argv[2])
if not isinstance(value, str) or not value:
    raise SystemExit(1)
print(value)
PY
}

safe_remove_temporary_directory() {
  local path="$1"
  case "$path" in
    "$ROBOTACTILE_DEPLOY_ROOT/runtime/tmp/"*) rm -rf -- "$path" ;;
    '') ;;
    *) die "refusing to remove non-temporary path: $path" ;;
  esac
}

ensure_pinned_git_source() {
  local repository="$1"
  local commit="$2"
  local destination="$3"
  local required_relative_path="$4"
  local stage=""
  local resolved=""
  local tracked_changes=""
  require_command git

  if [ ! -e "$destination" ]; then
    stage="$(mktemp -d "$ROBOTACTILE_DEPLOY_ROOT/runtime/tmp/source-clone.XXXXXX")"
    if ! git clone --no-checkout --filter=blob:none "$repository" "$stage"; then
      safe_remove_temporary_directory "$stage"
      die "failed to clone pinned source: $repository"
    fi
    if ! git -C "$stage" checkout --detach "$commit"; then
      safe_remove_temporary_directory "$stage"
      die "failed to check out pinned commit: $commit"
    fi
    resolved="$(git -C "$stage" rev-parse HEAD)" || {
      safe_remove_temporary_directory "$stage"
      die "failed to resolve cloned source commit"
    }
    if [ "$resolved" != "$commit" ]; then
      safe_remove_temporary_directory "$stage"
      die "cloned source resolved to $resolved instead of $commit"
    fi
    if [ ! -e "$stage/$required_relative_path" ]; then
      safe_remove_temporary_directory "$stage"
      die "pinned source lacks required path: $required_relative_path"
    fi
    if ! mv "$stage" "$destination"; then
      safe_remove_temporary_directory "$stage"
      die "failed to publish pinned source at $destination"
    fi
  fi

  [ -d "$destination/.git" ] || die "existing source is not a Git checkout: $destination"
  resolved="$(git -C "$destination" rev-parse HEAD)" || \
    die "failed to resolve existing source commit: $destination"
  [ "$resolved" = "$commit" ] || \
    die "refusing existing source at commit $resolved; expected $commit"
  tracked_changes="$(git -C "$destination" status --porcelain --untracked-files=all)" || \
    die "failed to inspect source worktree: $destination"
  [ -z "$tracked_changes" ] || die "refusing a modified source worktree: $destination"
  [ -e "$destination/$required_relative_path" ] || \
    die "pinned source lacks required path: $required_relative_path"
}
