#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 INTEGRATION_ID [DESTINATION] [--dry-run]" >&2
  exit 2
fi

INTEGRATION_ID="$1"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PIN_RESOLVER="$SCRIPT_DIR/resolve_pin.py"
PYTHON_BIN="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "required command is unavailable: $PYTHON_BIN" >&2
  exit 2
}
REPOSITORY_URL="$($PYTHON_BIN "$PIN_RESOLVER" "$INTEGRATION_ID" repository_url)"
COMMIT_SHA="$($PYTHON_BIN "$PIN_RESOLVER" "$INTEGRATION_ID" commit_sha)"
LICENSE_SPDX="$($PYTHON_BIN "$PIN_RESOLVER" "$INTEGRATION_ID" license_spdx)"
SOURCE_DIRECTORY="$($PYTHON_BIN "$PIN_RESOLVER" "$INTEGRATION_ID" source_directory)"
DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$REPOSITORY_ROOT/deployment}"
DESTINATION=""
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      printf 'usage: %s [DESTINATION] [--dry-run]\n' "$(basename "$0")"
      exit 0
      ;;
    --dry-run)
      DRY_RUN=true
      ;;
    *)
      [[ -z "$DESTINATION" ]] || {
        echo "only one destination may be supplied" >&2
        exit 2
      }
      DESTINATION="$1"
      ;;
  esac
  shift
done
DESTINATION="${DESTINATION:-$DEPLOY_ROOT/sources/$SOURCE_DIRECTORY}"

case "$REPOSITORY_URL" in
  https://github.com/*) ;;
  *) echo "repository must use the frozen GitHub HTTPS origin" >&2; exit 2 ;;
esac
if [[ ! "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "commit must be a lowercase 40-character Git SHA" >&2
  exit 2
fi
case "$DESTINATION" in
  /*) ;;
  *) echo "destination must be absolute" >&2; exit 2 ;;
esac
[[ ! -L "$DESTINATION" ]] || {
  echo "destination cannot be a symlink" >&2
  exit 2
}

if [[ "$DRY_RUN" == true ]]; then
  printf '{"commit_sha":"%s","destination":"%s","integration_id":"%s","license_spdx":"%s","repository_url":"%s","writes_performed":false}\n' \
    "$COMMIT_SHA" "$DESTINATION" "$INTEGRATION_ID" "$LICENSE_SPDX" "$REPOSITORY_URL"
  exit 0
fi
case "$DESTINATION" in
  /|/data|/data1|/data2|/mnt|/mnt/data)
  echo "destination is too broad" >&2
  exit 2
  ;;
esac

normalize_url() {
  local value="$1"
  value="${value%/}"
  value="${value%.git}"
  printf '%s' "$value"
}

verify_checkout() {
  local root="$1"
  [[ -d "$root/.git" ]] || { echo "existing destination is not a Git checkout" >&2; return 1; }
  local actual_commit actual_origin dirty
  actual_commit="$(git -C "$root" rev-parse HEAD)"
  actual_origin="$(git -C "$root" remote get-url origin)"
  dirty="$(git -C "$root" status --porcelain=v1 --untracked-files=all)"
  [[ "$actual_commit" == "$COMMIT_SHA" ]] || { echo "checkout commit mismatch" >&2; return 1; }
  [[ "$(normalize_url "$actual_origin")" == "$(normalize_url "$REPOSITORY_URL")" ]] || {
    echo "checkout origin mismatch" >&2
    return 1
  }
  [[ -z "$dirty" ]] || { echo "checkout is dirty" >&2; return 1; }
}

write_receipt() {
  local receipt="${DESTINATION}.robotactile-install.json"
  local candidate
  candidate="$(mktemp "${receipt}.candidate.XXXXXX")"
  printf '{"commit_sha":"%s","integration_id":"%s","license_spdx":"%s","repository_url":"%s","schema_version":"robotactile-external-install-v1"}\n' \
    "$COMMIT_SHA" "$INTEGRATION_ID" "$LICENSE_SPDX" "$REPOSITORY_URL" >"$candidate"
  if [[ -e "$receipt" ]]; then
    cmp -s "$candidate" "$receipt" || {
      echo "existing install receipt differs; refusing overwrite" >&2
      rm -f "$candidate"
      return 1
    }
    rm -f "$candidate"
  else
    mv "$candidate" "$receipt"
  fi
}

if [[ -e "$DESTINATION" ]]; then
  verify_checkout "$DESTINATION"
  write_receipt
  exit 0
fi

PARENT="$(dirname "$DESTINATION")"
mkdir -p "$PARENT"
STAGE_ROOT="$(mktemp -d "$PARENT/.robotactile-install.XXXXXX")"
cleanup() {
  rm -rf "$STAGE_ROOT"
}
trap cleanup EXIT

git clone --no-checkout "$REPOSITORY_URL" "$STAGE_ROOT/source"
git -C "$STAGE_ROOT/source" checkout --detach "$COMMIT_SHA"
verify_checkout "$STAGE_ROOT/source"
mv "$STAGE_ROOT/source" "$DESTINATION"
write_receipt
