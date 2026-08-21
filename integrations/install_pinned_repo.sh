#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 INTEGRATION_ID REPOSITORY_URL COMMIT_SHA LICENSE_SPDX DESTINATION" >&2
  exit 2
fi

INTEGRATION_ID="$1"
REPOSITORY_URL="$2"
COMMIT_SHA="$3"
LICENSE_SPDX="$4"
DESTINATION="$5"

case "$INTEGRATION_ID" in
  act_runtime|n0_twam|univtac) ;;
  *) echo "unknown integration id" >&2; exit 2 ;;
esac
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
if [[ "$DESTINATION" == "/" ]]; then
  echo "destination cannot be filesystem root" >&2
  exit 2
fi

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
