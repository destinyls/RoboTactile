#!/usr/bin/env bash
# Create or verify a detached, source-pinned official N0-TWAM checkout.
set -Eeuo pipefail

readonly OFFICIAL_URL="https://github.com/neoteai/N0-TWAM.git"
readonly OFFICIAL_COMMIT="cdd87b6a141667123ad2c25f452478afdb71e287"
readonly ALLOWED_CHANGE="n0_twam/configs/twam_posttrain_cfg.py"

die() {
  printf 'checkout preparation error: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 1 ]] || die "usage: $0 /absolute/path/to/N0-TWAM"
target="$1"
[[ "$target" == /* ]] || die "checkout path must be absolute"

if [[ ! -e "$target" ]]; then
  mkdir -p "$(dirname "$target")"
  git clone --filter=blob:none --no-checkout "$OFFICIAL_URL" "$target"
  git -C "$target" fetch --depth=1 origin "$OFFICIAL_COMMIT"
  git -C "$target" checkout --detach "$OFFICIAL_COMMIT"
fi

[[ -d "$target/.git" ]] || die "target is not a Git checkout: $target"
actual_origin="$(git -C "$target" remote get-url origin)"
[[ "${actual_origin%/}" == "${OFFICIAL_URL%/}" ]] || die "origin mismatch: $actual_origin"
actual_head="$(git -C "$target" rev-parse HEAD)"
[[ "$actual_head" == "$OFFICIAL_COMMIT" ]] || die "HEAD mismatch: $actual_head"
while IFS= read -r status_line; do
  [[ -z "$status_line" || "${status_line:3}" == "$ALLOWED_CHANGE" ]] || \
    die "forbidden upstream worktree change: $status_line"
done < <(git -C "$target" status --porcelain=v1 --untracked-files=all)

printf 'OFFICIAL_CHECKOUT_OK repo=%s commit=%s allowed_change=%s\n' \
  "$target" "$actual_head" "$ALLOWED_CHANGE"
