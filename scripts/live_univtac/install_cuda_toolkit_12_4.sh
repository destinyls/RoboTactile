#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

case "${ROBOTACTILE_CUDA_TOOLKIT_PROFILE:-12.4}" in
  12.4)
    CUDA_TOOLKIT_VERSION="12.4.1"
    CUDA_TOOLKIT_SERIES="12.4"
    EXPECTED_NVCC_RELEASE="12.4"
    EXPECTED_NVCC_BUILD="V12.4.131"
    ;;
  12.8)
    CUDA_TOOLKIT_VERSION="12.8.1"
    CUDA_TOOLKIT_SERIES="12.8"
    EXPECTED_NVCC_RELEASE="12.8"
    EXPECTED_NVCC_BUILD="V12.8.93"
    ;;
  *) die "unsupported CUDA Toolkit installer profile" ;;
esac
DEPLOY_ROOT="$(default_deployment_root)"
RUNFILE=""
EXPECTED_SHA256=""
NVCC_RELEASE=""
NVCC_BUILD=""
NVCC_IDENTITY=""
RUNFILE_STARTED="false"
SYSTEM_CUDA_LINK="/usr/local/cuda"
SYSTEM_CUDA_LINK_TEST_OVERRIDE="false"
SYSTEM_CUDA_LINK_OBSERVED="false"
SYSTEM_CUDA_LINK_RESTORATION="not_observed"

usage() {
  cat <<EOF
Usage: install_cuda_toolkit_${CUDA_TOOLKIT_SERIES//./_}.sh --runfile FILE --sha256 HEX [--root PATH]

Installs only CUDA Toolkit $CUDA_TOOLKIT_VERSION from the verified official NVIDIA runfile
below the RoboTactile deployment root. The NVIDIA driver, man page, and
/usr/local are not requested installation targets. The installer rejects a
pre-existing /usr/local/cuda and removes that path after the runfile only when
it is a symlink to the deployment-local toolkit, allowing the vendor's single
trailing slash. Foreign destinations are never overwritten or removed.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || die "--root requires a value"
      DEPLOY_ROOT="$2"
      shift 2
      ;;
    --runfile)
      [ "$#" -ge 2 ] || die "--runfile requires a value"
      RUNFILE="$2"
      shift 2
      ;;
    --sha256)
      [ "$#" -ge 2 ] || die "--sha256 requires a value"
      EXPECTED_SHA256="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$RUNFILE" ] || die "--runfile is required"
[ -f "$RUNFILE" ] || die "CUDA Toolkit runfile is unavailable: $RUNFILE"
[ -n "$EXPECTED_SHA256" ] || die "--sha256 is required"
validate_sha256 "$EXPECTED_SHA256"
EXPECTED_SHA256="$(printf '%s' "$EXPECTED_SHA256" | tr 'A-F' 'a-f')"

initialize_layout "$DEPLOY_ROOT"
DEPLOY_ROOT="$(cd "$DEPLOY_ROOT" && pwd -P)"
initialize_layout "$DEPLOY_ROOT"
require_command sh
require_command python3

INSTALL_PATH="$DEPLOY_ROOT/runtime/cuda-toolkit-$CUDA_TOOLKIT_SERIES"
RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/cuda_toolkit_install.json"
INCOMPLETE_MARKER="$DEPLOY_ROOT/runtime/.cuda-toolkit-$CUDA_TOOLKIT_SERIES.robotactile-install-incomplete.json"
ACTUAL_SHA256="$(sha256_file "$RUNFILE")"
RUNFILE_ARGUMENTS="--silent --toolkit --toolkitpath=runtime/cuda-toolkit-$CUDA_TOOLKIT_SERIES --no-man-page --tmpdir=runtime/tmp"

case "${ROBOTACTILE_TEST_SYSTEM_CUDA_LINK_SANDBOX:-}" in
  "") ;;
  1)
    [ -n "${PYTEST_CURRENT_TEST:-}" ] || \
      die "test-only system CUDA link sandbox is available only under pytest"
    # Test-only override: the path is fixed below the canonical deployment
    # root. No caller-controlled deletion target is accepted.
    SYSTEM_CUDA_LINK="$DEPLOY_ROOT/runtime/.robotactile-test-only-usr-local-cuda"
    SYSTEM_CUDA_LINK_TEST_OVERRIDE="true"
    export ROBOTACTILE_TEST_SYSTEM_CUDA_LINK_PATH="$SYSTEM_CUDA_LINK"
    ;;
  *) die "ROBOTACTILE_TEST_SYSTEM_CUDA_LINK_SANDBOX accepts only the exact value 1" ;;
esac

system_cuda_link_exists() {
  [ -e "$SYSTEM_CUDA_LINK" ] || [ -L "$SYSTEM_CUDA_LINK" ]
}

restore_system_cuda_link() {
  local target
  local normalized_target
  if ! system_cuda_link_exists; then
    SYSTEM_CUDA_LINK_OBSERVED="false"
    SYSTEM_CUDA_LINK_RESTORATION="not_observed"
    return 0
  fi
  SYSTEM_CUDA_LINK_OBSERVED="true"
  if [ ! -L "$SYSTEM_CUDA_LINK" ]; then
    printf 'ERROR: CUDA runfile created a non-symlink at protected path %s; refusing to remove it\n' \
      "$SYSTEM_CUDA_LINK" >&2
    SYSTEM_CUDA_LINK_RESTORATION="refused_non_symlink"
    return 1
  fi
  target="$(readlink "$SYSTEM_CUDA_LINK")"
  normalized_target="${target%/}"
  if [ "$normalized_target" != "$INSTALL_PATH" ]; then
    printf 'ERROR: CUDA runfile created protected symlink %s -> %s instead of exact target %s; refusing to remove it\n' \
      "$SYSTEM_CUDA_LINK" "$target" "$INSTALL_PATH" >&2
    SYSTEM_CUDA_LINK_RESTORATION="refused_target_mismatch"
    return 1
  fi
  if ! rm -f -- "$SYSTEM_CUDA_LINK"; then
    printf 'ERROR: failed to remove verified protected symlink %s\n' \
      "$SYSTEM_CUDA_LINK" >&2
    SYSTEM_CUDA_LINK_RESTORATION="remove_failed"
    return 1
  fi
  if system_cuda_link_exists; then
    printf 'ERROR: protected symlink still exists after removal attempt: %s\n' \
      "$SYSTEM_CUDA_LINK" >&2
    SYSTEM_CUDA_LINK_RESTORATION="remove_failed"
    return 1
  fi
  SYSTEM_CUDA_LINK_RESTORATION="removed_exact_install_symlink"
}

cleanup() {
  local status="$?"
  trap - EXIT
  if [ "$RUNFILE_STARTED" = "true" ]; then
    restore_system_cuda_link || status=2
  fi
  release_lock
  exit "$status"
}

acquire_lock "cuda-toolkit-$CUDA_TOOLKIT_VERSION-install"
trap cleanup EXIT

if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  die "SHA-256 mismatch for $RUNFILE: expected $EXPECTED_SHA256, got $ACTUAL_SHA256"
fi

probe_nvcc() {
  local output
  local nvcc="$INSTALL_PATH/bin/nvcc"
  [ -x "$nvcc" ] || die "installed CUDA Toolkit nvcc is absent or not executable: $nvcc"
  output="$("$nvcc" --version 2>&1)" || die "installed CUDA Toolkit nvcc probe failed"
  NVCC_RELEASE="$(
    printf '%s\n' "$output" |
      sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\),.*/\1/p' |
      tail -n 1
  )"
  NVCC_BUILD="$(
    printf '%s\n' "$output" |
      sed -n 's/.*release [^,]*, *\(V[0-9][0-9.]*\).*/\1/p' |
      tail -n 1
  )"
  [ "$NVCC_RELEASE" = "$EXPECTED_NVCC_RELEASE" ] || \
    die "CUDA Toolkit nvcc release must be $EXPECTED_NVCC_RELEASE; detected ${NVCC_RELEASE:-unknown}"
  [ "$NVCC_BUILD" = "$EXPECTED_NVCC_BUILD" ] || \
    die "CUDA Toolkit nvcc build must be $EXPECTED_NVCC_BUILD; detected ${NVCC_BUILD:-unknown}"
  NVCC_IDENTITY="$NVCC_RELEASE|$NVCC_BUILD"
}

marker_matches_request() {
  receipt_matches \
    "$INCOMPLETE_MARKER" \
    "component=cuda_toolkit" \
    "version=$CUDA_TOOLKIT_VERSION" \
    "status=incomplete" \
    "source_sha256=$ACTUAL_SHA256" \
    "toolkit_path=$INSTALL_PATH" \
    "install_path=runtime/cuda-toolkit-$CUDA_TOOLKIT_SERIES"
}

if [ -e "$RECEIPT_PATH" ]; then
  if system_cuda_link_exists; then
    die "refusing to accept CUDA receipt while protected path exists: $SYSTEM_CUDA_LINK"
  fi
  RECORDED_SYSTEM_CUDA_LINK_OBSERVED="$(
    receipt_field "$RECEIPT_PATH" "system_cuda_link_observed_after_run"
  )" || die "CUDA Toolkit receipt lacks the protected system-link observation"
  RECORDED_SYSTEM_CUDA_LINK_RESTORATION="$(
    receipt_field "$RECEIPT_PATH" "system_cuda_link_restoration"
  )" || die "CUDA Toolkit receipt lacks the protected system-link restoration result"
  case "$RECORDED_SYSTEM_CUDA_LINK_OBSERVED|$RECORDED_SYSTEM_CUDA_LINK_RESTORATION" in
    false\|not_observed|true\|removed_exact_install_symlink) ;;
    *) die "CUDA Toolkit receipt records an unsafe protected system-link restoration result" ;;
  esac
  [ -d "$INSTALL_PATH" ] || die "receipt exists but CUDA Toolkit directory is absent"
  [ ! -L "$INSTALL_PATH" ] || die "installed CUDA Toolkit path must not be a symbolic link"
  probe_nvcc
  if receipt_matches \
    "$RECEIPT_PATH" \
    "component=cuda_toolkit" \
    "version=$CUDA_TOOLKIT_VERSION" \
    "status=installed" \
    "source_sha256=$ACTUAL_SHA256" \
    "toolkit_path=$INSTALL_PATH" \
    "install_path=runtime/cuda-toolkit-$CUDA_TOOLKIT_SERIES" \
    "nvcc_release=$NVCC_RELEASE" \
    "nvcc_build=$NVCC_BUILD" \
    "nvcc_identity=$NVCC_IDENTITY" \
    "runfile_arguments=$RUNFILE_ARGUMENTS" \
    "driver_install_requested=false" \
    "system_cuda_link_path=$SYSTEM_CUDA_LINK" \
    "system_cuda_link_preexisting=false" \
    "system_cuda_link_observed_after_run=$RECORDED_SYSTEM_CUDA_LINK_OBSERVED" \
    "system_cuda_link_restoration=$RECORDED_SYSTEM_CUDA_LINK_RESTORATION" \
    "system_cuda_link_test_override=$SYSTEM_CUDA_LINK_TEST_OVERRIDE" \
    "vendor_log_path=/var/log/cuda-installer.log" \
    "vendor_log_managed=false"; then
    if [ -e "$INCOMPLETE_MARKER" ]; then
      marker_matches_request || \
        die "installed receipt conflicts with the incomplete-install marker"
      rm -f -- "$INCOMPLETE_MARKER"
    fi
    info "CUDA Toolkit $CUDA_TOOLKIT_VERSION is already installed with the same runfile SHA-256"
    printf '%s\n' "$RECEIPT_PATH"
    exit 0
  fi
  die "existing CUDA Toolkit receipt does not match the requested runfile or nvcc identity"
fi

if system_cuda_link_exists; then
  die "refusing to run CUDA installer while protected path exists: $SYSTEM_CUDA_LINK"
fi

if [ -e "$INCOMPLETE_MARKER" ]; then
  marker_matches_request || \
    die "refusing to resume a CUDA Toolkit installation created from a different request"
  if [ -e "$INSTALL_PATH" ] || [ -L "$INSTALL_PATH" ]; then
    [ -d "$INSTALL_PATH" ] || \
      die "incomplete CUDA Toolkit destination is not a directory: $INSTALL_PATH"
    [ ! -L "$INSTALL_PATH" ] || \
      die "refusing to recover a symbolic-link CUDA Toolkit destination"
    info "removing the verified incomplete CUDA Toolkit destination before retry"
    rm -rf -- "$INSTALL_PATH"
  fi
elif [ -e "$INSTALL_PATH" ] || [ -L "$INSTALL_PATH" ]; then
  die "refusing to overwrite foreign CUDA Toolkit destination: $INSTALL_PATH"
else
  write_receipt \
    "$INCOMPLETE_MARKER" \
    "component=cuda_toolkit" \
    "version=$CUDA_TOOLKIT_VERSION" \
    "status=incomplete" \
    "source_sha256=$ACTUAL_SHA256" \
    "toolkit_path=$INSTALL_PATH" \
    "install_path=runtime/cuda-toolkit-$CUDA_TOOLKIT_SERIES"
fi

LOG_PATH="$(new_log_path "cuda-toolkit-$CUDA_TOOLKIT_VERSION-runfile")"
RUNFILE_STARTED="true"
RUNFILE_STATUS=0
run_logged \
  "$LOG_PATH" \
  sh "$RUNFILE" \
    --silent \
    --toolkit \
    "--toolkitpath=$INSTALL_PATH" \
    --no-man-page \
    "--tmpdir=$DEPLOY_ROOT/runtime/tmp" || RUNFILE_STATUS="$?"
if ! restore_system_cuda_link; then
  RUNFILE_STARTED="false"
  die "CUDA Toolkit runfile left an unsafe protected system path; manual inspection is required"
fi
RUNFILE_STARTED="false"
if [ "$RUNFILE_STATUS" -ne 0 ]; then
  die "CUDA Toolkit runfile failed; verified incomplete state retained for retry; see $LOG_PATH"
fi

[ -d "$INSTALL_PATH" ] || die "CUDA Toolkit runfile did not create $INSTALL_PATH"
[ ! -L "$INSTALL_PATH" ] || die "CUDA Toolkit runfile created a symbolic-link destination"
probe_nvcc

LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt \
  "$RECEIPT_PATH" \
  "component=cuda_toolkit" \
  "version=$CUDA_TOOLKIT_VERSION" \
  "status=installed" \
  "source_runfile=$(basename "$RUNFILE")" \
  "source_sha256=$ACTUAL_SHA256" \
  "toolkit_path=$INSTALL_PATH" \
  "install_path=runtime/cuda-toolkit-$CUDA_TOOLKIT_SERIES" \
  "nvcc_release=$NVCC_RELEASE" \
  "nvcc_build=$NVCC_BUILD" \
  "nvcc_identity=$NVCC_IDENTITY" \
  "runfile_arguments=$RUNFILE_ARGUMENTS" \
  "driver_install_requested=false" \
  "man_page_install_requested=false" \
  "system_prefix_install_requested=false" \
  "system_cuda_link_path=$SYSTEM_CUDA_LINK" \
  "system_cuda_link_preexisting=false" \
  "system_cuda_link_observed_after_run=$SYSTEM_CUDA_LINK_OBSERVED" \
  "system_cuda_link_restoration=$SYSTEM_CUDA_LINK_RESTORATION" \
  "system_cuda_link_test_override=$SYSTEM_CUDA_LINK_TEST_OVERRIDE" \
  "vendor_log_path=/var/log/cuda-installer.log" \
  "vendor_log_managed=false" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"
rm -f -- "$INCOMPLETE_MARKER"

info "CUDA Toolkit $CUDA_TOOLKIT_VERSION toolkit-only installation completed"
printf '%s\n' "$RECEIPT_PATH"
