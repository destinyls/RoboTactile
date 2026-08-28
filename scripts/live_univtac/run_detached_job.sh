#!/usr/bin/env bash

# Launch one long-running live job with immutable lifecycle receipts.

set -Eeuo pipefail
umask 022

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

run_worker() {
  [ "$#" -ge 8 ] || die "invalid detached worker invocation"
  local python_bin="$1" job_directory="$2" job_id="$3"
  local artifact_path="$4" log_path="$5" launch_path="$6"
  shift 6
  [ "$1" = "--" ] || die "detached worker command separator is absent"
  shift
  [ "$#" -gt 0 ] || die "detached worker command is empty"

  exec "$python_bin" - \
    "$job_directory" "$job_id" "$artifact_path" "$log_path" \
    "$launch_path" -- "$@" <<'PY'
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def publish_once(path: Path, document: dict[str, object]) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)

def parent_death_guard(parent_pid: int) -> None:
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        os._exit(125)
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)

arguments = sys.argv[1:]
separator = arguments.index("--")
fixed = arguments[:separator]
command = arguments[separator + 1 :]
if len(fixed) != 5 or not command:
    raise SystemExit("invalid detached worker arguments")

job_directory = Path(fixed[0])
job_id = fixed[1]
artifact_path = Path(fixed[2])
log_path = Path(fixed[3])
launch_path = Path(fixed[4])
exit_path = job_directory / "exit.json"
requested_signal: int | None = None
child: subprocess.Popen[bytes] | None = None

def forward_signal(signum: int, _frame: object) -> None:
    global requested_signal
    requested_signal = signum
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass

signal.signal(signal.SIGTERM, forward_signal)
signal.signal(signal.SIGINT, forward_signal)
deadline = time.monotonic() + 30.0
while not launch_path.is_file():
    if requested_signal is not None:
        raise SystemExit(128 + requested_signal)
    if time.monotonic() >= deadline:
        raise SystemExit("launch receipt was not published within 30 seconds")
    time.sleep(0.02)
launch = json.loads(launch_path.read_text(encoding="utf-8"))
expected_launch = {
    "artifact_path": str(artifact_path),
    "command": command,
    "job_id": job_id,
    "log_path": str(log_path),
    "worker_pgid": os.getpgrp(),
    "worker_pid": os.getpid(),
}
if any(launch.get(key) != value for key, value in expected_launch.items()):
    raise SystemExit("launch receipt does not match the detached worker")
if artifact_path.exists() or artifact_path.is_symlink():
    raise SystemExit("expected artifact appeared before command start")
started_at = utc_now()
started_monotonic = time.monotonic()
launch_error: str | None = None
print(f"DETACHED_JOB_START job_id={job_id} started_at_utc={started_at}", flush=True)
try:
    supervisor_pid = os.getpid()
    child = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        preexec_fn=lambda: parent_death_guard(supervisor_pid),
    )
    if requested_signal is not None and child.poll() is None:
        os.killpg(child.pid, requested_signal)
    process_status = child.wait()
except OSError as error:
    process_status = 127
    launch_error = f"{type(error).__name__}: {error}"
    print(f"DETACHED_JOB_LAUNCH_ERROR {launch_error}", flush=True)
ended_at = utc_now()
duration_seconds = time.monotonic() - started_monotonic
if process_status < 0:
    signal_number: int | None = -process_status
    signal_name: str | None = signal.Signals(signal_number).name
    return_code: int | None = None
else:
    signal_number = None
    signal_name = None
    return_code = process_status
forwarded_signal = (
    None if requested_signal is None else signal.Signals(requested_signal).name
)
artifact_present = artifact_path.exists()
if artifact_path.is_dir():
    artifact_kind = "directory"
elif artifact_path.is_file():
    artifact_kind = "file"
elif artifact_path.is_symlink():
    artifact_kind = "symlink"
else:
    artifact_kind = "missing"
root_receipt_present = artifact_path.is_dir() and (
    artifact_path / "root_receipt.json"
).is_file()
print(
    "DETACHED_JOB_END "
    f"job_id={job_id} return_code={return_code} signal={signal_name} "
    f"ended_at_utc={ended_at}",
    flush=True,
)
try:
    os.fsync(sys.stdout.fileno())
except OSError:
    pass
document: dict[str, object] = {
    "artifact_kind": artifact_kind,
    "artifact_path": str(artifact_path),
    "artifact_present": artifact_present,
    "artifact_root_receipt_present": root_receipt_present,
    "duration_seconds": duration_seconds,
    "ended_at_utc": ended_at,
    "forwarded_signal": forwarded_signal,
    "job_id": job_id,
    "launch_error": launch_error,
    "log_path": str(log_path),
    "log_sha256": sha256_file(log_path),
    "return_code": return_code,
    "schema_version": "robotactile.detached_job.exit.v1",
    "signal": signal_name,
    "signal_number": signal_number,
    "started_at_utc": started_at,
    "worker_pgid": os.getpgrp(),
    "worker_pid": os.getpid(),
}
publish_once(exit_path, document)
if signal_number is not None:
    raise SystemExit(128 + signal_number)
raise SystemExit(return_code)
PY
}

if [ "${1:-}" = "--_worker" ]; then
  shift
  run_worker "$@"
fi

DEPLOY_ROOT="$(default_deployment_root)"
JOB_ID=""
ARTIFACT_PATH=""
SYSTEM_PYTHON="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: run_detached_job.sh --root PATH --job-id ID --artifact PATH -- COMMAND [ARG ...]

Launches COMMAND under nohup and a dedicated worker session. The no-clobber
job directory is outputs/jobs/ID below the deployment root. launch.json is
published atomically before COMMAND starts; exit.json is published atomically
only after the real child process exits. A missing exit.json is incomplete
evidence, including after worker SIGKILL or host power loss.

Options:
  --root PATH       Absolute RoboTactile deployment root.
  --job-id ID       Unique job identity: letters, digits, dot, dash, underscore.
  --artifact PATH   Expected new file or directory below the deployment root.
  -h, --help        Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || die "--root requires a value"
      DEPLOY_ROOT="$2"
      shift 2
      ;;
    --job-id)
      [ "$#" -ge 2 ] || die "--job-id requires a value"
      JOB_ID="$2"
      shift 2
      ;;
    --artifact)
      [ "$#" -ge 2 ] || die "--artifact requires a value"
      ARTIFACT_PATH="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$JOB_ID" in
  ''|*[!A-Za-z0-9._-]*)
    die "job ID may contain only letters, digits, dot, dash, underscore"
    ;;
esac
[ -n "$ARTIFACT_PATH" ] || die "--artifact is required"
[ "$#" -gt 0 ] || die "COMMAND is required after --"

validate_absolute_root "$DEPLOY_ROOT"
mkdir -p "$DEPLOY_ROOT"
DEPLOY_ROOT="$(cd "$DEPLOY_ROOT" && pwd -P)"
initialize_layout "$DEPLOY_ROOT"
require_command "$SYSTEM_PYTHON"
require_command nohup
require_command setsid
PYTHON_BIN="$(command -v "$SYSTEM_PYTHON")"

ARTIFACT_PATH="$("$PYTHON_BIN" - "$DEPLOY_ROOT" "$ARTIFACT_PATH" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
artifact = Path(sys.argv[2])
if not artifact.is_absolute():
    raise SystemExit("--artifact must be an absolute path")
artifact = artifact.resolve(strict=False)
try:
    artifact.relative_to(root)
except ValueError as error:
    raise SystemExit("--artifact must resolve below the deployment root") from error
if artifact == root:
    raise SystemExit("--artifact must not be the deployment root")
print(artifact)
PY
)"
JOBS_ROOT="$DEPLOY_ROOT/outputs/jobs"
JOB_DIRECTORY="$JOBS_ROOT/$JOB_ID"
mkdir -p "$JOBS_ROOT"
if ! mkdir "$JOB_DIRECTORY" 2>/dev/null; then
  die "refusing to reuse detached job directory: $JOB_DIRECTORY"
fi
if [ -e "$ARTIFACT_PATH" ] || [ -L "$ARTIFACT_PATH" ]; then
  die "expected artifact already exists: $ARTIFACT_PATH"
fi
LOG_PATH="$JOB_DIRECTORY/job.log"
LAUNCH_PATH="$JOB_DIRECTORY/launch.json"
EXIT_PATH="$JOB_DIRECTORY/exit.json"
if ! (set -C; : > "$LOG_PATH") 2>/dev/null; then
  die "refusing to overwrite detached job log: $LOG_PATH"
fi

nohup setsid "$SCRIPT_PATH" --_worker \
  "$PYTHON_BIN" "$JOB_DIRECTORY" "$JOB_ID" "$ARTIFACT_PATH" \
  "$LOG_PATH" "$LAUNCH_PATH" -- "$@" \
  </dev/null >>"$LOG_PATH" 2>&1 &
WORKER_PID=$!
WORKER_PGID="$WORKER_PID"

SCRIPT_SHA256="$(sha256_file "$SCRIPT_PATH")"
WORKING_DIRECTORY="$(pwd -P)"
if ! "$PYTHON_BIN" - \
  "$LAUNCH_PATH" "$JOB_ID" "$JOB_DIRECTORY" "$ARTIFACT_PATH" \
  "$LOG_PATH" "$EXIT_PATH" "$WORKER_PID" "$WORKER_PGID" "$$" \
  "$WORKING_DIRECTORY" "$SCRIPT_SHA256" -- "$@" <<'PY'
from __future__ import annotations

import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

arguments = sys.argv[1:]
separator = arguments.index("--")
fixed = arguments[:separator]
command = arguments[separator + 1 :]
if len(fixed) != 11 or not command:
    raise SystemExit("invalid launch receipt arguments")

target = Path(fixed[0])
document = {
    "artifact_path": fixed[3],
    "artifact_present_at_launch": False,
    "command": command,
    "exit_path": fixed[5],
    "job_directory": fixed[2],
    "job_id": fixed[1],
    "launched_at_utc": datetime.now(timezone.utc).isoformat(),
    "launcher_pid": int(fixed[8]),
    "launcher_sha256": fixed[10],
    "log_path": fixed[4],
    "schema_version": "robotactile.detached_job.launch.v1",
    "worker_pgid": int(fixed[7]),
    "worker_pid": int(fixed[6]),
    "working_directory": fixed[9],
}
payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
temporary = target.parent / f".{target.name}.{os.getpid()}.{secrets.token_hex(6)}"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    temporary.unlink(missing_ok=True)
PY
then
  kill -KILL "$WORKER_PID" 2>/dev/null || true
  wait "$WORKER_PID" 2>/dev/null || true
  die "failed to publish detached launch receipt: $LAUNCH_PATH"
fi

"$PYTHON_BIN" - \
  "$JOB_DIRECTORY" "$LAUNCH_PATH" "$EXIT_PATH" "$LOG_PATH" \
  "$WORKER_PID" "$WORKER_PGID" <<'PY'
import json
import sys

print(json.dumps({
    "exit_path": sys.argv[3],
    "job_directory": sys.argv[1],
    "launch_path": sys.argv[2],
    "log_path": sys.argv[4],
    "worker_pgid": int(sys.argv[6]),
    "worker_pid": int(sys.argv[5]),
}, sort_keys=True))
PY
