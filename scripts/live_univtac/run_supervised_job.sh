#!/usr/bin/env bash
# Manage one deployment-local supervisord daemon and foreground live program.
set -Eeuo pipefail
umask 022
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
usage() {
  cat <<'EOF'
Usage:
  run_supervised_job.sh launch [OPTIONS] -- COMMAND [ARG ...]
  run_supervised_job.sh {status|stop|shutdown} [OPTIONS]

Required options:
  --root PATH             Absolute RoboTactile deployment root.
  --job-id ID             Unique letters/digits/dot/dash/underscore identity.
  --supervisord PATH      Explicit absolute supervisord executable.
  --supervisorctl PATH    Explicit absolute supervisorctl executable.

Launch-only options:
  --artifact PATH         Expected new artifact below the deployment root.
  --timeout-seconds N     Overall startup/reset/trial timeout (default: 2400).

The launch action creates a no-clobber outputs/supervised-jobs/ID directory.
status, stop, and shutdown publish separate control receipts. Supervisor state,
including EXITED with code 0, is never treated as artifact validation.
EOF
}
publish_control_receipt() {
  local target="$1" action="$2" daemon_rc="$3" controller_rc="$4" output="$5"
  "$PYTHON_BIN" - "$target" "$action" "$JOB_ID" "$ARTIFACT_PATH" \
    "$PROGRAM_NAME" "$PID_PATH" "$TIMEOUT_SECONDS" "$daemon_rc" \
    "$controller_rc" "$output" <<'PY'
from __future__ import annotations
import json, os, re, secrets, sys
from datetime import datetime, timezone
from pathlib import Path
target = Path(sys.argv[1])
action, job_id = sys.argv[2:4]
artifact = Path(sys.argv[4])
program, pid_path = sys.argv[5:7]
timeout_seconds = int(sys.argv[7])
daemon_rc = None if not sys.argv[8] else int(sys.argv[8])
controller_rc, output = int(sys.argv[9]), sys.argv[10]
states = {"BACKOFF", "EXITED", "FATAL", "RUNNING", "STARTING", "STOPPED"}
state = next((token.rstrip(":,").upper() for token in output.split()
              if token.rstrip(":,").upper() in states), None)
match = re.search(r"exit status\s+(-?\d+)", output, flags=re.IGNORECASE)
reported_exit_code = None if match is None else int(match.group(1))
daemon_pid = None
if Path(pid_path).is_file():
    value = Path(pid_path).read_text(encoding="utf-8").strip()
    daemon_pid = int(value) if value.isdigit() else None
document = {
    "action": action, "artifact_path": str(artifact),
    "artifact_present": artifact.exists(),
    "artifact_root_receipt_present": (
        artifact.is_dir() and (artifact / "root_receipt.json").is_file()
    ),
    "artifact_success_claimed": False, "artifact_validation_claimed": False,
    "controller_output": output, "controller_return_code": controller_rc,
    "daemon_pid": daemon_pid, "job_id": job_id, "program_name": program,
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "schema_version": "robotactile.supervised_job.control.v1",
    "supervisor_reported_exit_code": reported_exit_code, "supervisor_state": state,
    "supervisor_state_is_artifact_success": False,
    "supervisord_return_code": daemon_rc, "timeout_kill_after_seconds": 90,
    "timeout_seconds": timeout_seconds,
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
print(payload.decode("utf-8"), end="")
PY
}
case "${1:-}" in
  launch|status|stop|shutdown) ACTION="$1"; shift ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; die "first argument must be launch, status, stop, or shutdown" ;;
esac
DEPLOY_ROOT="$(default_deployment_root)"
JOB_ID=""
ARTIFACT_INPUT=""
SUPERVISORD_INPUT=""
SUPERVISORCTL_INPUT=""
TIMEOUT_SECONDS="2400"
TIMEOUT_EXPLICIT="false"
SYSTEM_PYTHON="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) [ "$#" -ge 2 ] || die "--root requires a value"; DEPLOY_ROOT="$2"; shift 2 ;;
    --job-id) [ "$#" -ge 2 ] || die "--job-id requires a value"; JOB_ID="$2"; shift 2 ;;
    --artifact) [ "$#" -ge 2 ] || die "--artifact requires a value"; ARTIFACT_INPUT="$2"; shift 2 ;;
    --supervisord) [ "$#" -ge 2 ] || die "--supervisord requires a value"; SUPERVISORD_INPUT="$2"; shift 2 ;;
    --supervisorctl) [ "$#" -ge 2 ] || die "--supervisorctl requires a value"; SUPERVISORCTL_INPUT="$2"; shift 2 ;;
    --timeout-seconds) [ "$#" -ge 2 ] || die "--timeout-seconds requires a value"; TIMEOUT_SECONDS="$2"; TIMEOUT_EXPLICIT="true"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
case "$JOB_ID" in
  ''|*[!A-Za-z0-9._-]*) die "invalid job ID" ;;
esac
case "$TIMEOUT_SECONDS" in
  ''|*[!0-9]*) die "--timeout-seconds must be an integer" ;;
esac
[ "$TIMEOUT_SECONDS" -ge 60 ] || die "--timeout-seconds must be at least 60"
[ -n "$SUPERVISORD_INPUT" ] || die "--supervisord is required"
[ -n "$SUPERVISORCTL_INPUT" ] || die "--supervisorctl is required"
if [ "$ACTION" = launch ]; then
  [ -n "$ARTIFACT_INPUT" ] || die "launch requires --artifact"
  [ "$#" -gt 0 ] || die "launch requires COMMAND after --"
else
  [ -z "$ARTIFACT_INPUT" ] || die "--artifact is launch-only"
  [ "$TIMEOUT_EXPLICIT" = false ] || die "--timeout-seconds is launch-only"
  [ "$#" -eq 0 ] || die "$ACTION does not accept COMMAND"
fi
validate_absolute_root "$DEPLOY_ROOT"
mkdir -p "$DEPLOY_ROOT"
DEPLOY_ROOT="$(cd "$DEPLOY_ROOT" && pwd -P)"
initialize_layout "$DEPLOY_ROOT"
require_command "$SYSTEM_PYTHON"
PYTHON_BIN="$(command -v "$SYSTEM_PYTHON")"
mapfile -t RESOLVED_PATHS < <("$PYTHON_BIN" - "$SUPERVISORD_INPUT" "$SUPERVISORCTL_INPUT" <<'PY'
import os
import sys
from pathlib import Path

for raw in sys.argv[1:]:
    path = Path(raw)
    if not path.is_absolute():
        raise SystemExit("supervisor executable paths must be absolute")
    resolved = Path(os.path.abspath(path))
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise SystemExit(f"supervisor executable is not executable: {resolved}")
    if "\n" in str(resolved):
        raise SystemExit("supervisor executable path contains a newline")
    print(resolved)
PY
)
[ "${#RESOLVED_PATHS[@]}" -eq 2 ] || die "failed to resolve supervisor executables"
SUPERVISORD_PATH="${RESOLVED_PATHS[0]}"
SUPERVISORCTL_PATH="${RESOLVED_PATHS[1]}"
JOB_HASH="$("$PYTHON_BIN" - "$JOB_ID" <<'PY'
import hashlib
import sys
print(hashlib.sha256(sys.argv[1].encode("utf-8")).hexdigest()[:20])
PY
)"
PROGRAM_NAME="robotactile_$JOB_HASH"
JOBS_ROOT="$DEPLOY_ROOT/outputs/supervised-jobs"
JOB_DIRECTORY="$JOBS_ROOT/$JOB_ID"
SOCKET_DIRECTORY="$DEPLOY_ROOT/runtime/sv"
SOCKET_PATH="$SOCKET_DIRECTORY/$JOB_HASH.sock"
"$PYTHON_BIN" - "$SOCKET_PATH" <<'PY'
import os
import sys
length = len(os.fsencode(sys.argv[1]))
if length > 100:
    raise SystemExit(f"supervisor socket path exceeds 100 bytes: {length}")
PY
CONFIG_PATH="$JOB_DIRECTORY/supervisord.conf"
PROGRAM_PATH="$JOB_DIRECTORY/program.sh"
LAUNCH_PATH="$JOB_DIRECTORY/launch.json"
START_RESULT_PATH="$JOB_DIRECTORY/start_result.json"
PID_PATH="$JOB_DIRECTORY/supervisord.pid"
DAEMON_LOG_PATH="$JOB_DIRECTORY/supervisord.log"
PROGRAM_LOG_PATH="$JOB_DIRECTORY/program.log"
LAUNCH_LOG_PATH="$JOB_DIRECTORY/supervisord-launch.log"
if [ "$ACTION" = launch ]; then
  mkdir -p "$JOBS_ROOT" "$SOCKET_DIRECTORY"
  if ! mkdir "$JOB_DIRECTORY" 2>/dev/null; then
    die "refusing to reuse supervised job directory: $JOB_DIRECTORY"
  fi
  ARTIFACT_PATH="$("$PYTHON_BIN" - "$DEPLOY_ROOT" "$ARTIFACT_INPUT" <<'PY'
import sys
from pathlib import Path
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
  if [ -e "$ARTIFACT_PATH" ] || [ -L "$ARTIFACT_PATH" ]; then
    die "expected artifact already exists: $ARTIFACT_PATH"
  fi
  if [ -e "$SOCKET_PATH" ] || [ -L "$SOCKET_PATH" ]; then
    die "supervisor socket already exists: $SOCKET_PATH"
  fi
  for argument in "$@"; do
    case "$argument" in *$'\n'*) die "COMMAND arguments must not contain newlines" ;; esac
  done
  SUPERVISORD_SHA256="$(sha256_file "$SUPERVISORD_PATH")"
  SUPERVISORCTL_SHA256="$(sha256_file "$SUPERVISORCTL_PATH")"
  require_command timeout
  TIMEOUT_PATH="$(command -v timeout)"
  TIMEOUT_VERSION="$("$TIMEOUT_PATH" --version 2>&1)" || die "failed to query timeout version"
  case "$TIMEOUT_VERSION" in *"GNU coreutils"*) ;; *) die "GNU timeout is required" ;; esac
  TIMEOUT_SHA256="$(sha256_file "$TIMEOUT_PATH")"
  WORKING_DIRECTORY="$(pwd -P)"
  "$PYTHON_BIN" - "$PROGRAM_PATH" "$CONFIG_PATH" "$LAUNCH_PATH" "$JOB_ID" \
    "$ARTIFACT_PATH" "$PROGRAM_NAME" "$SOCKET_PATH" "$PID_PATH" \
    "$DAEMON_LOG_PATH" "$PROGRAM_LOG_PATH" "$LAUNCH_LOG_PATH" \
    "$WORKING_DIRECTORY" "$SUPERVISORD_PATH" "$SUPERVISORCTL_PATH" \
    "$TIMEOUT_PATH" "$SUPERVISORD_SHA256" "$SUPERVISORCTL_SHA256" \
    "$TIMEOUT_SHA256" "$TIMEOUT_SECONDS" -- "$@" <<'PY'
from __future__ import annotations
import hashlib, json, os, secrets, shlex, sys
from datetime import datetime, timezone
from pathlib import Path
args = sys.argv[1:]
separator = args.index("--")
fixed, command = args[:separator], args[separator + 1:]
if len(fixed) != 19 or not command:
    raise SystemExit("invalid supervised launch arguments")
(program_path, config_path, launch_path) = map(Path, fixed[:3])
(job_id, artifact, program, socket, pid, daemon_log, program_log,
 launch_log, working_directory, supervisord, supervisorctl, timeout_path,
 supervisord_sha, supervisorctl_sha, timeout_sha, timeout_seconds) = fixed[3:]
timeout_seconds_int = int(timeout_seconds)
def publish(path: Path, payload: bytes, mode: int) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
wrapped = [
    timeout_path, "--signal=TERM", "--kill-after=90s",
    f"{timeout_seconds_int}s", *command,
]
program_payload = (
    "#!/usr/bin/env bash\nset -Eeuo pipefail\nexec "
    + shlex.join(wrapped) + "\n"
).encode("utf-8")
publish(program_path, program_payload, 0o555)
escape = lambda value: value.replace("%", "%%")
config_payload = f"""[unix_http_server]
file={escape(socket)}
chmod=0700

[supervisord]
logfile={escape(daemon_log)}
logfile_maxbytes=0
pidfile={escape(pid)}
childlogdir={escape(str(config_path.parent))}
nodaemon=false
nocleanup=true

[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix://{escape(socket)}

[program:{program}]
command={escape(shlex.join([str(program_path)]))}
directory={escape(working_directory)}
autostart=true
autorestart=false
startsecs=1
startretries=0
stopsignal=TERM
stopwaitsecs=90
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={escape(program_log)}
stdout_logfile_maxbytes=0
environment=ROBOTACTILE_HARD_TIMEOUT_SECONDS="{timeout_seconds_int}",ROBOTACTILE_TIMEOUT_KILL_AFTER_SECONDS="90"
; timeout_path={escape(timeout_path)}
; timeout_seconds={timeout_seconds_int}
; timeout_kill_after_seconds=90
""".encode("utf-8")
publish(config_path, config_payload, 0o444)
document = {
    "artifact_path": artifact, "artifact_present_at_launch": False,
    "artifact_success_claimed": False, "command": command,
    "config_path": str(config_path),
    "config_sha256": hashlib.sha256(config_payload).hexdigest(),
    "daemon_log_path": daemon_log, "job_directory": str(config_path.parent),
    "job_id": job_id, "launch_log_path": launch_log,
    "launched_at_utc": datetime.now(timezone.utc).isoformat(),
    "pid_path": pid, "program_log_path": program_log,
    "program_name": program, "program_path": str(program_path),
    "program_sha256": hashlib.sha256(program_payload).hexdigest(),
    "schema_version": "robotactile.supervised_job.launch.v1",
    "socket_path": socket, "socket_path_bytes": len(os.fsencode(socket)),
    "supervisorctl_path": supervisorctl, "supervisorctl_sha256": supervisorctl_sha,
    "supervisord_path": supervisord, "supervisord_sha256": supervisord_sha,
    "timeout_kill_after_seconds": 90, "timeout_path": timeout_path,
    "timeout_seconds": timeout_seconds_int, "timeout_sha256": timeout_sha,
    "working_directory": working_directory,
}
launch_payload = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
publish(launch_path, launch_payload, 0o444)
PY
  if ! (set -C; : > "$LAUNCH_LOG_PATH") 2>/dev/null; then
    die "refusing to overwrite supervisor launch log"
  fi
  set +e
  "$SUPERVISORD_PATH" -c "$CONFIG_PATH" >>"$LAUNCH_LOG_PATH" 2>&1
  SUPERVISORD_RC=$?
  set -e
  CONTROLLER_RC=1
  CONTROLLER_OUTPUT=""
  attempt=0
  while [ "$SUPERVISORD_RC" -eq 0 ] && [ "$attempt" -lt 100 ]; do
    attempt=$((attempt + 1))
    set +e
    CONTROLLER_OUTPUT="$("$TIMEOUT_PATH" --signal=TERM --kill-after=2s 2s \
      "$SUPERVISORCTL_PATH" -c "$CONFIG_PATH" status "$PROGRAM_NAME" 2>&1)"
    CONTROLLER_RC=$?
    set -e
    [ "$CONTROLLER_RC" -eq 0 ] && break
    sleep 0.1
  done
  publish_control_receipt "$START_RESULT_PATH" start "$SUPERVISORD_RC" \
    "$CONTROLLER_RC" "$CONTROLLER_OUTPUT"
  [ "$SUPERVISORD_RC" -eq 0 ] || exit "$SUPERVISORD_RC"
  exit "$CONTROLLER_RC"
fi
[ -d "$JOB_DIRECTORY" ] || die "supervised job is absent: $JOB_DIRECTORY"
LOADED_FIELDS="$("$PYTHON_BIN" - "$LAUNCH_PATH" "$JOB_ID" \
  "$SUPERVISORD_PATH" "$SUPERVISORCTL_PATH" <<'PY'
import json
import sys
from pathlib import Path
document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if document.get("schema_version") != "robotactile.supervised_job.launch.v1":
    raise SystemExit("invalid supervised launch receipt")
if document.get("job_id") != sys.argv[2]:
    raise SystemExit("launch receipt job mismatch")
if document.get("supervisord_path") != sys.argv[3]:
    raise SystemExit("--supervisord does not match launch receipt")
if document.get("supervisorctl_path") != sys.argv[4]:
    raise SystemExit("--supervisorctl does not match launch receipt")
for key in ("artifact_path", "program_name", "pid_path", "config_path",
            "config_sha256", "program_path", "program_sha256",
            "supervisord_sha256", "supervisorctl_sha256", "timeout_seconds",
            "timeout_path", "timeout_sha256"):
    print(document[key])
PY
)" || die "failed to load supervised launch receipt"
mapfile -t LOADED <<<"$LOADED_FIELDS"
[ "${#LOADED[@]}" -eq 12 ] || die "invalid supervised launch receipt fields"
ARTIFACT_PATH="${LOADED[0]}"
PROGRAM_NAME="${LOADED[1]}"
PID_PATH="${LOADED[2]}"
CONFIG_PATH="${LOADED[3]}"
[ "$(sha256_file "$CONFIG_PATH")" = "${LOADED[4]}" ] || die "supervisor config hash mismatch"
PROGRAM_PATH="${LOADED[5]}"
[ "$(sha256_file "$PROGRAM_PATH")" = "${LOADED[6]}" ] || die "supervised program hash mismatch"
[ "$(sha256_file "$SUPERVISORD_PATH")" = "${LOADED[7]}" ] || die "supervisord hash mismatch"
[ "$(sha256_file "$SUPERVISORCTL_PATH")" = "${LOADED[8]}" ] || die "supervisorctl hash mismatch"
TIMEOUT_SECONDS="${LOADED[9]}"
TIMEOUT_PATH="${LOADED[10]}"
[ -x "$TIMEOUT_PATH" ] || die "recorded GNU timeout is not executable"
[ "$(sha256_file "$TIMEOUT_PATH")" = "${LOADED[11]}" ] || die "GNU timeout hash mismatch"
CONTROL_DIRECTORY="$JOB_DIRECTORY/control"
mkdir -p "$CONTROL_DIRECTORY"
RECEIPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CONTROL_RECEIPT="$CONTROL_DIRECTORY/$ACTION-$RECEIPT_ID.json"
set +e
if [ "$ACTION" = shutdown ]; then
  CONTROLLER_OUTPUT="$("$TIMEOUT_PATH" --signal=TERM --kill-after=5s 30s \
    "$SUPERVISORCTL_PATH" -c "$CONFIG_PATH" shutdown 2>&1)"
else
  CONTROLLER_OUTPUT="$("$TIMEOUT_PATH" --signal=TERM --kill-after=5s 30s \
    "$SUPERVISORCTL_PATH" -c "$CONFIG_PATH" "$ACTION" "$PROGRAM_NAME" 2>&1)"
fi
CONTROLLER_RC=$?
set -e
publish_control_receipt "$CONTROL_RECEIPT" "$ACTION" "" "$CONTROLLER_RC" \
  "$CONTROLLER_OUTPUT"
exit "$CONTROLLER_RC"
