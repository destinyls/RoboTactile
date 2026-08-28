"""Fake-Supervisor tests for the optional long-job launcher; no Isaac import."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/live_univtac/run_supervised_job.sh"


@contextmanager
def _short_workspace() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="rt-sv-", dir="/tmp") as directory:
        yield Path(directory).resolve()


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env python3\n{body}", encoding="utf-8")
    path.chmod(0o755)


def _fake_supervisor(base: Path) -> tuple[Path, Path, dict[str, str]]:
    fake_bin = base / "fake-bin"
    fake_bin.mkdir()
    supervisord = fake_bin / "supervisord"
    supervisorctl = fake_bin / "supervisorctl"
    timeout = fake_bin / "timeout"
    _write_executable(
        supervisord,
        """import configparser, os, pathlib, sys
args = sys.argv[1:]
config_path = pathlib.Path(args[args.index('-c') + 1])
config = configparser.RawConfigParser()
config.read(config_path)
pid_path = pathlib.Path(config['supervisord']['pidfile'])
socket_path = pathlib.Path(config['unix_http_server']['file'])
pid_path.write_text(str(os.getpid()), encoding='utf-8')
socket_path.touch()
(config_path.parent / 'fake.state').write_text('RUNNING', encoding='utf-8')
""",
    )
    _write_executable(
        supervisorctl,
        """import configparser, pathlib, sys
args = sys.argv[1:]
index = args.index('-c')
config_path = pathlib.Path(args[index + 1])
action_args = args[index + 2:]
action = action_args[0]
config = configparser.RawConfigParser()
config.read(config_path)
section = next(name for name in config.sections() if name.startswith('program:'))
program = section.split(':', 1)[1]
state_path = config_path.parent / 'fake.state'
state = state_path.read_text(encoding='utf-8').strip()
if action == 'status':
    if state == 'EXITED':
        print(f'{program} EXITED exit status 0')
    else:
        print(f'{program} {state} pid 4242, uptime 0:01:00')
elif action == 'stop':
    state_path.write_text('STOPPED', encoding='utf-8')
    print(f'{program}: stopped')
elif action == 'shutdown':
    state_path.write_text('SHUTDOWN', encoding='utf-8')
    pathlib.Path(config['unix_http_server']['file']).unlink(missing_ok=True)
    pathlib.Path(config['supervisord']['pidfile']).unlink(missing_ok=True)
    print('Shut down')
else:
    raise SystemExit(2)
""",
    )
    _write_executable(
        timeout,
        """import os, sys
args = sys.argv[1:]
if args == ['--version']:
    print('timeout (GNU coreutils) 9.0')
    raise SystemExit(0)
while args and args[0].startswith('--'):
    args.pop(0)
if args and args[0].endswith('s') and args[0][:-1].isdigit():
    args.pop(0)
os.execv(args[0], args)
""",
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["ROBOTACTILE_SYSTEM_PYTHON"] = sys.executable
    return supervisord, supervisorctl, environment


def _invoke(
    action: str,
    *,
    deployment: Path,
    job_id: str,
    supervisord: Path,
    supervisorctl: Path,
    environment: dict[str, str],
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            action,
            "--root",
            str(deployment),
            "--job-id",
            job_id,
            "--supervisord",
            str(supervisord),
            "--supervisorctl",
            str(supervisorctl),
            *(extra or []),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=15,
    )


def test_launch_is_local_no_clobber_foreground_and_hard_timed() -> None:
    with _short_workspace() as base:
        supervisord, supervisorctl, environment = _fake_supervisor(base)
        deployment = base / "deployment"
        artifact = deployment / "artifacts/live-univtac/fake-run"
        result = _invoke(
            "launch",
            deployment=deployment,
            job_id="paper-run-v1",
            supervisord=supervisord,
            supervisorctl=supervisorctl,
            environment=environment,
            extra=[
                "--artifact",
                str(artifact),
                "--timeout-seconds",
                "75",
                "--",
                "/usr/bin/true",
            ],
        )
        assert result.returncode == 0, result.stderr
        start = json.loads(result.stdout)
        job = deployment / "outputs/supervised-jobs/paper-run-v1"
        launch_path = job / "launch.json"
        launch = json.loads(launch_path.read_text(encoding="utf-8"))

        assert start["supervisor_state"] == "RUNNING"
        assert start["artifact_success_claimed"] is False
        assert launch["timeout_seconds"] == 75
        assert launch["timeout_kill_after_seconds"] == 90
        assert launch["artifact_present_at_launch"] is False
        assert launch["artifact_success_claimed"] is False
        assert launch["supervisord_path"] == str(supervisord)
        assert launch["supervisorctl_path"] == str(supervisorctl)
        assert launch["socket_path_bytes"] <= 100
        assert Path(launch["socket_path"]).is_relative_to(deployment)
        for key in (
            "config_path",
            "daemon_log_path",
            "job_directory",
            "launch_log_path",
            "pid_path",
            "program_log_path",
            "program_path",
            "socket_path",
        ):
            assert Path(launch[key]).is_relative_to(deployment)
        config = (job / "supervisord.conf").read_text(encoding="utf-8")
        program = (job / "program.sh").read_text(encoding="utf-8")
        assert "nodaemon=false" in config
        assert "autorestart=false" in config
        assert "stopasgroup=true" in config
        assert 'ROBOTACTILE_HARD_TIMEOUT_SECONDS="75"' in config
        assert "; timeout_seconds=75" in config
        assert "--signal=TERM --kill-after=90s 75s /usr/bin/true" in program
        assert not stat.S_IMODE(launch_path.stat().st_mode) & stat.S_IWUSR

        repeated = _invoke(
            "launch",
            deployment=deployment,
            job_id="paper-run-v1",
            supervisord=supervisord,
            supervisorctl=supervisorctl,
            environment=environment,
            extra=["--artifact", str(artifact), "--", "/usr/bin/true"],
        )
        assert repeated.returncode == 2
        assert "refusing to reuse supervised job directory" in repeated.stderr


def test_exited_zero_and_artifact_presence_never_claim_success() -> None:
    with _short_workspace() as base:
        supervisord, supervisorctl, environment = _fake_supervisor(base)
        deployment = base / "deployment"
        artifact = deployment / "artifacts/live-univtac/fake-exited"
        launched = _invoke(
            "launch",
            deployment=deployment,
            job_id="exited-v1",
            supervisord=supervisord,
            supervisorctl=supervisorctl,
            environment=environment,
            extra=["--artifact", str(artifact), "--", "/usr/bin/true"],
        )
        assert launched.returncode == 0, launched.stderr
        job = deployment / "outputs/supervised-jobs/exited-v1"
        artifact.mkdir(parents=True)
        (artifact / "root_receipt.json").write_text("{}\n", encoding="utf-8")
        (job / "fake.state").write_text("EXITED", encoding="utf-8")

        status = _invoke(
            "status",
            deployment=deployment,
            job_id="exited-v1",
            supervisord=supervisord,
            supervisorctl=supervisorctl,
            environment=environment,
        )
        assert status.returncode == 0, status.stderr
        receipt = json.loads(status.stdout)
        assert receipt["supervisor_state"] == "EXITED"
        assert receipt["supervisor_reported_exit_code"] == 0
        assert receipt["artifact_present"] is True
        assert receipt["artifact_root_receipt_present"] is True
        assert receipt["artifact_success_claimed"] is False
        assert receipt["artifact_validation_claimed"] is False
        assert receipt["supervisor_state_is_artifact_success"] is False

        stopped = _invoke(
            "stop",
            deployment=deployment,
            job_id="exited-v1",
            supervisord=supervisord,
            supervisorctl=supervisorctl,
            environment=environment,
        )
        assert json.loads(stopped.stdout)["supervisor_state"] == "STOPPED"
        shutdown = _invoke(
            "shutdown",
            deployment=deployment,
            job_id="exited-v1",
            supervisord=supervisord,
            supervisorctl=supervisorctl,
            environment=environment,
        )
        assert shutdown.returncode == 0
        assert json.loads(shutdown.stdout)["artifact_success_claimed"] is False
        assert not (job / "supervisord.pid").exists()
        control_receipts = list((job / "control").glob("*.json"))
        assert len(control_receipts) == 3


def test_rejects_short_timeout_existing_artifact_and_long_socket() -> None:
    with _short_workspace() as base:
        supervisord, supervisorctl, environment = _fake_supervisor(base)
        deployment = base / "deployment"
        artifact = deployment / "artifacts/live-univtac/preexisting"
        too_short = _invoke(
            "launch",
            deployment=deployment,
            job_id="short-v1",
            supervisord=supervisord,
            supervisorctl=supervisorctl,
            environment=environment,
            extra=[
                "--artifact",
                str(artifact),
                "--timeout-seconds",
                "59",
                "--",
                "/usr/bin/true",
            ],
        )
        assert too_short.returncode == 2
        assert "at least 60" in too_short.stderr

        artifact.mkdir(parents=True)
        existing = _invoke(
            "launch",
            deployment=deployment,
            job_id="artifact-v1",
            supervisord=supervisord,
            supervisorctl=supervisorctl,
            environment=environment,
            extra=["--artifact", str(artifact), "--", "/usr/bin/true"],
        )
        assert existing.returncode == 2
        assert "expected artifact already exists" in existing.stderr

        long_deployment = base / ("x" * 75) / "deployment"
        long_artifact = long_deployment / "artifacts/live-univtac/result"
        long_socket = _invoke(
            "launch",
            deployment=long_deployment,
            job_id="socket-v1",
            supervisord=supervisord,
            supervisorctl=supervisorctl,
            environment=environment,
            extra=["--artifact", str(long_artifact), "--", "/usr/bin/true"],
        )
        assert long_socket.returncode != 0
        assert "socket path exceeds 100 bytes" in long_socket.stderr


def test_script_has_optional_explicit_supervisor_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--supervisord" in text
    assert "--supervisorctl" in text
    assert "--timeout-seconds" in text
    assert 'TIMEOUT_SECONDS="2400"' in text
    assert '"--signal=TERM", "--kill-after=90s"' in text
    assert '"artifact_success_claimed": False' in text
    assert "supervisor socket path exceeds 100 bytes" in text
    assert "/opt/" not in text
    assert "isaac" not in text.lower()
