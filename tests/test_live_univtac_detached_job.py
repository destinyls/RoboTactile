"""Lifecycle tests for the detached live-job launcher (never starts Isaac)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/live_univtac/run_detached_job.sh"


def _wait_for(path: Path, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.02)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _launcher_environment(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    setsid = fake_bin / "setsid"
    setsid.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "os.setsid()\n"
        "os.execvp(sys.argv[1], sys.argv[1:])\n",
        encoding="utf-8",
    )
    setsid.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["ROBOTACTILE_SYSTEM_PYTHON"] = sys.executable
    return environment


def _launch(
    tmp_path: Path,
    *,
    job_id: str,
    artifact: Path,
    command: list[str],
) -> tuple[subprocess.CompletedProcess[str], Path]:
    deployment = tmp_path / "deployment"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--root",
            str(deployment),
            "--job-id",
            job_id,
            "--artifact",
            str(artifact),
            "--",
            *command,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_launcher_environment(tmp_path),
        timeout=10,
    )
    return result, deployment / "outputs/jobs" / job_id


def test_success_receipt_is_hash_bound_and_job_directory_is_no_clobber(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "deployment/artifacts/live-univtac/synthetic-success"
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            "p=Path(sys.argv[1]); p.mkdir(parents=True); "
            "(p/'root_receipt.json').write_text('{}\\n')"
        ),
        str(artifact),
    ]
    launched, job = _launch(
        tmp_path,
        job_id="success-v1",
        artifact=artifact,
        command=command,
    )

    assert launched.returncode == 0, launched.stderr
    summary = json.loads(launched.stdout)
    assert summary["worker_pid"] == summary["worker_pgid"]
    launch_path = job / "launch.json"
    exit_path = job / "exit.json"
    _wait_for(exit_path)
    launch = _load(launch_path)
    finished = _load(exit_path)

    assert launch["schema_version"] == "robotactile.detached_job.launch.v1"
    assert launch["artifact_present_at_launch"] is False
    assert launch["command"] == command
    assert launch["worker_pid"] == launch["worker_pgid"]
    assert not stat.S_IMODE(launch_path.stat().st_mode) & stat.S_IWUSR
    assert finished["schema_version"] == "robotactile.detached_job.exit.v1"
    assert finished["return_code"] == 0
    assert finished["signal"] is None
    assert finished["signal_number"] is None
    assert datetime.fromisoformat(str(finished["started_at_utc"])) <= (
        datetime.fromisoformat(str(finished["ended_at_utc"]))
    )
    assert float(finished["duration_seconds"]) >= 0
    assert finished["artifact_present"] is True
    assert finished["artifact_kind"] == "directory"
    assert finished["artifact_root_receipt_present"] is True
    log = job / "job.log"
    assert finished["log_sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()
    assert not stat.S_IMODE(exit_path.stat().st_mode) & stat.S_IWUSR

    launch_bytes = launch_path.read_bytes()
    repeated, _ = _launch(
        tmp_path,
        job_id="success-v1",
        artifact=artifact,
        command=command,
    )
    assert repeated.returncode == 2
    assert "refusing to reuse detached job directory" in repeated.stderr
    assert launch_path.read_bytes() == launch_bytes


def test_nonzero_exit_is_recorded_without_claiming_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "deployment/artifacts/live-univtac/never-created"
    launched, job = _launch(
        tmp_path,
        job_id="normal-failure-v1",
        artifact=artifact,
        command=[sys.executable, "-c", "raise SystemExit(7)"],
    )

    assert launched.returncode == 0, launched.stderr
    exit_path = job / "exit.json"
    _wait_for(exit_path)
    finished = _load(exit_path)
    assert finished["return_code"] == 7
    assert finished["signal"] is None
    assert finished["signal_number"] is None
    assert finished["artifact_present"] is False
    assert finished["artifact_kind"] == "missing"


def test_term_is_forwarded_and_real_signal_is_recorded(tmp_path: Path) -> None:
    artifact = tmp_path / "deployment/artifacts/live-univtac/term-artifact"
    launched, job = _launch(
        tmp_path,
        job_id="term-v1",
        artifact=artifact,
        command=[
            sys.executable,
            "-c",
            "import time; print('CHILD_READY', flush=True); time.sleep(60)",
        ],
    )

    assert launched.returncode == 0, launched.stderr
    launch = _load(job / "launch.json")
    log = job / "job.log"
    deadline = time.monotonic() + 10
    while "CHILD_READY" not in log.read_text(encoding="utf-8"):
        if time.monotonic() >= deadline:
            raise AssertionError("child never became ready")
        time.sleep(0.02)
    os.kill(int(launch["worker_pid"]), signal.SIGTERM)

    exit_path = job / "exit.json"
    _wait_for(exit_path)
    finished = _load(exit_path)
    assert finished["return_code"] is None
    assert finished["signal"] == "SIGTERM"
    assert finished["signal_number"] == signal.SIGTERM
    assert finished["forwarded_signal"] == "SIGTERM"
    assert finished["artifact_present"] is False


def test_sigkill_leaves_no_exit_receipt_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "deployment/artifacts/live-univtac/killed-artifact"
    child_pid_path = tmp_path / "child.pid"
    launched, job = _launch(
        tmp_path,
        job_id="sigkill-v1",
        artifact=artifact,
        command=[
            sys.executable,
            "-c",
            (
                "import os, pathlib, sys, time; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                "time.sleep(60)"
            ),
            str(child_pid_path),
        ],
    )

    assert launched.returncode == 0, launched.stderr
    launch = _load(job / "launch.json")
    _wait_for(child_pid_path)
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    os.kill(int(launch["worker_pid"]), signal.SIGKILL)
    time.sleep(0.3)
    assert not (job / "exit.json").exists()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child_pid, signal.SIGKILL)


def test_script_contract_does_not_depend_on_uv_or_isaac() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "nohup setsid" in text
    assert "os.killpg(child.pid, signum)" in text
    assert "os.link(temporary, target)" in text
    assert "os.link(temporary, path)" in text
    assert "signal.SIGTERM" in text
    assert "signal.SIGINT" in text
    assert "uv run" not in text
    assert "isaac" not in text.lower()
