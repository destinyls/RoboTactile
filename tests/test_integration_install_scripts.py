"""No-network help and dry-run behavior for generic source installers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("bash", str(ROOT / "integrations" / script), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )


def test_wrappers_support_help_without_network_or_writes() -> None:
    for script in (
        "install_act_runtime.sh",
        "install_dream_tac.sh",
        "install_n0_twam.sh",
        "install_n0_vtla.sh",
        "install_univtac.sh",
    ):
        completed = _run(script, "--help")
        assert completed.returncode == 0
        assert "--dry-run" in completed.stdout


def test_default_dry_run_resolves_lock_and_repo_contained_destination() -> None:
    deployment_existed = (ROOT / "deployment").exists()
    completed = _run("install_univtac.sh", "--dry-run")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["commit_sha"] == "05bcd3edb92237107efa40105292a24f1a9fd761"
    assert payload["destination"] == str(ROOT / "deployment/sources/UniVTAC")
    assert payload["writes_performed"] is False
    assert (ROOT / "deployment").exists() is deployment_existed
