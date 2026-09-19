#!/usr/bin/env python3
"""Run bounded RoboTactile Slurm smokes and emit an auditable receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], environment: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _cpu_smoke(output_dir: Path, environment: dict[str, str]) -> dict[str, Any]:
    replay_dir = output_dir / "replay"
    command = [
        sys.executable,
        "-m",
        "robotactile_benchmark.cli",
        "smoke-replay",
        "--operator",
        "T1_fixed_source_delay",
        "--severity",
        "3",
        "--output",
        str(replay_dir),
    ]
    run = _run(command, environment)
    report_path = replay_dir / "validation_report.json"
    report: dict[str, Any] | None = None
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    passed = run["returncode"] == 0 and bool(report and report.get("passed"))
    return {
        "passed": passed,
        "run": run,
        "validation_report": report,
        "validation_report_path": str(report_path),
        "validation_report_sha256": _sha256(report_path)
        if report_path.is_file()
        else None,
    }


def _gpu_smoke(environment: dict[str, str]) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    run = _run(command, environment)
    visible_rows = [line for line in run["stdout"].splitlines() if line.strip()]
    return {
        "passed": run["returncode"] == 0 and len(visible_rows) == 1,
        "allocated_gpu_count": len(visible_rows),
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "run": run,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--deploy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve(strict=True)
    deploy_root = args.deploy_root.resolve(strict=True)
    output_dir = args.output.resolve()
    if not (deploy_root == repo_root / "deployment"):
        raise SystemExit("deployment root must be the repo-local deployment directory")
    output_dir.mkdir(parents=True, exist_ok=False)

    environment = dict(os.environ)
    python_path = str(repo_root / "src")
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{python_path}{os.pathsep}{existing_python_path}"
        if existing_python_path
        else python_path
    )
    started = datetime.now(timezone.utc)
    result = (
        _cpu_smoke(output_dir, environment)
        if args.mode == "cpu"
        else _gpu_smoke(environment)
    )
    receipt = {
        "schema_version": 1,
        "evidence_level": "SMOKE",
        "mode": args.mode,
        "passed": result["passed"],
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo_root),
        "deployment_root": str(deploy_root),
        "output_directory": str(output_dir),
        "host": platform.node(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "slurm": {
            key: environment.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_QOS",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_NODELIST",
                "SLURM_CPUS_PER_TASK",
            )
        },
        "claims": {
            "code_smoke": args.mode == "cpu" and bool(result["passed"]),
            "gpu_allocation_smoke": args.mode == "gpu" and bool(result["passed"]),
            "model_inference": False,
            "closed_loop": False,
            "official_result": False,
        },
        "result": result,
    }
    receipt_path = output_dir / "smoke_receipt.json"
    _write_receipt(receipt_path, receipt)
    print(json.dumps({"passed": result["passed"], "receipt": str(receipt_path)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
