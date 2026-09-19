"""Isolate five-model acceptance checks in their selected Python runtimes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = {"act", "dream_tac", "ftp1_policy", "n0_twam", "n0_vtla"}
PATHS = (
    "binding_path",
    "code",
    "package",
    "live_artifact",
    "source_root",
    "runtime_python",
)


def worker_environment(row: dict[str, object]) -> dict[str, str]:
    """Mirror campaign policy environment without importing GPU-side packages."""
    env = dict(os.environ)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    env["PYTHONPATH"] = os.pathsep.join((str(row["package"]), str(row["code"])))
    if row["model"] == "act":
        return env
    binding = json.loads(Path(str(row["binding_path"])).read_text())
    if binding["model"] != row["model"]:
        raise ValueError("worker binding model mismatch")
    # Match scripts.retrained_evaluation.campaign.build_process_environments.
    # Preserve the caller's GPU selection rather than starting any campaign.
    if binding.get("shared_pythonpath"):
        env["PYTHONPATH"] += os.pathsep + binding["shared_pythonpath"]
    root = Path(binding["deployment_root"])
    if row["model"] == "ftp1_policy":
        env["OPENPI_DATA_HOME"] = str(root / "artifacts/openpi-data/ftp1-policy")
    elif row["model"] == "n0_vtla":
        env["OPENPI_DATA_HOME"] = str(
            root.parent / "deployment/artifacts/models/n0_vtla/data_cache"
        )
    elif row["model"] == "dream_tac":
        env["CUDNN_HOME"] = str(
            root / "runtime/ftp1-policy/lib/python3.11/site-packages/nvidia/cudnn"
        )
        env["CUDA_HOME"] = str(root / "runtime/cuda-toolkit-12.8")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(Path(env["CUDNN_HOME"]) / "lib"), env.get("LD_LIBRARY_PATH", ""))
        )
    return env


def load_manifest(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text())
    rows = value["models"]
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("manifest.models must contain exactly five rows")
    if {row["model"] for row in rows} != MODELS:
        raise ValueError("manifest must include each of the five model IDs once")
    for row in rows:
        for name in PATHS:
            if not isinstance(row.get(name), str) or not Path(row[name]).is_absolute():
                raise ValueError(f"{row['model']}.{name} must be an absolute path")
        for name in ("request_path", "server_receipt"):
            if row.get(name) is not None and not Path(row[name]).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new output directory"
    )
    parser.add_argument(
        "--mode",
        choices=("audit", "infer"),
        default="audit",
        help="infer requires already running exclusive policy services; never starts them",
    )
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    rows = load_manifest(args.manifest)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    for row in rows:
        model = row["model"]
        target = output / f"{model}.json"
        row_path = output / f"{model}.input.json"
        row_path.write_text(json.dumps(row, indent=2) + "\n")
        command = [
            str(row["runtime_python"]),
            "-m",
            "scripts.model_acceptance.probe",
            "--row",
            str(row_path),
            "--output",
            str(target),
            "--mode",
            args.mode,
        ]
        error = None
        returncode = None
        try:
            with (output / f"{model}.log").open("x") as log:
                env = worker_environment(row)
                result = subprocess.run(
                    command,
                    cwd=str(row["code"]),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
            returncode = result.returncode
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if target.exists():
            receipt = json.loads(target.read_text())
        else:
            receipt = {
                "model": model,
                "passed": False,
                "error": error or "worker exited without a receipt",
            }
            target.write_text(json.dumps(receipt, indent=2) + "\n")
        summary.append(
            {
                "model": model,
                "passed": returncode == 0 and receipt.get("passed") is True,
                "returncode": returncode,
                "error": error,
                "receipt": str(target),
                "log": str(output / f"{model}.log"),
            }
        )
    passed = all(item["passed"] for item in summary)
    (output / "summary.json").write_text(
        json.dumps({"mode": args.mode, "passed": passed, "models": summary}, indent=2)
        + "\n"
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
