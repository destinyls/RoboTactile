"""Run one seed of official N0 Clean/optical-14 on all UniVTAC tasks.

The preparation phase freezes new requests from a previous, source-bound Clean
template. The execution phase starts one official N0 server per task and runs
one paired Isaac session. Neither phase reuses a scored episode artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.execution.loading import load_live_univtac_request
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.fault_timing import (
    EARLY_RANDOM_ONSET_MODE,
    FIXED_FAULT_ONSET_MODE,
)
from robotactile_benchmark.n0_fault_campaign.generation import (
    N0FaultCampaignGenerationSpec,
    generate_n0_fault_campaign_bundle,
)

REGISTRY = "optical_marker_extreme_v1"
TASKS = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def _reference_task(root_a: Path, root_b: Path, task: str) -> Path:
    found = [root / task for root in (root_a, root_b) if (root / task).is_dir()]
    if len(found) != 1:
        raise ValueError(f"expected exactly one reference task for {task}: {found}")
    return found[0]


def _clean_template(reference: Path) -> Path:
    root = reference / "fault_campaign"
    manifest = json.loads((root / "campaign_manifest.json").read_text())
    clean = [cell for cell in manifest["cells"] if cell["condition"] == "clean"]
    if len(clean) != 1 or clean[0]["disposition"] != "live_request":
        raise ValueError(f"reference has no unique Clean request: {reference}")
    request_relpath = clean[0]["request_relpath"]
    if not isinstance(request_relpath, str):
        raise ValueError("reference Clean request path is not a string")
    return root / request_relpath


def prepare(
    *,
    repo: Path,
    output: Path,
    reference_a: Path,
    reference_b: Path,
    tasks: tuple[str, ...],
    seed: int,
    integration_label: str,
    port: int,
    capture_profile: str,
    isaac_python: Path,
    n0_python: Path | None = None,
    n0_source: Path | None = None,
    gpu_id: int = 0,
    severity_registry: str = REGISTRY,
    operator_ids: tuple[str, ...] = tuple(sorted(CORE_OPERATOR_IDS)),
    fault_onset_mode: str = EARLY_RANDOM_ONSET_MODE,
) -> None:
    if output.exists():
        raise FileExistsError(f"campaign already exists: {output}")
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError("tasks must be unique canonical UniVTAC IDs")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 1 <= port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if type(gpu_id) is not int or gpu_id < 0:
        raise ValueError("gpu_id must be a non-negative integer")
    if capture_profile not in {"paper_full_v1", "preview_v1", "metrics_only_v1"}:
        raise ValueError("unsupported capture profile")
    if (
        not operator_ids
        or len(set(operator_ids)) != len(operator_ids)
        or set(operator_ids) - CORE_OPERATOR_IDS
    ):
        raise ValueError("operator_ids must be unique canonical operators")
    if fault_onset_mode not in {
        EARLY_RANDOM_ONSET_MODE,
        FIXED_FAULT_ONSET_MODE,
    }:
        raise ValueError("unsupported fault onset mode")
    root = repo / "deployment-sm120"
    n0_python = n0_python or root / "runtime/n0-twam/bin/python"
    n0_source = n0_source or root / "sources/N0-TWAM"
    for required in (isaac_python, n0_python, n0_source):
        if not required.exists():
            raise FileNotFoundError(required)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "plan.json",
        {
            "schema": "robotactile-official-n0-fault-seed-v2",
            "repo": str(repo),
            "tasks": list(tasks),
            "seed": seed,
            "severity_registry": severity_registry,
            "operator_ids": list(operator_ids),
            "severity_level": 5,
            "fault_onset_mode": fault_onset_mode,
            "fault_start_index": 0,
            "fault_onset_max_index": 8,
            "capture_profile": capture_profile,
            "integration_label": integration_label,
            "port": port,
            "isaac_python": str(isaac_python),
            "n0_python": str(n0_python),
            "n0_source": str(n0_source),
            "gpu_id": gpu_id,
            "reference_roots": [str(reference_a), str(reference_b)],
            "prepared_unix": time.time(),
        },
    )
    for task in tasks:
        reference = _reference_task(reference_a, reference_b, task)
        task_root = output / task
        template_path = _clean_template(reference)
        template = load_live_univtac_request(template_path)
        config = (
            root
            / "artifacts/models/n0_twam/configs"
            / f"{task}-{integration_label}"
            / "integration_config.json"
        )
        rest = reference / "fault_campaign/rest_references" / task
        if not config.is_file() or not (rest / "rest_reference.json").is_file():
            raise FileNotFoundError(f"task config/rest reference missing: {task}")
        base = replace(
            template,
            initial_seed=seed,
            exogenous_seed=seed,
            output_dir=task_root / "template-clean-output",
            runtime_dir=(root / "runtime/live-univtac/n0" / task / output.name),
        )
        clean_path = task_root / "base_clean_request.json"
        write_json(clean_path, live_univtac_request_to_dict(base))
        campaign_id = f"{output.name}-{task}"
        status, loaded = generate_n0_fault_campaign_bundle(
            task_root / "fault_campaign",
            N0FaultCampaignGenerationSpec(
                campaign_id=campaign_id,
                base_clean_request_paths=(clean_path,),
                operator_ids=operator_ids,
                severity_levels=(5,),
                operator_seed_master=seed,
                fault_start_index=0,
                fault_stop_index=base.max_observation_steps,
                rest_reference_artifacts={task: rest},
                severity_registry=severity_registry,
                fault_onset_mode=fault_onset_mode,
                fault_onset_max_index=8,
            ),
        )
        unsupported = len(set(operator_ids) & {"A1_stream_absence", "A2_frame_erasure"})
        if (
            loaded.manifest.live_request_count != 1 + len(operator_ids) - unsupported
            or loaded.manifest.unsupported_contract_count != unsupported
        ):
            raise ValueError(f"unexpected N0 applicability inventory: {task}")
        write_json(
            task_root / "prepared.json",
            {
                "task": task,
                "reference_clean_request": str(template_path),
                "reference_rest": str(rest),
                "integration_config": str(config),
                "campaign_manifest_sha256": loaded.manifest.sha256,
                "generation_status": status,
                "live_request_count": loaded.manifest.live_request_count,
                "unsupported_contract_count": loaded.manifest.unsupported_contract_count,
            },
        )


def _wait_server(
    process: subprocess.Popen[bytes], port: int, seconds: int = 1200
) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"official N0 server exited: {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError("official N0 server did not open its port")


def _check_server_port_available(port: int) -> None:
    """Match asyncio server reuse semantics while rejecting live listeners."""
    with socket.socket() as probe:
        # A previous group may leave TIME_WAIT connections after its server exits.
        # Reusing that address is safe; another live listener still prevents bind.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def _stop_owned(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _run_command(command: list[str], log: Path, env: dict[str, str]) -> None:
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command exited {result.returncode}: {log}")


def official_server_command(
    *, n0_python: Path, repo: Path, port: int, save_root: Path
) -> list[str]:
    """Launch upstream NCCL setup with a single local torchrun rank."""

    return [
        str(n0_python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        str(repo / "scripts/n0_twam/serve_official.py"),
        "--port",
        str(port),
        "--save-root",
        str(save_root),
    ]


def official_server_environment(
    *, root: Path, task: str, save_root: Path, inherited: dict[str, str]
) -> dict[str, str]:
    """Bind the official task pool and delta checkpoint before config import."""

    model_root = root / "artifacts/models/n0_twam"
    pool = model_root / "serve-pools" / task
    manifest = json.loads((pool / "serve_bundle_manifest.json").read_text())
    if manifest.get("task_id") != task or manifest.get("action_mode") != "delta":
        raise ValueError(f"official N0 serve manifest does not match {task}")
    if Path(manifest["serve_pool_root"]).resolve() != pool.resolve():
        raise ValueError("official N0 serve pool path drifted")
    bundle = model_root / "serve-bundle"
    if not (bundle / "vae/config.json").is_file():
        raise FileNotFoundError("official N0 serve VAE is missing")
    result = dict(inherited)
    result.update(
        TWAM_SERVE_POOL=str(pool),
        TWAM_SERVE_TASK=manifest["serve_task_id"],
        TWAM_SERVE_ACTION_MODE="delta",
        TWAM_SERVE_BUNDLE=str(bundle),
        TWAM_SERVE_OUT=str(save_root),
    )
    return result


def _runtime_environment(
    *,
    deployment_root: Path,
    package_root: Path,
    n0_source: Path,
    gpu_id: int,
    inherited: dict[str, str],
) -> dict[str, str]:
    result = dict(inherited)
    result["PYTHONPATH"] = os.pathsep.join(
        (str(package_root), str(n0_source), result.get("PYTHONPATH", ""))
    )
    result["PYTHONUNBUFFERED"] = "1"
    result["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    result["TOKENIZERS_PARALLELISM"] = "false"
    result.setdefault(
        "ROBOTACTILE_N0_DIGEST_CACHE_DIR",
        str(
            deployment_root
            / "runtime/artifact-digest-cache/n0-twam"
            / socket.gethostname()
        ),
    )
    return result


def run(*, output: Path, package_root: Path) -> None:
    plan: dict[str, Any] = json.loads((output / "plan.json").read_text())
    repo = Path(plan["repo"])
    root = repo / "deployment-sm120"
    n0_source = Path(plan.get("n0_source", root / "sources/N0-TWAM"))
    n0_python = Path(plan.get("n0_python", root / "runtime/n0-twam/bin/python"))
    gpu_id = plan.get("gpu_id", 0)
    if type(gpu_id) is not int or gpu_id < 0:
        raise ValueError("plan gpu_id must be a non-negative integer")
    isaac = Path(plan["isaac_python"])
    for required in (isaac, n0_python, n0_source):
        if not required.exists():
            raise FileNotFoundError(required)
    if not (package_root / "robotactile_benchmark").is_dir():
        raise FileNotFoundError("isolated benchmark package is absent")
    env = _runtime_environment(
        deployment_root=root,
        package_root=package_root,
        n0_source=n0_source,
        gpu_id=gpu_id,
        inherited=dict(os.environ),
    )
    port = int(plan["port"])
    for task in plan["tasks"]:
        task_root = output / task
        if (task_root / "finished.json").exists():
            continue
        if (task_root / "started.json").exists():
            raise RuntimeError(f"task already started; inspect before resuming: {task}")
        prepared = json.loads((task_root / "prepared.json").read_text())
        _check_server_port_available(port)
        server_env = official_server_environment(
            root=root,
            task=task,
            save_root=task_root / "server_dumps",
            inherited=env,
        )
        server_env["PYTHONPATH"] = os.pathsep.join(
            (str(repo / "src"), str(n0_source), env.get("PYTHONPATH", ""))
        )
        write_json(
            task_root / "started.json", {"task": task, "started_unix": time.time()}
        )
        server_log = task_root / "server.log"
        with server_log.open("x", encoding="utf-8") as stream:
            server = subprocess.Popen(
                official_server_command(
                    n0_python=n0_python,
                    repo=repo,
                    port=port,
                    save_root=task_root / "server_dumps",
                ),
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=server_env,
                start_new_session=True,
            )
            try:
                _wait_server(server, port)
                _run_command(
                    [
                        str(isaac),
                        "-m",
                        "robotactile_benchmark.cli",
                        "run-n0-fault-campaign",
                        "--campaign-root",
                        str(task_root / "fault_campaign"),
                        "--integration-config",
                        prepared["integration_config"],
                        "--n0-source-root",
                        str(n0_source),
                        "--n0-port",
                        str(port),
                        "--capture-profile",
                        plan["capture_profile"],
                        "--action-execution-contract",
                        "robotactile_n0_training_60hz_ee_v1",
                    ],
                    task_root / "execution.log",
                    env,
                )
                _run_command(
                    [
                        str(n0_python),
                        "-m",
                        "robotactile_benchmark.cli",
                        "report-n0-fault-campaign",
                        "--campaign-root",
                        str(task_root / "fault_campaign"),
                        "--output",
                        str(task_root / "report"),
                        "--bootstrap-seed",
                        str(plan["seed"]),
                    ],
                    task_root / "report.log",
                    env,
                )
                write_json(
                    task_root / "finished.json",
                    {"task": task, "finished_unix": time.time()},
                )
            except Exception as exc:
                write_json(
                    task_root / "error.json",
                    {"task": task, "error": str(exc), "failed_unix": time.time()},
                )
                raise
            finally:
                _stop_owned(server)
    write_json(
        output / "finished.json",
        {"tasks": plan["tasks"], "finished_unix": time.time()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--repo", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--reference-a", type=Path, required=True)
    prep.add_argument("--reference-b", type=Path, required=True)
    prep.add_argument("--tasks", nargs="+", required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--integration-label", required=True)
    prep.add_argument("--port", type=int, required=True)
    prep.add_argument("--isaac-python", type=Path, required=True)
    prep.add_argument("--n0-python", type=Path)
    prep.add_argument("--n0-source", type=Path)
    prep.add_argument("--gpu-id", type=int, default=0)
    prep.add_argument(
        "--fault-onset-mode",
        choices=(FIXED_FAULT_ONSET_MODE, EARLY_RANDOM_ONSET_MODE),
        default=EARLY_RANDOM_ONSET_MODE,
    )
    prep.add_argument("--capture-profile", default="paper_full_v1")
    execute = sub.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            repo=args.repo.resolve(strict=True),
            output=args.output.absolute(),
            reference_a=args.reference_a.resolve(strict=True),
            reference_b=args.reference_b.resolve(strict=True),
            tasks=tuple(args.tasks),
            seed=args.seed,
            integration_label=args.integration_label,
            port=args.port,
            capture_profile=args.capture_profile,
            isaac_python=args.isaac_python.resolve(strict=True),
            n0_python=(
                args.n0_python.resolve(strict=True)
                if args.n0_python is not None
                else None
            ),
            n0_source=(
                args.n0_source.resolve(strict=True)
                if args.n0_source is not None
                else None
            ),
            gpu_id=args.gpu_id,
            fault_onset_mode=args.fault_onset_mode,
        )
    else:
        run(
            output=args.output.resolve(strict=True),
            package_root=args.package_root.resolve(strict=True),
        )


if __name__ == "__main__":
    main()
