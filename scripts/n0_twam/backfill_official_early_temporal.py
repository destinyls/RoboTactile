"""Backfill only unscored official N0 temporal cells without rerunning Clean."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from robotactile_benchmark.execution.loading import load_live_univtac_request
from robotactile_benchmark.n0_fault_campaign.generation import (
    N0FaultCampaignGenerationSpec,
    generate_n0_fault_campaign_bundle,
)
from robotactile_benchmark.n0_fault_campaign.io import (
    file_sha256,
    load_n0_fault_campaign_bundle,
)
from robotactile_benchmark.trials import Condition
from scripts.n0_twam.run_official_early_fault_seed import (
    REGISTRY,
    _run_command,
    _stop_owned,
    _wait_server,
    official_server_command,
    official_server_environment,
    write_json,
)

TEMPORAL_OPERATORS = (
    "T1_fixed_source_delay",
    "T2_held_last_freeze",
    "T3_inter_sensor_skew",
)


def _valid_scored_terminal(path: Path) -> bool:
    if not path.is_file():
        return False
    result: dict[str, Any] = json.loads(path.read_text())
    return (
        result.get("execution_status") in {"success", "timeout"}
        and result.get("validation_passed") is True
        and result.get("score_eligible") is True
        and type(result.get("score_success")) is bool
    )


def _pending_operators(prior_campaign: Path) -> tuple[str, ...]:
    loaded = load_n0_fault_campaign_bundle(prior_campaign)
    selected: list[str] = []
    for operator in TEMPORAL_OPERATORS:
        cells = [cell for cell in loaded.manifest.cells if cell.operator_id == operator]
        if len(cells) != 1 or cells[0].artifact_relpath is None:
            raise ValueError(f"prior campaign lacks one {operator} cell")
        terminal = loaded.root / cells[0].artifact_relpath / "terminal_result.json"
        if _valid_scored_terminal(terminal):
            continue
        if terminal.is_file():
            result: dict[str, Any] = json.loads(terminal.read_text())
            if result.get("failure_code") != "delivery_failed":
                raise ValueError(f"{operator} has an unclassified terminal outcome")
        selected.append(operator)
    return tuple(selected)


def prepare(prior: Path, output: Path) -> None:
    """Freeze a no-clobber successor from completed v6 tasks and valid Clean."""

    if output.exists():
        raise FileExistsError(output)
    if not (prior / "finished.json").is_file():
        raise RuntimeError("prior campaign must finish before backfill preparation")
    old_plan: dict[str, Any] = json.loads((prior / "plan.json").read_text())
    if old_plan.get("seed") != 5 or old_plan.get("severity_registry") != REGISTRY:
        raise ValueError("prior seed or optical registry differs")
    output.mkdir(parents=True, exist_ok=False)
    selected_tasks: list[str] = []
    for task in old_plan["tasks"]:
        prior_task = prior / task
        prior_campaign = prior_task / "fault_campaign"
        operators = _pending_operators(prior_campaign)
        if not operators:
            continue
        loaded_prior = load_n0_fault_campaign_bundle(prior_campaign)
        clean = [
            cell
            for cell in loaded_prior.manifest.cells
            if cell.condition is Condition.CLEAN
        ]
        if len(clean) != 1 or clean[0].artifact_relpath is None:
            raise ValueError(f"prior {task} Clean cell is not unique")
        clean_artifact = loaded_prior.root / clean[0].artifact_relpath
        if not _valid_scored_terminal(clean_artifact / "terminal_result.json"):
            raise ValueError(f"prior {task} Clean has no valid scored artifact")
        base_request = prior_task / "base_clean_request.json"
        loaded_request = load_live_univtac_request(base_request)
        task_root = output / task
        _, generated = generate_n0_fault_campaign_bundle(
            task_root / "fault_campaign",
            N0FaultCampaignGenerationSpec(
                campaign_id=f"{output.name}-{task}",
                base_clean_request_paths=(base_request,),
                operator_ids=operators,
                severity_levels=(5,),
                operator_seed_master=5,
                fault_start_index=0,
                fault_stop_index=loaded_request.max_observation_steps,
                rest_reference_artifacts={},
                severity_registry=REGISTRY,
                fault_onset_mode="early_random_onset_v1",
                fault_onset_max_index=8,
            ),
        )
        new_clean = [
            cell
            for cell in generated.manifest.cells
            if cell.condition is Condition.CLEAN
        ]
        if (
            len(new_clean) != 1
            or new_clean[0].artifact_relpath is None
            or new_clean[0].trial_manifest_sha256 != clean[0].trial_manifest_sha256
        ):
            raise ValueError(f"successor {task} Clean identity drifted")
        clean_target = generated.root / new_clean[0].artifact_relpath
        shutil.copytree(clean_artifact, clean_target, symlinks=False)
        write_json(
            task_root / "prepared.json",
            {
                "task": task,
                "operators": list(operators),
                "integration_config": json.loads(
                    (prior_task / "prepared.json").read_text()
                )["integration_config"],
                "inherited_clean_artifact": str(clean_artifact),
                "inherited_clean_root_receipt_sha256": file_sha256(
                    clean_artifact / "root_receipt.json"
                ),
                "campaign_manifest_sha256": generated.manifest.sha256,
            },
        )
        selected_tasks.append(task)
    write_json(
        output / "plan.json",
        {
            "schema": "robotactile-n0-temporal-backfill-v1",
            "prior": str(prior),
            "tasks": selected_tasks,
            "repo": old_plan["repo"],
            "seed": 5,
            "port": old_plan["port"],
            "capture_profile": old_plan["capture_profile"],
            "isaac_python": old_plan["isaac_python"],
            "prepared_unix": time.time(),
        },
    )


def run(output: Path, package_root: Path) -> None:
    """Execute only generated T cells; preserve all inherited Clean artifacts."""

    plan: dict[str, Any] = json.loads((output / "plan.json").read_text())
    if not (Path(plan["prior"]) / "finished.json").is_file():
        raise RuntimeError("prior campaign is not finished")
    repo = Path(plan["repo"])
    root = repo / "deployment-sm120"
    n0_source = root / "sources/N0-TWAM"
    n0_python = root / "runtime/n0-twam/bin/python"
    isaac = Path(plan["isaac_python"])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(package_root), str(n0_source), env.get("PYTHONPATH", ""))
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    port = int(plan["port"])
    for task in plan["tasks"]:
        task_root = output / task
        if (task_root / "finished.json").exists():
            continue
        if (task_root / "started.json").exists():
            raise RuntimeError(f"backfill already started: {task}")
        prepared: dict[str, Any] = json.loads((task_root / "prepared.json").read_text())
        campaign = load_n0_fault_campaign_bundle(task_root / "fault_campaign")
        cells = [
            cell
            for cell in campaign.manifest.cells
            if cell.operator_id in prepared["operators"]
        ]
        if len(cells) != len(prepared["operators"]):
            raise ValueError(f"backfill cell inventory mismatch: {task}")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
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
        with (task_root / "server.log").open("x", encoding="utf-8") as stream:
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
                for cell in cells:
                    if cell.request_relpath is None or cell.artifact_relpath is None:
                        raise ValueError(f"backfill request path missing: {task}")
                    terminal = (
                        campaign.root / cell.artifact_relpath / "terminal_result.json"
                    )
                    if terminal.exists():
                        raise FileExistsError(
                            f"backfill artifact already exists: {terminal}"
                        )
                    _run_command(
                        [
                            str(isaac),
                            "-m",
                            "robotactile_benchmark.cli",
                            "live-univtac-run",
                            "--request",
                            str(campaign.root / cell.request_relpath),
                            "--root",
                            str(root),
                            "--config",
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
                        task_root / f"{cell.operator_id}.log",
                        env,
                    )
                    if not _valid_scored_terminal(terminal):
                        raise RuntimeError(
                            f"backfill did not score: {cell.operator_id}"
                        )
                _run_command(
                    [
                        str(n0_python),
                        "-m",
                        "robotactile_benchmark.cli",
                        "report-n0-fault-campaign",
                        "--campaign-root",
                        str(campaign.root),
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
            except Exception as error:
                write_json(
                    task_root / "error.json", {"task": task, "error": str(error)}
                )
                raise
            finally:
                _stop_owned(server)
    write_json(
        output / "finished.json", {"tasks": plan["tasks"], "finished_unix": time.time()}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--prior", type=Path, required=True)
    prepare_command.add_argument("--output", type=Path, required=True)
    run_command = commands.add_parser("run")
    run_command.add_argument("--output", type=Path, required=True)
    run_command.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.prior.resolve(strict=True), args.output.absolute())
    else:
        run(args.output.resolve(strict=True), args.package_root.resolve(strict=True))


if __name__ == "__main__":
    main()
