"""Capture one excluded development Clean episode, never a campaign SR row."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, cast

from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import LiveUniVTACRunRequest
from robotactile_benchmark.execution.live_artifacts import load_live_univtac_artifact
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.n0_fault_campaign.io import file_sha256
from robotactile_benchmark.n0_fault_campaign.stress_group import write_once
from robotactile_benchmark.n0_fault_campaign.stress_metrics import eligible_terminal
from robotactile_benchmark.n0_fault_campaign.stress_protocol import validate_protocol
from robotactile_benchmark.trials import Condition
from scripts.n0_twam.noise_stress_queue import (
    check_port,
    gpu_lock,
    read_json,
    validate_runtime_binding,
    wait_server,
)
from scripts.n0_twam.run_official_early_fault_seed import (
    official_server_command,
    official_server_environment,
)
from scripts.retrained_evaluation.released_clean_seed_queue import (
    build_request,
    stop_owned,
    worker_command,
)


def source_receipt(source: Path, plan: dict[str, Any], seed: int) -> dict[str, Any]:
    artifact = load_live_univtac_artifact(source)
    trial = artifact.trial
    result = result_to_dict(artifact.evidence.result)
    finalization = artifact.evidence.finalization
    records = () if finalization is None else finalization.clean_records
    valid = (
        trial.condition is Condition.CLEAN
        and trial.task == plan["task"] == "lift_bottle"
        and trial.initial_seed == trial.exogenous_seed == seed
        and seed in plan["excluded_seeds"]
        and seed not in plan["seeds"]
        and trial.checkpoint_sha256 == plan["binding"]["model_sha256"]
        and trial.dataset_sha256 == plan["binding"]["dataset_sha256"]
        and artifact.capture_profile is LiveCaptureProfile.PAPER_FULL
        and eligible_terminal(result)
        and len(records) >= 61
        and all(r.observation.step_index == i for i, r in enumerate(records))
    )
    return {
        "source_path": str(source.resolve()),
        "source_sha256": artifact.external_root_sha256,
        "source_seed": seed,
        "terminal_status": result["terminal_status"],
        "clean_record_count": len(records),
        "valid_for_replay": valid,
        "not_scored_in_campaign": True,
        "source_selected_independent_of_success": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu_id < 0 or not 1 <= args.port <= 65535 or args.timeout_s <= 0:
        raise ValueError("invalid capture GPU, port or deadline")
    for name in (
        "protocol",
        "model_root",
        "code",
        "package",
        "config",
        "n0_source",
        "digest_cache",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve(strict=True))
    plan = validate_protocol(read_json(args.protocol))
    if (
        plan["task"] != "lift_bottle"
        or args.seed not in plan["excluded_seeds"]
        or args.seed in plan["seeds"]
    ):
        raise ValueError("capture requires an excluded development lift_bottle seed")
    args.output = args.output.absolute()
    for name in ("isaac_python", "n0_python"):
        setattr(args, name, Path(getattr(args, name)).absolute())
    isaac_hash = file_sha256(args.isaac_python.resolve(strict=True))
    if isaac_hash != args.isaac_python_sha256:
        raise ValueError("Isaac launcher digest differs from declared runtime")
    module_file = __import__("robotactile_benchmark").__file__
    if module_file is None or Path(module_file).resolve().parent.parent != args.package:
        raise ValueError("capture runner imported a different package")
    binding = {**plan["binding"], "code_sha256": args.code_sha256}
    runtime = validate_runtime_binding(args, {"binding": binding})
    args.root, args.model, args.task = args.model_root, "n0_twam", "lift_bottle"
    args.dataset_sha256 = binding["dataset_sha256"]
    args.episode_timeout_s = args.timeout_s
    request = live_univtac_request_to_dict(
        cast(LiveUniVTACRunRequest, build_request(args, args.seed, args.output))
    )
    frozen = {
        "schema": "n0_development_clean_capture_plan_v1",
        "source_protocol_sha256": plan["protocol_sha256"],
        "binding": binding,
        "runtime": runtime,
        "seed": args.seed,
        "request": request,
        "request_sha256": canonical_hash(request),
        "capture_profile": "paper_full_v1",
        "minimum_contiguous_prefix_last_step": 60,
        "not_scored_in_campaign": True,
        "success_rate_claimed": False,
        "max_attempts": 1,
        "isaac_python_resolved": str(args.isaac_python.resolve(strict=True)),
        "n0_python_resolved": str(args.n0_python.resolve(strict=True)),
        "isaac_launcher_sha256": isaac_hash,
        "n0_launcher_sha256": file_sha256(args.n0_python.resolve(strict=True)),
        "n0_server_sha256": file_sha256(args.n0_source / "n0_twam/n0_twam_server.py"),
    }
    frozen["capture_plan_sha256"] = canonical_hash(frozen)
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu_id),
        PYTHONUNBUFFERED="1",
        PYTHONPATH=os.pathsep.join(
            (str(args.package), str(args.code), str(args.n0_source))
        ),
        ROBOTACTILE_REPOSITORY_ROOT=str(args.code),
        ROBOTACTILE_PACKAGE_PATH=str(args.package),
        ROBOTACTILE_N0_DIGEST_CACHE_DIR=str(args.digest_cache),
        TOKENIZERS_PARALLELISM="false",
    )
    server_env = official_server_environment(
        root=args.model_root,
        task=args.task,
        save_root=args.output / "server/dumps",
        inherited=env,
    )
    server_env["PYTHONPATH"] = env["PYTHONPATH"]
    server_command = official_server_command(
        n0_python=args.n0_python,
        repo=args.code,
        port=args.port,
        save_root=args.output / "server/dumps",
    )
    command = worker_command(args, args.output / "request.json")
    command[command.index("--capture-profile") + 1] = "paper_full_v1"
    server = worker = None
    previous = {
        s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)
    }

    def deadline(_signum: int, _frame: object) -> None:
        raise TimeoutError("development capture deadline exceeded")

    def interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(f"development capture interrupted by signal {signum}")

    with gpu_lock(args.gpu_lock.absolute()):
        check_port(args.port)
        memory = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu_id),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        if int(memory) >= 1024:
            raise RuntimeError(
                "capture GPU is occupied; cannot share or interrupt campaigns"
            )
        args.output.mkdir(parents=True, exist_ok=False)
        write_once(args.output / "plan.json", frozen)
        write_once(args.output / "request.json", request)
        try:
            signal.signal(signal.SIGALRM, deadline)
            signal.signal(signal.SIGINT, interrupted)
            signal.signal(signal.SIGTERM, interrupted)
            signal.alarm(args.timeout_s)
            (args.output / "server").mkdir()
            for role, argv, process_env in (
                ("server", server_command, server_env),
                ("worker", command, env),
            ):
                with (args.output / f"{role}.log").open("xb") as log:
                    process = subprocess.Popen(
                        argv,
                        cwd=args.code,
                        env=process_env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                if role == "server":
                    server = process
                else:
                    worker = process
                write_once(
                    args.output / f"{role}-launch.json",
                    {"pid": process.pid, "argv": argv},
                )
                if role == "server":
                    wait_server(process, args.port, min(1800, args.timeout_s))
            assert worker is not None
            returncode = worker.wait(timeout=args.timeout_s)
            receipt = source_receipt(args.output / "artifact", plan, args.seed)
            receipt.update(
                capture_plan_sha256=frozen["capture_plan_sha256"],
                worker_returncode=returncode,
            )
            if returncode != 0:
                receipt["valid_for_replay"] = False
            write_once(args.output / "receipt.json", receipt)
            if not receipt["valid_for_replay"]:
                raise ValueError(
                    "captured source failed replay eligibility; retained without rerun"
                )
            return receipt
        except BaseException as error:
            if not (args.output / "receipt.json").exists():
                write_once(
                    args.output / "receipt.json",
                    {
                        "source_path": str(args.output / "artifact"),
                        "source_sha256": None,
                        "source_seed": args.seed,
                        "terminal_status": None,
                        "valid_for_replay": False,
                        "not_scored_in_campaign": True,
                        "capture_plan_sha256": frozen["capture_plan_sha256"],
                        "error_type": type(error).__name__,
                    },
                )
            write_once(
                args.output / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "not_scored_in_campaign": True,
                },
            )
            raise
        finally:
            signal.alarm(0)
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            try:
                stop_owned(worker)
            finally:
                stop_owned(server)
                write_once(
                    args.output / "exit.json",
                    {
                        "worker_returncode": None
                        if worker is None
                        else worker.returncode,
                        "server_returncode": None
                        if server is None
                        else server.returncode,
                    },
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "protocol",
        "model-root",
        "code",
        "package",
        "config",
        "isaac-python",
        "n0-python",
        "n0-source",
        "digest-cache",
        "output",
        "gpu-lock",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("code-sha256", "isaac-python-sha256"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("seed", "gpu-id", "port"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--timeout-s", type=int, default=7200)
    receipt = run(parser.parse_args())
    print(
        f"valid_for_replay={receipt['valid_for_replay']}; not_scored_in_campaign=true"
    )


if __name__ == "__main__":
    main()
