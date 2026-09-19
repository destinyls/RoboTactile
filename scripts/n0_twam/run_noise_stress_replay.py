"""Own an offline-only N0 server for frozen-input Clean/Clean/Noise diagnostics."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.n0_fault_campaign.io import file_sha256
from robotactile_benchmark.n0_fault_campaign.stress_group import write_once
from robotactile_benchmark.n0_fault_campaign.stress_provenance import (
    verify_runtime_code,
)
from robotactile_benchmark.n0_fault_campaign.stress_replay import run_fixed_input_replay
from robotactile_benchmark.n0_fault_campaign.stress_replay_evidence import (
    load_replay_inputs,
    summarize_replay_probes,
)
from robotactile_benchmark.transport.n0_official import load_official_n0_rpc
from scripts.n0_twam.noise_stress_queue import (
    check_port,
    gpu_lock,
    validate_runtime_binding,
    wait_server,
)
from scripts.n0_twam.run_official_early_fault_seed import (
    official_server_command,
    official_server_environment,
)
from scripts.retrained_evaluation.released_clean_seed_queue import stop_owned


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu_id < 0 or not 1 <= args.port <= 65535 or args.timeout_s <= 0:
        raise ValueError("invalid offline GPU, port or deadline")
    if (
        not args.target_steps
        or args.target_steps != sorted(set(args.target_steps))
        or any(s not in (12, 36, 60) for s in args.target_steps)
    ):
        raise ValueError("offline targets must be predeclared members of 12,36,60")
    for name in (
        "group",
        "clean_artifact",
        "model_root",
        "code",
        "package",
        "config",
        "n0_source",
        "digest_cache",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve(strict=True))
    args.output = args.output.absolute()
    args.n0_python = (
        args.n0_python.absolute()
    )  # keep the venv launcher, hash its real target
    verify_runtime_code(args.code, args.package, args.code_sha256)
    module_file = __import__("robotactile_benchmark").__file__
    if module_file is None:
        raise ValueError("offline runner package has no source file")
    actual_package = Path(module_file).resolve().parent.parent
    if actual_package != args.package:
        raise ValueError("offline runner imported a different package")
    plan, provenance, clean, variants = load_replay_inputs(
        args.group, args.clean_artifact, max(args.target_steps)
    )
    # Diagnostic code has its own snapshot; it does not relabel the old live protocol.
    binding = {**plan["binding"], "code_sha256": args.code_sha256}
    runtime = validate_runtime_binding(args, {"binding": binding})
    launch_hashes = {
        "n0_launcher_sha256": file_sha256(args.n0_python.resolve(strict=True)),
        "n0_server_sha256": file_sha256(args.n0_source / "n0_twam/n0_twam_server.py"),
    }
    frozen = {
        "schema": "n0_offline_noise_replay_plan_v1",
        **provenance,
        "binding": binding,
        "source_protocol_sha256": plan["protocol_sha256"],
        "target_steps": args.target_steps,
        "branches": ["clean-reference", "clean-repeat", *variants],
        "actions_executed": False,
        "success_rate_claimed": False,
    }
    frozen["diagnostic_plan_sha256"] = canonical_hash(frozen)
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
    save_root = args.output / "server/dumps"
    env = official_server_environment(
        root=args.model_root, task="lift_bottle", save_root=save_root, inherited=env
    )
    traces = args.output / "server/input-traces"
    command = official_server_command(
        n0_python=args.n0_python, repo=args.code, port=args.port, save_root=save_root
    )
    command += [
        "--diagnostic-input-trace-root",
        str(traces),
        "--diagnostic-input-trace-limit",
        "8",
        "--diagnostic-input-capture-arrays",
    ]
    server = None
    previous_handlers = {
        s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)
    }

    def deadline(_signum: int, _frame: object) -> None:
        raise TimeoutError("offline replay deadline exceeded")

    def interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(f"offline replay interrupted by signal {signum}")

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
                "offline GPU is occupied; do not share or interrupt live evaluation"
            )
        args.output.mkdir(parents=True, exist_ok=False)
        write_once(args.output / "plan.json", frozen)
        try:
            signal.signal(signal.SIGALRM, deadline)
            signal.signal(signal.SIGINT, interrupted)
            signal.signal(signal.SIGTERM, interrupted)
            signal.alarm(args.timeout_s)
            log_path = args.output / "server/server.log"
            log_path.parent.mkdir(exist_ok=False)
            with log_path.open("xb") as log:
                server = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
            write_once(
                args.output / "server/launch.json",
                {
                    "purpose": "offline_fixed_input_replay_only",
                    "pid": server.pid,
                    "port": args.port,
                    "gpu_id": args.gpu_id,
                    "argv": command,
                    "diagnostic_plan_sha256": frozen["diagnostic_plan_sha256"],
                    **runtime,
                    **launch_hashes,
                },
            )
            wait_server(server, args.port, min(1800, args.timeout_s))
            report = run_fixed_input_replay(
                clean_records=clean,
                fault_records=variants,
                rpc_factory=lambda: load_official_n0_rpc(
                    source_root=args.n0_source,
                    host="127.0.0.1",
                    port=args.port,
                    api_key=os.environ.get("N0_TWAM_API_KEY"),
                ),
                seed=provenance["model_seed"],
                source_sha256=provenance["source_sha256"],
                protocol_sha256=frozen["diagnostic_plan_sha256"],
                target_steps=tuple(args.target_steps),
            )
            report["source_provenance"] = provenance
            report["model_probe"] = summarize_replay_probes(
                traces,
                source_sha256=provenance["source_sha256"],
                protocol_sha256=frozen["diagnostic_plan_sha256"],
                conditions=list(variants),
            )
            report["rng_verification"] = report["model_probe"][
                "rng_after_reset_equal_and_complete"
            ]
            write_once(args.output / "report.json", report)
            return report
        except BaseException as error:
            write_once(
                args.output / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "actions_executed": False,
                },
            )
            raise
        finally:
            signal.alarm(0)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            if server is not None:
                stop_owned(server)
                write_once(
                    args.output / "server/exit.json", {"returncode": server.returncode}
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "group",
        "clean-artifact",
        "model-root",
        "code",
        "package",
        "config",
        "n0-python",
        "n0-source",
        "digest-cache",
        "output",
        "gpu-lock",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument("--target-steps", type=int, nargs="+", default=[12, 36, 60])
    report = run(parser.parse_args())
    print(
        f"offline comparisons={len(report['comparisons'])}; no actions executed; no SR claim"
    )


if __name__ == "__main__":
    main()
