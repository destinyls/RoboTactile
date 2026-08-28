#!/usr/bin/env python3
"""Probe two fresh UniVTAC task runtimes inside one live Isaac application."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_factory import (
    N0_UNIVTAC_ANTIALIASING_MODE,
    launch_univtac_app_host,
)
from robotactile_benchmark.backends.univtac_host import UniVTACSimulationAppHost
from robotactile_benchmark.backends.univtac_isaac import (
    UniVTACIsaacBackend,
    UniVTACTaskRuntime,
)
from robotactile_benchmark.clean_baseline.io import write_canonical_no_clobber
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)

_DEFAULT_INITIAL_SEEDS = (4_000_000, 4_000_001)
_ANTIALIASING_MODES = ("Off", "FXAA", "DLSS", "TAA", "DLAA")
_RENDERING_MODES = ("balanced", "performance", "quality")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preclose-output", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--initial-seed",
        action="append",
        dest="initial_seeds",
        type=int,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--rendering-mode", choices=_RENDERING_MODES, default="balanced"
    )
    parser.add_argument(
        "--antialiasing-mode",
        choices=_ANTIALIASING_MODES,
        default=N0_UNIVTAC_ANTIALIASING_MODE,
    )
    return parser


def _absolute_directory(path: Path, name: str) -> Path:
    selected = Path(path).absolute()
    if selected.is_symlink() or not selected.is_dir():
        raise ValueError(f"{name} must be a non-symlink existing directory")
    return selected


def _output_path(path: Path) -> Path:
    selected = Path(path).absolute()
    if selected.exists() or selected.is_symlink():
        raise FileExistsError("output must not already exist")
    if selected.parent.is_symlink() or not selected.parent.is_dir():
        raise ValueError("output parent must be a non-symlink existing directory")
    return selected


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output = _output_path(args.output)
    requested = args.preclose_output
    preclose = _output_path(
        output.with_name(f"{output.stem}.preclose.json")
        if requested is None
        else requested
    )
    if preclose == output:
        raise ValueError("preclose output must differ from final output")
    return output, preclose


def _initial_seeds(args: argparse.Namespace) -> tuple[int, ...]:
    raw = args.initial_seeds
    seeds = _DEFAULT_INITIAL_SEEDS if raw is None else tuple(raw)
    if len(seeds) < 2:
        raise ValueError("initial_seed must be supplied at least twice")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError("initial_seed values must be non-negative integers")
    return seeds


def _failure(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "error_message": str(error),
        "error_type": type(error).__name__,
        "stage": stage,
    }


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} is not a lowercase SHA256")
    return value


def _runtime_directory(
    runtime_root: Path, task_id: str, ordinal: int, seed: int
) -> Path:
    runtime_dir = runtime_root / f"{task_id}-{ordinal:04d}-seed-{seed}"
    if runtime_dir.exists() or runtime_dir.is_symlink():
        raise FileExistsError(f"runtime directory already exists: {runtime_dir}")
    return runtime_dir


def run(
    args: argparse.Namespace,
    *,
    before_app_close: Optional[Callable[[dict[str, object]], None]] = None,
) -> tuple[int, dict[str, object]]:
    process_id = os.getpid()
    started = time.perf_counter()
    app_stages: list[str] = []
    runs: list[dict[str, object]] = []
    cleanup_errors: list[dict[str, str]] = []
    host: Optional[UniVTACSimulationAppHost] = None
    app_launch_duration_s: Optional[float] = None
    app_close_duration_s: Optional[float] = None
    app_close_status = "not_started"
    requested_exit_code: Optional[int] = None
    seeds: tuple[int, ...] = ()
    task_id = str(getattr(args, "task", ""))
    stage = "argument_validation"
    failure: Optional[dict[str, str]] = None

    def build_document(*, preclose: bool) -> tuple[int, dict[str, object]]:
        runtime_count = 0 if host is None else host.runtime_count
        completed = sum(item["status"] == "passed" for item in runs)
        runtimes_passed = (
            failure is None
            and len(runs) == len(seeds)
            and completed == len(seeds)
            and runtime_count == len(seeds)
        )
        passed = runtimes_passed and not preclose
        process_ids = {item["process_id"] for item in runs}
        exit_code = 0 if passed else (1 if preclose else (requested_exit_code or 1))
        status = (
            "runtimes_passed_app_close_pending"
            if preclose and runtimes_passed
            else ("passed" if passed else "failed")
        )
        document: dict[str, object] = {
            "antialiasing_mode": str(getattr(args, "antialiasing_mode", "")),
            "app_close_duration_s": None if preclose else app_close_duration_s,
            "app_close_status": "pending" if preclose else app_close_status,
            "app_launch_duration_s": app_launch_duration_s,
            "app_lifecycle_stages": tuple(app_stages),
            "app_process_id": process_id,
            "cleanup_errors": tuple(cleanup_errors),
            "completed_runtime_count": completed,
            "device": str(getattr(args, "device", "")),
            "evidence_level": "live_univtac_same_app_fresh_runtime_probe_v1",
            "exit_code": None if preclose and runtimes_passed else exit_code,
            "failure": failure,
            "initial_seeds": seeds,
            "policy_loaded": False,
            "receipt_stage": "preclose" if preclose else "final",
            "rendering_mode": str(getattr(args, "rendering_mode", "")),
            "runs": tuple(runs),
            "runtime_count": runtime_count,
            "same_process_confirmed": runtimes_passed and process_ids == {process_id},
            "status": status,
            "task_id": task_id,
            "total_duration_s": time.perf_counter() - started,
        }
        document["content_sha256"] = canonical_hash(document)
        return exit_code, document

    try:
        seeds = _initial_seeds(args)
        upstream_root = _absolute_directory(args.upstream_root, "upstream_root")
        runtime_root = _absolute_directory(args.runtime_root, "runtime_root")
        if not task_id:
            raise ValueError("task must be non-empty")

        stage = "backend_configuration"
        config = build_univtac_backend_config(task_id, action_spec=EE8_ACTION_SPEC)
        launcher_args = production_univtac_launcher_args()
        launcher_args["rendering_mode"] = args.rendering_mode

        stage = "app_launch"
        app_stages.append("app_launch_start")
        app_started = time.perf_counter()
        host = launch_univtac_app_host(
            config,
            upstream_root=upstream_root,
            initial_seed=seeds[0],
            launcher_args=launcher_args,
            device=args.device,
            antialiasing_mode=args.antialiasing_mode,
            stage_observer=app_stages.append,
        )
        app_launch_duration_s = time.perf_counter() - app_started
        app_stages.append("app_launch_complete")

        for ordinal, seed in enumerate(seeds):
            runtime_stages: list[str] = []
            runtime: Optional[UniVTACTaskRuntime] = None
            backend: Optional[UniVTACIsaacBackend] = None
            runtime_dir = _runtime_directory(runtime_root, task_id, ordinal, seed)
            result: dict[str, object] = {
                "close_duration_s": None,
                "construction_duration_s": None,
                "initial_clean_record_sha256": None,
                "initial_seed": seed,
                "initial_state_sha256": None,
                "lifecycle_stages": runtime_stages,
                "ordinal": ordinal,
                "process_id": os.getpid(),
                "reset_duration_s": None,
                "reset_receipt_sha256": None,
                "runtime_count": None,
                "runtime_dir": str(runtime_dir),
                "status": "running",
            }
            runs.append(result)
            runtime_failure: Optional[dict[str, str]] = None
            try:
                stage = f"runtime[{ordinal}].construction"
                runtime_stages.append("runtime_construction_start")
                construction_started = time.perf_counter()
                runtime = host.create_runtime(
                    runtime_dir=runtime_dir,
                    initial_seed=seed,
                    stage_observer=runtime_stages.append,
                )
                result["construction_duration_s"] = (
                    time.perf_counter() - construction_started
                )
                result["runtime_count"] = host.runtime_count
                runtime_stages.append("runtime_construction_complete")

                stage = f"runtime[{ordinal}].backend_initialization"
                backend = UniVTACIsaacBackend(config, runtime)
                context = PolicyEpisodeContext(
                    episode_id=f"same-app-reuse-{task_id}-{ordinal}-{seed}",
                    task=task_id,
                    initial_seed=seed,
                    exogenous_seed=seed,
                    instruction=config.task.prompt,
                    action_spec=EE8_ACTION_SPEC,
                )

                stage = f"runtime[{ordinal}].reset"
                runtime_stages.append("backend_reset_start")
                reset_started = time.perf_counter()
                receipt = backend.reset(context)
                result["reset_duration_s"] = time.perf_counter() - reset_started
                runtime_stages.append("backend_reset_complete")

                stage = f"runtime[{ordinal}].observe"
                record = backend.observe()
                runtime_stages.append("initial_observation_captured")
                state_sha256 = _require_sha256(
                    receipt.simulator_state_sha256, "initial state"
                )
                clean_sha256 = _require_sha256(
                    record.clean_record_sha256, "initial clean record"
                )
                if backend.latest_state_sha256 != state_sha256:
                    raise RuntimeError("backend initial state SHA256 mismatch")
                if backend.initial_clean_record_sha256 != clean_sha256:
                    raise RuntimeError("backend initial clean record SHA256 mismatch")
                result.update(
                    {
                        "initial_clean_record_sha256": clean_sha256,
                        "initial_state_sha256": state_sha256,
                        "reset_receipt_sha256": _require_sha256(
                            receipt.sha256, "reset receipt"
                        ),
                    }
                )
            except Exception as error:
                runtime_failure = _failure(stage, error)
                result["failure"] = runtime_failure
            finally:
                close_stage = f"runtime[{ordinal}].close"
                runtime_stages.append("backend_close_start")
                close_started = time.perf_counter()
                try:
                    if backend is not None:
                        backend.close()
                    elif runtime is not None:
                        runtime.close_runtime()
                    runtime_stages.append("backend_close_complete")
                except Exception as error:
                    close_failure = _failure(close_stage, error)
                    cleanup_errors.append(close_failure)
                    runtime_stages.append("backend_close_failed")
                    if runtime_failure is None:
                        runtime_failure = close_failure
                        result["failure"] = close_failure
                result["close_duration_s"] = time.perf_counter() - close_started
                result["lifecycle_stages"] = tuple(runtime_stages)
                result["status"] = "failed" if runtime_failure else "passed"

            if runtime_failure is not None:
                failure = runtime_failure
                break
    except Exception as error:
        failure = _failure(stage, error)
    finally:
        if host is not None:
            app_stages.append("app_close_pending")
            if before_app_close is not None:
                try:
                    _, preclose_document = build_document(preclose=True)
                    before_app_close(preclose_document)
                except Exception as error:
                    receipt_failure = _failure("preclose_receipt", error)
                    cleanup_errors.append(receipt_failure)
                    if failure is None:
                        failure = receipt_failure
            app_stages.append("app_close_start")
            app_close_started = time.perf_counter()
            try:
                host.close()
                app_stages.append("app_close_complete")
                app_close_status = "returned"
            except SystemExit as error:
                raw_exit_code = error.code
                system_exit_code = 0 if raw_exit_code is None else 1
                if isinstance(raw_exit_code, int):
                    system_exit_code = int(raw_exit_code)
                if system_exit_code == 0:
                    app_stages.append("app_close_system_exit_zero")
                    app_close_status = "system_exit_zero"
                else:
                    close_failure = _failure("app_close", error)
                    cleanup_errors.append(close_failure)
                    app_stages.append("app_close_system_exit_nonzero")
                    app_close_status = "system_exit_nonzero"
                    requested_exit_code = system_exit_code
                    if failure is None:
                        failure = close_failure
            except KeyboardInterrupt as error:
                close_failure = _failure("app_close", error)
                cleanup_errors.append(close_failure)
                app_stages.append("app_close_interrupted")
                app_close_status = "interrupted"
                requested_exit_code = 130
                if failure is None:
                    failure = close_failure
            except Exception as error:
                close_failure = _failure("app_close", error)
                cleanup_errors.append(close_failure)
                app_stages.append("app_close_failed")
                app_close_status = "failed"
                if failure is None:
                    failure = close_failure
            app_close_duration_s = time.perf_counter() - app_close_started

    return build_document(preclose=False)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output, preclose_output = _output_paths(args)

    def write_preclose(document: dict[str, object]) -> None:
        write_canonical_no_clobber(preclose_output, document)

    exit_code, document = run(args, before_app_close=write_preclose)
    write_canonical_no_clobber(output, document)
    serialized = json.dumps(document, sort_keys=True, separators=(",", ":"))
    print(serialized, file=sys.stdout if exit_code == 0 else sys.stderr, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
