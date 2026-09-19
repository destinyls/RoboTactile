"""Launch exactly one source-bound Dream-Tac optimizer step on one HCU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast

from scripts.dream_tac.training.launch_training import (
    build_dataset_contract,
    build_overrides,
)
from scripts.dream_tac.training.training_request import (
    executable_path,
    sha256_regular_file,
)

from .optimizer_step_preflight import write_optimizer_step_preflight_receipt
from .optimizer_step_request import ACCELERATOR_CONTRACT, HcuOptimizerStepRequest
from .optimizer_step_validators import checkpoint_evidence
from .receipt import signed_receipt, verify_receipt, write_or_verify_receipt

ENVIRONMENT_RECEIPT_NAME: Final[str] = "optimizer_step_environment_receipt.json"
RESOLVED_EXPERIMENT_NAME: Final[str] = "optimizer_step_resolved_experiment.json"
LAUNCH_PLAN_NAME: Final[str] = "optimizer_step_launch_plan.json"
LAUNCH_RESULT_NAME: Final[str] = "optimizer_step_launch_result.json"
LOG_NAME: Final[str] = "optimizer_step.log"
_BASH_BOOTSTRAP: Final[str] = 'set -e\nsource "$1"\nshift\nexec "$@"'
_BASH_ARGV0: Final[str] = "robotactile-hcu-bootstrap"


def _hcu_overrides(request: HcuOptimizerStepRequest) -> tuple[str, ...]:
    common = tuple(
        item
        for item in build_overrides(request)
        if not item.startswith("checkpoint.load_path=")
    )
    return (
        *common,
        "~trainer.callbacks.device_monitor",
        f"checkpoint.load_path={request.base_dcp_root}",
        f"model.config.tokenizer.vae_pth={request.tokenizer_checkpoint.path}",
        "model.config.fsdp_shard_size=1",
    )


def _training_payload(request: HcuOptimizerStepRequest) -> tuple[str, ...]:
    return (
        str(request.python_executable),
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node_rank=0",
        "--nproc_per_node=1",
        "--master_addr=127.0.0.1",
        f"--master_port={request.master_port}",
        "-m",
        "cosmos_policy.scripts.train",
        "--config=cosmos_policy/config/config.py",
        "--",
        *_hcu_overrides(request),
    )


def _wrapped_command(
    request: HcuOptimizerStepRequest, payload: Sequence[str]
) -> tuple[str, ...]:
    """Source content-bound HCU state without interpolating request data."""

    return (
        "bash",
        "--noprofile",
        "--norc",
        "-c",
        _BASH_BOOTSTRAP,
        _BASH_ARGV0,
        str(request.hcu_environment_script.path),
        *payload,
    )


def build_command(request: HcuOptimizerStepRequest) -> tuple[str, ...]:
    """Return the environment-bootstrapped exact single-rank command."""

    return _wrapped_command(request, _training_payload(request))


def explicit_environment(request: HcuOptimizerStepRequest) -> dict[str, str]:
    """Return deliberate HCU overrides recorded in the launch plan."""

    device = str(request.hcu_device)
    pythonpath = ":".join(
        (
            str(request.dream_tac_root),
            *(str(p) for p in request.runtime_pythonpath_roots),
        )
    )
    return {
        "COSMOS_TACTILE_SELF_ATTN_BACKEND": "flashbias_sdpa",
        "CUDA_VISIBLE_DEVICES": device,
        "DREAM_TAC_BASE_CHECKPOINT": str(request.base_dcp_root),
        "HIP_VISIBLE_DEVICES": device,
        "IMAGINAIRE_OUTPUT_ROOT": str(request.output_root),
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": pythonpath,
        "ROBOTACTILE_HCU_TRAINING": "1",
        "ROCR_VISIBLE_DEVICES": device,
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }


def _process_environment(request: HcuOptimizerStepRequest) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("ROBOTACTILE_HCU_CONFIG_PROBE_ONLY", None)
    environment.update(explicit_environment(request))
    return environment


def _environment_probe_script() -> str:
    return "\n".join(
        (
            "import importlib.util,json,platform,sys,torch",
            "assert platform.system() == 'Linux', 'HCU training requires Linux'",
            "assert sys.version_info[:2] == (3, 10), 'runtime must use Python 3.10'",
            "required=('cosmos_policy','flash_attn','transformer_engine','h5py','cv2','transformers','pynvml')",
            "optional=('natten','xformers')",
            "missing=[name for name in required if importlib.util.find_spec(name) is None]",
            "assert not missing, f'missing Dream-Tac modules: {missing}'",
            "assert torch.version.hip, 'PyTorch is not an HCU/HIP build'",
            "assert torch.cuda.is_available(), 'HCU compatibility device API unavailable'",
            "assert torch.cuda.device_count() == 1, 'single-device visibility required'",
            "assert torch.distributed.is_nccl_available(), 'NCCL/RCCL backend unavailable'",
            "p=torch.cuda.get_device_properties(0)",
            "payload={'platform':platform.platform(),'python_version':platform.python_version(),'torch_version':torch.__version__,'hip_version':torch.version.hip,'visible_device_count':torch.cuda.device_count(),'device_name':torch.cuda.get_device_name(0),'total_memory_bytes':p.total_memory,'bf16_reported_supported':torch.cuda.is_bf16_supported(),'nccl_rccl_available':torch.distributed.is_nccl_available(),'optional_modules':{name:importlib.util.find_spec(name) is not None for name in optional}}",
            "print(json.dumps(payload,sort_keys=True))",
        )
    )


def probe_environment(request: HcuOptimizerStepRequest) -> dict[str, object]:
    """Probe the exact request runtime without launching a model or optimizer."""

    python = executable_path(request.python_executable)
    completed = subprocess.run(
        list(
            _wrapped_command(
                request,
                (str(python), "-c", _environment_probe_script()),
            )
        ),
        cwd=request.dream_tac_root,
        env=_process_environment(request),
        check=True,
        capture_output=True,
        text=True,
    )
    payload: object | None = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if not isinstance(payload, dict):
        raise ValueError("Dream-Tac HCU environment probe returned invalid JSON")
    return signed_receipt(
        {
            "schema_version": 1,
            "status": "passed",
            "protocol_id": "dream_tac_hcu_optimizer_step_environment_v1",
            "claim_boundary": "runtime_probe_only_not_optimizer_step",
            "request_sha256": request.request_sha256,
            "accelerator_contract": ACCELERATOR_CONTRACT,
            "python_executable": str(python),
            "hcu_environment_script": request.hcu_environment_script.to_dict(),
            "runtime_pythonpath_roots": [
                str(path) for path in request.runtime_pythonpath_roots
            ],
            "environment": payload,
        }
    )


def build_resolved_experiment(request: HcuOptimizerStepRequest) -> dict[str, object]:
    dataset = build_dataset_contract(request)
    return signed_receipt(
        {
            "schema_version": 1,
            "status": "resolved",
            "protocol_id": "dream_tac_hcu_optimizer_step_resolved_v1",
            "claim_boundary": "configuration_only_not_optimizer_step",
            "request_sha256": request.request_sha256,
            "dataset_bindings": {
                "dataloader_train.dataset": dict(dataset),
                "dataloader_train.sampler.dataset": dict(dataset),
            },
            "vision_tactile_proprio_always_on": True,
            "max_iter": 1,
            "batch_size": 1,
            "hcu_environment_script": request.hcu_environment_script.to_dict(),
            "checkpoint": {
                "consolidated_source": request.base_checkpoint.to_dict(),
                "dcp_load_root": str(request.base_dcp_root),
                "dcp_receipt": request.base_dcp_receipt.to_dict(),
                "tokenizer": request.tokenizer_checkpoint.to_dict(),
            },
            "overrides": list(_hcu_overrides(request)),
        }
    )


def _load_existing_result(path: Path, request: HcuOptimizerStepRequest) -> int | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("existing HCU optimizer-step result is not an object")
    verify_receipt(payload)
    if payload.get("request_sha256") != request.request_sha256:
        raise ValueError("existing HCU optimizer-step result request mismatch")
    return int(cast(int, payload["exit_code"]))


def execute(request: HcuOptimizerStepRequest) -> int:
    """Run all gates once, launch once, and validate real optimizer evidence."""

    result_path = request.receipt_root / LAUNCH_RESULT_NAME
    existing = _load_existing_result(result_path, request)
    if existing is not None:
        return existing
    _, artifact_preflight = write_optimizer_step_preflight_receipt(request)
    environment = probe_environment(request)
    environment_path = request.receipt_root / ENVIRONMENT_RECEIPT_NAME
    write_or_verify_receipt(environment_path, environment)
    resolved = build_resolved_experiment(request)
    resolved_path = request.receipt_root / RESOLVED_EXPERIMENT_NAME
    write_or_verify_receipt(resolved_path, resolved)
    plan_path = request.receipt_root / LAUNCH_PLAN_NAME
    if plan_path.exists():
        raise RuntimeError(
            "launch plan exists without a result; inspect it instead of relaunching"
        )
    log_path = request.receipt_root / LOG_NAME
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite launch log: {log_path}")
    plan = signed_receipt(
        {
            "schema_version": 1,
            "status": "planned",
            "protocol_id": "dream_tac_hcu_optimizer_step_launch_v1",
            "claim_boundary": "planned_only_not_optimizer_step",
            "request_sha256": request.request_sha256,
            "artifact_preflight_receipt_sha256": artifact_preflight["receipt_sha256"],
            "environment_receipt_sha256": environment["receipt_sha256"],
            "resolved_experiment_receipt_sha256": resolved["receipt_sha256"],
            "command": list(build_command(request)),
            "hcu_environment_script": request.hcu_environment_script.to_dict(),
            "environment": explicit_environment(request),
            "cwd": str(request.dream_tac_root),
            "log_path": str(log_path),
        }
    )
    write_or_verify_receipt(plan_path, plan)
    with log_path.open("x", encoding="utf-8") as stream:
        process = subprocess.run(
            list(build_command(request)),
            cwd=request.dream_tac_root,
            env=_process_environment(request),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    evidence = checkpoint_evidence(request, log_path)
    validated = process.returncode == 0 and evidence["status"] == "passed"
    exit_code = 0 if validated else (process.returncode or 2)
    result = signed_receipt(
        {
            "schema_version": 1,
            "status": "completed" if validated else "failed",
            "protocol_id": "dream_tac_hcu_optimizer_step_result_v1",
            "claim_boundary": "one_optimizer_step_only_not_model_quality",
            "request_sha256": request.request_sha256,
            "launch_plan_receipt_sha256": plan["receipt_sha256"],
            "process_returncode": process.returncode,
            "optimizer_step_completed": validated,
            "exit_code": exit_code,
            "log_path": str(log_path),
            "log_sha256": sha256_regular_file(log_path),
            "evidence": evidence,
        }
    )
    write_or_verify_receipt(result_path, result)
    return exit_code


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-command", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    request = HcuOptimizerStepRequest.load(args.request)
    if args.print_command:
        print(
            json.dumps(
                {
                    "command": list(build_command(request)),
                    "environment": explicit_environment(request),
                    "gates_checked": False,
                    "writes_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.preflight_only:
        path, receipt = write_optimizer_step_preflight_receipt(request)
        environment = probe_environment(request)
        environment_path = request.receipt_root / ENVIRONMENT_RECEIPT_NAME
        write_or_verify_receipt(environment_path, environment)
        print(
            json.dumps(
                {
                    "artifact_preflight": str(path),
                    "artifact_status": receipt["status"],
                    "environment_preflight": str(environment_path),
                    "environment_status": environment["status"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return execute(request)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_command",
    "build_resolved_experiment",
    "execute",
    "explicit_environment",
    "main",
    "probe_environment",
]
