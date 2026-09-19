"""Fail-closed NVIDIA/CUDA and artifact gates for Dream-Tac P2/P3."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from .identity import (
    canonical_json_sha256,
    load_json_object,
    receipt_sha256,
    signed_payload,
    write_or_verify_json,
)
from .statistics import (
    validate_dataset_statistics,
    validate_post_normalization_statistics,
)
from .training_constants import (
    DATASET_DIR_NAME,
    DATASET_POST_NORM_STATS_NAME,
    DATASET_STATS_NAME,
    GLOBAL_RECEIPT_NAME,
)
from .training_request import (
    ACCELERATOR_CONTRACT,
    DreamTacTrainingRequest,
    executable_path,
    sha256_regular_file,
    verify_bound_file,
)

PREFLIGHT_RECEIPT_NAME: Final[str] = "preflight_receipt.json"
SOURCE_MANIFEST_NAME: Final[str] = "source_split_manifest.json"
T5_RECEIPT_NAME: Final[str] = "t5_cache_receipt.json"
PROMPT_MANIFEST_NAME: Final[str] = "prompt_manifest.json"
T5_REQUEST_NAME: Final[str] = "t5_cache_request.json"
EnvironmentProbe = Callable[[DreamTacTrainingRequest], Mapping[str, object]]


def _run_text(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None
) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_checkout(request: DreamTacTrainingRequest) -> dict[str, object]:
    root = request.dream_tac_root.resolve(strict=True)
    if not root.is_dir() or not (root / ".git").is_dir():
        raise ValueError("dream_tac_root is not a Git checkout")
    commit = _run_text(("git", "rev-parse", "HEAD"), cwd=root)
    if commit != request.to_dict()["dream_tac_commit"]:
        raise ValueError("Dream-Tac checkout commit mismatch")
    dirty = _run_text(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=root
    )
    if dirty:
        raise ValueError("Dream-Tac checkout must be clean for source-bound training")
    config_path = root / "cosmos_policy" / "config" / "config.py"
    train_path = root / "cosmos_policy" / "scripts" / "train.py"
    for path in (config_path, train_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"pinned Dream-Tac entrypoint is absent: {path}")
    return {
        "commit": commit,
        "config_path": str(config_path),
        "config_sha256": sha256_regular_file(config_path),
        "train_entrypoint": str(train_path),
        "train_entrypoint_sha256": sha256_regular_file(train_path),
        "working_tree": "clean",
    }


def _manifest_identity(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    claimed = unsigned.pop("manifest_sha256", None)
    actual = canonical_json_sha256(unsigned)
    if claimed != actual:
        raise ValueError("Dream-Tac source manifest digest is invalid")
    return actual


def _data_artifacts(request: DreamTacTrainingRequest) -> dict[str, object]:
    root = request.materialization_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("materialization_root is not a directory")
    source = load_json_object(root / SOURCE_MANIFEST_NAME)
    source_sha = _manifest_identity(source)
    if source_sha != request.source_manifest_sha256:
        raise ValueError("Dream-Tac source manifest identity mismatch")
    receipt = load_json_object(root / GLOBAL_RECEIPT_NAME)
    receipt_sha = receipt_sha256(receipt, field="receipt_sha256")
    if receipt_sha != request.materialization_receipt_sha256:
        raise ValueError("Dream-Tac materialization receipt identity mismatch")
    expected = {
        "status": "complete",
        "source_manifest_sha256": source_sha,
        "materialized_episode_count": 759,
        "training_ready": False,
        "t5_cache_status": "external_generation_required",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Dream-Tac train759 receipt contract mismatch")
    dataset_root = root / DATASET_DIR_NAME
    train_root = dataset_root / "train"
    if not train_root.is_dir() or (dataset_root / "val").exists():
        raise ValueError("Dream-Tac dataset must expose train759 only")
    statistics_path = dataset_root / DATASET_STATS_NAME
    statistics_sha = sha256_regular_file(statistics_path)
    if statistics_sha != receipt.get("dataset_statistics_sha256"):
        raise ValueError("Dream-Tac dataset statistics identity mismatch")
    statistics = validate_dataset_statistics(load_json_object(statistics_path))
    post_statistics_path = dataset_root / DATASET_POST_NORM_STATS_NAME
    post_statistics_sha = sha256_regular_file(post_statistics_path)
    validate_post_normalization_statistics(
        statistics, load_json_object(post_statistics_path)
    )
    prompts = load_json_object(root / PROMPT_MANIFEST_NAME)
    prompt_sha = receipt_sha256(prompts, field="prompt_manifest_sha256")
    cache_keys = prompts.get("t5_cache_keys")
    if (
        prompts.get("source_manifest_sha256") != source_sha
        or prompts.get("task_count") != 8
        or prompts.get("prompt_count") != 8
        or not isinstance(cache_keys, list)
        or len(cache_keys) != 8
        or len(set(cache_keys)) != 8
        or any(not isinstance(item, str) or not item for item in cache_keys)
    ):
        raise ValueError("Dream-Tac prompt manifest contract mismatch")
    t5_request = load_json_object(root / T5_REQUEST_NAME)
    t5_request_sha = receipt_sha256(t5_request, field="t5_request_sha256")
    if (
        t5_request.get("dream_tac_commit") != request.to_dict()["dream_tac_commit"]
        or t5_request.get("prompt_manifest_sha256") != prompt_sha
        or t5_request.get("expected_cache_keys") != cache_keys
    ):
        raise ValueError("Dream-Tac T5 request contract mismatch")
    t5_receipt = load_json_object(root / T5_RECEIPT_NAME)
    receipt_sha256(t5_receipt, field="t5_cache_receipt_sha256")
    t5_path = dataset_root / "t5_embeddings.pkl"
    t5_actual_sha = sha256_regular_file(t5_path)
    t5_cache_sha = t5_receipt.get("cache_sha256")
    t5_expected = {
        "status": "complete",
        "dream_tac_commit": request.to_dict()["dream_tac_commit"],
        "prompt_manifest_sha256": prompt_sha,
        "t5_request_sha256": t5_request_sha,
        "cache_sha256": t5_actual_sha,
        "cache_keys": cache_keys,
    }
    if any(t5_receipt.get(key) != value for key, value in t5_expected.items()):
        raise ValueError("Dream-Tac T5 receipt contract mismatch")
    if t5_cache_sha != request.t5_cache_sha256:
        raise ValueError("Dream-Tac T5 cache identity mismatch")
    return {
        "dataset_root": str(dataset_root),
        "dataset_statistics_sha256": statistics_sha,
        "dataset_statistics_post_norm_sha256": post_statistics_sha,
        "materialization_receipt_sha256": receipt_sha,
        "materialized_episode_count": 759,
        "source_manifest_sha256": source_sha,
        "t5_cache_sha256": t5_cache_sha,
        "t5_request_sha256": t5_request_sha,
        "t5_receipt_sha256": t5_receipt.get("t5_cache_receipt_sha256"),
    }


def _resume_identity(request: DreamTacTrainingRequest) -> dict[str, object]:
    marker = request.latest_marker
    bound = request.resume_checkpoint
    if marker.exists():
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("Dream-Tac latest checkpoint marker is not regular")
        checkpoint_name = marker.read_text(encoding="utf-8").strip()
        if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError("Dream-Tac latest checkpoint marker is unsafe")
        latest = marker.parent / checkpoint_name
        if bound is None:
            raise ValueError(
                "existing latest checkpoint requires an explicit resume_checkpoint"
            )
        requested = verify_bound_file(bound, "resume checkpoint")
        if requested != latest.resolve(strict=True):
            raise ValueError(
                "resume_checkpoint must match the same-job latest checkpoint"
            )
        return {
            "mode": "same_job_latest",
            "checkpoint_path": str(requested),
            "checkpoint_sha256": bound.sha256,
            "latest_marker": str(marker),
            "latest_marker_sha256": sha256_regular_file(marker),
        }
    if bound is None:
        return {"mode": "new_run", "latest_marker": str(marker)}
    checkpoint = verify_bound_file(bound, "resume checkpoint")
    return {
        "mode": "explicit_checkpoint",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": bound.sha256,
        "latest_marker": str(marker),
    }


def _python_probe_script(expected_devices: int) -> str:
    return "\n".join(
        (
            "import importlib.util,json,platform,sys,torch",
            "assert platform.system() == 'Linux', 'Dream-Tac training requires Linux'",
            "assert sys.version_info[:2] == (3, 10), 'Dream-Tac runtime must use Python 3.10'",
            "required=('cosmos_policy','flash_attn','transformer_engine','natten','xformers','h5py','cv2','transformers')",
            "missing=[name for name in required if importlib.util.find_spec(name) is None]",
            "assert not missing, f'missing Dream-Tac modules: {missing}'",
            "assert torch.version.cuda is not None, 'PyTorch is not a CUDA build'",
            "assert torch.cuda.is_available(), 'CUDA is unavailable'",
            f"assert torch.cuda.device_count() == {expected_devices}, 'visible CUDA device count mismatch'",
            "payload={'platform':platform.platform(),'python_version':platform.python_version(),'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,'cuda_device_count':torch.cuda.device_count(),'cuda_devices':[{'index':i,'name':torch.cuda.get_device_name(i),'capability':list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]}",
            "print(json.dumps(payload,sort_keys=True))",
        )
    )


def probe_nvidia_environment(
    request: DreamTacTrainingRequest,
) -> Mapping[str, object]:
    """Require Linux, Python 3.10, NVIDIA CUDA, and the upstream dependency set."""

    python = executable_path(request.python_executable)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise ValueError("nvidia-smi is required; HCU/HPU is not supported")
    root = request.dream_tac_root.resolve(strict=True)
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not existing else f"{root}:{existing}"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(device) for device in request.cuda_devices
    )
    raw = _run_text(
        (str(python), "-c", _python_probe_script(request.nproc_per_node)),
        cwd=root,
        env=env,
    )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Dream-Tac Python environment probe returned invalid JSON")
    gpu_query = _run_text(
        (
            nvidia_smi,
            "-i",
            ",".join(str(device) for device in request.cuda_devices),
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        cwd=root,
        env=env,
    )
    lines = [line.strip() for line in gpu_query.splitlines() if line.strip()]
    if len(lines) != request.nproc_per_node:
        raise ValueError("nvidia-smi did not identify every requested GPU")
    return {
        **payload,
        "accelerator_contract": ACCELERATOR_CONTRACT,
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "nvidia_smi": lines,
        "python_executable": str(python),
    }


def build_preflight_receipt(
    request: DreamTacTrainingRequest,
    *,
    environment_probe: EnvironmentProbe = probe_nvidia_environment,
) -> dict[str, object]:
    """Run all immutable gates and return a deterministic signed receipt."""

    source = _source_checkout(request)
    data = _data_artifacts(request)
    base = verify_bound_file(request.base_checkpoint, "base checkpoint")
    resume = _resume_identity(request)
    environment = dict(environment_probe(request))
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "protocol_id": "dream_tac_univtac_training_preflight_v1",
        "request_sha256": request.request_sha256,
        "phase": request.phase,
        "run_name": request.run_name,
        "accelerator_contract": ACCELERATOR_CONTRACT,
        "source": source,
        "data": data,
        "base_checkpoint": {
            "path": str(base),
            "sha256": request.base_checkpoint.sha256,
        },
        "resume": resume,
        "environment": environment,
    }
    return signed_payload(unsigned, field="preflight_receipt_sha256")


def write_preflight_receipt(
    request: DreamTacTrainingRequest,
    *,
    environment_probe: EnvironmentProbe = probe_nvidia_environment,
) -> tuple[Path, dict[str, object]]:
    receipt = build_preflight_receipt(request, environment_probe=environment_probe)
    path = request.receipt_root / PREFLIGHT_RECEIPT_NAME
    write_or_verify_json(path, receipt)
    return path, receipt


__all__ = [
    "PREFLIGHT_RECEIPT_NAME",
    "build_preflight_receipt",
    "probe_nvidia_environment",
    "write_preflight_receipt",
]
