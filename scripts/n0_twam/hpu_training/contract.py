"""Fail-closed contracts for source-pinned official N0-TWAM training."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

OFFICIAL_REPOSITORY_URL = "https://github.com/neoteai/N0-TWAM.git"
OFFICIAL_COMMIT = "cdd87b6a141667123ad2c25f452478afdb71e287"
ALLOWED_OFFICIAL_CHANGE = "n0_twam/configs/twam_posttrain_cfg.py"
PROVEN_DOCKER_IMAGE = "docker-registry-sh.tencentcloudcr.com/hg/hg_wm:v3.5"
PROVEN_DOCKER_IMAGE_ID = (
    "sha256:09822e7616a3f9794284341aee19e1dfd531c2d2cb58168a40b14fcf818fdf9a"
)
PROVEN_RCCL_PLUGIN_FILE = Path(
    "/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so.0.0.0"
)
PROVEN_RCCL_PLUGIN_SHA256 = (
    "20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"
)
RUNTIME_PYTHON_PATH = Path(
    "/mnt/data/task/n0_twam_track32_franka_20260810/"
    "runtime/venv_py310_24ec50d/bin/python"
)
RUNTIME_PYTHON_OVERLAYS = (
    Path(
        "/mnt/data/task/n0_twam_track32_franka_20260810/deploy/"
        "wa_track3_clearup_s21000_20260817/python-overlay"
    ),
    Path(
        "/mnt/data/task/n0_twam_track32_franka_xyzw_20260817_v1/"
        "runtime/lerobot_0_3_3_overlay_v4"
    ),
    Path(
        "/mnt/data/task/n0_twam_track31_retrain_vision_tactile_20260829_v1/"
        "runtime/python-overlay-official-cdd87b6-v1"
    ),
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
PROVEN_MAX_LATENT_FRAMES = 5


@dataclass(frozen=True)
class TrainingSpec:
    """Only the paths and bounded hyperparameters exposed by the wrapper."""

    dataset_path: Path
    base_model_path: Path
    released_checkpoint_path: Path
    empty_embedding_path: Path
    norm_stat_path: Path
    per_repo_norm_stat_path: Path
    latent_inventory_path: Path
    prompt: str
    max_latent_frames: int
    num_steps: int
    save_interval: int
    gradient_accumulation_steps: int
    batch_size: int
    load_worker: int
    learning_rate: float
    warmup_steps: int
    gc_interval: int
    seed: int


@dataclass(frozen=True)
class ClusterSpec:
    """Validated multi-node topology with 8 accelerator processes per node."""

    nodes: tuple[str, ...]
    ssh_user: str
    ssh_port: int
    master_addr: str
    master_port: int


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _absolute_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute: {value!r}")
    return path


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def load_training_spec(path: Path) -> TrainingSpec:
    """Load a train-only spec; validation-related knobs are not accepted."""
    payload = _load_object(path, label="training spec")
    allowed = {
        "dataset_path",
        "base_model_path",
        "released_checkpoint_path",
        "empty_embedding_path",
        "norm_stat_path",
        "per_repo_norm_stat_path",
        "latent_inventory_path",
        "prompt",
        "max_latent_frames",
        "num_steps",
        "save_interval",
        "gradient_accumulation_steps",
        "batch_size",
        "load_worker",
        "learning_rate",
        "warmup_steps",
        "gc_interval",
        "seed",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported training spec keys: {unknown}")

    dataset_path = _absolute_path(payload.get("dataset_path"), field="dataset_path")
    if dataset_path.name != "train759" or "frozen40" in dataset_path.parts:
        raise ValueError("dataset_path must point exactly at a train759 directory")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    per_repo_path = _absolute_path(
        payload.get("per_repo_norm_stat_path"), field="per_repo_norm_stat_path"
    )
    learning_rate = payload.get("learning_rate", 1e-4)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or float(learning_rate) <= 0.0
    ):
        raise ValueError("learning_rate must be positive")

    base_model_path = _absolute_path(
        payload.get("base_model_path"), field="base_model_path"
    )
    released_checkpoint_path = _absolute_path(
        payload.get("released_checkpoint_path"),
        field="released_checkpoint_path",
    )
    if released_checkpoint_path != base_model_path:
        raise ValueError(
            "released_checkpoint_path must equal base_model_path so training starts "
            "from the official pretrained base, not a UniVTAC post-trained model"
        )

    max_latent_frames = _positive_int(
        payload.get("max_latent_frames", PROVEN_MAX_LATENT_FRAMES),
        field="max_latent_frames",
    )
    if max_latent_frames != PROVEN_MAX_LATENT_FRAMES:
        raise ValueError("the source-pinned 2-node recipe requires max_latent_frames=5")

    return TrainingSpec(
        dataset_path=dataset_path,
        base_model_path=base_model_path,
        released_checkpoint_path=released_checkpoint_path,
        empty_embedding_path=_absolute_path(
            payload.get("empty_embedding_path"), field="empty_embedding_path"
        ),
        norm_stat_path=_absolute_path(
            payload.get("norm_stat_path"), field="norm_stat_path"
        ),
        per_repo_norm_stat_path=per_repo_path,
        latent_inventory_path=_absolute_path(
            payload.get("latent_inventory_path"), field="latent_inventory_path"
        ),
        prompt=prompt.strip(),
        max_latent_frames=max_latent_frames,
        num_steps=_positive_int(payload.get("num_steps", 10_000), field="num_steps"),
        save_interval=_positive_int(
            payload.get("save_interval", 500), field="save_interval"
        ),
        gradient_accumulation_steps=_positive_int(
            payload.get("gradient_accumulation_steps", 4),
            field="gradient_accumulation_steps",
        ),
        batch_size=_positive_int(payload.get("batch_size", 1), field="batch_size"),
        load_worker=_nonnegative_int(
            payload.get("load_worker", 4), field="load_worker"
        ),
        learning_rate=float(learning_rate),
        warmup_steps=_nonnegative_int(
            payload.get("warmup_steps", 20), field="warmup_steps"
        ),
        gc_interval=_positive_int(payload.get("gc_interval", 50), field="gc_interval"),
        seed=_nonnegative_int(payload.get("seed", 20260829), field="seed"),
    )


def load_cluster_spec(path: Path) -> ClusterSpec:
    """Load an explicit two-node topology with eight processes per node."""
    payload = _load_object(path, label="cluster spec")
    allowed = {
        "nodes",
        "ssh_user",
        "ssh_port",
        "master_addr",
        "master_port",
        "processes_per_node",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported cluster spec keys: {unknown}")
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) != 2:
        raise ValueError("cluster must explicitly contain exactly two nodes")
    parsed_nodes: list[str] = []
    for value in raw_nodes:
        if not isinstance(value, str):
            raise ValueError("cluster node addresses must be strings")
        parsed_nodes.append(str(ipaddress.ip_address(value)))
    nodes = tuple(parsed_nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError("cluster nodes must be unique")
    if payload.get("processes_per_node", 8) != 8:
        raise ValueError("processes_per_node is fixed at 8")
    ssh_user = payload.get("ssh_user", "root")
    if not isinstance(ssh_user, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_-]*", ssh_user
    ):
        raise ValueError("ssh_user has an invalid format")
    raw_master_addr = payload.get("master_addr", nodes[0])
    if not isinstance(raw_master_addr, str):
        raise ValueError("master_addr must be a string IP address")
    master_addr = str(ipaddress.ip_address(raw_master_addr))
    if master_addr not in nodes:
        raise ValueError("master_addr must be one of the cluster nodes")
    return ClusterSpec(
        nodes=nodes,
        ssh_user=ssh_user,
        ssh_port=_positive_int(payload.get("ssh_port", 36000), field="ssh_port"),
        master_addr=master_addr,
        master_port=_positive_int(
            payload.get("master_port", 29620), field="master_port"
        ),
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\r\n")


def verify_official_checkout(repo: Path) -> dict[str, object]:
    """Prove the checkout is pinned and only the training config may differ."""
    if not repo.is_dir() or not (repo / ".git").exists():
        raise ValueError(f"not a Git checkout: {repo}")
    head = _git(repo, "rev-parse", "HEAD")
    if head != OFFICIAL_COMMIT:
        raise ValueError(f"official checkout HEAD mismatch: {head}")
    origin = _git(repo, "remote", "get-url", "origin")
    if origin.rstrip("/") != OFFICIAL_REPOSITORY_URL.rstrip("/"):
        raise ValueError(f"official checkout origin mismatch: {origin}")
    status_lines = tuple(
        line
        for line in _git(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    )
    unexpected = [line for line in status_lines if line[3:] != ALLOWED_OFFICIAL_CHANGE]
    if unexpected:
        raise ValueError(f"official checkout has forbidden changes: {unexpected}")
    return {
        "repository_url": origin,
        "commit": head,
        "allowed_change": ALLOWED_OFFICIAL_CHANGE,
        "status": list(status_lines),
    }


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    return run_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_paths_exist(paths: Sequence[tuple[str, Path]]) -> None:
    missing = [f"{label}={path}" for label, path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required paths: " + ", ".join(missing))


def json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
