"""Strict parsing for the packaged UniVTAC task registry."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Tuple, cast

from robotactile_benchmark.adapters.univtac import (
    DepthPhaseTracker,
    UniVTACAliasManifest,
)
from robotactile_benchmark.backends.univtac_contracts import (
    REGISTRY_ID,
    REGISTRY_RESOURCE_SHA256,
    REGISTRY_SEMANTIC_VERSION,
    UPSTREAM_COMMIT,
    UniVTACBackendConfig,
    UniVTACContractError,
    UniVTACTaskRegistry,
    UniVTACTaskSpec,
)
from robotactile_benchmark.contracts import canonical_hash

_TASK_KEYS = frozenset(
    {
        "task_id",
        "prompt_id",
        "prompt",
        "module_name",
        "class_name",
        "success_predicate_id",
        "task_source_sha256",
        "action_horizon",
        "early_stop_capable",
        "predicate_note",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "action_spec",
        "action_mode",
        "force",
        "sim_hz",
        "decimation",
        "physics_steps_per_action",
        "canonical_joint_names",
        "action_lower_bounds",
        "action_upper_bounds",
        "head_shape",
        "wrist_shape",
        "tactile_rgb_shape",
        "tactile_depth_shape",
    }
)
_ALIAS_KEYS = frozenset(
    {
        "head_camera",
        "wrist_camera",
        "left_tactile",
        "right_tactile",
        "left_physical_source_id",
        "right_physical_source_id",
        "sensor_type",
        "far_plane_mm",
        "calibration_id",
        "calibration_config_sha256",
        "tactile_payload",
    }
)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        raise UniVTACContractError(f"{name} fields do not match the frozen schema")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UniVTACContractError(f"{name} must be a positive integer")
    return int(value)


def _shape(value: Any, name: str, rank: int) -> Tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise UniVTACContractError(f"{name} must be an integer sequence")
    result = tuple(_positive_int(item, name) for item in value)
    if len(result) != rank:
        raise UniVTACContractError(f"{name} must have rank {rank}")
    return result


def _float_tuple(value: Any, name: str, length: int) -> Tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise UniVTACContractError(f"{name} must be a numeric sequence")
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise UniVTACContractError(f"{name} must contain {length} finite values")
    return result


def _resource_bytes() -> bytes:
    packaged = resources.files("robotactile_benchmark").joinpath(
        "configs/univtac/tasks_v1.json"
    )
    if packaged.is_file():
        return packaged.read_bytes()
    return (
        Path(__file__).resolve().parents[3] / "configs/univtac/tasks_v1.json"
    ).read_bytes()


def load_registry(
    document: Optional[Mapping[str, Any]] = None,
) -> UniVTACTaskRegistry:
    """Load and strictly validate the source-bound v1 task registry."""

    if document is None:
        payload = _resource_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != REGISTRY_RESOURCE_SHA256:
            raise UniVTACContractError("packaged UniVTAC registry hash mismatch")
        parsed = json.loads(payload.decode("utf-8"))
    else:
        parsed = dict(document)
        digest = canonical_hash(parsed)
    if not isinstance(parsed, Mapping):
        raise UniVTACContractError("registry must be a JSON object")
    _exact_keys(
        parsed,
        frozenset(
            {
                "registry_id",
                "semantic_version",
                "upstream_repository",
                "upstream_commit",
                "runtime",
                "aliases",
                "phase_tracker",
                "tasks",
            }
        ),
        "registry",
    )
    runtime = cast(Mapping[str, Any], parsed["runtime"])
    aliases = cast(Mapping[str, Any], parsed["aliases"])
    phase = cast(Mapping[str, Any], parsed["phase_tracker"])
    _exact_keys(runtime, _RUNTIME_KEYS, "runtime")
    _exact_keys(aliases, _ALIAS_KEYS, "aliases")
    _exact_keys(
        phase,
        frozenset({"on_threshold_mm", "off_threshold_mm"}),
        "phase tracker",
    )
    raw_tasks = parsed["tasks"]
    if not isinstance(raw_tasks, list):
        raise UniVTACContractError("tasks must be a list")
    tasks = []
    for raw_task in raw_tasks:
        if not isinstance(raw_task, Mapping):
            raise UniVTACContractError("task entries must be objects")
        _exact_keys(raw_task, _TASK_KEYS, "task")
        tasks.append(UniVTACTaskSpec(**raw_task))
    if len(tasks) != 8 or len({task.task_id for task in tasks}) != 8:
        raise UniVTACContractError("registry must contain exactly eight unique tasks")
    if (
        parsed["registry_id"] != REGISTRY_ID
        or parsed["semantic_version"] != REGISTRY_SEMANTIC_VERSION
    ):
        raise UniVTACContractError("registry identity/version mismatch")
    if parsed["upstream_commit"] != UPSTREAM_COMMIT:
        raise UniVTACContractError("registry upstream commit mismatch")
    return UniVTACTaskRegistry(
        registry_id=cast(str, parsed["registry_id"]),
        semantic_version=cast(str, parsed["semantic_version"]),
        upstream_repository=cast(str, parsed["upstream_repository"]),
        upstream_commit=cast(str, parsed["upstream_commit"]),
        resource_sha256=digest,
        runtime=dict(runtime),
        aliases=dict(aliases),
        phase_tracker=dict(phase),
        tasks=tuple(tasks),
    )


def build_config(task_id: str) -> UniVTACBackendConfig:
    """Build one typed backend config from the strict packaged registry."""

    registry = load_registry()
    runtime = registry.runtime
    return UniVTACBackendConfig(
        task=registry.task(task_id),
        registry_resource_sha256=registry.resource_sha256,
        upstream_commit=registry.upstream_commit,
        action_spec=cast(str, runtime["action_spec"]),
        action_mode=cast(str, runtime["action_mode"]),
        force=cast(bool, runtime["force"]),
        sim_hz=_positive_int(runtime["sim_hz"], "sim_hz"),
        decimation=_positive_int(runtime["decimation"], "decimation"),
        physics_steps_per_action=_positive_int(
            runtime["physics_steps_per_action"], "physics_steps_per_action"
        ),
        canonical_joint_names=tuple(
            cast(Sequence[str], runtime["canonical_joint_names"])
        ),
        action_lower_bounds=_float_tuple(
            runtime["action_lower_bounds"], "action lower bounds", 8
        ),
        action_upper_bounds=_float_tuple(
            runtime["action_upper_bounds"], "action upper bounds", 8
        ),
        head_shape=_shape(runtime["head_shape"], "head shape", 3),
        wrist_shape=_shape(runtime["wrist_shape"], "wrist shape", 3),
        tactile_rgb_shape=_shape(runtime["tactile_rgb_shape"], "tactile RGB shape", 3),
        tactile_depth_shape=_shape(
            runtime["tactile_depth_shape"], "tactile depth shape", 2
        ),
        aliases=UniVTACAliasManifest(**dict(registry.aliases)),
        phase_tracker=DepthPhaseTracker(**dict(registry.phase_tracker)),
    )
