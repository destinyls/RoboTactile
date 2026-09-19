"""Persistent N0 + Isaac execution for one task/condition across many seeds.

The supervisor starts one task-bound N0-TWAM server and one Isaac application.
The Isaac worker then reconstructs a fresh UniVTAC task, policy object, and fault
delivery session for every seed.  This is the condition-sharded execution unit
used to map Clean and the 14 failure conditions onto independent GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Optional, cast

from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
from robotactile_benchmark.constants import operator_requires_rest_reference
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.same_task_worker_runtime import (
    SameTaskIsaacSession,
)
from robotactile_benchmark.integrations.n0_twam.retrained import (
    load_retrained,
    read_object,
)
from robotactile_benchmark.policies.tactile_availability import (
    TactileAvailabilityMode,
)
from scripts.retrained_evaluation.campaign import (
    build_process_environments,
    resolve_isaac_python,
    stop_owned,
    wait_ready,
)
from scripts.retrained_evaluation.group import (
    AVAILABILITY_OPERATORS,
    FAULT_WINDOW_MODES,
    OPERATORS,
    SUPPORTED_REGISTRIES,
    policy_factory,
    prepare_group,
    write_json,
)

SCHEMA = "robotactile-persistent-condition-work-v1"
SESSION_SCHEMA = "robotactile-persistent-condition-session-v1"
EPISODE_SCHEMA = "robotactile-persistent-condition-episode-v1"
EVIDENCE_MODES = ("diagnostic", "statistics", "claim")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or Path(value).name != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be a path-free non-empty identifier")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _relative_path(value: object, name: str) -> str:
    if type(value) is not str or "\\" in value:
        raise ValueError(f"{name} must be a POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a safe relative path")
    return value


def _member(root: Path, relative: str, name: str) -> Path:
    candidate = (root / _relative_path(relative, name)).absolute()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escaped the campaign root") from error
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} cannot traverse a symlink")
    return candidate


@dataclass(frozen=True)
class ConditionWorkItem:
    """One immutable seed/request dispatch inside a condition worker."""

    sequence: int
    seed: int
    group_relpath: str
    request_relpath: str
    request_index: int
    request_file_sha256: str
    trial_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in ("sequence", "seed", "request_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _relative_path(self.group_relpath, "group_relpath")
        _relative_path(self.request_relpath, "request_relpath")
        _require_sha256(self.request_file_sha256, "request_file_sha256")
        _require_sha256(self.trial_manifest_sha256, "trial_manifest_sha256")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> "ConditionWorkItem":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("condition work item fields mismatch")
        return cls(**cast(dict[str, Any], dict(value)))


@dataclass(frozen=True)
class ConditionWorkManifest:
    """Hash-bound work assigned to one persistent task/condition worker."""

    task_id: str
    condition_id: str
    capture_profile: str
    evidence_mode: str
    binding_relpath: str
    binding_file_sha256: str
    model_identity_sha256: str
    reset_equivalence_receipt: Optional[str]
    reset_equivalence_receipt_sha256: Optional[str]
    items: tuple[ConditionWorkItem, ...]
    content_sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.condition_id, "condition_id")
        LiveCaptureProfile(self.capture_profile)
        if self.evidence_mode not in EVIDENCE_MODES:
            raise ValueError("unsupported persistent evidence mode")
        _relative_path(self.binding_relpath, "binding_relpath")
        _require_sha256(self.binding_file_sha256, "binding_file_sha256")
        _require_sha256(self.model_identity_sha256, "model_identity_sha256")
        if (self.reset_equivalence_receipt is None) != (
            self.reset_equivalence_receipt_sha256 is None
        ):
            raise ValueError("reset-equivalence receipt identity is incomplete")
        if self.reset_equivalence_receipt is not None:
            if (
                type(self.reset_equivalence_receipt) is not str
                or not self.reset_equivalence_receipt
                or not Path(self.reset_equivalence_receipt).is_absolute()
            ):
                raise ValueError("reset_equivalence_receipt must be an absolute path")
            _require_sha256(
                self.reset_equivalence_receipt_sha256,
                "reset_equivalence_receipt_sha256",
            )
        if self.evidence_mode != "diagnostic" and (
            self.reset_equivalence_receipt is None
        ):
            raise ValueError(
                "statistics/claim execution requires reset-equivalence evidence"
            )
        if (
            self.evidence_mode == "claim"
            and LiveCaptureProfile(self.capture_profile)
            is not LiveCaptureProfile.PAPER_FULL
        ):
            raise ValueError("claim execution requires paper_full_v1 capture")
        items = tuple(self.items)
        if not items:
            raise ValueError("persistent condition work cannot be empty")
        if tuple(item.sequence for item in items) != tuple(range(len(items))):
            raise ValueError("work item sequence must be contiguous")
        if len({item.seed for item in items}) != len(items):
            raise ValueError("persistent condition seeds must be unique")
        object.__setattr__(self, "items", items)
        if self.schema != SCHEMA:
            raise ValueError("persistent condition work schema mismatch")
        _require_sha256(self.content_sha256, "content_sha256")
        if self.content_sha256 != canonical_hash(self._content_dict()):
            raise ValueError("persistent condition work content hash mismatch")

    def _content_dict(self) -> dict[str, object]:
        return {
            "binding_file_sha256": self.binding_file_sha256,
            "binding_relpath": self.binding_relpath,
            "capture_profile": self.capture_profile,
            "condition_id": self.condition_id,
            "evidence_mode": self.evidence_mode,
            "items": [item.to_dict() for item in self.items],
            "model_identity_sha256": self.model_identity_sha256,
            "reset_equivalence_receipt": self.reset_equivalence_receipt,
            "reset_equivalence_receipt_sha256": (self.reset_equivalence_receipt_sha256),
            "schema": self.schema,
            "task_id": self.task_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._content_dict(), "content_sha256": self.content_sha256}

    @classmethod
    def build(cls, **values: object) -> "ConditionWorkManifest":
        items = cast(tuple[ConditionWorkItem, ...], values["items"])
        content = {
            **values,
            "items": [item.to_dict() for item in items],
            "schema": SCHEMA,
        }
        return cls(
            **cast(dict[str, Any], values),
            schema=SCHEMA,
            content_sha256=canonical_hash(content),
        )

    @classmethod
    def from_dict(cls, value: object) -> "ConditionWorkManifest":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("persistent condition work fields mismatch")
        parsed = dict(value)
        raw_items = parsed["items"]
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
            raise TypeError("persistent condition items must be a sequence")
        parsed["items"] = tuple(ConditionWorkItem.from_dict(item) for item in raw_items)
        return cls(**cast(dict[str, Any], parsed))


def _validate_reset_equivalence(path: Path, task_id: str) -> str:
    selected = Path(path).absolute()
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("reset-equivalence receipt must be a regular file")
    receipt = read_object(selected)
    declared_content_sha256 = receipt.pop("content_sha256", None)
    if declared_content_sha256 != canonical_hash(receipt):
        raise ValueError("reset-equivalence receipt content hash mismatch")
    if (
        receipt.get("evidence_level") != "live_univtac_same_app_fresh_runtime_probe_v1"
        or receipt.get("status") != "passed"
        or receipt.get("task_id") != task_id
        or receipt.get("same_process_confirmed") is not True
        or receipt.get("app_close_status") not in {"returned", "system_exit_zero"}
    ):
        raise ValueError("reset-equivalence receipt is not a passed task proof")
    runtime_count = receipt.get("runtime_count")
    if type(runtime_count) is not int or runtime_count < 2:
        raise ValueError("reset-equivalence proof must cover at least two runtimes")
    return _sha256_file(selected)


def _model_identity_sha256(
    binding: Mapping[str, Any], artifact: Mapping[str, Any], task_id: str
) -> str:
    tasks = artifact.get("tasks")
    if not isinstance(tasks, Mapping) or not isinstance(tasks.get(task_id), Mapping):
        raise ValueError("retrained artifact does not contain the selected task")
    task = cast(Mapping[str, Any], tasks[task_id])
    return canonical_hash(
        {
            "action_hz": artifact.get("action_hz"),
            "action_per_frame": artifact.get("action_per_frame"),
            "artifact_sha256": artifact.get("artifact_sha256"),
            "checkpoint_sha256": artifact.get("checkpoint_sha256"),
            "dataset_sha256": binding.get("dataset_sha256"),
            "model": binding.get("model"),
            "normalizer_sha256": task.get("normalizer_sha256"),
            "source_tree_sha256": artifact.get("source_tree_sha256"),
            "task_config_sha256": task.get("config_sha256"),
            "task_id": task_id,
        }
    )


def _select_request(group: Path, condition_id: str) -> tuple[int, str, Path]:
    plan = read_object(group / "group.json")
    ordered = plan.get("ordered_requests")
    if not isinstance(ordered, list):
        raise TypeError("group ordered_requests must be a list")
    matches = [
        (index, relative)
        for index, relative in enumerate(ordered)
        if isinstance(relative, str) and Path(relative).stem == condition_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"condition {condition_id!r} matched {len(matches)} group requests"
        )
    index, relative = matches[0]
    return index, relative, _member(group, relative, "request")


def prepare_condition_work(
    *,
    binding_path: Path,
    task_id: str,
    condition_id: str,
    campaign: Path,
    seeds: Sequence[int],
    capture_profile: LiveCaptureProfile,
    evidence_mode: str,
    severity_registry: str,
    severity_level: int,
    fault_window_mode: str,
    fault_onset_max_index: int,
    reset_equivalence_receipt: Optional[Path],
    tactile_availability_mode: Optional[TactileAvailabilityMode] = None,
    tactile_zero_shape: Optional[tuple[int, int, int]] = None,
    port_override: Optional[int] = None,
) -> Path:
    """Freeze one no-clobber condition shard without launching GPU processes."""

    selected_campaign = Path(campaign).absolute()
    if selected_campaign.exists() or selected_campaign.is_symlink():
        raise FileExistsError(f"refusing to overwrite campaign: {selected_campaign}")
    task = _require_identifier(task_id, "task_id")
    condition = _require_identifier(condition_id, "condition_id")
    if condition not in ("clean", *OPERATORS, *AVAILABILITY_OPERATORS):
        raise ValueError("condition must be clean or one registered failure operator")
    seed_tuple = tuple(seeds)
    if (
        not seed_tuple
        or len(set(seed_tuple)) != len(seed_tuple)
        or any(type(seed) is not int or seed < 0 for seed in seed_tuple)
    ):
        raise ValueError("seeds must be unique non-negative integers")
    selected_profile = LiveCaptureProfile(capture_profile)
    if evidence_mode not in EVIDENCE_MODES:
        raise ValueError("unsupported evidence mode")
    if severity_registry not in SUPPORTED_REGISTRIES:
        raise ValueError("unsupported severity registry")
    if type(severity_level) is not int or severity_level not in range(1, 6):
        raise ValueError("severity level must be an integer from 1 to 5")
    if fault_window_mode not in FAULT_WINDOW_MODES:
        raise ValueError("unsupported fault window mode")
    if type(fault_onset_max_index) is not int or fault_onset_max_index < 1:
        raise ValueError("fault onset maximum must be a positive integer")

    binding = read_object(Path(binding_path).absolute())
    if binding.get("model") != "n0_twam":
        raise ValueError("persistent condition execution currently requires n0_twam")
    if port_override is not None:
        if type(port_override) is not int or not 1 <= port_override <= 65535:
            raise ValueError("N0 port override must be in [1, 65535]")
        binding["port"] = port_override
    tasks = binding.get("tasks")
    if not isinstance(tasks, Mapping) or task not in tasks:
        raise ValueError("task is absent from the N0 binding")
    evaluation = binding.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise TypeError("binding evaluation must be an object")
    selected_evaluation = dict(evaluation)
    selected_evaluation.update(
        capture_profile=selected_profile.value,
        severity_registries=[severity_registry],
        severity_level=severity_level,
        operators=[
            condition if condition != "clean" else "F5_contact_shape_distortion"
        ],
        fault_window_mode=fault_window_mode,
        measure_n0_rest=False,
    )
    selected_evaluation.pop("fault_start_index", None)
    if fault_window_mode == "early_random_onset_v1":
        selected_evaluation["fault_onset_max_index"] = fault_onset_max_index
    else:
        selected_evaluation.pop("fault_onset_max_index", None)
        selected_evaluation["fault_start_index"] = (
            0 if fault_window_mode == "full_episode_v1" else 20
        )
    if tactile_availability_mode is not None:
        selected_evaluation["tactile_availability_mode"] = (
            tactile_availability_mode.value
        )
    if tactile_zero_shape is not None:
        selected_evaluation["tactile_zero_shape"] = list(tactile_zero_shape)
    selected_mode = TactileAvailabilityMode(
        selected_evaluation.get("tactile_availability_mode", "required")
    )
    if condition in AVAILABILITY_OPERATORS and (
        selected_mode is TactileAvailabilityMode.REQUIRED
    ):
        raise ValueError(
            "A1/A2 require an explicit native_missing or zero_fill input contract"
        )
    rest_references = binding.get("rest_references", {})
    if not isinstance(rest_references, Mapping):
        raise TypeError("binding rest_references must be an object")
    if condition != "clean" and operator_requires_rest_reference(
        condition, severity_registry=severity_registry
    ):
        reference = rest_references.get(task)
        if not isinstance(reference, str) or not reference:
            raise ValueError(
                f"{condition} requires a precomputed task-bound rest reference"
            )

    reset_path: Optional[str] = None
    reset_sha256: Optional[str] = None
    if reset_equivalence_receipt is not None:
        proof = Path(reset_equivalence_receipt).absolute()
        reset_sha256 = _validate_reset_equivalence(proof, task)
        reset_path = str(proof)
    elif evidence_mode != "diagnostic":
        raise ValueError(
            "statistics/claim execution requires --reset-equivalence-receipt"
        )
    if evidence_mode == "claim" and selected_profile is not (
        LiveCaptureProfile.PAPER_FULL
    ):
        raise ValueError("claim execution requires paper_full_v1 capture")

    server_receipt = selected_campaign / "server/worker/server_receipt.json"
    binding["evaluation"] = selected_evaluation
    binding["server_receipt"] = str(server_receipt)
    binding.pop("binding_sha256", None)
    binding["binding_sha256"] = canonical_hash(binding)
    n0_artifact = load_retrained(Path(binding["artifact"]))
    selected_campaign.mkdir(parents=True)
    frozen_binding = selected_campaign / "binding.json"
    write_json(frozen_binding, binding)

    items: list[ConditionWorkItem] = []
    for sequence, seed in enumerate(seed_tuple):
        group = (
            selected_campaign / "seeds" / f"seed-{seed:06d}" / "groups/n0_twam" / task
        )
        prepare_group(
            binding,
            task,
            group,
            seed,
            n0_artifact=n0_artifact,
        )
        index, relative, request_path = _select_request(group, condition)
        request = load_live_univtac_request(request_path)
        loaded = load_live_univtac_run(request)
        if loaded.trial.initial_seed != seed or loaded.trial.task != task:
            raise ValueError("prepared request seed/task identity mismatch")
        items.append(
            ConditionWorkItem(
                sequence=sequence,
                seed=seed,
                group_relpath=group.relative_to(selected_campaign).as_posix(),
                request_relpath=relative,
                request_index=index,
                request_file_sha256=_sha256_file(request_path),
                trial_manifest_sha256=loaded.trial.sha256,
            )
        )
    manifest = ConditionWorkManifest.build(
        task_id=task,
        condition_id=condition,
        capture_profile=selected_profile.value,
        evidence_mode=evidence_mode,
        binding_relpath=frozen_binding.relative_to(selected_campaign).as_posix(),
        binding_file_sha256=_sha256_file(frozen_binding),
        model_identity_sha256=_model_identity_sha256(binding, n0_artifact, task),
        reset_equivalence_receipt=reset_path,
        reset_equivalence_receipt_sha256=reset_sha256,
        items=tuple(items),
    )
    output = selected_campaign / "work_manifest.json"
    write_json(output, manifest.to_dict())
    return output


def load_work_manifest(path: Path) -> ConditionWorkManifest:
    """Load and verify one frozen work manifest."""

    return ConditionWorkManifest.from_dict(read_object(path))


def _validate_item(
    campaign: Path,
    manifest: ConditionWorkManifest,
    item: ConditionWorkItem,
) -> tuple[Path, Path]:
    group = _member(campaign, item.group_relpath, "group")
    if group.is_symlink() or not group.is_dir():
        raise ValueError("work group must be a real directory")
    index, relative, request_path = _select_request(group, manifest.condition_id)
    if index != item.request_index or relative != item.request_relpath:
        raise ValueError("work item request selection changed after freeze")
    if _sha256_file(request_path) != item.request_file_sha256:
        raise ValueError("work item request file changed after freeze")
    loaded = load_live_univtac_run(load_live_univtac_request(request_path))
    if (
        loaded.trial.sha256 != item.trial_manifest_sha256
        or loaded.trial.task != manifest.task_id
        or loaded.trial.initial_seed != item.seed
    ):
        raise ValueError("work item trial identity changed after freeze")
    result_path = group / "results" / f"{item.request_index:02d}.json"
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError("condition result already exists; never repeat it")
    request = loaded.request
    if request.output_dir is None:
        raise ValueError("persistent condition request requires output_dir")
    if request.output_dir.exists() or request.output_dir.is_symlink():
        raise FileExistsError("condition artifact already exists; never replace it")
    return request_path, result_path


def execute_condition_work(campaign: Path, manifest_path: Path) -> dict[str, object]:
    """Run all frozen seeds in one Isaac application process."""

    root = Path(campaign).resolve(strict=True)
    work_path = Path(manifest_path).resolve(strict=True)
    if work_path.parent != root:
        raise ValueError("work manifest must live at the campaign root")
    manifest = load_work_manifest(work_path)
    binding_path = _member(root, manifest.binding_relpath, "binding")
    if _sha256_file(binding_path) != manifest.binding_file_sha256:
        raise ValueError("persistent binding changed after work freeze")
    binding = read_object(binding_path)
    if binding.get("model") != "n0_twam":
        raise ValueError("persistent worker binding must select n0_twam")
    if manifest.reset_equivalence_receipt is not None:
        proof = Path(manifest.reset_equivalence_receipt)
        if _validate_reset_equivalence(proof, manifest.task_id) != (
            manifest.reset_equivalence_receipt_sha256
        ):
            raise ValueError("reset-equivalence receipt changed after work freeze")

    validated = [_validate_item(root, manifest, item) for item in manifest.items]
    n0_artifact = load_retrained(Path(binding["artifact"]))
    n0_server_receipt = read_object(Path(binding["server_receipt"]))
    if _model_identity_sha256(binding, n0_artifact, manifest.task_id) != (
        manifest.model_identity_sha256
    ):
        raise ValueError("N0 model identity changed after work freeze")
    first_request = load_live_univtac_request(validated[0][0])
    bootstrap = load_live_univtac_run(first_request)
    session_id = f"persistent-condition-{uuid.uuid4().hex}"
    started_unix = time.time()
    started_monotonic = time.monotonic()
    capture_profile = LiveCaptureProfile(manifest.capture_profile)
    exporter = partial(write_live_univtac_artifact, capture_profile=capture_profile)
    session: Optional[SameTaskIsaacSession] = None
    dispatches: list[dict[str, object]] = []
    close_status = "not_started"

    def build_session_receipt(
        *, app_close_status: str, receipt_stage: str, runtime_count: int
    ) -> dict[str, object]:
        receipt: dict[str, object] = {
            "app_close_status": app_close_status,
            "capture_profile": capture_profile.value,
            "condition_id": manifest.condition_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "dispatches": dispatches,
            "evidence_mode": manifest.evidence_mode,
            "finished_unix": time.time(),
            "isaac_application_launch_count": 1,
            "model_identity_sha256": manifest.model_identity_sha256,
            "n0_port": binding["port"],
            "n0_endpoint_reset_before_observation": all(
                dispatch["policy_reset_completed"] is True for dispatch in dispatches
            ),
            "policy_instance_per_episode": True,
            "process_group_id": os.getpgrp(),
            "process_id": os.getpid(),
            "receipt_stage": receipt_stage,
            "runtime_count": runtime_count,
            "schema": SESSION_SCHEMA,
            "session_id": session_id,
            "started_unix": started_unix,
            "task_id": manifest.task_id,
            "task_runtime_per_episode": True,
            "total_duration_s": time.monotonic() - started_monotonic,
            "work_manifest_sha256": manifest.content_sha256,
        }
        receipt["content_sha256"] = canonical_hash(receipt)
        return receipt

    try:
        session = SameTaskIsaacSession(
            bootstrap=bootstrap,
            action_execution_contract=N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
        )
        for item, (request_path, result_path) in zip(manifest.items, validated):
            episode_started = time.monotonic()
            request = load_live_univtac_request(request_path)
            execution = session.execute(
                request,
                policy_factory=partial(
                    policy_factory,
                    binding,
                    n0_artifact=n0_artifact,
                    n0_server_receipt=n0_server_receipt,
                ),
                artifact_exporter=exporter,
            )
            assert request.output_dir is not None
            artifact = load_live_univtac_artifact(request.output_dir)
            if (
                artifact.evidence.result != execution.evidence.result
                or artifact.run_content_sha256 != execution.loaded.content_sha256
                or artifact.capture_profile is not capture_profile
            ):
                raise RuntimeError("persistent episode artifact cross-link mismatch")
            reset_witness = artifact.evidence.initial_diagnostics.get("reset_witness")
            reset_witness_sha256 = (
                canonical_hash(reset_witness)
                if isinstance(reset_witness, Mapping)
                else None
            )
            row = {
                **result_to_dict(artifact.evidence.result),
                "artifact": str(request.output_dir),
                "capture_profile": capture_profile.value,
                "condition": manifest.condition_id,
                "elapsed_episode_s": time.monotonic() - episode_started,
                "group": item.group_relpath,
                "model": "n0_twam",
                "persistent_session_id": session_id,
                "request": item.request_relpath,
                "root_receipt_sha256": artifact.root_receipt_sha256,
                "runtime_ordinal": session.runtime_count - 1,
                "seed": item.seed,
                "task": manifest.task_id,
            }
            write_json(result_path, row)
            episode_receipt = {
                "artifact_root_sha256": artifact.root_receipt_sha256,
                "condition_id": manifest.condition_id,
                "initial_state_sha256": artifact.evidence.result.initial_state_sha256,
                "policy_reset_completed": (
                    artifact.evidence.result.observation_count > 0
                ),
                "request_file_sha256": item.request_file_sha256,
                "request_index": item.request_index,
                "reset_viable": artifact.evidence.initial_diagnostics.get(
                    "reset_viable"
                ),
                "reset_witness_sha256": reset_witness_sha256,
                "runtime_ordinal": session.runtime_count - 1,
                "schema": EPISODE_SCHEMA,
                "seed": item.seed,
                "sequence": item.sequence,
                "session_id": session_id,
                "trial_manifest_sha256": item.trial_manifest_sha256,
            }
            episode_receipt["content_sha256"] = canonical_hash(episode_receipt)
            write_json(
                result_path.parent.parent / "persistent_condition_episode.json",
                episode_receipt,
            )
            dispatches.append(episode_receipt)
            print(
                json.dumps(
                    {
                        "condition": manifest.condition_id,
                        "elapsed_episode_s": row["elapsed_episode_s"],
                        "runtime_count": session.runtime_count,
                        "seed": item.seed,
                        "sequence": item.sequence,
                        "status": "episode_completed",
                        "success": row["score_success"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if (
                row.get("score_eligible") is not True
                or row.get("validation_passed") is not True
                or row.get("failure_stage") is not None
            ):
                raise RuntimeError(
                    "persistent episode produced invalid infrastructure/validation "
                    "evidence; refusing to consume later seeds"
                )
        runtime_count = session.runtime_count
        write_json(
            root / "persistent_condition_session.preclose.json",
            build_session_receipt(
                app_close_status="pending",
                receipt_stage="preclose",
                runtime_count=runtime_count,
            ),
        )
        try:
            session.close()
            close_status = "returned"
        except SystemExit as error:
            if error.code not in {None, 0}:
                raise
            close_status = "system_exit_zero"
        session = None
    except BaseException as error:
        if session is not None:
            with suppress(BaseException):
                session.close()
        failure = {
            "condition_id": manifest.condition_id,
            "dispatch_count": len(dispatches),
            "error_message": str(error),
            "error_type": type(error).__name__,
            "schema": "robotactile-persistent-condition-failure-v1",
            "session_id": session_id,
            "task_id": manifest.task_id,
            "unix_time": time.time(),
            "work_manifest_sha256": manifest.content_sha256,
        }
        failure["content_sha256"] = canonical_hash(failure)
        write_json(root / "worker_failure.json", failure)
        raise

    receipt = build_session_receipt(
        app_close_status=close_status,
        receipt_stage="final",
        runtime_count=runtime_count,
    )
    write_json(root / "persistent_condition_session.json", receipt)
    return receipt


def _summary(campaign: Path, manifest: ConditionWorkManifest) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for item in manifest.items:
        group = _member(campaign, item.group_relpath, "group")
        path = group / "results" / f"{item.request_index:02d}.json"
        if path.is_file():
            rows.append(read_object(path))
    eligible = [
        row
        for row in rows
        if row.get("score_eligible") is True
        and row.get("validation_passed") is True
        and row.get("failure_stage") is None
    ]
    success = sum(row.get("score_success") is True for row in eligible)
    return {
        "capture_profile": manifest.capture_profile,
        "completed_episode_count": len(rows),
        "condition_id": manifest.condition_id,
        "eligible_episode_count": len(eligible),
        "evidence_mode": manifest.evidence_mode,
        "invalid_episode_count": len(rows) - len(eligible),
        "missing_episode_count": len(manifest.items) - len(rows),
        "planned_episode_count": len(manifest.items),
        "schema": "robotactile-persistent-condition-summary-v1",
        "seeds": [item.seed for item in manifest.items],
        "success_count": success,
        "success_rate": success / len(eligible) if eligible else None,
        "task_id": manifest.task_id,
        "work_manifest_sha256": manifest.content_sha256,
    }


def _verified_hashed_document(path: Path, schema: str) -> dict[str, Any]:
    document = read_object(path)
    declared = document.pop("content_sha256", None)
    if document.get("schema") != schema or declared != canonical_hash(document):
        raise ValueError(f"invalid content-addressed document: {path}")
    document["content_sha256"] = declared
    return document


def aggregate_condition_shards(
    campaigns: Sequence[Path],
    output: Path,
    *,
    require_all_conditions: bool = True,
) -> dict[str, object]:
    """Aggregate seed-paired condition shards without pooling invalid episodes."""

    roots = tuple(Path(path).resolve(strict=True) for path in campaigns)
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("condition campaigns must be non-empty and unique")
    selected_output = Path(output).absolute()
    if selected_output.exists() or selected_output.is_symlink():
        raise FileExistsError("aggregate output must not already exist")
    manifests: dict[str, tuple[Path, ConditionWorkManifest]] = {}
    sessions: dict[str, dict[str, Any]] = {}
    for root in roots:
        manifest = load_work_manifest(root / "work_manifest.json")
        if manifest.condition_id in manifests:
            raise ValueError("duplicate condition campaign")
        session = _verified_hashed_document(
            root / "persistent_condition_session.json", SESSION_SCHEMA
        )
        if (
            session.get("task_id") != manifest.task_id
            or session.get("condition_id") != manifest.condition_id
            or session.get("work_manifest_sha256") != manifest.content_sha256
            or session.get("runtime_count") != len(manifest.items)
            or session.get("isaac_application_launch_count") != 1
            or session.get("model_identity_sha256") != manifest.model_identity_sha256
        ):
            raise ValueError("persistent session/work manifest identity mismatch")
        manifests[manifest.condition_id] = (root, manifest)
        sessions[manifest.condition_id] = session
    task_ids = {manifest.task_id for _, manifest in manifests.values()}
    if len(task_ids) != 1:
        raise ValueError("condition campaigns must share exactly one task")
    task_id = next(iter(task_ids))
    model_identities = {
        manifest.model_identity_sha256 for _, manifest in manifests.values()
    }
    if len(model_identities) != 1:
        raise ValueError("condition campaigns do not share one N0 model identity")
    model_identity_sha256 = next(iter(model_identities))
    expected_conditions = {"clean", *OPERATORS, *AVAILABILITY_OPERATORS}
    actual_conditions = set(manifests)
    if require_all_conditions and actual_conditions != expected_conditions:
        raise ValueError(
            "complete aggregation requires Clean and all 14 failure conditions"
        )
    seed_sets = {
        tuple(item.seed for item in manifest.items)
        for _, manifest in manifests.values()
    }
    if len(seed_sets) != 1:
        raise ValueError("condition campaigns do not share one ordered seed set")
    seeds = next(iter(seed_sets))

    episode_rows: dict[str, dict[int, dict[str, Any]]] = {}
    cells: list[dict[str, object]] = []
    for condition in sorted(
        actual_conditions, key=lambda value: (value != "clean", value)
    ):
        root, manifest = manifests[condition]
        rows: dict[int, dict[str, Any]] = {}
        for item in manifest.items:
            group = _member(root, item.group_relpath, "group")
            result_path = group / "results" / f"{item.request_index:02d}.json"
            if not result_path.is_file():
                continue
            row = read_object(result_path)
            if (
                row.get("task") != task_id
                or row.get("condition") != condition
                or row.get("seed") != item.seed
            ):
                raise ValueError("condition result identity mismatch")
            rows[item.seed] = row
        episode_rows[condition] = rows
        eligible = [
            row
            for row in rows.values()
            if row.get("score_eligible") is True
            and row.get("validation_passed") is True
            and row.get("failure_stage") is None
        ]
        success = sum(row.get("score_success") is True for row in eligible)
        cells.append(
            {
                "completed_episode_count": len(rows),
                "condition_id": condition,
                "eligible_episode_count": len(eligible),
                "invalid_episode_count": len(rows) - len(eligible),
                "missing_episode_count": len(seeds) - len(rows),
                "success_count": success,
                "success_rate": success / len(eligible) if eligible else None,
            }
        )

    clean_rows = episode_rows.get("clean", {})
    paired_effects: list[dict[str, object]] = []
    all_initial_states_match = True
    for condition in sorted(actual_conditions - {"clean"}):
        matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
        mismatch_seeds: list[int] = []
        for seed in seeds:
            clean = clean_rows.get(seed)
            fault = episode_rows[condition].get(seed)
            if clean is None or fault is None:
                continue
            same_initial_state = isinstance(
                clean.get("initial_state_sha256"), str
            ) and clean.get("initial_state_sha256") == fault.get("initial_state_sha256")
            if not same_initial_state:
                mismatch_seeds.append(seed)
                continue
            if all(
                row.get("score_eligible") is True
                and row.get("validation_passed") is True
                and row.get("failure_stage") is None
                for row in (clean, fault)
            ):
                matched.append((clean, fault))
        all_initial_states_match &= not mismatch_seeds
        clean_success = sum(clean.get("score_success") is True for clean, _ in matched)
        fault_success = sum(fault.get("score_success") is True for _, fault in matched)
        paired_count = len(matched)
        paired_effects.append(
            {
                "clean_success_count": clean_success,
                "condition_id": condition,
                "fault_induced_failure_count": sum(
                    clean.get("score_success") is True
                    and fault.get("score_success") is not True
                    for clean, fault in matched
                ),
                "fault_success_count": fault_success,
                "initial_state_mismatch_count": len(mismatch_seeds),
                "initial_state_mismatch_seeds": mismatch_seeds,
                "paired_episode_count": paired_count,
                "paired_success_rate_drop": (
                    (clean_success - fault_success) / paired_count
                    if paired_count
                    else None
                ),
            }
        )

    all_complete = all(
        cell["missing_episode_count"] == 0 and cell["invalid_episode_count"] == 0
        for cell in cells
    )
    all_reset = all(
        session.get("n0_endpoint_reset_before_observation") is True
        for session in sessions.values()
    )
    aggregate: dict[str, object] = {
        "aggregate_valid": (
            all_complete
            and all_initial_states_match
            and all_reset
            and (not require_all_conditions or actual_conditions == expected_conditions)
        ),
        "all_episode_initial_states_matched": all_initial_states_match,
        "all_n0_episode_resets_completed": all_reset,
        "cells": cells,
        "condition_count": len(actual_conditions),
        "conditions": sorted(actual_conditions),
        "model_identity_sha256": model_identity_sha256,
        "paired_effects": paired_effects,
        "required_condition_count": 15 if require_all_conditions else None,
        "schema": "robotactile-persistent-condition-aggregate-v1",
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "task_id": task_id,
    }
    aggregate["content_sha256"] = canonical_hash(aggregate)
    write_json(selected_output, aggregate)
    return aggregate


def _finalize_worker_session(
    campaign: Path,
    manifest: ConditionWorkManifest,
    returncode: int,
) -> dict[str, object]:
    if returncode != 0:
        raise RuntimeError(
            f"persistent Isaac worker exited {returncode}; see worker.log"
        )

    final_path = campaign / "persistent_condition_session.json"
    if final_path.is_file() and not final_path.is_symlink():
        return read_object(final_path)
    if final_path.exists() or final_path.is_symlink():
        raise RuntimeError("persistent worker final receipt must be a regular file")

    preclose_path = campaign / "persistent_condition_session.preclose.json"
    if preclose_path.is_symlink() or not preclose_path.is_file():
        raise RuntimeError("persistent worker exited without a final preclose receipt")
    receipt = _verified_hashed_document(preclose_path, SESSION_SCHEMA)
    expected = {
        "app_close_status": "pending",
        "condition_id": manifest.condition_id,
        "isaac_application_launch_count": 1,
        "model_identity_sha256": manifest.model_identity_sha256,
        "n0_endpoint_reset_before_observation": True,
        "policy_instance_per_episode": True,
        "receipt_stage": "preclose",
        "runtime_count": len(manifest.items),
        "task_id": manifest.task_id,
        "task_runtime_per_episode": True,
        "work_manifest_sha256": manifest.content_sha256,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise RuntimeError(
                f"persistent worker preclose has invalid {field}: "
                f"{receipt.get(field)!r}"
            )
    dispatches = receipt.get("dispatches")
    session_id = receipt.get("session_id")
    if (
        not isinstance(session_id, str)
        or not session_id
        or not isinstance(dispatches, list)
        or len(dispatches) != len(manifest.items)
    ):
        raise RuntimeError("persistent worker preclose has invalid dispatch evidence")
    for dispatch in dispatches:
        if not isinstance(dispatch, dict):
            raise RuntimeError("persistent worker preclose dispatch must be an object")
        unhashed = dict(dispatch)
        declared = unhashed.pop("content_sha256", None)
        if (
            declared != canonical_hash(unhashed)
            or dispatch.get("schema") != EPISODE_SCHEMA
            or dispatch.get("condition_id") != manifest.condition_id
            or dispatch.get("session_id") != session_id
            or dispatch.get("policy_reset_completed") is not True
        ):
            raise RuntimeError(
                "persistent worker preclose has invalid dispatch evidence"
            )
    total_duration_s = receipt.get("total_duration_s")
    if isinstance(total_duration_s, bool) or not isinstance(
        total_duration_s, (int, float)
    ):
        raise RuntimeError("persistent worker preclose has invalid total duration")

    close_duration_s = max(0.0, time.time() - preclose_path.stat().st_mtime)
    receipt.update(
        {
            "app_close_status": "system_exit_zero",
            "finished_unix": time.time(),
            "receipt_stage": "final",
            "total_duration_s": total_duration_s + close_duration_s,
        }
    )
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = canonical_hash(receipt)
    write_json(final_path, receipt)
    return receipt


def supervise(
    *,
    binding_path: Path,
    task_id: str,
    condition_id: str,
    campaign: Path,
    code: Path,
    package: Path,
    seeds: Sequence[int],
    capture_profile: LiveCaptureProfile,
    evidence_mode: str,
    severity_registry: str,
    severity_level: int,
    fault_window_mode: str,
    fault_onset_max_index: int,
    reset_equivalence_receipt: Optional[Path],
    tactile_availability_mode: Optional[TactileAvailabilityMode] = None,
    tactile_zero_shape: Optional[tuple[int, int, int]] = None,
    gpu_id: int = 0,
    port_override: Optional[int] = None,
) -> dict[str, object]:
    """Own exactly one N0 process and one Isaac process for a condition shard."""

    if type(gpu_id) is not int or gpu_id < 0:
        raise ValueError("gpu_id must be a non-negative integer")
    selected_campaign = Path(campaign).absolute()
    selected_code = Path(code).resolve(strict=True)
    selected_package = Path(package).resolve(strict=True)
    work_path = prepare_condition_work(
        binding_path=binding_path,
        task_id=task_id,
        condition_id=condition_id,
        campaign=selected_campaign,
        seeds=seeds,
        capture_profile=capture_profile,
        evidence_mode=evidence_mode,
        severity_registry=severity_registry,
        severity_level=severity_level,
        fault_window_mode=fault_window_mode,
        fault_onset_max_index=fault_onset_max_index,
        reset_equivalence_receipt=reset_equivalence_receipt,
        tactile_availability_mode=tactile_availability_mode,
        tactile_zero_shape=tactile_zero_shape,
        port_override=port_override,
    )
    manifest = load_work_manifest(work_path)
    frozen_binding = selected_campaign / manifest.binding_relpath
    binding = read_object(frozen_binding)
    runtime = Path(binding["runtime_python"])
    if not runtime.is_file():
        raise FileNotFoundError(f"repo-local N0 runtime missing: {runtime}")
    isaac = resolve_isaac_python(binding)
    port = binding.get("port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("N0 binding port must be in [1, 65535]")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    server_env, isaac_env = build_process_environments(
        binding, selected_code, selected_package
    )
    for environment in (server_env, isaac_env):
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    server_dir = selected_campaign / "server"
    receipt = Path(binding["server_receipt"])
    if receipt != server_dir / "worker/server_receipt.json":
        raise ValueError("persistent N0 receipt escaped its campaign directory")
    server_dir.mkdir(parents=True, exist_ok=False)
    server_command = [
        str(runtime),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        str(selected_code / "scripts/n0_twam/serve_retrained.py"),
        "--artifact",
        binding["artifact"],
        "--task",
        task_id,
        "--output",
        str(receipt.parent),
        "--port",
        str(port),
    ]
    worker_command = [
        str(isaac),
        "-m",
        "scripts.retrained_evaluation.persistent_condition",
        "worker",
        "--campaign",
        str(selected_campaign),
        "--work-manifest",
        str(work_path),
    ]
    started = time.monotonic()
    n0_process: Optional[subprocess.Popen[bytes]] = None
    worker_process: Optional[subprocess.Popen[bytes]] = None
    try:
        with (server_dir / "server.log").open("xb") as server_log:
            n0_process = subprocess.Popen(
                server_command,
                cwd=selected_code,
                env=server_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_json(
                server_dir / "launch.json",
                {
                    "argv": server_command,
                    "hostname": socket.gethostname(),
                    "pid": n0_process.pid,
                    "started_unix": time.time(),
                },
            )
            wait_ready(n0_process, port, receipt)
            write_json(
                server_dir / "ready.json",
                {
                    "receipt": str(receipt),
                    "startup_s": time.monotonic() - started,
                },
            )
            worker_started = time.monotonic()
            with (selected_campaign / "worker.log").open("xb") as worker_log:
                worker_process = subprocess.Popen(
                    worker_command,
                    cwd=selected_code,
                    env=isaac_env,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                write_json(
                    selected_campaign / "worker_launch.json",
                    {
                        "argv": worker_command,
                        "hostname": socket.gethostname(),
                        "isaac_python": str(isaac),
                        "pid": worker_process.pid,
                        "started_unix": time.time(),
                    },
                )
                returncode = worker_process.wait()
            write_json(
                selected_campaign / "worker_exit.json",
                {
                    "elapsed_s": time.monotonic() - worker_started,
                    "returncode": returncode,
                },
            )
            _finalize_worker_session(selected_campaign, manifest, returncode)
            if n0_process.poll() is not None:
                raise RuntimeError(
                    "N0 server exited before persistent condition completion"
                )
    except BaseException as error:
        write_json(
            selected_campaign / "supervisor_failure.json",
            {
                "error_message": str(error),
                "error_type": type(error).__name__,
                "finished_unix": time.time(),
                "work_manifest_sha256": manifest.content_sha256,
            },
        )
        raise
    finally:
        if worker_process is not None:
            stop_owned(worker_process)
        if n0_process is not None:
            stop_owned(n0_process)

    summary = _summary(selected_campaign, manifest)
    summary["n0_server_launch_count"] = 1
    summary["isaac_application_launch_count"] = 1
    summary["gpu_id"] = gpu_id
    summary["n0_port"] = port
    summary["total_supervisor_duration_s"] = time.monotonic() - started
    if summary["completed_episode_count"] != summary["planned_episode_count"]:
        raise RuntimeError("persistent condition worker left missing episode results")
    write_json(selected_campaign / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--binding", type=Path, required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--condition", required=True)
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--code", type=Path, required=True)
    run.add_argument("--package", type=Path, required=True)
    run.add_argument("--seed-start", type=int, default=0)
    run.add_argument("--seed-count", type=int, default=100)
    run.add_argument(
        "--capture-profile",
        choices=[profile.value for profile in LiveCaptureProfile],
        default=LiveCaptureProfile.METRICS_ONLY.value,
    )
    run.add_argument("--evidence-mode", choices=EVIDENCE_MODES, default="diagnostic")
    run.add_argument(
        "--severity-registry",
        choices=SUPPORTED_REGISTRIES,
        default="optical_marker_extreme_v1",
    )
    run.add_argument("--severity-level", type=int, default=5)
    run.add_argument(
        "--fault-window",
        choices=FAULT_WINDOW_MODES,
        default="early_random_onset_v1",
    )
    run.add_argument("--fault-onset-max-index", type=int, default=8)
    run.add_argument("--reset-equivalence-receipt", type=Path)
    run.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="physical GPU exposed as logical cuda:0 to both N0 and Isaac",
    )
    run.add_argument(
        "--port",
        type=int,
        help="condition-local N0 RPC port; use a distinct port per concurrent shard",
    )
    run.add_argument(
        "--tactile-availability-mode",
        choices=[mode.value for mode in TactileAvailabilityMode],
    )
    run.add_argument("--tactile-zero-shape", type=int, nargs=3)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--campaign", type=Path, required=True)
    worker.add_argument("--work-manifest", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--campaign", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--allow-subset", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "worker":
        execute_condition_work(args.campaign, args.work_manifest)
        return 0
    if args.command == "aggregate":
        result = aggregate_condition_shards(
            args.campaign,
            args.output,
            require_all_conditions=not args.allow_subset,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.seed_count < 1 or args.seed_start < 0:
        raise ValueError("seed-start/count must define a non-empty non-negative range")
    shape = (
        None
        if args.tactile_zero_shape is None
        else cast(tuple[int, int, int], tuple(args.tactile_zero_shape))
    )
    summary = supervise(
        binding_path=args.binding,
        task_id=args.task,
        condition_id=args.condition,
        campaign=args.campaign,
        code=args.code,
        package=args.package,
        seeds=range(args.seed_start, args.seed_start + args.seed_count),
        capture_profile=LiveCaptureProfile(args.capture_profile),
        evidence_mode=args.evidence_mode,
        severity_registry=args.severity_registry,
        severity_level=args.severity_level,
        fault_window_mode=args.fault_window,
        fault_onset_max_index=args.fault_onset_max_index,
        reset_equivalence_receipt=args.reset_equivalence_receipt,
        tactile_availability_mode=(
            None
            if args.tactile_availability_mode is None
            else TactileAvailabilityMode(args.tactile_availability_mode)
        ),
        tactile_zero_shape=shape,
        gpu_id=args.gpu_id,
        port_override=args.port,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConditionWorkItem",
    "ConditionWorkManifest",
    "aggregate_condition_shards",
    "execute_condition_work",
    "load_work_manifest",
    "main",
    "prepare_condition_work",
    "supervise",
]
