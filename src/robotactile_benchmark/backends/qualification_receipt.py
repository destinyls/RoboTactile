"""Self-validating and no-clobber UniVTAC qualification receipts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple

import numpy as np

from robotactile_benchmark.backends.univtac_contracts import (
    REGISTRY_RESOURCE_SHA256,
    UPSTREAM_COMMIT,
    UniVTACBackendConfig,
    build_univtac_backend_config,
)
from robotactile_benchmark.contracts import (
    canonical_hash,
    canonical_json,
    freeze_value,
    thaw_value,
)

CPU_FAKE_EVIDENCE_LEVEL = "cpu_fake_upstream_qualification"
QUALIFICATION_SCHEMA_VERSION = "robotactile-univtac-qualification-v1"
QUALIFICATION_CHECKS = (
    "contract_identity",
    "fresh_reset_determinism",
    "stale_reset_return_ignored",
    "joint9_reorder_shared_qpos8",
    "cuda_conversion_sequence",
    "action_conditioned_state_change",
    "native_step_continuity",
    "terminal_priority",
    "malformed_input_fail_closed",
    "resource_cleanup",
)
QUALIFICATION_FAILURE_CODES = frozenset(
    {
        "contract_identity_mismatch",
        "reset_hash_mismatch",
        "initial_record_hash_mismatch",
        "stale_reset_return_used",
        "joint_reorder_mismatch",
        "cuda_sequence_mismatch",
        "action_state_unchanged",
        "native_step_mismatch",
        "terminal_priority_mismatch",
        "malformed_input_gate_failed",
        "resource_cleanup_failed",
    }
)
_ENVIRONMENT_KEYS = frozenset(
    {
        "python_implementation",
        "python_version",
        "platform_system",
        "platform_machine",
        "numpy_version",
        "accelerator",
        "isaac_sim_executed",
        "runtime_kind",
    }
)


class UniVTACQualificationError(ValueError):
    """Fail-closed CPU qualification or receipt validation error."""


def expected_terminal_priority_results(early_stop_capable: bool) -> Dict[str, str]:
    """Return task-aware terminal evidence without inventing capabilities."""

    return {
        "early_over_timeout": "early_stop" if early_stop_capable else "not_applicable",
        "failure_over_early": "task_failure",
        "success_over_failure": "success",
        "timeout_fallback": "timeout",
    }


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UniVTACQualificationError(f"{name} must be a lowercase SHA256")
    return value


def qualification_descriptor(value: Any) -> Mapping[str, Any]:
    """Materialize a deterministic JSON descriptor for hash recomputation."""

    parsed = json.loads(canonical_json(value))
    if not isinstance(parsed, dict):
        raise UniVTACQualificationError("qualification descriptor must be a mapping")
    return parsed


def action_progress_witness(
    first_state: str,
    action_state: str,
    first_native_step: int,
    action_native_step: int,
    action_benchmark_step: int,
    action_sha256: str,
) -> str:
    """Hash the full native/dense action-progress relation."""

    return canonical_hash(
        {
            "first_state_sha256": first_state,
            "action_state_sha256": action_state,
            "first_native_step_id": first_native_step,
            "action_native_step_id": action_native_step,
            "action_benchmark_step": action_benchmark_step,
            "action_sha256": action_sha256,
        }
    )


def joint_reorder_witness(
    live_joint_names: Tuple[str, ...],
    canonical_joint_names: Tuple[str, ...],
    canonical_joint9: Tuple[float, ...],
    model_visible_qpos8: Tuple[float, ...],
) -> str:
    """Hash names, raw canonical joint9, and folded policy qpos8 together."""

    joint9 = np.asarray(canonical_joint9, dtype=np.float32)
    qpos8 = np.asarray(model_visible_qpos8, dtype=np.float32)
    return canonical_hash(
        {
            "live_joint_names": live_joint_names,
            "canonical_joint_names": canonical_joint_names,
            "canonical_joint9": joint9,
            "model_visible_qpos8": qpos8,
            "shared_gripper_qpos": float(joint9[-2]),
        }
    )


@dataclass(frozen=True)
class UniVTACQualificationReceipt:
    """Typed CPU evidence that cannot be mistaken for a live Isaac run."""

    task_id: str
    success_predicate_id: str
    upstream_commit: str
    task_source_sha256: str
    registry_resource_sha256: str
    config_sha256: str
    config_descriptor: Mapping[str, Any]
    handshake_sha256: str
    handshake_descriptor: Mapping[str, Any]
    first_reset_state_sha256: str
    second_reset_state_sha256: str
    first_clean_record_sha256: str
    second_clean_record_sha256: str
    first_initial_native_step_id: int
    second_initial_native_step_id: int
    first_joint_reorder_witness_sha256: str
    second_joint_reorder_witness_sha256: str
    live_joint_names: Tuple[str, ...]
    canonical_joint_names: Tuple[str, ...]
    initial_canonical_joint9: Tuple[float, ...]
    initial_model_visible_qpos8: Tuple[float, ...]
    action_state_sha256: str
    action_native_step_id: int
    action_benchmark_step: int
    action_sha256: str
    action_progress_witness_sha256: str
    terminal_priority_results: Mapping[str, str]
    terminal_priority_witness_sha256: str
    predicate_note: str
    environment: Mapping[str, Any]
    checks: Tuple[str, ...]
    passed: bool
    failure_codes: Tuple[str, ...]
    schema_version: str = QUALIFICATION_SCHEMA_VERSION
    evidence_level: str = CPU_FAKE_EVIDENCE_LEVEL

    def __post_init__(self) -> None:
        self._validate_header_and_hashes()
        config = build_univtac_backend_config(self.task_id)
        self._validate_identity(config)
        self._validate_steps(config.physics_steps_per_action)
        self._validate_status()
        terminal = freeze_value(self.terminal_priority_results)
        environment = freeze_value(self.environment)
        config_descriptor = freeze_value(self.config_descriptor)
        handshake_descriptor = freeze_value(self.handshake_descriptor)
        self._validate_environment(environment)
        self._validate_terminal(
            terminal,
            expected_terminal_priority_results(config.task.early_stop_capable),
        )
        live_names, canonical_names, joint9, qpos8 = self._validate_joint_payload(
            config.canonical_joint_names
        )
        expected_config = freeze_value(qualification_descriptor(config))
        expected_handshake = freeze_value(
            qualification_descriptor(config.expected_handshake(live_names))
        )
        if config_descriptor != expected_config:
            raise UniVTACQualificationError("config identity descriptor mismatch")
        if handshake_descriptor != expected_handshake:
            raise UniVTACQualificationError("handshake identity descriptor mismatch")
        if canonical_hash(config_descriptor) != self.config_sha256:
            raise UniVTACQualificationError("config descriptor hash mismatch")
        if canonical_hash(handshake_descriptor) != self.handshake_sha256:
            raise UniVTACQualificationError("handshake descriptor hash mismatch")
        expected_joint = joint_reorder_witness(
            live_names, canonical_names, joint9, qpos8
        )
        if (
            self.first_joint_reorder_witness_sha256 != expected_joint
            or self.second_joint_reorder_witness_sha256 != expected_joint
        ):
            raise UniVTACQualificationError("joint witness hash mismatch")
        self._freeze_fields(
            terminal,
            environment,
            config_descriptor,
            handshake_descriptor,
            live_names,
            canonical_names,
            joint9,
            qpos8,
        )
        self._validate_witnesses(terminal)

    def _validate_header_and_hashes(self) -> None:
        if self.schema_version != QUALIFICATION_SCHEMA_VERSION:
            raise UniVTACQualificationError("qualification schema version mismatch")
        if self.evidence_level != CPU_FAKE_EVIDENCE_LEVEL:
            raise UniVTACQualificationError("qualification evidence level mismatch")
        for name in ("task_id", "success_predicate_id", "predicate_note"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise UniVTACQualificationError(f"{name} must be non-empty")
        for field in fields(self):
            if field.name.endswith("_sha256"):
                object.__setattr__(
                    self, field.name, _sha256(getattr(self, field.name), field.name)
                )

    def _validate_identity(self, config: UniVTACBackendConfig) -> None:
        if (
            self.upstream_commit != UPSTREAM_COMMIT
            or self.registry_resource_sha256 != REGISTRY_RESOURCE_SHA256
            or self.task_source_sha256 != config.task.task_source_sha256
            or self.success_predicate_id != config.task.success_predicate_id
            or self.predicate_note != config.task.predicate_note
            or self.config_sha256 != config.sha256
        ):
            raise UniVTACQualificationError("qualification identity mismatch")

    def _validate_steps(self, increment: int) -> None:
        for name in (
            "first_initial_native_step_id",
            "second_initial_native_step_id",
            "action_native_step_id",
            "action_benchmark_step",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UniVTACQualificationError(
                    "qualification native step IDs must be non-negative integers"
                )
        if (
            self.first_initial_native_step_id != self.second_initial_native_step_id
            or self.action_native_step_id
            != self.first_initial_native_step_id + increment
            or self.action_benchmark_step != 1
        ):
            raise UniVTACQualificationError(
                "qualification native step continuity mismatch"
            )

    def _validate_status(self) -> None:
        if type(self.passed) is not bool:
            raise UniVTACQualificationError("passed must be bool")
        codes = tuple(self.failure_codes)
        if len(codes) != len(set(codes)):
            raise UniVTACQualificationError("failure codes must be unique")
        if set(codes) - QUALIFICATION_FAILURE_CODES:
            raise UniVTACQualificationError("unknown failure code in receipt")
        if self.passed != (not codes):
            raise UniVTACQualificationError("passed and failure codes disagree")
        checks = tuple(self.checks)
        if self.passed and checks != QUALIFICATION_CHECKS:
            raise UniVTACQualificationError("passed receipt lacks required checks")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "failure_codes", codes)

    @staticmethod
    def _validate_environment(environment: Mapping[str, Any]) -> None:
        text_keys = _ENVIRONMENT_KEYS - {"isaac_sim_executed"}
        if (
            set(environment) != _ENVIRONMENT_KEYS
            or any(
                not isinstance(environment[key], str) or not environment[key]
                for key in text_keys
            )
            or environment["accelerator"] != "none"
            or environment["isaac_sim_executed"] is not False
            or environment["runtime_kind"] != "upstream_faithful_cpu_fake"
        ):
            raise UniVTACQualificationError("CPU qualification environment mismatch")

    @staticmethod
    def _validate_terminal(
        terminal: Mapping[str, Any], expected: Mapping[str, str]
    ) -> None:
        if dict(terminal) != dict(expected):
            raise UniVTACQualificationError("terminal priority results mismatch")

    def _validate_joint_payload(
        self, expected_names: Tuple[str, ...]
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[float, ...], Tuple[float, ...]]:
        live_names = tuple(self.live_joint_names)
        canonical_names = tuple(self.canonical_joint_names)
        joint9 = tuple(float(value) for value in self.initial_canonical_joint9)
        qpos8 = tuple(float(value) for value in self.initial_model_visible_qpos8)
        if (
            len(live_names) != 9
            or len(set(live_names)) != 9
            or canonical_names != expected_names
            or set(live_names) != set(canonical_names)
            or len(joint9) != 9
            or len(qpos8) != 8
            or not all(np.isfinite(value) for value in joint9 + qpos8)
            or not np.array_equal(
                np.asarray(qpos8, dtype=np.float32),
                np.asarray(joint9[:7] + joint9[-2:-1], dtype=np.float32),
            )
        ):
            raise UniVTACQualificationError("joint witness payload mismatch")
        return live_names, canonical_names, joint9, qpos8

    def _freeze_fields(
        self,
        terminal: Any,
        environment: Any,
        config_descriptor: Any,
        handshake_descriptor: Any,
        live_names: Tuple[str, ...],
        canonical_names: Tuple[str, ...],
        joint9: Tuple[float, ...],
        qpos8: Tuple[float, ...],
    ) -> None:
        for name, value in (
            ("terminal_priority_results", terminal),
            ("environment", environment),
            ("config_descriptor", config_descriptor),
            ("handshake_descriptor", handshake_descriptor),
            ("live_joint_names", live_names),
            ("canonical_joint_names", canonical_names),
            ("initial_canonical_joint9", joint9),
            ("initial_model_visible_qpos8", qpos8),
        ):
            object.__setattr__(self, name, value)

    def _validate_witnesses(self, terminal: Mapping[str, Any]) -> None:
        expected_action = action_progress_witness(
            self.first_reset_state_sha256,
            self.action_state_sha256,
            self.first_initial_native_step_id,
            self.action_native_step_id,
            self.action_benchmark_step,
            self.action_sha256,
        )
        if self.action_progress_witness_sha256 != expected_action:
            raise UniVTACQualificationError("action witness hash mismatch")
        if self.terminal_priority_witness_sha256 != canonical_hash(terminal):
            raise UniVTACQualificationError("terminal witness hash mismatch")
        if self.passed and (
            self.first_reset_state_sha256 != self.second_reset_state_sha256
            or self.first_clean_record_sha256 != self.second_clean_record_sha256
            or self.first_reset_state_sha256 == self.action_state_sha256
        ):
            raise UniVTACQualificationError("passed receipt has inconsistent witnesses")

    @property
    def sha256(self) -> str:
        return canonical_hash(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return strict JSON content plus its externally stored content hash."""

        document = {
            field.name: thaw_value(getattr(self, field.name)) for field in fields(self)
        }
        document["receipt_sha256"] = self.sha256
        return document
