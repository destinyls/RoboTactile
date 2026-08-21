"""Strict self-validating CPU policy qualification receipt."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple, cast

from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC, PolicyIdentity
from robotactile_benchmark.contracts import canonical_hash, freeze_value, thaw_value
from robotactile_benchmark.qualification.sources import (
    ACT_RUNTIME_SOURCE_PATH,
    ACT_RUNTIME_SOURCE_SHA256,
    TASK6A_SOURCE_HASHES,
    TASK6B_SOURCE_HASHES,
    source_manifest_sha256,
)

SCHEMA_VERSION = "robotactile-policy-qualification-v1"
ACT_EVIDENCE_TYPE = "cpu_fake_act_policy_protocol"
N0_EVIDENCE_TYPE = "cpu_fake_n0_gateway_protocol"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BOUNDARY_FLAGS = (
    "live_model_executed",
    "twam_server_executed",
    "isaac_sim_executed",
    "task_success_measured",
    "tls_authenticated",
)
ACT_CHECKS = (
    "policy_identity_bound",
    "task6a_sources_bound",
    "act_runtime_source_bound_not_loaded",
    "strict_observation_routing",
    "runtime_lifecycle",
    "no_touch_artifact_unavailable",
    "structural_absence_zero_effects",
    "evidence_boundary",
)
N0_CHECKS = (
    "policy_identity_bound",
    "task6a_sources_bound",
    "task6b_sources_bound",
    "cross_package_two_phase_commit",
    "prepare_zero_engine_effect",
    "finalize_exactly_once",
    "no_touch_artifact_unavailable",
    "structural_absence_zero_effects",
    "evidence_boundary",
)


class PolicyQualificationError(ValueError):
    """Fail-closed policy qualification or receipt validation error."""


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PolicyQualificationError(f"{name} must be a lowercase SHA256")
    return value


def _validate_source_map(
    value: Mapping[str, object], expected: Mapping[str, str], name: str
) -> Mapping[str, Any]:
    if set(value) != set(expected):
        raise PolicyQualificationError(f"{name} source fields mismatch")
    for path, expected_digest in expected.items():
        if _sha256(value[path], f"{name} source hash") != expected_digest:
            raise PolicyQualificationError(f"{name} source identity mismatch")
    return cast(Mapping[str, Any], freeze_value(value))


def expected_absence_results() -> Mapping[str, Any]:
    result = {
        operator_id: {
            "terminal_status": "unsupported_contract",
            "backend_reset_count": 0,
            "policy_effect_count": 0,
            "transport_effect_count": 0,
        }
        for operator_id in ("A1_stream_absence", "A2_frame_erasure")
    }
    return cast(Mapping[str, Any], freeze_value(result))


@dataclass(frozen=True)
class PolicyQualificationReceipt:
    """CPU-only evidence that cannot be upgraded into live/simulator claims."""

    evidence_type: str
    policy_kind: str
    task_id: str
    policy_identity: Mapping[str, Any]
    policy_identity_sha256: str
    task6a_source_hashes: Mapping[str, Any]
    task6a_source_manifest_sha256: str
    task6b_source_hashes: Mapping[str, Any]
    task6b_source_manifest_sha256: Optional[str]
    act_runtime_source_path: Optional[str]
    act_runtime_source_sha256: Optional[str]
    artifact_status: str
    no_touch_status: str
    structural_absence_results: Mapping[str, Any]
    structural_absence_sha256: str
    protocol_result: Mapping[str, Any]
    protocol_result_sha256: str
    checks: Tuple[str, ...]
    passed: bool
    failure_codes: Tuple[str, ...]
    live_model_executed: bool
    twam_server_executed: bool
    isaac_sim_executed: bool
    task_success_measured: bool
    tls_authenticated: bool
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise PolicyQualificationError("qualification schema version mismatch")
        if self.evidence_type not in {ACT_EVIDENCE_TYPE, N0_EVIDENCE_TYPE}:
            raise PolicyQualificationError("qualification evidence type mismatch")
        expected_kind = "act" if self.evidence_type == ACT_EVIDENCE_TYPE else "n0"
        if self.policy_kind != expected_kind:
            raise PolicyQualificationError("policy kind and evidence type mismatch")
        if type(self.task_id) is not str or not self.task_id:
            raise PolicyQualificationError("task_id must be non-empty")
        identity = self._validate_identity()
        task6a = _validate_source_map(
            self.task6a_source_hashes, TASK6A_SOURCE_HASHES, "Task 6A"
        )
        task6b = self._validate_kind_sources()
        absence = freeze_value(self.structural_absence_results)
        protocol = freeze_value(self.protocol_result)
        if absence != expected_absence_results():
            raise PolicyQualificationError("structural absence evidence mismatch")
        if _sha256(self.structural_absence_sha256, "absence hash") != canonical_hash(
            absence
        ):
            raise PolicyQualificationError("structural absence hash mismatch")
        if _sha256(self.protocol_result_sha256, "protocol hash") != canonical_hash(
            protocol
        ):
            raise PolicyQualificationError("protocol result hash mismatch")
        self._validate_protocol(protocol)
        self._validate_status()
        for name in _BOUNDARY_FLAGS:
            if getattr(self, name) is not False:
                raise PolicyQualificationError(
                    "CPU evidence boundary cannot be upgraded"
                )
        for name, value in (
            ("policy_identity", identity),
            ("task6a_source_hashes", task6a),
            ("task6b_source_hashes", task6b),
            ("structural_absence_results", absence),
            ("protocol_result", protocol),
        ):
            object.__setattr__(self, name, value)

    def _validate_identity(self) -> Mapping[str, Any]:
        expected_fields = {
            "system_id",
            "checkpoint_sha256",
            "config_sha256",
            "action_spec",
            "consumes_tactile",
            "supports_structural_absence",
        }
        if set(self.policy_identity) != expected_fields:
            raise PolicyQualificationError("policy identity fields mismatch")
        try:
            identity = PolicyIdentity(**dict(self.policy_identity))
        except (TypeError, ValueError) as error:
            raise PolicyQualificationError("policy identity is invalid") from error
        if (
            identity.action_spec != ACTION_SPEC
            or not identity.consumes_tactile
            or identity.supports_structural_absence
        ):
            raise PolicyQualificationError("policy identity capability mismatch")
        frozen = freeze_value(self.policy_identity)
        if _sha256(
            self.policy_identity_sha256, "policy identity hash"
        ) != canonical_hash(frozen):
            raise PolicyQualificationError("policy identity hash mismatch")
        return cast(Mapping[str, Any], frozen)

    def _validate_kind_sources(self) -> Mapping[str, Any]:
        expected_manifest = source_manifest_sha256(TASK6A_SOURCE_HASHES)
        if self.task6a_source_manifest_sha256 != expected_manifest:
            raise PolicyQualificationError("Task 6A source manifest mismatch")
        if self.policy_kind == "act":
            if (
                self.task6b_source_hashes
                or self.task6b_source_manifest_sha256 is not None
            ):
                raise PolicyQualificationError(
                    "ACT receipt may not claim Task 6B sources"
                )
            if (
                self.act_runtime_source_path != ACT_RUNTIME_SOURCE_PATH
                or self.act_runtime_source_sha256 != ACT_RUNTIME_SOURCE_SHA256
            ):
                raise PolicyQualificationError("ACT runtime source identity mismatch")
            return cast(Mapping[str, Any], freeze_value({}))
        if TASK6B_SOURCE_HASHES is None:
            raise PolicyQualificationError(
                "Task 6B source hashes are not review-locked"
            )
        frozen = _validate_source_map(
            self.task6b_source_hashes, TASK6B_SOURCE_HASHES, "Task 6B"
        )
        if self.task6b_source_manifest_sha256 != source_manifest_sha256(
            TASK6B_SOURCE_HASHES
        ):
            raise PolicyQualificationError("Task 6B source manifest mismatch")
        if (
            self.act_runtime_source_path is not None
            or self.act_runtime_source_sha256 is not None
        ):
            raise PolicyQualificationError(
                "N0 receipt may not claim ACT runtime source"
            )
        return frozen

    def _validate_protocol(self, protocol: Mapping[str, Any]) -> None:
        if protocol.get("protocol_status") != "passed":
            raise PolicyQualificationError("policy protocol did not pass")
        if self.policy_kind == "act":
            expected_fields = {
                "protocol_status",
                "runtime_reset_count",
                "runtime_infer_count",
                "runtime_commit_model_calls",
                "action_plan_sha256",
                "routed_input_sha256",
            }
            if (
                set(protocol) != expected_fields
                or protocol["runtime_reset_count"] != 1
                or protocol["runtime_infer_count"] != 1
                or protocol["runtime_commit_model_calls"] != 0
            ):
                raise PolicyQualificationError("ACT protocol witness mismatch")
            _sha256(protocol["action_plan_sha256"], "action plan hash")
            _sha256(protocol["routed_input_sha256"], "routed input hash")
            return
        expected_fields = {
            "protocol_status",
            "request_operations",
            "prepare_engine_commit_count",
            "final_engine_commit_count",
            "final_cache_position",
            "client_state",
            "handshake_sha256",
            "executed_actions_sha256",
            "grounding_frames_sha256",
            "grounding_frame_count",
            "fake_gateway_executed",
        }
        if (
            set(protocol) != expected_fields
            or tuple(protocol["request_operations"])
            != ("reset", "infer", "prepare_commit", "finalize_commit")
            or protocol["prepare_engine_commit_count"] != 0
            or protocol["final_engine_commit_count"] != 1
            or protocol["final_cache_position"] != 1
            or protocol["client_state"] != "ready"
            or protocol["grounding_frame_count"] != 8
            or protocol["fake_gateway_executed"] is not True
        ):
            raise PolicyQualificationError("N0 protocol witness mismatch")
        for name in (
            "handshake_sha256",
            "executed_actions_sha256",
            "grounding_frames_sha256",
        ):
            _sha256(protocol[name], name)

    def _validate_status(self) -> None:
        if self.artifact_status != "artifact_not_loaded":
            raise PolicyQualificationError("model artifact status mismatch")
        if self.no_touch_status != "artifact_unavailable":
            raise PolicyQualificationError("no-touch status mismatch")
        expected_checks = ACT_CHECKS if self.policy_kind == "act" else N0_CHECKS
        if tuple(self.checks) != expected_checks:
            raise PolicyQualificationError("qualification checks mismatch")
        if type(self.passed) is not bool or self.passed != (not self.failure_codes):
            raise PolicyQualificationError("qualification pass/failure mismatch")
        if not self.passed or self.failure_codes:
            raise PolicyQualificationError("CPU policy qualification did not pass")
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "failure_codes", tuple(self.failure_codes))

    @property
    def sha256(self) -> str:
        return canonical_hash(self)

    def to_dict(self) -> Dict[str, Any]:
        document = {
            field.name: thaw_value(getattr(self, field.name)) for field in fields(self)
        }
        document["receipt_sha256"] = self.sha256
        return document

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "PolicyQualificationReceipt":
        expected = {field.name for field in fields(cls)} | {"receipt_sha256"}
        if set(document) != expected:
            raise PolicyQualificationError("qualification receipt fields mismatch")
        values = {
            key: value for key, value in document.items() if key != "receipt_sha256"
        }
        if _sha256(document["receipt_sha256"], "receipt hash") != canonical_hash(
            values
        ):
            raise PolicyQualificationError("qualification receipt hash mismatch")
        receipt = cls(**values)
        return receipt
