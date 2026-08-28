"""Postprocess existing runtime evidence into one source-bound parity artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from robotactile_benchmark.closed_loop.artifact_io import strict_json_bytes
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    load_n0_twam_artifact_manifest,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity_contracts import (
    ObservationEvidenceBinding,
    ObservationParityArtifact,
    ObservationParityError,
    ObservationParityGate,
    ObservationParityStatus,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity_io import (
    resolve_evidence_member,
    sha256_file,
    write_observation_parity_artifact,
)
from robotactile_benchmark.runtime_source import (
    RuntimeSourceBinding,
    current_integrations_lock_sha256,
    current_n0_input_profile_sha256,
)

EXPERIMENT_LOCK_EVIDENCE_LEVEL = "n0_univtac_experiment_preregistration_v1"
EXPERIMENT_LOCK_SCHEMA_VERSION = "robotactile-n0-univtac-experiment-lock-v1"
TEACHER_FORCED_EVIDENCE_LEVEL = (
    "univtac_n0_simulator_timestamped_teacher_forced_alignment_v3"
)
OBSERVATION_PARITY_LIMITATIONS = (
    "This verifies source-bound observation contracts, not pixel-registered ground truth.",
    "The legacy HDF5 JPEG decode is a training proxy, not the exact LeRobot H.264 frame or training latent.",
    "The public 84.5 percent checkpoint lineage is not identified by upstream release metadata.",
)
_REQUIRED_SERVER_DOMAINS = (
    "hdf5_n0_server_pixels",
    "live_reset_n0_server_pixels",
    "live_replayed_n0_server_pixels",
)
_REQUIRED_STREAMS = ("top", "wrist_l", "tactile_a", "tactile_b")


@dataclass(frozen=True)
class ObservationParityInputs:
    """Deployment-relative inputs consumed by the parity postprocessor."""

    experiment_lock: str
    teacher_forced_probe: str
    isaac_install_receipt: str
    n0_client_install_receipt: str
    n0_runtime_receipt: str
    tacex_install_receipt: str
    n0_artifact_manifest: str


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ObservationParityError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ObservationParityError(f"{name} must be a sequence")
    return cast(Sequence[object], value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ObservationParityError(f"{name} must be a non-empty string")
    return value


def _canonical_document(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = strict_json_bytes(path.read_bytes(), name)
    except (OSError, TypeError, ValueError) as error:
        raise ObservationParityError(f"{name} must be canonical JSON") from error
    document = _mapping(value, name)
    claimed = document.get("content_sha256")
    without_hash = dict(document)
    without_hash.pop("content_sha256", None)
    if claimed != canonical_hash(without_hash):
        raise ObservationParityError(f"{name} content hash mismatch")
    return document


def _receipt_document(path: Path, name: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ObservationParityError(f"{name} must be a regular file")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ObservationParityError(f"{name} has a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObservationParityError(f"{name} must be JSON") from error
    return _mapping(value, name)


def _expect_fields(
    document: Mapping[str, object], expected: Mapping[str, str], name: str
) -> None:
    for key, value in expected.items():
        if document.get(key) != value:
            raise ObservationParityError(f"{name} identity mismatch: {key}")


def _evidence(
    root: Path,
    kind: str,
    relative: str,
    *,
    content_sha256: str | None,
) -> ObservationEvidenceBinding:
    path = resolve_evidence_member(root, relative, f"{kind} path")
    return ObservationEvidenceBinding(
        kind=kind,
        relpath=relative,
        sha256=sha256_file(path),
        content_sha256=content_sha256,
    )


def _camera_renderer_passed(
    probe: Mapping[str, object], lock: Mapping[str, object]
) -> bool:
    camera = probe.get("camera_contract")
    renderer = probe.get("renderer_settings")
    runtime = probe.get("camera_runtime")
    if not isinstance(camera, Mapping) or not camera:
        return False
    if not isinstance(renderer, Mapping) or not renderer:
        return False
    if not isinstance(runtime, Mapping):
        return False
    freshness = runtime.get("freshness")
    if not isinstance(freshness, Mapping) or not freshness:
        return False
    if not all(
        isinstance(item, Mapping) and item.get("passed") is True
        for item in freshness.values()
    ):
        return False
    simulator = _mapping(lock.get("simulator_contract"), "simulator contract")
    return probe.get("antialiasing_mode") == simulator.get(
        "antialiasing_mode"
    ) and isinstance(probe.get("rendering_mode"), str)


def _model_boundary_passed(probe: Mapping[str, object]) -> bool:
    arrays = probe.get("arrays")
    if not isinstance(arrays, Mapping):
        return False
    domains = arrays.get("domains")
    if not isinstance(domains, Mapping):
        return False
    for name in _REQUIRED_SERVER_DOMAINS:
        stream_map = domains.get(name)
        if not isinstance(stream_map, Mapping) or set(stream_map) != set(
            _REQUIRED_STREAMS
        ):
            return False
    contract = probe.get("n0_server_pixel_contract")
    if not isinstance(contract, Mapping):
        return False
    return (
        contract.get("camera_shape_hwc") == [256, 256, 3]
        and contract.get("tactile_shape_hwc") == [128, 128, 3]
        and contract.get("interpolation") == "cv2.INTER_AREA"
        and contract.get("production_resize_location") == "pinned_n0_server"
    )


def _gate(
    gate_id: str,
    passed: bool,
    *,
    failure_code: str,
    passed_detail: str,
    failed_detail: str,
) -> ObservationParityGate:
    return ObservationParityGate(
        gate_id=gate_id,
        status=(
            ObservationParityStatus.PASS if passed else ObservationParityStatus.FAIL
        ),
        code="passed" if passed else failure_code,
        detail=passed_detail if passed else failed_detail,
    )


def build_observation_parity_artifact(
    *,
    deployment_root: Path,
    task_id: str,
    inputs: ObservationParityInputs,
) -> ObservationParityArtifact:
    """Cross-check existing receipts and derive one fail-closed task artifact."""

    root = Path(deployment_root).resolve(strict=True)
    paths = {
        name: resolve_evidence_member(root, value, f"{name} path")
        for name, value in inputs.__dict__.items()
    }
    experiment = _canonical_document(paths["experiment_lock"], "experiment lock")
    probe = _canonical_document(paths["teacher_forced_probe"], "teacher-forced probe")
    if (
        experiment.get("schema_version") != EXPERIMENT_LOCK_SCHEMA_VERSION
        or experiment.get("evidence_level") != EXPERIMENT_LOCK_EVIDENCE_LEVEL
    ):
        raise ObservationParityError("experiment lock identity mismatch")
    if probe.get("evidence_level") != TEACHER_FORCED_EVIDENCE_LEVEL:
        raise ObservationParityError("teacher-forced probe identity mismatch")
    trial = _mapping(experiment.get("trial"), "experiment trial")
    if trial.get("task_id") != task_id or probe.get("task_id") != task_id:
        raise ObservationParityError("observation evidence task mismatch")
    if probe.get("pixel_registered_ground_truth") is not False:
        raise ObservationParityError(
            "teacher-forced evidence must preserve its non-registered limitation"
        )

    components = _mapping(experiment.get("components"), "experiment components")
    source_component = _mapping(
        components.get("robotactile_source_manifest"), "source manifest component"
    )
    wheel_component = _mapping(components.get("robotactile_wheel"), "wheel component")
    source_sha256 = _string(source_component.get("sha256"), "source SHA256")
    wheel_sha256 = _string(wheel_component.get("sha256"), "wheel SHA256")
    if probe.get("source_manifest_sha256") != source_sha256:
        raise ObservationParityError("probe source manifest mismatch")

    external = _mapping(experiment.get("external_sources"), "external sources")
    univtac = _mapping(external.get("univtac"), "UniVTAC source")
    n0 = _mapping(external.get("n0_twam"), "N0-TWAM source")
    univtac_commit = _string(univtac.get("commit_sha"), "UniVTAC commit")
    n0_commit = _string(n0.get("commit_sha"), "N0-TWAM commit")
    manifest = load_n0_twam_artifact_manifest(paths["n0_artifact_manifest"])
    if manifest.task_id != task_id or manifest.external_commit != n0_commit:
        raise ObservationParityError("N0 artifact source/task mismatch")
    artifact_component = _mapping(
        components.get("artifact_manifest"), "artifact manifest component"
    )
    if artifact_component.get("sha256") != sha256_file(paths["n0_artifact_manifest"]):
        raise ObservationParityError("experiment lock N0 manifest mismatch")

    model_contract = _mapping(
        experiment.get("model_contract"), "experiment model contract"
    )
    input_profile = _mapping(model_contract.get("input_profile"), "input profile")
    input_profile_sha256 = canonical_hash(input_profile)
    if input_profile_sha256 != current_n0_input_profile_sha256():
        raise ObservationParityError("experiment input profile mismatch")
    simulator = _mapping(experiment.get("simulator_contract"), "simulator contract")
    source_binding = RuntimeSourceBinding(
        robotactile_source_manifest_sha256=source_sha256,
        robotactile_wheel_sha256=wheel_sha256,
        integrations_lock_sha256=current_integrations_lock_sha256(),
        univtac_source_commit=univtac_commit,
        n0_source_commit=n0_commit,
        checkpoint_sha256=manifest.checkpoint_sha256,
        config_sha256=manifest.config_sha256,
        normalizer_sha256=manifest.normalizer_sha256,
        serve_bundle_sha256=manifest.serve_bundle_sha256,
        prompt_manifest_sha256=manifest.prompt_manifest_sha256,
        input_profile_sha256=input_profile_sha256,
        action_execution_contract=_string(
            simulator.get("action_execution_contract"),
            "action execution contract",
        ),
        native_step_contract=_string(
            simulator.get("native_step_contract"), "native step contract"
        ),
    )

    receipts = {
        "isaac_install_receipt": _receipt_document(
            paths["isaac_install_receipt"], "Isaac install receipt"
        ),
        "n0_client_install_receipt": _receipt_document(
            paths["n0_client_install_receipt"], "N0 client install receipt"
        ),
        "n0_runtime_receipt": _receipt_document(
            paths["n0_runtime_receipt"], "N0 runtime receipt"
        ),
        "tacex_install_receipt": _receipt_document(
            paths["tacex_install_receipt"], "TacEx install receipt"
        ),
    }
    _expect_fields(
        receipts["isaac_install_receipt"],
        {
            "component": "robotactile_isaac",
            "source_manifest_sha256": source_sha256,
            "status": "installed",
            "wheel_sha256": wheel_sha256,
        },
        "Isaac install receipt",
    )
    _expect_fields(
        receipts["n0_client_install_receipt"],
        {
            "component": "n0_twam_robotactile_client",
            "source_manifest_sha256": source_sha256,
            "status": "installed",
            "wheel_sha256": wheel_sha256,
        },
        "N0 client install receipt",
    )
    _expect_fields(
        receipts["n0_runtime_receipt"],
        {
            "component": "n0_twam_runtime",
            "source_commit": n0_commit,
            "status": "installed",
        },
        "N0 runtime receipt",
    )
    _expect_fields(
        receipts["tacex_install_receipt"],
        {
            "component": "tacex",
            "status": "installed",
            "univtac_source_commit": univtac_commit,
        },
        "TacEx install receipt",
    )
    if receipts["n0_client_install_receipt"].get(
        "runtime_receipt_sha256"
    ) != sha256_file(paths["n0_runtime_receipt"]):
        raise ObservationParityError("N0 client/runtime receipt binding mismatch")

    replay = _mapping(probe.get("teacher_forced_replay"), "teacher-forced replay")
    replay_passed = replay.get("execution_success") is True
    camera_passed = _camera_renderer_passed(probe, experiment)
    boundary_passed = _model_boundary_passed(probe)
    state_registered = probe.get("robot_state_registered") is True
    gates = (
        _gate(
            "source_identity",
            True,
            failure_code="source_identity_mismatch",
            passed_detail="All source, wheel, checkout, and model hashes agree.",
            failed_detail="One or more source identities disagree.",
        ),
        _gate(
            "experiment_lock",
            True,
            failure_code="experiment_lock_invalid",
            passed_detail="The preregistration lock is canonical and content addressed.",
            failed_detail="The experiment lock is invalid.",
        ),
        _gate(
            "teacher_forced_probe",
            replay_passed,
            failure_code="teacher_forced_replay_failed",
            passed_detail="The official qpos teacher-forced replay executed.",
            failed_detail="The official qpos teacher-forced replay did not execute.",
        ),
        _gate(
            "camera_renderer",
            camera_passed,
            failure_code="camera_renderer_unverified",
            passed_detail="Camera freshness and renderer contracts are recorded and agree.",
            failed_detail="Camera freshness or renderer identity is not verified.",
        ),
        _gate(
            "model_boundary",
            boundary_passed,
            failure_code="model_boundary_unverified",
            passed_detail="All four streams are captured at the pinned N0 server boundary.",
            failed_detail="The required N0 server pixel domains are incomplete.",
        ),
        _gate(
            "state_registration",
            state_registered,
            failure_code="robot_state_not_registered",
            passed_detail="The live robot state matches the selected expert timestamp.",
            failed_detail="The live robot state does not match the expert timestamp.",
        ),
    )
    evidence = (
        _evidence(
            root,
            "experiment_lock",
            inputs.experiment_lock,
            content_sha256=_string(
                experiment.get("content_sha256"), "experiment content SHA256"
            ),
        ),
        _evidence(
            root,
            "teacher_forced_probe",
            inputs.teacher_forced_probe,
            content_sha256=_string(probe.get("content_sha256"), "probe content SHA256"),
        ),
        _evidence(
            root,
            "isaac_install_receipt",
            inputs.isaac_install_receipt,
            content_sha256=None,
        ),
        _evidence(
            root,
            "n0_client_install_receipt",
            inputs.n0_client_install_receipt,
            content_sha256=None,
        ),
        _evidence(
            root,
            "n0_runtime_receipt",
            inputs.n0_runtime_receipt,
            content_sha256=None,
        ),
        _evidence(
            root,
            "tacex_install_receipt",
            inputs.tacex_install_receipt,
            content_sha256=None,
        ),
        _evidence(
            root,
            "n0_artifact_manifest",
            inputs.n0_artifact_manifest,
            content_sha256=None,
        ),
    )
    return ObservationParityArtifact.build(
        task_id=task_id,
        source_binding=source_binding,
        evidence=evidence,
        gates=gates,
        limitations=OBSERVATION_PARITY_LIMITATIONS,
    )


def build_and_write_observation_parity_artifact(
    *,
    deployment_root: Path,
    task_id: str,
    inputs: ObservationParityInputs,
    output_path: Path,
) -> ObservationParityArtifact:
    artifact = build_observation_parity_artifact(
        deployment_root=deployment_root,
        task_id=task_id,
        inputs=inputs,
    )
    write_observation_parity_artifact(output_path, artifact)
    return artifact


__all__ = [
    "EXPERIMENT_LOCK_EVIDENCE_LEVEL",
    "EXPERIMENT_LOCK_SCHEMA_VERSION",
    "OBSERVATION_PARITY_LIMITATIONS",
    "ObservationParityInputs",
    "TEACHER_FORCED_EVIDENCE_LEVEL",
    "build_and_write_observation_parity_artifact",
    "build_observation_parity_artifact",
]
