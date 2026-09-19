"""Content-bound live request generation for the official UniVTAC ACT."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_success_profiles import (
    UniVTACSuccessProfile,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    sha256_bytes,
)
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
    production_univtac_launcher_args,
)
from robotactile_benchmark.execution.loading import load_live_univtac_run
from robotactile_benchmark.execution.request_values import (
    live_univtac_request_to_dict,
)
from robotactile_benchmark.integrations.act.artifacts import ACTArtifactManifest
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.policies.univtac_official_act_loading import (
    validate_official_univtac_act_artifact,
)
from robotactile_benchmark.trials import Condition, system_manifest_hash


@dataclass(frozen=True)
class GeneratedACTRequest:
    """One typed ACT request and its canonical file identity."""

    request: LiveUniVTACRunRequest
    request_path: Path
    request_file_sha256: str
    trial_manifest_sha256: str


def _system_id(manifest: ACTArtifactManifest) -> str:
    return (
        f"official-univtac-act.{manifest.task_id}.{manifest.profile.value}."
        "policy_last.v1"
    )


def _validate_manifest_pair(
    base_manifest: ACTArtifactManifest,
    execution_manifest: ACTArtifactManifest,
) -> None:
    if execution_manifest.task_id != base_manifest.task_id:
        raise ValueError("ACT base/execution manifest task mismatch")
    shared_fields = (
        "artifact_root",
        "upstream_root",
        "upstream_commit",
        "source_path",
        "source_sha256",
        "encoder_path",
        "encoder_sha256",
    )
    for name in shared_fields:
        if getattr(execution_manifest, name) != getattr(base_manifest, name):
            raise ValueError(f"ACT base/execution manifest {name} mismatch")


def _select_execution_manifest(
    *,
    base_manifest: ACTArtifactManifest,
    execution_manifest: Optional[ACTArtifactManifest],
    condition: Condition,
) -> ACTArtifactManifest:
    if base_manifest.profile is not OfficialACTProfile.UNIVTAC:
        raise ValueError("ACT base manifest must select the univtac profile")
    if condition is Condition.NO_TOUCH:
        if execution_manifest is None:
            raise ValueError("ACT no-touch requires a vision_only execution manifest")
        if type(execution_manifest) is not ACTArtifactManifest:
            raise TypeError("execution_manifest must be an exact ACTArtifactManifest")
        if execution_manifest.profile is not OfficialACTProfile.VISION_ONLY:
            raise ValueError("ACT no-touch execution manifest must select vision_only")
        _validate_manifest_pair(base_manifest, execution_manifest)
        return execution_manifest
    if execution_manifest is not None and execution_manifest != base_manifest:
        raise ValueError(
            "ACT clean/faulted execution must use the univtac base manifest"
        )
    return base_manifest


def build_official_act_request(
    *,
    base_manifest: ACTArtifactManifest,
    execution_manifest: Optional[ACTArtifactManifest] = None,
    layout: DeploymentLayout,
    condition: Condition,
    dataset_sha256: str,
    initial_seed: int,
    exogenous_seed: int,
    max_control_cycles: int,
    max_observation_steps: int,
    wall_timeout_s: float,
    act_device_name: str,
    simulator_device: str,
    live_output_dir: Path,
    fault_manifest_path: Optional[Path] = None,
    rest_references_path: Optional[Path] = None,
    success_profile_id: UniVTACSuccessProfile = UniVTACSuccessProfile.OFFICIAL_V1,
) -> LiveUniVTACRunRequest:
    """Build one official ACT Clean, Faulted, or matched No-touch request."""

    if type(base_manifest) is not ACTArtifactManifest:
        raise TypeError("base_manifest must be an exact ACTArtifactManifest")
    if type(layout) is not DeploymentLayout:
        raise TypeError("layout must be an exact DeploymentLayout")
    selected_condition = Condition(condition)
    selected_manifest = _select_execution_manifest(
        base_manifest=base_manifest,
        execution_manifest=execution_manifest,
        condition=selected_condition,
    )
    validate_official_univtac_act_artifact(base_manifest)
    if selected_manifest is not base_manifest:
        validate_official_univtac_act_artifact(selected_manifest)

    base_system_id = _system_id(base_manifest)
    base_system_manifest_sha256 = system_manifest_hash(
        base_system_id,
        base_manifest.checkpoint_sha256,
        base_manifest.config_sha256,
        QPOS8_ACTION_SPEC,
    )
    no_touch = selected_condition is Condition.NO_TOUCH
    request = LiveUniVTACRunRequest(
        task_id=base_manifest.task_id,
        condition=selected_condition,
        policy_kind=LivePolicyKind.ACT,
        base_system_id=base_system_id,
        dataset_sha256=dataset_sha256,
        checkpoint_sha256=selected_manifest.checkpoint_sha256,
        config_sha256=selected_manifest.config_sha256,
        base_system_manifest_sha256=base_system_manifest_sha256,
        initial_seed=initial_seed,
        exogenous_seed=exogenous_seed,
        max_control_cycles=max_control_cycles,
        max_observation_steps=max_observation_steps,
        execute_action_steps=1,
        wall_timeout_s=wall_timeout_s,
        upstream_root=selected_manifest.upstream_root,
        runtime_dir=(
            layout.runtime
            / "live-univtac"
            / "act"
            / base_manifest.task_id
            / selected_manifest.profile.value
        ),
        output_dir=Path(live_output_dir).absolute(),
        fault_manifest_path=(
            None
            if fault_manifest_path is None
            else Path(fault_manifest_path).absolute()
        ),
        rest_references_path=(
            None
            if rest_references_path is None
            else Path(rest_references_path).absolute()
        ),
        matched_no_touch_system_id=(
            _system_id(selected_manifest) if no_touch else None
        ),
        matched_no_touch_artifact_path=(
            selected_manifest.checkpoint_path if no_touch else None
        ),
        act_device_name=act_device_name,
        simulator_device=simulator_device,
        launcher_args=production_univtac_launcher_args(),
        success_profile_id=success_profile_id,
        n0_source_commit=None,
        n0_normalizer_sha256=None,
        n0_serve_bundle_sha256=None,
        n0_prompt_manifest_sha256=None,
    )
    load_live_univtac_run(request)
    return request


def write_official_act_request(
    request_path: Path,
    request: LiveUniVTACRunRequest,
) -> GeneratedACTRequest:
    """Publish canonical request bytes idempotently without clobbering drift."""

    if request.policy_kind is not LivePolicyKind.ACT:
        raise ValueError("request must select policy_kind=act")
    target = Path(request_path).absolute()
    payload = canonical_json_bytes(live_univtac_request_to_dict(request))
    _publish_canonical_json(target, payload)
    loaded = load_live_univtac_run(request)
    return GeneratedACTRequest(
        request=request,
        request_path=target,
        request_file_sha256=sha256_bytes(payload),
        trial_manifest_sha256=loaded.trial.sha256,
    )


def _publish_canonical_json(target: Path, payload: bytes) -> None:
    if target.is_symlink():
        raise ValueError("ACT request path cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different ACT request")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError:
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(
                "concurrent ACT request publication disagrees"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "GeneratedACTRequest",
    "build_official_act_request",
    "write_official_act_request",
]
