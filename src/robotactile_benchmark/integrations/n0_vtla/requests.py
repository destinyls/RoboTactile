"""Content-bound live request generation for the official N0-VTLA policy."""

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
from robotactile_benchmark.integrations.n0_vtla.artifacts import (
    N0VTLAArtifactManifest,
    validate_n0_vtla_artifact,
)
from robotactile_benchmark.trials import Condition, system_manifest_hash


@dataclass(frozen=True)
class GeneratedN0VTLARequest:
    """One typed request and its canonical file identity."""

    request: LiveUniVTACRunRequest
    request_path: Path
    request_file_sha256: str
    trial_manifest_sha256: str


def build_official_n0_vtla_request(
    *,
    manifest: N0VTLAArtifactManifest,
    layout: DeploymentLayout,
    condition: Condition,
    dataset_sha256: str,
    initial_seed: int,
    exogenous_seed: int,
    max_control_cycles: int,
    max_observation_steps: int,
    wall_timeout_s: float,
    simulator_device: str,
    live_output_dir: Path,
    fault_manifest_path: Optional[Path] = None,
    success_profile_id: UniVTACSuccessProfile = UniVTACSuccessProfile.OFFICIAL_V1,
) -> LiveUniVTACRunRequest:
    """Bind source, checkpoint, prompt, and fault identities into one request."""

    if type(manifest) is not N0VTLAArtifactManifest:
        raise TypeError("manifest must be an exact N0VTLAArtifactManifest")
    if type(layout) is not DeploymentLayout:
        raise TypeError("layout must be an exact DeploymentLayout")
    selected_condition = Condition(condition)
    validate_n0_vtla_artifact(manifest)
    system_id = (
        f"official-n0-vtla.{manifest.task_id}.{manifest.checkpoint_revision[:16]}"
    )
    base_manifest = system_manifest_hash(
        system_id,
        manifest.checkpoint_sha256,
        manifest.config_sha256,
        QPOS8_ACTION_SPEC,
    )
    request = LiveUniVTACRunRequest(
        task_id=manifest.task_id,
        condition=selected_condition,
        policy_kind=LivePolicyKind.N0_VTLA,
        base_system_id=system_id,
        dataset_sha256=dataset_sha256,
        checkpoint_sha256=manifest.checkpoint_sha256,
        config_sha256=manifest.config_sha256,
        base_system_manifest_sha256=base_manifest,
        initial_seed=initial_seed,
        exogenous_seed=exogenous_seed,
        max_control_cycles=max_control_cycles,
        max_observation_steps=max_observation_steps,
        execute_action_steps=50,
        wall_timeout_s=wall_timeout_s,
        upstream_root=layout.sources / "UniVTAC",
        runtime_dir=(layout.runtime / "live-univtac" / "n0-vtla" / manifest.task_id),
        output_dir=Path(live_output_dir).absolute(),
        fault_manifest_path=(
            None
            if fault_manifest_path is None
            else Path(fault_manifest_path).absolute()
        ),
        rest_references_path=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name=None,
        simulator_device=simulator_device,
        launcher_args=production_univtac_launcher_args(),
        success_profile_id=success_profile_id,
        n0_source_commit=manifest.external_commit,
        n0_normalizer_sha256=manifest.normalizer_sha256,
        n0_serve_bundle_sha256=manifest.serve_bundle_sha256,
        n0_prompt_manifest_sha256=manifest.prompt_manifest_sha256,
    )
    load_live_univtac_run(request)
    return request


def write_official_n0_vtla_request(
    request_path: Path,
    request: LiveUniVTACRunRequest,
) -> GeneratedN0VTLARequest:
    """Publish canonical request bytes idempotently without clobbering drift."""

    if request.policy_kind is not LivePolicyKind.N0_VTLA:
        raise ValueError("request must select policy_kind=n0_vtla")
    target = Path(request_path).absolute()
    payload = canonical_json_bytes(live_univtac_request_to_dict(request))
    _publish_canonical_json(target, payload)
    loaded = load_live_univtac_run(request)
    return GeneratedN0VTLARequest(
        request=request,
        request_path=target,
        request_file_sha256=sha256_bytes(payload),
        trial_manifest_sha256=loaded.trial.sha256,
    )


def _publish_canonical_json(target: Path, payload: bytes) -> None:
    if target.is_symlink():
        raise ValueError("N0-VTLA request path cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different N0-VTLA request")
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
                "concurrent N0-VTLA request publication disagrees"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "GeneratedN0VTLARequest",
    "build_official_n0_vtla_request",
    "write_official_n0_vtla_request",
]
