"""Content-bound clean request generation for official N0-TWAM smoke runs."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    REGISTRY_ID,
    UPSTREAM_COMMIT,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    sha256_bytes,
)
from robotactile_benchmark.closed_loop.contracts import (
    InitialStatePolicy,
    WallTimeoutRole,
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
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    N0TWAMArtifactManifest,
    validate_n0_twam_artifact,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
)
from robotactile_benchmark.trials import Condition, system_manifest_hash

DEFAULT_N0_WATCHDOG_SECONDS_PER_ACTION = 15.0


@dataclass(frozen=True)
class GeneratedN0CleanRequest:
    """One typed request and the hash of its canonical JSON representation."""

    request: LiveUniVTACRunRequest
    request_path: Path
    request_file_sha256: str
    trial_manifest_sha256: str


@dataclass(frozen=True)
class GeneratedN0TrialSet:
    """One frozen trial-set manifest and its canonical content identity."""

    manifest_path: Path
    sha256: str


def derive_n0_infrastructure_watchdog_timeout(
    *,
    requested_floor_s: float,
    action_horizon: int,
    seconds_per_action: float = DEFAULT_N0_WATCHDOG_SECONDS_PER_ACTION,
) -> float:
    """Freeze a non-scoring watchdog budget from the official action horizon."""

    for value, name in (
        (requested_floor_s, "requested_floor_s"),
        (seconds_per_action, "seconds_per_action"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if isinstance(action_horizon, bool) or not isinstance(action_horizon, int):
        raise TypeError("action_horizon must be an integer")
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    return max(
        float(requested_floor_s),
        float(action_horizon) * float(seconds_per_action),
    )


def _publish_canonical_json(target: Path, payload: bytes, label: str) -> None:
    """Publish canonical bytes idempotently without replacing divergent content."""

    target = Path(target).absolute()
    if target.is_symlink():
        raise ValueError(f"{label} path cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace a different {label}")
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
            raise FileExistsError(f"concurrent {label} publication disagrees") from None
    finally:
        temporary.unlink(missing_ok=True)


def write_official_n0_trial_set(
    *,
    manifest_path: Path,
    task_id: str,
    initial_seed: int,
    exogenous_seed: int,
    max_control_cycles: int,
    max_observation_steps: int,
) -> GeneratedN0TrialSet:
    """Freeze the exact task/seeds/horizon contract used by an N0 request."""

    if not task_id:
        raise ValueError("task_id cannot be empty")
    for name, value in (
        ("initial_seed", initial_seed),
        ("exogenous_seed", exogenous_seed),
        ("max_control_cycles", max_control_cycles),
        ("max_observation_steps", max_observation_steps),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if max_control_cycles < 1:
        raise ValueError("max_control_cycles must be positive")
    if max_observation_steps != max_control_cycles + 1:
        raise ValueError("max_observation_steps must equal max_control_cycles + 1")
    document = {
        "identity_kind": "frozen_trial_set_manifest",
        "task_registry_id": REGISTRY_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "task_id": task_id,
        "trial_count": 1,
        "initial_seed": initial_seed,
        "exogenous_seed": exogenous_seed,
        "action_horizon": max_control_cycles,
        "max_observation_steps": max_observation_steps,
        "semantic_version": "1.0",
    }
    payload = canonical_json_bytes(document)
    target = Path(manifest_path).absolute()
    _publish_canonical_json(target, payload, "N0 trial-set manifest")
    return GeneratedN0TrialSet(
        manifest_path=target,
        sha256=sha256_bytes(payload),
    )


def build_official_n0_clean_request(
    *,
    manifest: N0TWAMArtifactManifest,
    layout: DeploymentLayout,
    dataset_sha256: str,
    initial_seed: int,
    exogenous_seed: int,
    max_control_cycles: int,
    max_observation_steps: int,
    wall_timeout_s: float,
    simulator_device: str,
    live_output_dir: Path,
    initial_state_policy: InitialStatePolicy = InitialStatePolicy.OFFICIAL_REPRODUCTION,
    wall_timeout_role: WallTimeoutRole = WallTimeoutRole.SCORING_BOUNDARY_V1,
) -> LiveUniVTACRunRequest:
    """Bind official source/weight identities into one clean live request."""

    if type(manifest) is not N0TWAMArtifactManifest:
        raise TypeError("manifest must be an exact N0TWAMArtifactManifest")
    if type(layout) is not DeploymentLayout:
        raise TypeError("layout must be an exact DeploymentLayout")
    validate_n0_twam_artifact(manifest)
    system_id = (
        f"official-n0-twam.{manifest.task_id}.univtac-delta."
        f"{N0_LIVE_UNIVTAC_INPUT_PROFILE.profile_id}"
    )
    base_manifest = system_manifest_hash(
        system_id,
        manifest.checkpoint_sha256,
        manifest.config_sha256,
        EE8_ACTION_SPEC,
    )
    request = LiveUniVTACRunRequest(
        task_id=manifest.task_id,
        condition=Condition.CLEAN,
        policy_kind=LivePolicyKind.N0,
        base_system_id=system_id,
        dataset_sha256=dataset_sha256,
        checkpoint_sha256=manifest.checkpoint_sha256,
        config_sha256=manifest.config_sha256,
        base_system_manifest_sha256=base_manifest,
        initial_seed=initial_seed,
        exogenous_seed=exogenous_seed,
        max_control_cycles=max_control_cycles,
        max_observation_steps=max_observation_steps,
        execute_action_steps=24,
        wall_timeout_s=wall_timeout_s,
        upstream_root=layout.sources / "UniVTAC",
        runtime_dir=layout.runtime / "live-univtac" / "n0" / manifest.task_id,
        output_dir=Path(live_output_dir).absolute(),
        fault_manifest_path=None,
        rest_references_path=None,
        restoration_index=None,
        restoration_mode=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name=None,
        simulator_device=simulator_device,
        launcher_args=production_univtac_launcher_args(),
        n0_source_commit=manifest.external_commit,
        n0_normalizer_sha256=manifest.normalizer_sha256,
        n0_serve_bundle_sha256=manifest.serve_bundle_sha256,
        n0_prompt_manifest_sha256=manifest.prompt_manifest_sha256,
        initial_state_policy=initial_state_policy,
        wall_timeout_role=wall_timeout_role,
    )
    load_live_univtac_run(request)
    return request


def write_official_n0_clean_request(
    request_path: Path, request: LiveUniVTACRunRequest
) -> GeneratedN0CleanRequest:
    """Publish canonical request bytes idempotently without clobbering drift."""

    target = Path(request_path).absolute()
    payload = canonical_json_bytes(live_univtac_request_to_dict(request))
    _publish_canonical_json(target, payload, "N0 request")
    loaded = load_live_univtac_run(request)
    return GeneratedN0CleanRequest(
        request=request,
        request_path=target,
        request_file_sha256=sha256_bytes(payload),
        trial_manifest_sha256=loaded.trial.sha256,
    )


__all__ = [
    "DEFAULT_N0_WATCHDOG_SECONDS_PER_ACTION",
    "GeneratedN0CleanRequest",
    "GeneratedN0TrialSet",
    "build_official_n0_clean_request",
    "derive_n0_infrastructure_watchdog_timeout",
    "write_official_n0_clean_request",
    "write_official_n0_trial_set",
]
