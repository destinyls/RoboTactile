#!/usr/bin/env python3
"""Prepare canonical official ACT Clean/Faulted/No-touch requests."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from robotactile_benchmark.backends.univtac_contracts import (
    resolve_univtac_full_horizon_budget,
)
from robotactile_benchmark.backends.univtac_success_profiles import (
    UniVTACSuccessProfile,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    resolve_deployment_root,
)
from robotactile_benchmark.integrations.act.requests import (
    GeneratedACTRequest,
    build_official_act_request,
    write_official_act_request,
)
from robotactile_benchmark.integrations.runtime_config import (
    ACTRuntimeArtifacts,
    resolve_act_runtime_artifacts,
)
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.trials import Condition

SUMMARY_SCHEMA = "robotactile-act-request-generation-summary-v1"
EVIDENCE_LEVEL = "request_generation_only_no_model_or_simulator_execution_v1"


@dataclass(frozen=True)
class PreparationSpec:
    deployment_root: Path
    task_id: str
    dataset_sha256: str
    initial_seed: int
    exogenous_seed: int
    max_control_cycles: int
    max_observation_steps: int
    wall_timeout_s: float
    simulator_device: str
    univtac_config_path: Optional[Path] = None
    vision_only_config_path: Optional[Path] = None
    include_no_touch: bool = False
    fault_manifest_path: Optional[Path] = None
    rest_references_path: Optional[Path] = None
    output_root: Optional[Path] = None
    live_output_root: Optional[Path] = None
    success_profile_id: UniVTACSuccessProfile = UniVTACSuccessProfile.OFFICIAL_V1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--initial-seed", required=True, type=int)
    parser.add_argument("--exogenous-seed", required=True, type=int)
    parser.add_argument(
        "--max-control-cycles",
        type=int,
        help="defaults to the selected task's frozen action horizon",
    )
    parser.add_argument(
        "--max-observation-steps",
        type=int,
        help="defaults to max-control-cycles + 1",
    )
    parser.add_argument("--wall-timeout-s", type=float, default=1800.0)
    parser.add_argument("--simulator-device", default="cuda:0")
    parser.add_argument("--univtac-config", type=Path)
    parser.add_argument("--vision-only-config", type=Path)
    parser.add_argument("--include-no-touch", action="store_true")
    parser.add_argument("--fault-manifest", type=Path)
    parser.add_argument("--rest-references", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--live-output-root", type=Path)
    parser.add_argument(
        "--success-profile",
        choices=tuple(item.value for item in UniVTACSuccessProfile),
        default=UniVTACSuccessProfile.OFFICIAL_V1.value,
    )
    return parser


def _default_config(
    layout: DeploymentLayout,
    task_id: str,
    profile: OfficialACTProfile,
) -> Path:
    return (
        layout.model_artifacts
        / "act"
        / "configs"
        / task_id
        / profile.value
        / "integration_config.json"
    )


def _load_profile(
    path: Path,
    *,
    task_id: str,
    profile: OfficialACTProfile,
) -> ACTRuntimeArtifacts:
    runtime = resolve_act_runtime_artifacts(Path(path).expanduser().absolute())
    if runtime.manifest.task_id != task_id:
        raise ValueError(f"ACT {profile.value} config task mismatch")
    if runtime.manifest.profile is not profile:
        raise ValueError(f"ACT config must select profile={profile.value}")
    return runtime


def _build_request(
    *,
    spec: PreparationSpec,
    layout: DeploymentLayout,
    base: ACTRuntimeArtifacts,
    condition: Condition,
    live_output_dir: Path,
    execution: Optional[ACTRuntimeArtifacts] = None,
) -> GeneratedACTRequest:
    request = build_official_act_request(
        base_manifest=base.manifest,
        execution_manifest=None if execution is None else execution.manifest,
        layout=layout,
        condition=condition,
        dataset_sha256=spec.dataset_sha256,
        initial_seed=spec.initial_seed,
        exogenous_seed=spec.exogenous_seed,
        max_control_cycles=spec.max_control_cycles,
        max_observation_steps=spec.max_observation_steps,
        wall_timeout_s=spec.wall_timeout_s,
        act_device_name=base.device if execution is None else execution.device,
        simulator_device=spec.simulator_device,
        live_output_dir=live_output_dir,
        fault_manifest_path=(
            spec.fault_manifest_path if condition is Condition.FAULTED else None
        ),
        rest_references_path=(
            spec.rest_references_path if condition is Condition.FAULTED else None
        ),
        success_profile_id=spec.success_profile_id,
    )
    if spec.output_root is None:
        raise ValueError("normalized ACT preparation spec lacks output_root")
    request_root = Path(spec.output_root).expanduser().absolute()
    return write_official_act_request(
        request_root / f"{condition.value}.json",
        request,
    )


def _profile_summary(
    runtime: ACTRuntimeArtifacts,
    integration_config_path: Path,
) -> dict[str, object]:
    manifest = runtime.manifest
    return {
        "artifact_manifest_path": str(runtime.manifest_path),
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_sha256": manifest.config_sha256,
        "device": runtime.device,
        "integration_config_path": str(integration_config_path),
        "manifest_profile": manifest.profile.value,
        "stats_sha256": manifest.stats_sha256,
    }


def _request_summary(generated: GeneratedACTRequest) -> dict[str, object]:
    request = generated.request
    return {
        "condition": request.condition.value,
        "live_output_dir": str(request.output_dir),
        "matched_no_touch_artifact": (
            None
            if request.matched_no_touch_artifact_path is None
            else str(request.matched_no_touch_artifact_path)
        ),
        "matched_no_touch_system_id": request.matched_no_touch_system_id,
        "request_file_sha256": generated.request_file_sha256,
        "request_path": str(generated.request_path),
        "trial_manifest_sha256": generated.trial_manifest_sha256,
    }


def _publish_summary(path: Path, summary: dict[str, object]) -> None:
    target = Path(path).absolute()
    if target.is_symlink():
        raise ValueError("ACT request summary path cannot be a symlink")
    payload = canonical_json_bytes(summary)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different ACT summary")
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
                "concurrent ACT request summary publication disagrees"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def prepare_requests(spec: PreparationSpec) -> dict[str, object]:
    """Load validated profile configs and publish the selected request group."""

    if type(spec) is not PreparationSpec:
        raise TypeError("spec must be an exact PreparationSpec")
    if spec.rest_references_path is not None and spec.fault_manifest_path is None:
        raise ValueError("--rest-references requires --fault-manifest")
    root = Path(spec.deployment_root).expanduser().absolute()
    layout = DeploymentLayout(root)
    request_root = (
        layout.requests / "act" / spec.task_id
        if spec.output_root is None
        else Path(spec.output_root).expanduser().absolute()
    )
    live_root = (
        layout.artifacts / "live-univtac" / "act" / spec.task_id
        if spec.live_output_root is None
        else Path(spec.live_output_root).expanduser().absolute()
    )
    normalized = PreparationSpec(
        deployment_root=root,
        task_id=spec.task_id,
        dataset_sha256=spec.dataset_sha256,
        initial_seed=spec.initial_seed,
        exogenous_seed=spec.exogenous_seed,
        max_control_cycles=spec.max_control_cycles,
        max_observation_steps=spec.max_observation_steps,
        wall_timeout_s=spec.wall_timeout_s,
        simulator_device=spec.simulator_device,
        univtac_config_path=spec.univtac_config_path,
        vision_only_config_path=spec.vision_only_config_path,
        include_no_touch=spec.include_no_touch,
        fault_manifest_path=(
            None
            if spec.fault_manifest_path is None
            else Path(spec.fault_manifest_path).expanduser().absolute()
        ),
        rest_references_path=(
            None
            if spec.rest_references_path is None
            else Path(spec.rest_references_path).expanduser().absolute()
        ),
        output_root=request_root,
        live_output_root=live_root,
        success_profile_id=spec.success_profile_id,
    )
    base_config = (
        Path(
            spec.univtac_config_path
            or _default_config(layout, spec.task_id, OfficialACTProfile.UNIVTAC)
        )
        .expanduser()
        .absolute()
    )
    base = _load_profile(
        base_config,
        task_id=spec.task_id,
        profile=OfficialACTProfile.UNIVTAC,
    )
    include_no_touch = spec.include_no_touch or spec.vision_only_config_path is not None
    vision: Optional[ACTRuntimeArtifacts] = None
    vision_config: Optional[Path] = None
    if include_no_touch:
        vision_config = (
            Path(
                spec.vision_only_config_path
                or _default_config(layout, spec.task_id, OfficialACTProfile.VISION_ONLY)
            )
            .expanduser()
            .absolute()
        )
        vision = _load_profile(
            vision_config,
            task_id=spec.task_id,
            profile=OfficialACTProfile.VISION_ONLY,
        )

    generated = [
        _build_request(
            spec=normalized,
            layout=layout,
            base=base,
            condition=Condition.CLEAN,
            live_output_dir=live_root / Condition.CLEAN.value,
        )
    ]
    if normalized.fault_manifest_path is not None:
        generated.append(
            _build_request(
                spec=normalized,
                layout=layout,
                base=base,
                condition=Condition.FAULTED,
                live_output_dir=live_root / Condition.FAULTED.value,
            )
        )
    if vision is not None:
        generated.append(
            _build_request(
                spec=normalized,
                layout=layout,
                base=base,
                execution=vision,
                condition=Condition.NO_TOUCH,
                live_output_dir=live_root / Condition.NO_TOUCH.value,
            )
        )

    summary_path = request_root / "request_generation_summary.json"
    summary: dict[str, object] = {
        "budgets": {
            "max_control_cycles": normalized.max_control_cycles,
            "max_observation_steps": normalized.max_observation_steps,
            "wall_timeout_s": normalized.wall_timeout_s,
        },
        "dataset_sha256": normalized.dataset_sha256,
        "evidence_level": EVIDENCE_LEVEL,
        "exogenous_seed": normalized.exogenous_seed,
        "initial_seed": normalized.initial_seed,
        "live_execution_claimed": False,
        "policy_kind": "act",
        "profiles": {
            "univtac": _profile_summary(base, base_config),
            "vision_only": (
                None
                if vision is None or vision_config is None
                else _profile_summary(vision, vision_config)
            ),
        },
        "request_count": len(generated),
        "requests": [_request_summary(item) for item in generated],
        "schema_version": SUMMARY_SCHEMA,
        "simulator_device": normalized.simulator_device,
        "status": "ready",
        "summary_path": str(summary_path),
        "task_id": normalized.task_id,
    }
    _publish_summary(summary_path, summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    root = resolve_deployment_root(args.root)
    max_control_cycles, max_observation_steps = resolve_univtac_full_horizon_budget(
        args.task,
        max_control_cycles=args.max_control_cycles,
        max_observation_steps=args.max_observation_steps,
    )
    summary = prepare_requests(
        PreparationSpec(
            deployment_root=root,
            task_id=args.task,
            dataset_sha256=args.dataset_sha256,
            initial_seed=args.initial_seed,
            exogenous_seed=args.exogenous_seed,
            max_control_cycles=max_control_cycles,
            max_observation_steps=max_observation_steps,
            wall_timeout_s=args.wall_timeout_s,
            simulator_device=args.simulator_device,
            univtac_config_path=args.univtac_config,
            vision_only_config_path=args.vision_only_config,
            include_no_touch=args.include_no_touch,
            fault_manifest_path=args.fault_manifest,
            rest_references_path=args.rest_references,
            output_root=args.output_root,
            live_output_root=args.live_output_root,
            success_profile_id=UniVTACSuccessProfile(args.success_profile),
        )
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
