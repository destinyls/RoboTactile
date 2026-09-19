#!/usr/bin/env python3
"""Prepare one no-clobber FTP-1 Clean/Faulted paired request group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.calibration import load_rest_reference_artifact
from robotactile_benchmark.calibration.contracts import (
    REST_REFERENCE_PATH,
    ROOT_RECEIPT_PATH,
    VALIDATION_PATH,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.constants import (
    DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
    SENSOR_SLOTS,
    SEVERITY_REGISTRY_ID,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    resolve_deployment_root,
)
from robotactile_benchmark.execution.contracts import LiveUniVTACRunRequest
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.request_values import (
    live_univtac_request_to_dict,
)
from robotactile_benchmark.integrations.ftp1_policy.artifacts import (
    TASK_RELEASES,
    FTP1PolicyArtifactManifest,
)
from robotactile_benchmark.integrations.ftp1_policy.requests import (
    build_official_ftp1_policy_request,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_ftp1_policy_runtime_artifacts,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.severity import severity_value
from robotactile_benchmark.trials import Condition

SCHEMA_VERSION = "robotactile-ftp1-robustness-plan-v1"
EVIDENCE_LEVEL = "request_generation_only_no_model_or_simulator_execution_v1"
PLAN_RECEIPT = "robustness_plan_receipt.json"
EXECUTABLE_OPERATORS = (
    "F1_global_response_drift",
    "F2_spatial_sensitivity_loss",
    "F3_persistent_surface_artifact",
    "F4_local_nonresponsive_patch",
    "F5_contact_shape_distortion",
    "F6_history_residual_imprint",
    "F7_high_load_saturation",
    "T1_fixed_source_delay",
    "T2_held_last_freeze",
    "T3_inter_sensor_skew",
    "C1_sensor_identity_misrouting",
    "C2_frame_misregistration",
)
UNSUPPORTED_OPERATORS = ("A1_stream_absence", "A2_frame_erasure")


@dataclass(frozen=True)
class SeverityProfile:
    profile_id: str
    registry_id: str
    level: int
    evidence_scope: str

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_scope": self.evidence_scope,
            "profile_id": self.profile_id,
            "registry_id": self.registry_id,
            "severity_level": self.level,
        }


@dataclass(frozen=True)
class PreparationSpec:
    deployment_root: Path
    config_path: Path
    task_id: str
    dataset_sha256: str
    initial_seed: int
    exogenous_seed: int
    severity_profile: str
    rest_reference_root: Path
    operator_seed_master: int
    fault_start_index: int
    fault_stop_index: Optional[int]
    max_control_cycles: Optional[int]
    max_observation_steps: Optional[int]
    wall_timeout_s: float
    device: str
    output_root: Optional[Path]
    live_output_root: Optional[Path]
    paired_receipt: Optional[Path]


def _severity_profile(value: str) -> SeverityProfile:
    aliases = {f"s{i}": f"registered_s{i}" for i in range(1, 6)}
    normalized = aliases.get(value, value)
    registered = {f"registered_s{i}": i for i in range(1, 6)}
    if normalized in registered:
        return SeverityProfile(
            normalized,
            SEVERITY_REGISTRY_ID,
            registered[normalized],
            "registered_s1_s5",
        )
    if normalized == "diagnostic_stress_max":
        return SeverityProfile(
            normalized,
            DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
            5,
            "diagnostic_non_paper",
        )
    raise ValueError("unsupported severity profile")


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a non-symlink regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def _tree_hashes(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise FileExistsError(f"output must be a non-symlink directory: {root}")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FileExistsError(f"output tree cannot contain symlinks: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _file_sha256(path)
    return result


def _publish_tree(staging: Path, output: Path) -> str:
    if output.is_symlink() or output.parent.is_symlink():
        raise FileExistsError("output path cannot be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.publish.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if output.exists():
            if _tree_hashes(output) == _tree_hashes(staging):
                return "already_present"
            raise FileExistsError(f"refusing to replace different plan: {output}")
        os.rename(staging, output)
        return "created"
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def _relative(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start=start)).as_posix()


def _write_request(
    staging: Path,
    relative_path: str,
    request: LiveUniVTACRunRequest,
) -> tuple[str, LoadedLiveUniVTACRun]:
    target = staging / relative_path
    document = live_univtac_request_to_dict(request)
    for field in ("fault_manifest_path", "rest_references_path"):
        path = getattr(request, field)
        document[field] = None if path is None else _relative(path, target.parent)
    _write_json(target, document)
    loaded = load_live_univtac_run(load_live_univtac_request(target))
    return _file_sha256(target), loaded


def _operator_seed(spec: PreparationSpec, pair_key: str, operator_id: str) -> int:
    digest = canonical_hash(
        {
            "master_seed": spec.operator_seed_master,
            "operator_id": operator_id,
            "pair_key": pair_key,
            "task_id": spec.task_id,
        }
    )
    return int(digest[:8], 16) & 0x7FFFFFFF


def _fault(
    operator_id: str,
    profile: SeverityProfile,
    operator_seed: int,
    start: int,
    stop: int,
    rest_sha256: str,
) -> FaultManifest:
    parameters: dict[str, object] = {}
    if operator_requires_rest_reference(
        operator_id, severity_registry=profile.registry_id
    ):
        parameters["rest_reference_sha256"] = rest_sha256
    if operator_id == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    return FaultManifest(
        operator_id=operator_id,
        severity_level=profile.level,
        operator_seed=operator_seed,
        start_index=start,
        stop_index=stop,
        sensor_slots=SENSOR_SLOTS,
        observability=Observability.BLIND,
        parameters=parameters,
        severity_registry=profile.registry_id,
    )


def _operator_start_index(
    operator_id: str,
    profile: SeverityProfile,
    requested_start: int,
) -> int:
    """Resolve the earliest causally valid start for one operator."""

    if operator_id != "T1_fixed_source_delay":
        return requested_start
    lag = int(
        severity_value(
            operator_id,
            profile.level,
            registry_id=profile.registry_id,
        )
    )
    return max(requested_start, lag)


def _copy_rest_reference(source: Path, staging: Path) -> Path:
    target = staging / "rest_reference_artifact"
    target.mkdir()
    for name in (REST_REFERENCE_PATH, VALIDATION_PATH, ROOT_RECEIPT_PATH):
        with (target / name).open("xb") as stream:
            stream.write((source / name).read_bytes())
    return target


def _build_request(
    *,
    manifest: FTP1PolicyArtifactManifest,
    layout: DeploymentLayout,
    spec: PreparationSpec,
    max_cycles: int,
    max_observations: int,
    condition: Condition,
    live_output_dir: Path,
    fault_manifest_path: Optional[Path] = None,
    rest_references_path: Optional[Path] = None,
) -> LiveUniVTACRunRequest:
    return build_official_ftp1_policy_request(
        manifest=manifest,
        layout=layout,
        condition=condition,
        dataset_sha256=spec.dataset_sha256,
        initial_seed=spec.initial_seed,
        exogenous_seed=spec.exogenous_seed,
        max_control_cycles=max_cycles,
        max_observation_steps=max_observations,
        wall_timeout_s=spec.wall_timeout_s,
        simulator_device=spec.device,
        live_output_dir=live_output_dir,
        fault_manifest_path=fault_manifest_path,
        rest_references_path=rest_references_path,
    )


def prepare_robustness_group(spec: PreparationSpec) -> dict[str, object]:
    """Generate and atomically publish one directly runnable paired plan."""

    layout = DeploymentLayout(Path(spec.deployment_root).expanduser().absolute())
    config_path = Path(spec.config_path).expanduser().absolute()
    runtime = resolve_ftp1_policy_runtime_artifacts(config_path)
    manifest = runtime.manifest
    if spec.task_id != manifest.task_id or spec.task_id not in TASK_RELEASES:
        raise ValueError("task does not match the released FTP-1 artifact")
    profile = _severity_profile(spec.severity_profile)
    backend = build_univtac_backend_config(spec.task_id, action_spec=QPOS8_ACTION_SPEC)
    max_cycles = (
        backend.task.action_horizon
        if spec.max_control_cycles is None
        else spec.max_control_cycles
    )
    max_observations = (
        backend.task.action_horizon + 1
        if spec.max_observation_steps is None
        else spec.max_observation_steps
    )
    fault_stop = (
        max_observations if spec.fault_stop_index is None else spec.fault_stop_index
    )
    if not 0 <= spec.fault_start_index < fault_stop <= max_observations:
        raise ValueError("fault window must lie inside the observation budget")
    if max_cycles >= max_observations:
        raise ValueError("max_control_cycles must be below max_observation_steps")
    operator_start_indices = {
        operator_id: _operator_start_index(
            operator_id,
            profile,
            spec.fault_start_index,
        )
        for operator_id in EXECUTABLE_OPERATORS
    }
    invalid_starts = {
        operator_id: start_index
        for operator_id, start_index in operator_start_indices.items()
        if start_index >= fault_stop
    }
    if invalid_starts:
        details = ", ".join(
            f"{operator_id}={start_index}"
            for operator_id, start_index in invalid_starts.items()
        )
        raise ValueError(
            "operator fault start must be below fault_stop_index "
            f"({fault_stop}): {details}"
        )
    source_rest = Path(spec.rest_reference_root).expanduser().absolute()
    rest = load_rest_reference_artifact(source_rest)
    if rest.validation.task != spec.task_id:
        raise ValueError("rest-reference task does not match FTP-1 task")
    plan_contract = {
        "budgets": {
            "max_control_cycles": max_cycles,
            "max_observation_steps": max_observations,
            "simulator_device": spec.device,
            "wall_timeout_s": spec.wall_timeout_s,
        },
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_sha256": manifest.config_sha256,
        "dataset_sha256": spec.dataset_sha256,
        "fault_window": {
            "operator_start_indices": operator_start_indices,
            "start_index": spec.fault_start_index,
            "stop_index": fault_stop,
        },
        "initial_seed": spec.initial_seed,
        "exogenous_seed": spec.exogenous_seed,
        "operators": EXECUTABLE_OPERATORS,
        "rest_reference_sha256": rest.references.sha256,
        "severity_profile": profile.to_dict(),
        "task_id": spec.task_id,
        "unsupported_operators": UNSUPPORTED_OPERATORS,
    }
    contract_sha = canonical_hash(plan_contract)
    plan_id = f"ftp1-{spec.task_id}-{contract_sha[:16]}"
    output = (
        layout.requests / "ftp1-policy" / "robustness" / plan_id
        if spec.output_root is None
        else Path(spec.output_root).expanduser().absolute()
    )
    live_root = (
        layout.outputs / "ftp1-policy" / "robustness" / plan_id
        if spec.live_output_root is None
        else Path(spec.live_output_root).expanduser().absolute()
    )
    paired_receipt = (
        live_root / "paired_execution_receipt.json"
        if spec.paired_receipt is None
        else Path(spec.paired_receipt).expanduser().absolute()
    )
    if (
        output == live_root
        or output in live_root.parents
        or live_root in output.parents
    ):
        raise ValueError("plan and live output roots must be disjoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{plan_id}.staging-", dir=output.parent))
    try:
        copied_rest_root = _copy_rest_reference(source_rest, staging)
        copied_rest = load_rest_reference_artifact(copied_rest_root)
        if copied_rest.root_receipt_sha256 != rest.root_receipt_sha256:
            raise RuntimeError("copied rest-reference identity changed")
        rest_bundle = copied_rest_root / REST_REFERENCE_PATH
        clean_request = _build_request(
            manifest=manifest,
            layout=layout,
            spec=spec,
            max_cycles=max_cycles,
            max_observations=max_observations,
            condition=Condition.CLEAN,
            live_output_dir=live_root / "00-clean",
        )
        clean_rel = "requests/00-clean.json"
        clean_hash, clean_loaded = _write_request(staging, clean_rel, clean_request)
        pair_key = clean_loaded.trial.pair_key
        ordered: list[dict[str, object]] = [
            {
                "condition": "clean",
                "fault_manifest_path": None,
                "fault_manifest_sha256": None,
                "fault_start_index": None,
                "live_output_dir": str(clean_request.output_dir),
                "operator_id": None,
                "ordinal": 0,
                "request_file_sha256": clean_hash,
                "request_path": clean_rel,
                "severity_level": None,
                "trial_manifest_sha256": clean_loaded.trial.sha256,
            }
        ]
        for ordinal, operator_id in enumerate(EXECUTABLE_OPERATORS, start=1):
            fault = _fault(
                operator_id,
                profile,
                _operator_seed(spec, pair_key, operator_id),
                operator_start_indices[operator_id],
                fault_stop,
                copied_rest.references.sha256,
            )
            fault_rel = f"fault_manifests/{ordinal:02d}-{operator_id}.json"
            _write_json(staging / fault_rel, fault.to_dict())
            request_rel = f"requests/{ordinal:02d}-{operator_id}.json"
            request = _build_request(
                manifest=manifest,
                layout=layout,
                spec=spec,
                max_cycles=max_cycles,
                max_observations=max_observations,
                condition=Condition.FAULTED,
                live_output_dir=live_root / f"{ordinal:02d}-{operator_id}",
                fault_manifest_path=staging / fault_rel,
                rest_references_path=(
                    rest_bundle
                    if operator_requires_rest_reference(
                        operator_id, severity_registry=profile.registry_id
                    )
                    else None
                ),
            )
            request_hash, loaded = _write_request(staging, request_rel, request)
            if loaded.trial.pair_key != pair_key:
                raise RuntimeError("generated request changed the paired snapshot key")
            ordered.append(
                {
                    "condition": "faulted",
                    "fault_manifest_path": fault_rel,
                    "fault_manifest_sha256": fault.sha256,
                    "fault_start_index": fault.start_index,
                    "live_output_dir": str(request.output_dir),
                    "operator_id": operator_id,
                    "ordinal": ordinal,
                    "request_file_sha256": request_hash,
                    "request_path": request_rel,
                    "severity_level": profile.level,
                    "trial_manifest_sha256": loaded.trial.sha256,
                }
            )
        unsupported: list[dict[str, object]] = []
        for operator_id in UNSUPPORTED_OPERATORS:
            relpath = f"unsupported_contracts/{operator_id}.json"
            document = {
                "applicability": "N/A",
                "black_frame_used": False,
                "fault_manifest_generated": False,
                "operator_id": operator_id,
                "pair_key": pair_key,
                "policy_kind": "ftp1_policy",
                "reason_code": "ftp1_requires_both_tactile_streams_v1",
                "request_generated": False,
                "status": "unsupported_contract",
                "substituted_payload": False,
                "task_id": spec.task_id,
            }
            _write_json(staging / relpath, document)
            unsupported.append(
                {
                    **document,
                    "path": relpath,
                    "file_sha256": _file_sha256(staging / relpath),
                }
            )
        member_sha256 = _tree_hashes(staging)
        receipt = {
            "artifact_manifest": {
                "checkpoint_revision": manifest.checkpoint_revision,
                "checkpoint_sha256": manifest.checkpoint_sha256,
                "config_sha256": manifest.config_sha256,
                "integration_config_path": str(config_path),
                "integration_config_sha256": _file_sha256(config_path),
                "manifest_path": str(runtime.manifest_path.absolute()),
                "manifest_sha256": _file_sha256(runtime.manifest_path),
                "source_commit": manifest.external_commit,
            },
            "base_system_id": clean_loaded.trial.base_system_id,
            "budgets": plan_contract["budgets"],
            "dataset_sha256": spec.dataset_sha256,
            "evidence_level": EVIDENCE_LEVEL,
            "exogenous_seed": spec.exogenous_seed,
            "fault_window": {
                "operator_start_indices": operator_start_indices,
                "start_index": spec.fault_start_index,
                "stop_index": fault_stop,
            },
            "initial_seed": spec.initial_seed,
            "member_sha256": member_sha256,
            "ordered_requests": ordered,
            "pair_key": pair_key,
            "paired_run": {
                "receipt_path": str(paired_receipt),
                "request_paths": [item["request_path"] for item in ordered],
                "subcommand": "live-univtac-paired-run",
            },
            "plan_contract_sha256": contract_sha,
            "plan_id": plan_id,
            "policy_kind": "ftp1_policy",
            "rest_reference": {
                "bundled_path": "rest_reference_artifact",
                "no_contact_validation_sha256": rest.validation.sha256,
                "rest_reference_sha256": rest.references.sha256,
                "root_receipt_sha256": rest.root_receipt_sha256,
            },
            "schema_version": SCHEMA_VERSION,
            "severity_profile": profile.to_dict(),
            "simulator_execution_claimed": False,
            "task_id": spec.task_id,
            "task_success_claimed": False,
            "unsupported_contracts": unsupported,
        }
        _write_json(staging / PLAN_RECEIPT, receipt)
        status = _publish_tree(staging, output)
        return {
            "ordered_request_paths": [
                str(output / cast(str, item["request_path"])) for item in ordered
            ],
            "output_root": str(output),
            "paired_receipt": str(paired_receipt),
            "plan_id": plan_id,
            "receipt_file_sha256": _file_sha256(output / PLAN_RECEIPT),
            "status": status,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_RELEASES), required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--initial-seed", type=int, required=True)
    parser.add_argument("--exogenous-seed", type=int, required=True)
    parser.add_argument("--severity-profile", default="registered_s5")
    parser.add_argument("--rest-reference", type=Path, required=True)
    parser.add_argument("--operator-seed", type=int, default=20260830)
    parser.add_argument("--fault-start-index", type=int, default=16)
    parser.add_argument("--fault-stop-index", type=int)
    parser.add_argument("--max-control-cycles", type=int)
    parser.add_argument("--max-observation-steps", type=int)
    parser.add_argument("--wall-timeout-s", type=float, default=1800.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--live-output-root", type=Path)
    parser.add_argument("--paired-receipt", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    root = resolve_deployment_root(cast(Optional[Path], args.root))
    spec = PreparationSpec(
        deployment_root=root,
        config_path=cast(Path, args.config).expanduser().absolute(),
        task_id=cast(str, args.task),
        dataset_sha256=cast(str, args.dataset_sha256),
        initial_seed=cast(int, args.initial_seed),
        exogenous_seed=cast(int, args.exogenous_seed),
        severity_profile=cast(str, args.severity_profile),
        rest_reference_root=cast(Path, args.rest_reference),
        operator_seed_master=cast(int, args.operator_seed),
        fault_start_index=cast(int, args.fault_start_index),
        fault_stop_index=cast(Optional[int], args.fault_stop_index),
        max_control_cycles=cast(Optional[int], args.max_control_cycles),
        max_observation_steps=cast(Optional[int], args.max_observation_steps),
        wall_timeout_s=cast(float, args.wall_timeout_s),
        device=cast(str, args.device),
        output_root=cast(Optional[Path], args.output_root),
        live_output_root=cast(Optional[Path], args.live_output_root),
        paired_receipt=cast(Optional[Path], args.paired_receipt),
    )
    try:
        summary = prepare_robustness_group(spec)
    except (FileExistsError, OSError, TypeError, ValueError, RuntimeError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    sys.stdout.write(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
