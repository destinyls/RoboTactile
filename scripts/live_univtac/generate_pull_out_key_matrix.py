#!/usr/bin/env python3
"""Generate one immutable four-condition pull-out-key live request matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from pull_out_key_matrix_io import file_sha256, publish_tree, tree_hashes, write_json
from pull_out_key_matrix_values import artifact_document, trial_set_document
from pull_out_key_rest_reference import (
    RestReferenceBinding,
    copy_rest_reference,
    fault_parameters,
    load_rest_reference_binding,
    request_rest_reference_path,
    rest_reference_receipt,
)

from robotactile_benchmark.backends.univtac_contracts import (
    REGISTRY_ID,
    UPSTREAM_COMMIT,
)
from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC
from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.policies.univtac_official_act_loading import (
    OfficialUniVTACACTArtifactManifest,
)
from robotactile_benchmark.trials import (
    Condition,
    RestorationMode,
    build_paired_trial_grid,
    system_manifest_hash,
)

TASK_ID = "pull_out_key"
ACTION_HORIZON = 300
MAX_OBSERVATION_STEPS = ACTION_HORIZON + 1
BASE_SYSTEM_ID = "official-univtac-act.pull_out_key.univtac.policy_last.v1"
NO_TOUCH_SYSTEM_ID = "official-univtac-act.pull_out_key.vision_only.policy_last.v1"
EVIDENCE_LEVEL = "request_generation_only_no_simulator_execution"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--univtac-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--tactile-checkpoint-sha256", required=True)
    parser.add_argument("--vision-checkpoint-sha256", required=True)
    parser.add_argument("--stats-sha256", required=True)
    parser.add_argument("--encoder-sha256", required=True)
    parser.add_argument("--initial-seed", type=int, required=True)
    parser.add_argument("--exogenous-seed", type=int, required=True)
    parser.add_argument(
        "--operator", choices=sorted(CORE_OPERATOR_IDS), default="T1_fixed_source_delay"
    )
    parser.add_argument("--severity", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--operator-seed", type=int, default=20260821)
    parser.add_argument("--fault-start", type=int, default=16)
    parser.add_argument("--restoration-index", type=int, default=180)
    parser.add_argument("--rest-reference-artifact", type=Path)
    parser.add_argument("--wall-timeout-s", type=float, default=1800.0)
    parser.add_argument("--output", type=Path)
    return parser


def _absolute(path: Path) -> Path:
    return path.expanduser().absolute()


def _artifact_manifest(
    *,
    profile: OfficialACTProfile,
    checkpoint_root: Path,
    univtac_root: Path,
    checkpoint_sha256: str,
    stats_sha256: str,
    encoder_sha256: str,
) -> OfficialUniVTACACTArtifactManifest:
    return OfficialUniVTACACTArtifactManifest.for_shared_root(
        task_id=TASK_ID,
        profile=profile,
        artifact_root=checkpoint_root,
        upstream_root=univtac_root,
        checkpoint_sha256=checkpoint_sha256,
        stats_sha256=stats_sha256,
        encoder_sha256=encoder_sha256,
    )


def _faults(
    args: argparse.Namespace,
    rest_binding: Optional[RestReferenceBinding],
) -> tuple[FaultManifest, FaultManifest]:
    if not 0 <= args.fault_start < args.restoration_index < MAX_OBSERVATION_STEPS:
        raise ValueError(
            "fault/restoration window must satisfy 0 <= start < restoration < 301"
        )
    common = {
        "operator_id": args.operator,
        "severity_level": args.severity,
        "operator_seed": args.operator_seed,
        "start_index": args.fault_start,
        "sensor_slots": ("left", "right"),
        "observability": Observability.BLIND,
        "parameters": fault_parameters(args.operator, rest_binding),
    }
    persistent = FaultManifest(stop_index=MAX_OBSERVATION_STEPS, **common)
    restored = FaultManifest(stop_index=args.restoration_index, **common)
    return persistent, restored


def _request_document(
    *,
    condition: Condition,
    matrix_id: str,
    deployment_root: Path,
    univtac_root: Path,
    dataset_sha256: str,
    tactile: OfficialUniVTACACTArtifactManifest,
    vision: OfficialUniVTACACTArtifactManifest,
    base_manifest_sha256: str,
    initial_seed: int,
    exogenous_seed: int,
    wall_timeout_s: float,
    restoration_index: int,
    rest_binding: Optional[RestReferenceBinding],
) -> dict[str, Any]:
    no_touch = condition is Condition.NO_TOUCH
    faulted = condition in {Condition.FAULTED, Condition.RESTORED}
    profile = vision if no_touch else tactile
    return {
        "task_id": TASK_ID,
        "condition": condition.value,
        "policy_kind": "act",
        "base_system_id": BASE_SYSTEM_ID,
        "dataset_sha256": dataset_sha256,
        "checkpoint_sha256": profile.checkpoint_sha256,
        "config_sha256": profile.config_sha256,
        "base_system_manifest_sha256": base_manifest_sha256,
        "initial_seed": initial_seed,
        "exogenous_seed": exogenous_seed,
        "max_control_cycles": ACTION_HORIZON,
        "max_observation_steps": MAX_OBSERVATION_STEPS,
        "execute_action_steps": 1,
        "wall_timeout_s": wall_timeout_s,
        "upstream_root": str(univtac_root),
        "runtime_dir": str(
            deployment_root / "runtime/live_univtac" / matrix_id / condition.value
        ),
        "output_dir": str(
            deployment_root / "artifacts/live_univtac" / matrix_id / condition.value
        ),
        "fault_manifest_path": (
            f"../fault_manifests/{'restored' if condition is Condition.RESTORED else 'persistent'}.json"
            if faulted
            else None
        ),
        "rest_references_path": request_rest_reference_path(faulted, rest_binding),
        "restoration_index": restoration_index
        if condition is Condition.RESTORED
        else None,
        "restoration_mode": (
            RestorationMode.VALID_STREAM_RESUME.value
            if condition is Condition.RESTORED
            else None
        ),
        "matched_no_touch_system_id": NO_TOUCH_SYSTEM_ID if no_touch else None,
        "matched_no_touch_artifact_path": (
            str(vision.checkpoint_path) if no_touch else None
        ),
        "act_device_name": "cuda:0",
        "simulator_device": "cuda:0",
        "launcher_args": {"enable_cameras": True, "headless": True},
        "n0_source_commit": None,
        "n0_normalizer_sha256": None,
        "n0_serve_bundle_sha256": None,
        "n0_prompt_manifest_sha256": None,
        "semantic_version": "1.0",
    }


def _generate(args: argparse.Namespace) -> dict[str, Any]:
    deployment_root = _absolute(args.deployment_root)
    univtac_root = _absolute(args.univtac_root)
    checkpoint_root = _absolute(args.checkpoint_root)
    tactile = _artifact_manifest(
        profile=OfficialACTProfile.UNIVTAC,
        checkpoint_root=checkpoint_root,
        univtac_root=univtac_root,
        checkpoint_sha256=args.tactile_checkpoint_sha256,
        stats_sha256=args.stats_sha256,
        encoder_sha256=args.encoder_sha256,
    )
    vision = _artifact_manifest(
        profile=OfficialACTProfile.VISION_ONLY,
        checkpoint_root=checkpoint_root,
        univtac_root=univtac_root,
        checkpoint_sha256=args.vision_checkpoint_sha256,
        stats_sha256=args.stats_sha256,
        encoder_sha256=args.encoder_sha256,
    )
    rest_binding = load_rest_reference_binding(
        args.rest_reference_artifact,
        args.operator,
        TASK_ID,
    )
    persistent, restored = _faults(args, rest_binding)
    trial_set = trial_set_document(
        initial_seed=args.initial_seed,
        exogenous_seed=args.exogenous_seed,
        task_registry_id=REGISTRY_ID,
        upstream_commit=UPSTREAM_COMMIT,
        task_id=TASK_ID,
        action_horizon=ACTION_HORIZON,
        max_observation_steps=MAX_OBSERVATION_STEPS,
    )
    dataset_sha256 = canonical_hash(trial_set)
    base_manifest_sha256 = system_manifest_hash(
        BASE_SYSTEM_ID,
        tactile.checkpoint_sha256,
        tactile.config_sha256,
        ACTION_SPEC,
    )
    matrix_contract = {
        "task_id": TASK_ID,
        "dataset_sha256": dataset_sha256,
        "base_system_manifest_sha256": base_manifest_sha256,
        "tactile_artifact": canonical_hash(artifact_document(tactile)),
        "vision_only_artifact": canonical_hash(artifact_document(vision)),
        "persistent_fault": persistent.sha256,
        "restored_fault": restored.sha256,
        "restoration_index": args.restoration_index,
        "rest_reference_artifact_root_sha256": (
            None if rest_binding is None else rest_binding.loaded.root_receipt_sha256
        ),
        "semantic_version": "1.0",
    }
    matrix_id = f"pull_out_key-{canonical_hash(matrix_contract)[:16]}"
    output = (
        _absolute(args.output)
        if args.output is not None
        else (deployment_root / "requests/live_univtac" / matrix_id)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{matrix_id}.staging-", dir=output.parent))
    try:
        write_json(staging / "trial_set_manifest.json", trial_set)
        write_json(staging / "fault_manifests/persistent.json", persistent.to_dict())
        write_json(staging / "fault_manifests/restored.json", restored.to_dict())
        if rest_binding is not None:
            copy_rest_reference(staging, rest_binding)
        write_json(
            staging / "policy_artifacts/univtac.json", artifact_document(tactile)
        )
        write_json(
            staging / "policy_artifacts/vision_only.json", artifact_document(vision)
        )
        request_documents: dict[Condition, dict[str, Any]] = {}
        for condition in Condition:
            document = _request_document(
                condition=condition,
                matrix_id=matrix_id,
                deployment_root=deployment_root,
                univtac_root=univtac_root,
                dataset_sha256=dataset_sha256,
                tactile=tactile,
                vision=vision,
                base_manifest_sha256=base_manifest_sha256,
                initial_seed=args.initial_seed,
                exogenous_seed=args.exogenous_seed,
                wall_timeout_s=args.wall_timeout_s,
                restoration_index=args.restoration_index,
                rest_binding=rest_binding,
            )
            request_documents[condition] = document
            write_json(staging / f"requests/{condition.value}.json", document)
        loaded = {
            condition: load_live_univtac_run(
                load_live_univtac_request(staging / f"requests/{condition.value}.json")
            )
            for condition in Condition
        }
        grid = build_paired_trial_grid(
            clean=loaded[Condition.CLEAN].trial,
            faulted_manifest=persistent,
            restored_manifest=restored,
            restoration_index=args.restoration_index,
            restoration_mode=RestorationMode.VALID_STREAM_RESUME,
            no_touch_system_id=NO_TOUCH_SYSTEM_ID,
            no_touch_checkpoint_sha256=vision.checkpoint_sha256,
            no_touch_config_sha256=vision.config_sha256,
        )
        if tuple(item.trial for item in loaded.values()) != grid:
            raise RuntimeError("generated requests do not match the paired trial grid")
        pair_keys = {item.trial.pair_key for item in loaded.values()}
        if len(pair_keys) != 1:
            raise RuntimeError("generated conditions do not share one pair key")
        member_hashes = tree_hashes(staging)
        receipt: dict[str, Any] = {
            "matrix_id": matrix_id,
            "task_id": TASK_ID,
            "pair_key": next(iter(pair_keys)),
            "evidence_level": EVIDENCE_LEVEL,
            "simulator_execution_claimed": False,
            "task_success_claimed": False,
            "dataset_identity": {
                "identity_kind": "frozen_trial_set_manifest_content_sha256",
                "path": "trial_set_manifest.json",
                "sha256": dataset_sha256,
                "is_raw_dataset_file_hash": False,
            },
            "base_system": {
                "system_id": BASE_SYSTEM_ID,
                "manifest_sha256": base_manifest_sha256,
                "profile": "univtac",
            },
            "matched_no_touch_system": {
                "system_id": NO_TOUCH_SYSTEM_ID,
                "profile": "vision_only",
            },
            "policy_artifacts": {
                profile: {
                    "path": f"policy_artifacts/{profile}.json",
                    "file_sha256": member_hashes[f"policy_artifacts/{profile}.json"],
                    "checkpoint_sha256": manifest.checkpoint_sha256,
                    "stats_sha256": manifest.stats_sha256,
                    "encoder_sha256": manifest.encoder_sha256,
                    "config_sha256": manifest.config_sha256,
                }
                for profile, manifest in (("univtac", tactile), ("vision_only", vision))
            },
            "fault_manifests": {
                "persistent": persistent.sha256,
                "restored": restored.sha256,
            },
            "rest_reference_artifact": rest_reference_receipt(rest_binding),
            "requests": {
                condition.value: {
                    "path": f"requests/{condition.value}.json",
                    "file_sha256": member_hashes[f"requests/{condition.value}.json"],
                    "trial_manifest_sha256": loaded[condition].trial.sha256,
                    "executed_system_id": loaded[condition].trial.executed_system_id,
                    "output_dir": request_documents[condition]["output_dir"],
                }
                for condition in Condition
            },
            "member_sha256": member_hashes,
            "matrix_contract_sha256": canonical_hash(matrix_contract),
            "semantic_version": "1.0",
        }
        write_json(staging / "matrix_receipt.json", receipt)
        status = publish_tree(staging, output)
        return {
            "status": status,
            "matrix_id": matrix_id,
            "pair_key": receipt["pair_key"],
            "output": str(output),
            "receipt_file_sha256": file_sha256(output / "matrix_receipt.json"),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = _generate(args)
    except (FileExistsError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
