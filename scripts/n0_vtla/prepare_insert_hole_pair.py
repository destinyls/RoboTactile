#!/usr/bin/env python3
"""Prepare one source-bound Clean/tactile-null N0-VTLA request pair."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from robotactile_benchmark.backends.univtac_success_profiles import (
    UniVTACSuccessProfile,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.constants import DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    resolve_deployment_root,
)
from robotactile_benchmark.integrations.n0_vtla.artifacts import file_sha256
from robotactile_benchmark.integrations.n0_vtla.requests import (
    build_official_n0_vtla_request,
    write_official_n0_vtla_request,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_vtla_runtime_artifacts,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.trials import Condition


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--campaign-id",
        default="n0-vtla-insert-hole-clean-vs-null-v1",
    )
    parser.add_argument("--initial-seed", type=int, default=17)
    parser.add_argument("--exogenous-seed", type=int, default=29)
    parser.add_argument("--max-control-cycles", type=int, default=6)
    parser.add_argument("--max-observation-steps", type=int, default=300)
    parser.add_argument("--wall-timeout-s", type=float, default=1800.0)
    parser.add_argument("--simulator-device", default="cuda:0")
    parser.add_argument(
        "--success-profile",
        choices=tuple(item.value for item in UniVTACSuccessProfile),
        default=UniVTACSuccessProfile.OFFICIAL_V1.value,
    )
    return parser


def _write_no_clobber(path: Path, payload: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace a different {label}: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    config_path = args.config or (
        layout.model_artifacts / "n0_vtla/configs/insert_hole/integration_config.json"
    )
    runtime = resolve_n0_vtla_runtime_artifacts(config_path)
    repository_root = Path(__file__).resolve().parents[2]
    task_registry = repository_root / "configs/univtac/tasks_v1.json"
    dataset_sha256 = file_sha256(task_registry, "UniVTAC task registry")
    request_root = layout.requests / "n0-vtla" / args.campaign_id
    output_root = layout.outputs / "n0-vtla" / args.campaign_id

    fault = FaultManifest(
        operator_id="F1_global_response_drift",
        severity_level=5,
        operator_seed=20260829,
        start_index=0,
        stop_index=args.max_observation_steps,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        parameters={},
        severity_registry=DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    )
    fault_path = request_root / "fault_manifest.tactile_null_black.json"
    _write_no_clobber(
        fault_path,
        canonical_json_bytes(fault.to_dict()),
        "N0-VTLA fault manifest",
    )

    common = {
        "manifest": runtime.manifest,
        "layout": layout,
        "dataset_sha256": dataset_sha256,
        "initial_seed": args.initial_seed,
        "exogenous_seed": args.exogenous_seed,
        "max_control_cycles": args.max_control_cycles,
        "max_observation_steps": args.max_observation_steps,
        "wall_timeout_s": args.wall_timeout_s,
        "simulator_device": args.simulator_device,
        "success_profile_id": UniVTACSuccessProfile(args.success_profile),
    }
    clean = build_official_n0_vtla_request(
        condition=Condition.CLEAN,
        live_output_dir=output_root / "clean",
        **common,
    )
    faulted = build_official_n0_vtla_request(
        condition=Condition.FAULTED,
        live_output_dir=output_root / "faulted_tactile_null_black",
        fault_manifest_path=fault_path,
        **common,
    )
    clean_file = write_official_n0_vtla_request(request_root / "clean.json", clean)
    faulted_file = write_official_n0_vtla_request(
        request_root / "faulted_tactile_null_black.json",
        faulted,
    )
    receipt_path = request_root / "paired_execution_receipt.json"
    payload = {
        "campaign_id": args.campaign_id,
        "clean_request": str(clean_file.request_path),
        "clean_request_sha256": clean_file.request_file_sha256,
        "dataset_sha256": dataset_sha256,
        "fault_manifest": str(fault_path),
        "fault_manifest_sha256": fault.sha256,
        "faulted_request": str(faulted_file.request_path),
        "faulted_request_sha256": faulted_file.request_file_sha256,
        "paired_execution_receipt": str(receipt_path),
        "policy_kind": "n0_vtla",
        "success_profile_id": args.success_profile,
        "semantic_version": "1.0",
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
