#!/usr/bin/env python3
"""Run matched Clean, tactile-absence, and Clean-action replay diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, cast

from robotactile_benchmark.backends.univtac_contracts import (
    N0_EE_ACTION_EXECUTION_CONTRACTS,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.backends.univtac_factory import launch_univtac_app_host
from robotactile_benchmark.backends.univtac_host import UniVTACSimulationAppHost
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.constants import (
    DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
)
from robotactile_benchmark.execution.action_replay import (
    ActionReplaySource,
    ActionTraceReplayReceipt,
    execute_action_trace_replay,
    write_action_trace_replay_receipt,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contact_ablation_reporting import (
    build_contact_ablation_pair_summary,
    build_contact_ablation_pilot_summary,
)
from robotactile_benchmark.execution.contracts import LiveUniVTACRunRequest
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.live_univtac import LiveBackendFactory
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.official_n0 import execute_official_n0_live_run
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    N0TWAMArtifactManifest,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)
from robotactile_benchmark.n0_fault_campaign.contracts import (
    N0FaultCampaignCellSpec,
    N0FaultCampaignError,
    N0FaultCellDisposition,
)
from robotactile_benchmark.n0_fault_campaign.io import (
    LoadedN0FaultCampaign,
    file_sha256,
    load_n0_fault_campaign_bundle,
    write_json,
)
from robotactile_benchmark.trials import Condition


@dataclass(frozen=True)
class _Pair:
    clean_cell: N0FaultCampaignCellSpec
    absence_cell: N0FaultCampaignCellSpec
    clean_request: LiveUniVTACRunRequest
    absence_request: LiveUniVTACRunRequest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--integration-config", type=Path, required=True)
    parser.add_argument("--n0-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n0-host", default="127.0.0.1")
    parser.add_argument("--n0-port", type=int, default=29601)
    parser.add_argument("--expected-pair-count", type=int, default=3)
    parser.add_argument(
        "--action-execution-contract",
        choices=sorted(N0_EE_ACTION_EXECUTION_CONTRACTS),
        default=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    )
    return parser


def _select_pairs(
    campaign: LoadedN0FaultCampaign, expected_pair_count: int
) -> Tuple[_Pair, ...]:
    manifest = campaign.manifest
    if manifest.severity_registry != (DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID):
        raise N0FaultCampaignError(
            "contact ablation requires observed-tactile absence campaign"
        )
    if expected_pair_count < 1 or manifest.pair_count != expected_pair_count:
        raise N0FaultCampaignError("campaign pair count differs from requested pilot")
    pairs: list[_Pair] = []
    for pair_key in sorted({cell.pair_key for cell in manifest.cells}):
        cells = tuple(cell for cell in manifest.cells if cell.pair_key == pair_key)
        clean = tuple(cell for cell in cells if cell.condition is Condition.CLEAN)
        absence = tuple(
            cell
            for cell in cells
            if cell.condition is Condition.FAULTED
            and cell.operator_id == "A1_stream_absence"
            and cell.severity_level == 5
            and cell.disposition is N0FaultCellDisposition.LIVE_REQUEST
        )
        if len(clean) != 1 or len(absence) != 1:
            raise N0FaultCampaignError(
                "every pilot pair requires one Clean and one live A1-S5 cell"
            )
        clean_request = _request_for_cell(campaign, clean[0])
        absence_request = _request_for_cell(campaign, absence[0])
        pairs.append(_Pair(clean[0], absence[0], clean_request, absence_request))
    return tuple(sorted(pairs, key=lambda item: item.clean_cell.initial_seed))


def _request_for_cell(
    campaign: LoadedN0FaultCampaign, cell: N0FaultCampaignCellSpec
) -> LiveUniVTACRunRequest:
    if cell.request_relpath is None:
        raise N0FaultCampaignError("live campaign cell lost its request path")
    return load_live_univtac_request(campaign.root / cell.request_relpath)


def _require_fresh_outputs(pairs: Tuple[_Pair, ...], output: Path) -> None:
    paths = [output]
    for pair in pairs:
        for request in (pair.clean_request, pair.absence_request):
            if request.output_dir is None:
                raise N0FaultCampaignError("pilot request has no output directory")
            paths.append(request.output_dir)
        paths.append(_replay_receipt_path(output.parent, pair.clean_cell.pair_key))
    for path in paths:
        if path.is_symlink():
            raise FileExistsError(f"refusing to rerun existing pilot output: {path}")
        occupied = (
            path.exists() and (not path.is_dir() or any(path.iterdir()))
            if path.suffix == ""
            else path.exists()
        )
        if occupied:
            raise FileExistsError(f"refusing to rerun existing pilot output: {path}")


def _replay_receipt_path(root: Path, pair_key: str) -> Path:
    return root / "action_replays" / pair_key / "receipt.json"


def _create_backend(
    host: UniVTACSimulationAppHost,
    loaded: LoadedLiveUniVTACRun,
    *,
    runtime_dir: Optional[Path] = None,
) -> UniVTACIsaacBackend:
    runtime = host.create_runtime(
        runtime_dir=loaded.request.runtime_dir if runtime_dir is None else runtime_dir,
        initial_seed=loaded.trial.initial_seed,
    )
    try:
        return UniVTACIsaacBackend(loaded.backend_config, runtime)
    except BaseException:
        runtime.close_runtime()
        raise


def _execute_n0(
    request: LiveUniVTACRunRequest,
    *,
    host: UniVTACSimulationAppHost,
    manifest: N0TWAMArtifactManifest,
    n0_source_root: Path,
    n0_host: str,
    n0_port: int,
    action_execution_contract: str,
) -> LoadedLiveUniVTACArtifact:
    loaded = load_live_univtac_run(request)

    def backend_factory(selected: LoadedLiveUniVTACRun) -> UniVTACIsaacBackend:
        if selected != loaded:
            raise ValueError("backend factory received a different loaded request")
        return _create_backend(host, selected)

    if type(manifest) is not N0TWAMArtifactManifest:
        raise TypeError("manifest must be an exact N0TWAMArtifactManifest")
    return execute_official_n0_live_run(
        request,
        manifest=manifest,
        source_root=n0_source_root,
        host=n0_host,
        port=n0_port,
        api_key=os.environ.get("N0_TWAM_API_KEY"),
        backend_factory=cast(LiveBackendFactory, backend_factory),
        action_execution_contract=action_execution_contract,
        capture_profile=LiveCaptureProfile.METRICS_ONLY,
    )


def _execute_replay(
    clean: LoadedLiveUniVTACArtifact,
    request: LiveUniVTACRunRequest,
    host: UniVTACSimulationAppHost,
    output_root: Path,
) -> tuple[ActionTraceReplayReceipt, str]:
    source = ActionReplaySource.from_artifact(clean)
    loaded = load_live_univtac_run(request)
    runtime_dir = request.runtime_dir.parent / "clean-action-replay"
    receipt = execute_action_trace_replay(
        source,
        _create_backend(host, loaded, runtime_dir=runtime_dir),
    )
    receipt_sha = write_action_trace_replay_receipt(
        _replay_receipt_path(output_root, source.trial.pair_key), receipt
    )
    return receipt, receipt_sha


def main() -> int:
    args = _parser().parse_args()
    campaign = load_n0_fault_campaign_bundle(args.campaign_root)
    output = args.output.expanduser().absolute()
    pairs = _select_pairs(campaign, args.expected_pair_count)
    _require_fresh_outputs(pairs, output)
    runtime = resolve_n0_runtime_artifacts(args.integration_config)
    if runtime.manifest.task_id != pairs[0].clean_cell.task:
        raise ValueError("integration manifest task differs from pilot campaign")
    bootstrap = load_live_univtac_run(pairs[0].clean_request)
    host = launch_univtac_app_host(
        bootstrap.backend_config,
        upstream_root=bootstrap.request.upstream_root,
        initial_seed=bootstrap.trial.initial_seed,
        launcher_args=bootstrap.request.launcher_args,
        device=bootstrap.request.simulator_device,
        n0_action_execution_contract=args.action_execution_contract,
    )
    rows: list[dict[str, object]] = []
    try:
        for pair in pairs:
            clean = _execute_n0(
                pair.clean_request,
                host=host,
                manifest=runtime.manifest,
                n0_source_root=args.n0_source_root.expanduser().absolute(),
                n0_host=args.n0_host,
                n0_port=args.n0_port,
                action_execution_contract=args.action_execution_contract,
            )
            replay, replay_file_sha256 = _execute_replay(
                clean, pair.clean_request, host, output.parent
            )
            absence = _execute_n0(
                pair.absence_request,
                host=host,
                manifest=runtime.manifest,
                n0_source_root=args.n0_source_root.expanduser().absolute(),
                n0_host=args.n0_host,
                n0_port=args.n0_port,
                action_execution_contract=args.action_execution_contract,
            )
            rows.append(
                build_contact_ablation_pair_summary(
                    pair_key=pair.clean_cell.pair_key,
                    initial_seed=pair.clean_cell.initial_seed,
                    exogenous_seed=pair.clean_cell.exogenous_seed,
                    clean=clean,
                    absence=absence,
                    replay=replay,
                    replay_file_sha256=replay_file_sha256,
                )
            )
        runtime_count = host.runtime_count
    finally:
        host.close()
    summary = build_contact_ablation_pilot_summary(
        campaign,
        rows,
        integration_manifest_sha256=file_sha256(runtime.manifest_path),
        runtime_count=runtime_count,
        action_execution_contract=args.action_execution_contract,
    )
    write_json(output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
