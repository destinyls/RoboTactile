"""CLI bridge for single and snapshot-paired live UniVTAC execution."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

from robotactile_benchmark.backends.univtac_contracts import (
    N0_EE_ACTION_EXECUTION_CONTRACTS,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    resolve_deployment_root,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.isaac_runtime_attestation import (
    IsaacAttestationRequest,
)
from robotactile_benchmark.execution.lifecycle_watchdog import LifecycleStageJournal
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.loading import load_live_univtac_request
from robotactile_benchmark.execution.official_act import (
    OfficialACTPairedLiveResult,
    execute_official_act_live_run,
    execute_official_act_paired_live_runs,
    official_act_live_summary,
    official_act_profile,
)
from robotactile_benchmark.execution.official_ftp1_policy import (
    OfficialFTP1PairedLiveResult,
    execute_official_ftp1_live_run,
    execute_official_ftp1_paired_live_runs,
)
from robotactile_benchmark.execution.official_n0 import (
    execute_official_n0_live_run,
    execute_official_n0_paired_live_runs,
)
from robotactile_benchmark.execution.official_n0_vtla import (
    OfficialN0VTLAPairedLiveResult,
    execute_official_n0_vtla_live_run,
    execute_official_n0_vtla_paired_live_runs,
)
from robotactile_benchmark.execution.paired_live_univtac import (
    PairedLiveUniVTACExecutionResult,
)
from robotactile_benchmark.execution.paired_receipt_io import (
    write_paired_execution_receipt,
)
from robotactile_benchmark.integrations.runtime_config import (
    FTP1PolicyRuntimeArtifacts,
    N0RuntimeArtifacts,
    N0VTLARuntimeArtifacts,
    resolve_act_runtime_artifacts,
    resolve_ftp1_policy_runtime_artifacts,
    resolve_n0_runtime_artifacts,
    resolve_n0_vtla_runtime_artifacts,
)
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile


def add_live_execution_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the legacy one-cell path and the exact paired path."""

    live = subparsers.add_parser(
        "live-univtac-run", help="execute one unqualified live UniVTAC request"
    )
    _add_runtime_arguments(live)
    live.add_argument("--request", type=Path, required=True, help="request JSON")
    live.add_argument("--campaign-id")
    live.add_argument("--qualification", type=Path)
    live.add_argument("--n0-server-attestation", type=Path)
    live.add_argument("--n0-server-attestation-sha256")
    live.add_argument("--isaac-attestation-output", type=Path)
    live.add_argument(
        "--capture-profile",
        choices=tuple(item.value for item in LiveCaptureProfile),
        default=LiveCaptureProfile.PAPER_FULL.value,
    )
    live.add_argument(
        "--action-execution-contract",
        choices=sorted(N0_EE_ACTION_EXECUTION_CONTRACTS),
        help=(
            "explicit N0 EE execution surface; defaults to the training-aligned "
            "60 Hz contract"
        ),
    )
    paired = subparsers.add_parser(
        "live-univtac-paired-run",
        help="execute matched requests from one canonical simulator snapshot",
    )
    _add_runtime_arguments(paired)
    paired.add_argument(
        "--requests",
        type=Path,
        nargs="+",
        required=True,
        help="ordered request JSON files; clean must be first",
    )
    paired.add_argument(
        "--receipt", type=Path, required=True, help="no-clobber group receipt JSON"
    )
    paired.add_argument(
        "--capture-profile",
        choices=tuple(item.value for item in LiveCaptureProfile),
        default=LiveCaptureProfile.PAPER_FULL.value,
    )
    paired.add_argument(
        "--action-execution-contract",
        choices=sorted(N0_EE_ACTION_EXECUTION_CONTRACTS),
        help=(
            "explicit N0 EE execution surface; defaults to the training-aligned "
            "60 Hz contract"
        ),
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--official-act-artifact-root", type=Path)
    parser.add_argument("--stats-sha256")
    parser.add_argument("--encoder-sha256")
    parser.add_argument("--n0-source-root", type=Path)
    parser.add_argument("--n0-host", default="127.0.0.1")
    parser.add_argument("--n0-port", type=int, default=29601)
    parser.add_argument("--n0-vtla-source-root", type=Path)
    parser.add_argument("--n0-vtla-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--ftp1-source-root", type=Path)
    parser.add_argument("--ftp1-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument(
        "--lifecycle-journal",
        type=Path,
        help="pre-created identity-bound lifecycle journal directory",
    )


def handle_live_execution_command(
    args: argparse.Namespace,
) -> Optional[dict[str, object]]:
    """Execute a live CLI command or return ``None`` when unrelated."""

    if args.command not in {"live-univtac-run", "live-univtac-paired-run"}:
        return None
    if args.command == "live-univtac-run":
        capture_profile = LiveCaptureProfile(args.capture_profile)
        lifecycle_journal = _lifecycle_journal(args)
        if lifecycle_journal is not None:
            lifecycle_journal.record("request_loading")
            if _sha256_file(args.request) != (
                lifecycle_journal.identity.request_file_sha256
            ):
                raise ValueError("lifecycle journal request hash mismatch")
        request = load_live_univtac_request(args.request)
        if (
            lifecycle_journal is not None
            and lifecycle_journal.identity.task_id != request.task_id
        ):
            raise ValueError("lifecycle journal task identity mismatch")
        if request.policy_kind is LivePolicyKind.FTP1_POLICY:
            if args.action_execution_contract is not None:
                raise ValueError("action execution contract is fixed for FTP-1 qpos8")
            ftp1_layout, ftp1_runtime = _ftp1_runtime_artifacts(args, request.task_id)
            artifact = execute_official_ftp1_live_run(
                request,
                manifest=ftp1_runtime.manifest,
                source_root=(
                    args.ftp1_source_root or ftp1_layout.sources / "ftp1-policy"
                ),
                endpoint=args.ftp1_endpoint,
                lifecycle_journal=lifecycle_journal,
                capture_profile=capture_profile,
            )
            return official_act_live_summary(artifact)
        if request.policy_kind is LivePolicyKind.N0_VTLA:
            if args.action_execution_contract is not None:
                raise ValueError("action execution contract is fixed for N0-VTLA qpos8")
            vtla_layout, vtla_runtime = _n0_vtla_runtime_artifacts(args)
            artifact = execute_official_n0_vtla_live_run(
                request,
                manifest=vtla_runtime.manifest,
                source_root=(
                    args.n0_vtla_source_root or vtla_layout.sources / "N0-VTLA"
                ),
                endpoint=args.n0_vtla_endpoint,
                lifecycle_journal=lifecycle_journal,
                capture_profile=capture_profile,
            )
            return official_act_live_summary(artifact)
        if request.policy_kind is LivePolicyKind.N0:
            n0_layout, n0_runtime = _n0_runtime_artifacts(args, request.task_id)
            artifact = execute_official_n0_live_run(
                request,
                manifest=n0_runtime.manifest,
                source_root=args.n0_source_root or n0_layout.sources / "N0-TWAM",
                host=args.n0_host,
                port=args.n0_port,
                api_key=os.environ.get("N0_TWAM_API_KEY"),
                lifecycle_journal=lifecycle_journal,
                isaac_attestation_request=_isaac_attestation_request(
                    args, n0_layout.root
                ),
                action_execution_contract=(
                    args.action_execution_contract
                    or N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
                ),
                capture_profile=capture_profile,
            )
            return official_act_live_summary(artifact)
        if args.action_execution_contract is not None:
            raise ValueError("action execution contract is only valid for N0")
        if lifecycle_journal is not None:
            raise ValueError("lifecycle watchdog currently requires policy_kind=n0")
        if request.policy_kind is not LivePolicyKind.ACT:
            raise ValueError(
                f"unsupported live policy kind: {request.policy_kind.value}"
            )
        artifact_root, stats_sha256, encoder_sha256 = _runtime_artifacts(
            args,
            request.task_id,
            official_act_profile(request.condition).value,
            request.act_device_name,
            request.checkpoint_sha256,
            request.config_sha256,
        )
        artifact = execute_official_act_live_run(
            request,
            artifact_root=artifact_root,
            stats_sha256=stats_sha256,
            encoder_sha256=encoder_sha256,
            capture_profile=capture_profile,
        )
        return official_act_live_summary(artifact)
    if args.lifecycle_journal is not None:
        raise ValueError("paired live execution does not accept a lifecycle journal")
    requests = tuple(load_live_univtac_request(path) for path in args.requests)
    capture_profile = LiveCaptureProfile(
        getattr(args, "capture_profile", LiveCaptureProfile.PAPER_FULL.value)
    )
    action_execution_contract = getattr(args, "action_execution_contract", None)
    if requests and requests[0].policy_kind is LivePolicyKind.FTP1_POLICY:
        if any(item.policy_kind is not LivePolicyKind.FTP1_POLICY for item in requests):
            raise ValueError("paired live requests cannot mix policy kinds")
        if action_execution_contract is not None:
            raise ValueError("action execution contract is fixed for FTP-1 qpos8")
        ftp1_layout, ftp1_runtime = _ftp1_runtime_artifacts(args, requests[0].task_id)

        def publish_ftp1_before_close(result: OfficialFTP1PairedLiveResult) -> None:
            write_paired_execution_receipt(args.receipt, result.paired)

        ftp1_result = execute_official_ftp1_paired_live_runs(
            requests,
            manifest=ftp1_runtime.manifest,
            source_root=(args.ftp1_source_root or ftp1_layout.sources / "ftp1-policy"),
            endpoint=args.ftp1_endpoint,
            capture_profile=capture_profile,
            pre_close_publisher=publish_ftp1_before_close,
        )
        return _paired_payload(
            args.receipt,
            ftp1_result.paired,
            ftp1_result.artifacts,
        )
    if requests and requests[0].policy_kind is LivePolicyKind.N0_VTLA:
        if any(item.policy_kind is not LivePolicyKind.N0_VTLA for item in requests):
            raise ValueError("paired live requests cannot mix policy kinds")
        if action_execution_contract is not None:
            raise ValueError("action execution contract is fixed for N0-VTLA qpos8")
        vtla_layout, vtla_runtime = _n0_vtla_runtime_artifacts(args)

        def publish_pair_before_close(result: OfficialN0VTLAPairedLiveResult) -> None:
            write_paired_execution_receipt(args.receipt, result.paired)

        vtla_result = execute_official_n0_vtla_paired_live_runs(
            requests,
            manifest=vtla_runtime.manifest,
            source_root=(args.n0_vtla_source_root or vtla_layout.sources / "N0-VTLA"),
            endpoint=args.n0_vtla_endpoint,
            capture_profile=capture_profile,
            pre_close_publisher=publish_pair_before_close,
        )
        return _paired_payload(
            args.receipt,
            vtla_result.paired,
            vtla_result.artifacts,
        )
    if requests and requests[0].policy_kind is LivePolicyKind.N0:
        if any(item.policy_kind is not LivePolicyKind.N0 for item in requests):
            raise ValueError("paired live requests cannot mix policy kinds")
        n0_layout, n0_runtime = _n0_runtime_artifacts(args, requests[0].task_id)
        n0_result = execute_official_n0_paired_live_runs(
            requests,
            manifest=n0_runtime.manifest,
            source_root=args.n0_source_root or n0_layout.sources / "N0-TWAM",
            host=args.n0_host,
            port=args.n0_port,
            api_key=os.environ.get("N0_TWAM_API_KEY"),
            action_execution_contract=(
                action_execution_contract or N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
            ),
            capture_profile=capture_profile,
        )
        return _paired_payload(args.receipt, n0_result.paired, n0_result.artifacts)
    if action_execution_contract is not None:
        raise ValueError("action execution contract is only valid for N0")
    if not requests:
        raise ValueError("paired live execution requires at least one request")
    if any(item.policy_kind is not LivePolicyKind.ACT for item in requests):
        raise ValueError("paired live requests cannot mix policy kinds")
    profiles = tuple(official_act_profile(item.condition) for item in requests)
    unique_profiles = frozenset(profiles)
    legacy = (
        args.official_act_artifact_root,
        args.stats_sha256,
        args.encoder_sha256,
    )
    if len(unique_profiles) > 1 and (
        args.config is not None or all(value is not None for value in legacy)
    ):
        raise ValueError(
            "mixed-profile ACT pairing requires canonical per-profile configs"
        )
    runtime_values = tuple(
        _runtime_artifacts(
            args,
            request.task_id,
            profile.value,
            request.act_device_name,
            request.checkpoint_sha256,
            request.config_sha256,
        )
        for request, profile in zip(requests, profiles)
    )
    artifact_roots = frozenset(value[0] for value in runtime_values)
    encoder_digests = frozenset(value[2] for value in runtime_values)
    if len(artifact_roots) != 1 or len(encoder_digests) != 1:
        raise ValueError("paired ACT profiles must share artifact root and encoder")
    profile_stats_sha256: dict[OfficialACTProfile, str] = {}
    for profile, (_, digest, _) in zip(profiles, runtime_values):
        previous = profile_stats_sha256.setdefault(profile, digest)
        if previous != digest:
            raise ValueError("paired ACT profile statistics digests disagree")
    artifact_root = next(iter(artifact_roots))
    encoder_sha256 = next(iter(encoder_digests))

    def publish_act_before_close(result: OfficialACTPairedLiveResult) -> None:
        write_paired_execution_receipt(args.receipt, result.paired)

    act_result = execute_official_act_paired_live_runs(
        requests,
        artifact_root=artifact_root,
        stats_sha256=profile_stats_sha256,
        encoder_sha256=encoder_sha256,
        capture_profile=capture_profile,
        pre_close_publisher=publish_act_before_close,
    )
    return _paired_payload(args.receipt, act_result.paired, act_result.artifacts)


def _paired_payload(
    receipt: Path,
    paired: PairedLiveUniVTACExecutionResult,
    artifacts: Sequence[LoadedLiveUniVTACArtifact],
) -> dict[str, object]:
    written = write_paired_execution_receipt(receipt, paired)
    payload = paired.to_dict()
    payload.update(
        {
            "artifact_root_sha256": tuple(
                item.external_root_sha256 for item in artifacts
            ),
            "receipt_file": str(written.path),
            "receipt_file_sha256": written.file_sha256,
        }
    )
    return payload


def _runtime_artifacts(
    args: argparse.Namespace,
    task_id: str,
    profile: str,
    device: Optional[str],
    checkpoint_sha256: str,
    config_sha256: str,
) -> Tuple[Path, str, str]:
    legacy = (
        args.official_act_artifact_root,
        args.stats_sha256,
        args.encoder_sha256,
    )
    if args.config is not None or any(value is None for value in legacy):
        layout = DeploymentLayout(resolve_deployment_root(args.root))
        config_path = args.config
        if config_path is None:
            canonical = (
                layout.model_artifacts
                / f"act/configs/{task_id}/{profile}/integration_config.json"
            )
            legacy_config_path = layout.model_artifacts / "act/integration_config.json"
            config_path = (
                legacy_config_path
                if legacy_config_path.is_file() and not canonical.exists()
                else canonical
            )
        resolved = resolve_act_runtime_artifacts(config_path)
        if device != resolved.device:
            raise ValueError("ACT request device does not match integration config")
        manifest = resolved.manifest
        expected = (
            manifest.task_id == task_id,
            manifest.profile.value == profile,
            manifest.checkpoint_sha256 == checkpoint_sha256,
            manifest.config_sha256 == config_sha256,
        )
        if not all(expected):
            raise ValueError("ACT request does not match integration artifact manifest")
        return resolved.artifact_root, resolved.stats_sha256, resolved.encoder_sha256
    artifact_root, stats_sha256, encoder_sha256 = legacy
    assert artifact_root is not None
    assert stats_sha256 is not None
    assert encoder_sha256 is not None
    return artifact_root, stats_sha256, encoder_sha256


def _n0_runtime_artifacts(
    args: argparse.Namespace, task_id: str
) -> tuple[DeploymentLayout, N0RuntimeArtifacts]:
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    config_path = args.config or (
        layout.model_artifacts / f"n0_twam/configs/{task_id}/integration_config.json"
    )
    return layout, resolve_n0_runtime_artifacts(config_path)


def _n0_vtla_runtime_artifacts(
    args: argparse.Namespace,
) -> tuple[DeploymentLayout, N0VTLARuntimeArtifacts]:
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    config_path = args.config or (
        layout.model_artifacts / "n0_vtla/configs/insert_hole/integration_config.json"
    )
    return layout, resolve_n0_vtla_runtime_artifacts(config_path)


def _ftp1_runtime_artifacts(
    args: argparse.Namespace,
    task_id: str,
) -> tuple[DeploymentLayout, FTP1PolicyRuntimeArtifacts]:
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    config_path = args.config or (
        layout.model_artifacts
        / f"ftp1_policy/configs/{task_id}/integration_config.json"
    )
    return layout, resolve_ftp1_policy_runtime_artifacts(config_path)


def _lifecycle_journal(
    args: argparse.Namespace,
) -> Optional[LifecycleStageJournal]:
    requested = args.lifecycle_journal
    if requested is None:
        return None
    layout = DeploymentLayout(resolve_deployment_root(args.root))
    journal = LifecycleStageJournal.open(requested)
    resolved = journal.path.resolve(strict=True)
    outputs = layout.outputs.resolve(strict=True)
    try:
        resolved.relative_to(outputs)
    except ValueError as error:
        raise ValueError("lifecycle journal must remain below outputs/") from error
    return journal


def _isaac_attestation_request(
    args: argparse.Namespace,
    deployment_root: Path,
) -> Optional[IsaacAttestationRequest]:
    values = (
        args.campaign_id,
        args.qualification,
        args.n0_server_attestation,
        args.n0_server_attestation_sha256,
        args.isaac_attestation_output,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("source-bound Isaac attestation arguments must be complete")
    return IsaacAttestationRequest(
        campaign_id=args.campaign_id,
        deployment_root=deployment_root,
        output_path=args.isaac_attestation_output,
        qualification_path=args.qualification,
        n0_server_attestation_path=args.n0_server_attestation,
        n0_server_attestation_sha256=args.n0_server_attestation_sha256,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["add_live_execution_subcommands", "handle_live_execution_command"]
