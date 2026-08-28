"""Build a hash-bound clean campaign from strict live requests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from robotactile_benchmark.backends.univtac_registry import load_registry
from robotactile_benchmark.clean_baseline.contracts import (
    CLEAN_CAMPAIGN_SAMPLING_SEMANTIC_VERSION,
    CLEAN_CAMPAIGN_SEMANTIC_VERSION,
    CLEAN_POLICY_SEED_ROLE,
    CLEAN_SEED_DERIVATION,
    CLEAN_SIMULATOR_SEED_ROLE,
    CleanCampaignError,
    CleanCampaignManifest,
    CleanCampaignProtocol,
    CleanCampaignSamplingSpec,
    CleanCampaignTrialSpec,
)
from robotactile_benchmark.clean_baseline.io import read_canonical_json_file
from robotactile_benchmark.closed_loop.artifact_io import sha256_bytes
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.trials import Condition

CLEAN_REQUEST_FILENAME = "request.json"
CLEAN_MAX_DISCOVERY_ENTRIES = 100_000


def discover_clean_request_paths(request_root: Path) -> tuple[Path, ...]:
    """Recursively discover only ``request.json`` while rejecting symlinks."""

    root = Path(request_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise CleanCampaignError("request_root must be a real existing directory")
    discovered: list[Path] = []
    entry_count = 0

    def visit(directory: Path) -> None:
        nonlocal entry_count
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise CleanCampaignError("request_root cannot be scanned") from error
        for entry in entries:
            entry_count += 1
            if entry_count > CLEAN_MAX_DISCOVERY_ENTRIES:
                raise CleanCampaignError("request discovery exceeds the entry cap")
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                raise CleanCampaignError(
                    f"symlink is forbidden below request_root: {relative}"
                )
            if entry.is_dir(follow_symlinks=False):
                visit(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise CleanCampaignError(
                    f"non-regular request tree entry is forbidden: {relative}"
                )
            if entry.name == CLEAN_REQUEST_FILENAME:
                discovered.append(path.absolute())

    visit(root)
    result = tuple(sorted(discovered, key=lambda path: path.as_posix()))
    if not result:
        raise CleanCampaignError("request_root contains no request.json files")
    if len(result) != len(set(result)):
        raise CleanCampaignError("request discovery produced duplicate paths")
    return result


def build_clean_campaign_manifest(
    *,
    deployment_root: Path,
    request_paths: Sequence[Path],
    campaign_id: str,
    protocol_id: CleanCampaignProtocol,
    master_seed: int,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260823,
    sampling: CleanCampaignSamplingSpec | None = None,
) -> CleanCampaignManifest:
    """Freeze canonical clean requests and their exact future artifact paths."""

    root = _real_deployment_root(deployment_root)
    paths = tuple(Path(path).absolute() for path in request_paths)
    if not paths:
        raise CleanCampaignError("a clean campaign requires at least one request")
    if len(paths) != len(set(paths)):
        raise CleanCampaignError("duplicate explicit clean request path")
    protocol = (
        protocol_id
        if isinstance(protocol_id, CleanCampaignProtocol)
        else CleanCampaignProtocol(protocol_id)
    )
    loaded_trials: list[tuple[str, int, int, CleanCampaignTrialSpec]] = []
    policy_kinds: set[LivePolicyKind] = set()
    for path in paths:
        request_relpath = _relative_below(root, path, "request", "requests", True)
        _, request_raw = read_canonical_json_file(path, "clean live request")
        request = load_live_univtac_request(path)
        _, after_raw = read_canonical_json_file(path, "clean live request")
        if request_raw != after_raw:
            raise CleanCampaignError("clean live request changed during loading")
        loaded = load_live_univtac_run(request)
        _validate_clean_request(loaded)
        if request.output_dir is None:
            raise CleanCampaignError("clean request must precommit output_dir")
        artifact_relpath = _relative_below(
            root, request.output_dir, "output_dir", "artifacts", False
        )
        policy_kinds.add(request.policy_kind)
        spec = CleanCampaignTrialSpec(
            ordinal=0,
            task=loaded.trial.task,
            initial_seed=loaded.trial.initial_seed,
            exogenous_seed=loaded.trial.exogenous_seed,
            base_system_id=loaded.trial.base_system_id,
            dataset_sha256=loaded.trial.dataset_sha256,
            base_system_manifest_sha256=loaded.trial.base_system_manifest_sha256,
            checkpoint_sha256=loaded.trial.checkpoint_sha256,
            config_sha256=loaded.trial.config_sha256,
            trial_manifest_sha256=loaded.trial.sha256,
            pair_key=loaded.trial.pair_key,
            run_spec_sha256=loaded.run_spec.sha256,
            run_content_sha256=loaded.content_sha256,
            request_file_sha256=sha256_bytes(request_raw),
            request_relpath=request_relpath,
            artifact_relpath=artifact_relpath,
            max_control_cycles=loaded.run_spec.max_control_cycles,
            max_observation_steps=loaded.run_spec.max_observation_steps,
            execute_action_steps=loaded.run_spec.execute_action_steps,
        )
        loaded_trials.append((spec.task, spec.initial_seed, spec.exogenous_seed, spec))
    if len(policy_kinds) != 1:
        raise CleanCampaignError("a clean campaign must use one policy kind")
    ordered = sorted(loaded_trials, key=lambda item: item[:3] + (item[3].pair_key,))
    trials = tuple(
        _with_ordinal(spec, ordinal) for ordinal, (_, _, _, spec) in enumerate(ordered)
    )
    return CleanCampaignManifest(
        campaign_id=campaign_id,
        protocol_id=protocol,
        policy_kind=next(iter(policy_kinds)),
        task_registry_sha256=load_registry().resource_sha256,
        master_seed=master_seed,
        seed_derivation=(
            CLEAN_SEED_DERIVATION if sampling is None else sampling.seed_protocol
        ),
        simulator_seed_role=CLEAN_SIMULATOR_SEED_ROLE,
        policy_seed_role=CLEAN_POLICY_SEED_ROLE,
        planned_trial_count=len(trials),
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        trials=trials,
        semantic_version=(
            CLEAN_CAMPAIGN_SEMANTIC_VERSION
            if sampling is None
            else CLEAN_CAMPAIGN_SAMPLING_SEMANTIC_VERSION
        ),
        sampling=sampling,
    )


def _real_deployment_root(root: Path) -> Path:
    selected = Path(root).absolute()
    if selected.is_symlink() or not selected.is_dir():
        raise CleanCampaignError("deployment_root must be a real existing directory")
    return selected.resolve(strict=True)


def _relative_below(
    root: Path,
    path: Path,
    name: str,
    prefix: str,
    require_file: bool,
) -> str:
    candidate = Path(path).absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise CleanCampaignError(f"{name} must remain below deployment_root") from error
    if not relative.parts or relative.parts[0] != prefix:
        raise CleanCampaignError(f"{name} must remain below {prefix}/")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise CleanCampaignError(f"{name} path cannot traverse a symlink")
    if require_file and not candidate.is_file():
        raise CleanCampaignError(f"{name} must be a regular file")
    if not require_file and candidate.exists() and not candidate.is_dir():
        raise CleanCampaignError(f"{name} must identify an artifact directory")
    resolved = candidate.resolve(strict=require_file)
    try:
        resolved_relative = resolved.relative_to(root)
    except ValueError as error:
        raise CleanCampaignError(f"{name} resolves outside deployment_root") from error
    if resolved_relative != relative:
        raise CleanCampaignError(f"{name} must use one canonical path")
    return relative.as_posix()


def _validate_clean_request(loaded: object) -> None:
    from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun

    if type(loaded) is not LoadedLiveUniVTACRun:
        raise TypeError("loaded must be an exact LoadedLiveUniVTACRun")
    request = loaded.request
    if (
        request.condition is not Condition.CLEAN
        or loaded.trial.condition is not Condition.CLEAN
        or request.fault_manifest_path is not None
        or request.rest_references_path is not None
        or request.restoration_index is not None
        or request.restoration_mode is not None
        or request.matched_no_touch_system_id is not None
        or request.matched_no_touch_artifact_path is not None
        or loaded.fault_manifest is not None
        or loaded.rest_references is not None
    ):
        raise CleanCampaignError(
            "clean campaign requests cannot carry fault/no-touch/restoration state"
        )


def _with_ordinal(spec: CleanCampaignTrialSpec, ordinal: int) -> CleanCampaignTrialSpec:
    values = spec.to_dict()
    values["ordinal"] = ordinal
    return CleanCampaignTrialSpec.from_dict(values)


__all__ = [
    "CLEAN_REQUEST_FILENAME",
    "build_clean_campaign_manifest",
    "discover_clean_request_paths",
]
