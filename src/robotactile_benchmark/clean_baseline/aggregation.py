"""Strict multi-artifact inventory and clean success aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Tuple, cast

from robotactile_benchmark.clean_baseline.artifact_validation import (
    validate_clean_artifact,
)
from robotactile_benchmark.clean_baseline.attempts import (
    CleanAttemptDisposition,
    LoadedCleanAttempt,
    candidate_attempt_present,
    load_candidate_attempt,
    load_successful_attempt,
)
from robotactile_benchmark.clean_baseline.contracts import (
    CLEAN_BASELINE_EVIDENCE_LEVEL,
    CleanCampaignError,
    CleanCampaignManifest,
    CleanCampaignProtocol,
    CleanCampaignTrialSpec,
)
from robotactile_benchmark.clean_baseline.io import read_canonical_json_file
from robotactile_benchmark.clean_baseline.summary import (
    CleanBaselineSummary,
    CleanCandidateClassification,
    CleanCandidateProvenance,
    CleanTaskSummary,
    ProtocolInvalidTrial,
)
from robotactile_benchmark.closed_loop.artifact_io import sha256_bytes
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.reporting.statistics import (
    task_stratified_success_bootstrap,
    wilson_score_interval,
)
from robotactile_benchmark.runtime_source import RuntimeSourceBinding
from robotactile_benchmark.trials import Condition


class IncompleteCleanCampaignError(CleanCampaignError):
    """A complete report was requested before all planned evidence was valid."""


@dataclass(frozen=True)
class VerifiedCleanArtifact:
    trial_spec: CleanCampaignTrialSpec
    artifact: LoadedLiveUniVTACArtifact
    attempt_receipt_sha256: str
    disposition: CleanAttemptDisposition = CleanAttemptDisposition.VALID_OUTCOME
    exception_code: str | None = None
    attempt_semantic_version: str = "1.0"
    qualification_relpath: str | None = None
    qualification_sha256: str | None = None
    runtime_source_binding: RuntimeSourceBinding | None = None
    isaac_attestation_relpath: str | None = None
    isaac_attestation_sha256: str | None = None
    n0_server_attestation_relpath: str | None = None
    n0_server_attestation_sha256: str | None = None


@dataclass(frozen=True)
class CleanArtifactInventory:
    campaign_manifest_sha256: str
    artifacts: Tuple[VerifiedCleanArtifact, ...]
    missing_trials: Tuple[CleanCampaignTrialSpec, ...]
    protocol_invalid_trials: Tuple[ProtocolInvalidTrial, ...]
    candidate_provenance: Tuple[CleanCandidateProvenance, ...] = ()
    semantic_version: str = "1.0"

    def __post_init__(self) -> None:
        planned = {item.trial_spec.trial_manifest_sha256 for item in self.artifacts}
        missing = {item.trial_manifest_sha256 for item in self.missing_trials}
        invalid = {item.trial_manifest_sha256 for item in self.protocol_invalid_trials}
        if planned & missing or planned & invalid or missing & invalid:
            raise CleanCampaignError("clean artifact inventory identities overlap")
        if self.semantic_version not in {"1.0", "2.0"}:
            raise CleanCampaignError("unsupported clean artifact inventory version")
        provenance = tuple(self.candidate_provenance)
        if self.semantic_version == "1.0" and provenance:
            raise CleanCampaignError("v1 inventory cannot carry candidate provenance")
        if self.semantic_version == "2.0" and (
            not provenance
            or tuple(item.ordinal for item in provenance)
            != tuple(range(len(provenance)))
        ):
            raise CleanCampaignError("v2 candidate provenance must be ordinal-complete")
        object.__setattr__(self, "candidate_provenance", provenance)


def load_clean_artifact_inventory(
    deployment_root: Path,
    manifest: CleanCampaignManifest,
    *,
    allow_compact: bool = False,
) -> CleanArtifactInventory:
    """Load every planned request, artifact, and successful command receipt."""

    if type(manifest) is not CleanCampaignManifest:
        raise TypeError("manifest must be an exact CleanCampaignManifest")
    if type(allow_compact) is not bool:
        raise TypeError("allow_compact must be bool")
    if allow_compact and manifest.protocol_id is not CleanCampaignProtocol.DIAGNOSTIC:
        raise CleanCampaignError(
            "compact capture aggregation requires a diagnostic_v1 campaign"
        )
    root = _real_root(deployment_root)
    if manifest.semantic_version == "2.0":
        return _load_v2_artifact_inventory(
            root,
            manifest,
            allow_compact=allow_compact,
        )
    return _load_v1_artifact_inventory(
        root,
        manifest,
        allow_compact=allow_compact,
    )


def _load_v1_artifact_inventory(
    root: Path,
    manifest: CleanCampaignManifest,
    *,
    allow_compact: bool,
) -> CleanArtifactInventory:
    verified: list[VerifiedCleanArtifact] = []
    missing: list[CleanCampaignTrialSpec] = []
    invalid: list[ProtocolInvalidTrial] = []
    for spec in manifest.trials:
        run = _load_expected_request(root, manifest, spec)
        try:
            artifact_path = _member_path(root, spec.artifact_relpath, "artifact")
        except CleanCampaignError:
            invalid.append(_invalid(spec, "artifact_validation_failed"))
            continue
        if artifact_path.is_symlink() or (
            artifact_path.exists() and not artifact_path.is_dir()
        ):
            invalid.append(_invalid(spec, "artifact_validation_failed"))
            continue
        if not artifact_path.exists():
            missing.append(spec)
            continue
        try:
            artifact = load_live_univtac_artifact(artifact_path)
            validate_clean_artifact(
                run,
                spec,
                artifact,
                allow_compact=allow_compact,
            )
        except (OSError, TypeError, ValueError):
            invalid.append(_invalid(spec, "artifact_validation_failed"))
            continue
        receipt, reason = load_successful_attempt(root, manifest, spec, artifact)
        if receipt is None:
            assert reason is not None
            invalid.append(_invalid(spec, reason))
            continue
        verified.append(
            VerifiedCleanArtifact(
                trial_spec=spec,
                artifact=artifact,
                attempt_receipt_sha256=receipt,
            )
        )
    roots = [item.artifact.external_root_sha256 for item in verified]
    if len(roots) != len(set(roots)):
        raise CleanCampaignError("one live artifact is reused by multiple trials")
    return CleanArtifactInventory(
        campaign_manifest_sha256=manifest.sha256,
        artifacts=tuple(verified),
        missing_trials=tuple(missing),
        protocol_invalid_trials=tuple(invalid),
    )


def _load_v2_artifact_inventory(
    root: Path,
    manifest: CleanCampaignManifest,
    *,
    allow_compact: bool,
) -> CleanArtifactInventory:
    sampling = manifest.sampling
    if sampling is None:
        raise CleanCampaignError("v2 clean campaign lacks sampling")
    verified: list[VerifiedCleanArtifact] = []
    missing: list[CleanCampaignTrialSpec] = []
    invalid: list[ProtocolInvalidTrial] = []
    provenance_by_ordinal: dict[int, CleanCandidateProvenance] = {}
    tasks = sorted({item.task for item in manifest.trials})
    for task in tasks:
        candidates = tuple(item for item in manifest.trials if item.task == task)
        valid_count = 0
        blocked = False
        for spec in candidates:
            run = _load_expected_request(root, manifest, spec)
            artifact_path = _member_path(root, spec.artifact_relpath, "artifact")
            if blocked:
                if artifact_path.exists() or candidate_attempt_present(
                    root, manifest, spec
                ):
                    raise CleanCampaignError(
                        "official candidate overrun after a required-candidate gap"
                    )
                provenance_by_ordinal[spec.ordinal] = _candidate_provenance(
                    spec,
                    CleanCandidateClassification.MISSING_REQUIRED_CANDIDATE,
                )
                continue
            if valid_count == sampling.target_valid_trials_per_task:
                if artifact_path.exists() or candidate_attempt_present(
                    root, manifest, spec
                ):
                    raise CleanCampaignError(
                        "official candidate overrun after first target-valid stop"
                    )
                provenance_by_ordinal[spec.ordinal] = _candidate_provenance(
                    spec, CleanCandidateClassification.UNUSED_RESERVE
                )
                continue
            artifact, artifact_reason = _load_optional_artifact(
                run,
                spec,
                artifact_path,
                allow_compact=allow_compact,
            )
            if artifact_reason is not None:
                invalid.append(_invalid(spec, artifact_reason))
                provenance_by_ordinal[spec.ordinal] = _candidate_provenance(
                    spec, CleanCandidateClassification.MISSING_REQUIRED_CANDIDATE
                )
                blocked = True
                continue
            attempt, attempt_reason = load_candidate_attempt(
                root, manifest, spec, artifact
            )
            if attempt is None:
                assert attempt_reason is not None
                provenance_by_ordinal[spec.ordinal] = _candidate_provenance(
                    spec, CleanCandidateClassification.MISSING_REQUIRED_CANDIDATE
                )
                if artifact is None and attempt_reason == "attempt_receipt_missing":
                    missing.append(spec)
                else:
                    invalid.append(_invalid(spec, attempt_reason))
                blocked = True
                continue
            if attempt.disposition is CleanAttemptDisposition.VALID_OUTCOME:
                if artifact is None:
                    raise CleanCampaignError("valid candidate lost its bound artifact")
                valid_count += 1
                verified.append(_verified_from_attempt(spec, artifact, attempt))
                provenance_by_ordinal[spec.ordinal] = _candidate_provenance(
                    spec,
                    CleanCandidateClassification.VALID_OUTCOME,
                    artifact=artifact,
                    attempt_sha256=attempt.receipt_sha256,
                )
                continue
            verified_artifact = None
            if artifact is not None:
                verified_artifact = _verified_from_attempt(spec, artifact, attempt)
                verified.append(verified_artifact)
            provenance_by_ordinal[spec.ordinal] = _candidate_provenance(
                spec,
                CleanCandidateClassification.EXCEPTION_REPLACED,
                artifact=None
                if verified_artifact is None
                else verified_artifact.artifact,
                attempt_sha256=attempt.receipt_sha256,
                exception_code=attempt.exception_code,
            )
    provenance = tuple(
        provenance_by_ordinal[index] for index in range(len(manifest.trials))
    )
    roots = [item.artifact.external_root_sha256 for item in verified]
    if len(roots) != len(set(roots)):
        raise CleanCampaignError("one live artifact is reused by multiple candidates")
    return CleanArtifactInventory(
        campaign_manifest_sha256=manifest.sha256,
        artifacts=tuple(verified),
        missing_trials=tuple(missing),
        protocol_invalid_trials=tuple(invalid),
        candidate_provenance=provenance,
        semantic_version="2.0",
    )


def _verified_from_attempt(
    spec: CleanCampaignTrialSpec,
    artifact: LoadedLiveUniVTACArtifact,
    attempt: LoadedCleanAttempt,
) -> VerifiedCleanArtifact:
    return VerifiedCleanArtifact(
        trial_spec=spec,
        artifact=artifact,
        attempt_receipt_sha256=attempt.receipt_sha256,
        disposition=attempt.disposition,
        exception_code=attempt.exception_code,
        attempt_semantic_version=attempt.semantic_version,
        qualification_relpath=attempt.qualification_relpath,
        qualification_sha256=attempt.qualification_sha256,
        runtime_source_binding=attempt.runtime_source_binding,
        isaac_attestation_relpath=attempt.isaac_attestation_relpath,
        isaac_attestation_sha256=attempt.isaac_attestation_sha256,
        n0_server_attestation_relpath=attempt.n0_server_attestation_relpath,
        n0_server_attestation_sha256=attempt.n0_server_attestation_sha256,
    )


def _load_optional_artifact(
    run: LoadedLiveUniVTACRun,
    spec: CleanCampaignTrialSpec,
    artifact_path: Path,
    *,
    allow_compact: bool,
) -> tuple[LoadedLiveUniVTACArtifact | None, str | None]:
    if artifact_path.is_symlink() or (
        artifact_path.exists() and not artifact_path.is_dir()
    ):
        return None, "artifact_validation_failed"
    if not artifact_path.exists():
        return None, None
    try:
        artifact = load_live_univtac_artifact(artifact_path)
        validate_clean_artifact(
            run,
            spec,
            artifact,
            allow_compact=allow_compact,
        )
    except (OSError, TypeError, ValueError):
        return None, "artifact_validation_failed"
    return artifact, None


def _candidate_provenance(
    spec: CleanCampaignTrialSpec,
    classification: CleanCandidateClassification,
    *,
    artifact: LoadedLiveUniVTACArtifact | None = None,
    attempt_sha256: str | None = None,
    exception_code: str | None = None,
) -> CleanCandidateProvenance:
    return CleanCandidateProvenance(
        task=spec.task,
        ordinal=spec.ordinal,
        trial_manifest_sha256=spec.trial_manifest_sha256,
        classification=classification,
        attempt_receipt_sha256=attempt_sha256,
        artifact_root_sha256=(
            None if artifact is None else artifact.external_root_sha256
        ),
        exception_code=exception_code,
    )


def aggregate_clean_campaign(
    deployment_root: Path,
    manifest: CleanCampaignManifest,
    *,
    allow_partial: bool = False,
    allow_compact: bool = False,
) -> CleanBaselineSummary:
    """Return a summary, failing closed on incomplete evidence by default."""

    if type(allow_partial) is not bool:
        raise TypeError("allow_partial must be bool")
    inventory = load_clean_artifact_inventory(
        deployment_root,
        manifest,
        allow_compact=allow_compact,
    )
    incomplete = bool(inventory.missing_trials or inventory.protocol_invalid_trials)
    if manifest.semantic_version == "2.0":
        sampling = manifest.sampling
        if sampling is None:
            raise CleanCampaignError("v2 clean campaign lacks sampling")
        valid_count = sum(
            item.classification is CleanCandidateClassification.VALID_OUTCOME
            for item in inventory.candidate_provenance
        )
        task_count = len({item.task for item in manifest.trials})
        incomplete = incomplete or valid_count != (
            sampling.target_valid_trials_per_task * task_count
        )
    if not allow_partial and incomplete:
        raise IncompleteCleanCampaignError(
            "clean campaign is incomplete; pass allow_partial=True for a "
            "non-claimable diagnostic summary"
        )
    return build_clean_baseline_summary(manifest, inventory)


def build_clean_baseline_summary(
    manifest: CleanCampaignManifest,
    inventory: CleanArtifactInventory,
) -> CleanBaselineSummary:
    if inventory.campaign_manifest_sha256 != manifest.sha256:
        raise CleanCampaignError("inventory belongs to another campaign manifest")
    artifacts_by_task: dict[str, list[VerifiedCleanArtifact]] = defaultdict(list)
    missing_by_task = Counter(item.task for item in inventory.missing_trials)
    spec_by_hash = {item.trial_manifest_sha256: item for item in manifest.trials}
    invalid_by_task: Counter[str] = Counter()
    for invalid_trial in inventory.protocol_invalid_trials:
        try:
            invalid_by_task[spec_by_hash[invalid_trial.trial_manifest_sha256].task] += 1
        except KeyError as error:
            raise CleanCampaignError(
                "protocol-invalid trial is absent from the manifest"
            ) from error
    for verified_artifact in inventory.artifacts:
        artifacts_by_task[verified_artifact.trial_spec.task].append(verified_artifact)
    task_summaries = []
    outcomes_by_task: dict[str, tuple[bool, ...]] = {}
    planned_by_task = Counter(item.task for item in manifest.trials)
    campaign_statuses: Counter[str] = Counter()
    is_v2 = manifest.semantic_version == "2.0"
    sampling = manifest.sampling
    if is_v2 and sampling is None:
        raise CleanCampaignError("v2 clean campaign lacks sampling")
    for task in sorted(planned_by_task):
        artifacts = artifacts_by_task[task]
        results = [item.artifact.evidence.result for item in artifacts]
        statuses = Counter(result.terminal_status.value for result in results)
        campaign_statuses.update(statuses)
        outcomes = tuple(
            bool(item.artifact.evidence.result.score_success)
            for item in artifacts
            if (
                not is_v2
                and item.artifact.evidence.result.score_eligible
                or is_v2
                and item.disposition is CleanAttemptDisposition.VALID_OUTCOME
            )
        )
        if outcomes:
            outcomes_by_task[task] = outcomes
        successes = sum(outcomes)
        eligible = len(outcomes)
        candidate_counts = Counter(
            item.classification
            for item in inventory.candidate_provenance
            if item.task == task
        )
        valid_candidates = candidate_counts[CleanCandidateClassification.VALID_OUTCOME]
        replaced_candidates = candidate_counts[
            CleanCandidateClassification.EXCEPTION_REPLACED
        ]
        unused_candidates = candidate_counts[
            CleanCandidateClassification.UNUSED_RESERVE
        ]
        missing_candidates = candidate_counts[
            CleanCandidateClassification.MISSING_REQUIRED_CANDIDATE
        ]
        task_summaries.append(
            CleanTaskSummary(
                task=task,
                planned=planned_by_task[task],
                loaded=len(artifacts),
                missing=missing_by_task[task],
                protocol_invalid=invalid_by_task[task],
                eligible=eligible,
                successes=successes,
                failures=eligible - successes,
                ineligible=len(artifacts) - eligible,
                eligible_success_rate=(None if not outcomes else successes / eligible),
                wilson=(
                    None
                    if not outcomes
                    else wilson_score_interval(
                        successes,
                        eligible,
                        confidence_level=manifest.confidence_level,
                    )
                ),
                unique_initial_state_count=len(
                    {
                        result.initial_state_sha256
                        for result in results
                        if result.initial_state_sha256 is not None
                    }
                ),
                terminal_status_counts=dict(sorted(statuses.items())),
                target_valid_trial_count=(
                    None if sampling is None else sampling.target_valid_trials_per_task
                ),
                candidate_trial_count=(
                    None if sampling is None else planned_by_task[task]
                ),
                attempted_candidate_count=(
                    None if sampling is None else valid_candidates + replaced_candidates
                ),
                exception_replacement_count=(
                    None if sampling is None else replaced_candidates
                ),
                unused_reserve_count=(None if sampling is None else unused_candidates),
                valid_outcome_count=(None if sampling is None else valid_candidates),
                missing_required_candidate_count=(
                    None if sampling is None else missing_candidates
                ),
            )
        )
    eligible_count = sum(item.eligible for item in task_summaries)
    success_count = sum(item.successes for item in task_summaries)
    macro_rate = (
        None
        if not outcomes_by_task
        else sum(sum(values) / len(values) for values in outcomes_by_task.values())
        / len(outcomes_by_task)
    )
    bootstrap = (
        None
        if not outcomes_by_task
        else task_stratified_success_bootstrap(
            outcomes_by_task,
            n_resamples=manifest.bootstrap_resamples,
            confidence_level=manifest.confidence_level,
            seed=manifest.bootstrap_seed,
        )
    )
    invalid_count = len(inventory.protocol_invalid_trials)
    missing_count = len(inventory.missing_trials)
    ineligible_count = len(inventory.artifacts) - eligible_count
    classification_counts = Counter(
        item.classification for item in inventory.candidate_provenance
    )
    valid_outcome_count = classification_counts[
        CleanCandidateClassification.VALID_OUTCOME
    ]
    exception_replacement_count = classification_counts[
        CleanCandidateClassification.EXCEPTION_REPLACED
    ]
    unused_reserve_count = classification_counts[
        CleanCandidateClassification.UNUSED_RESERVE
    ]
    missing_required_count = classification_counts[
        CleanCandidateClassification.MISSING_REQUIRED_CANDIDATE
    ]
    target_valid_count = (
        None
        if sampling is None
        else sampling.target_valid_trials_per_task * len(planned_by_task)
    )
    blockers = []
    if not manifest.protocol_allows_paper_claim:
        blockers.append("protocol_not_paper")
    if missing_count:
        blockers.append("missing_artifacts")
    if ineligible_count and not is_v2:
        blockers.append("ineligible_outcomes")
    if invalid_count:
        blockers.append("protocol_invalid_trials")
    if is_v2 and valid_outcome_count != target_valid_count:
        blockers.append("replacement_pool_exhausted")
    blockers.append("simulator_not_qualified")
    return CleanBaselineSummary(
        campaign_id=manifest.campaign_id,
        campaign_manifest_sha256=manifest.sha256,
        protocol_id=manifest.protocol_id,
        protocol_allows_paper_claim=manifest.protocol_allows_paper_claim,
        paper_claim_eligible=False,
        paper_claim_blockers=tuple(blockers),
        evidence_level=CLEAN_BASELINE_EVIDENCE_LEVEL,
        simulator_qualification_claimed=False,
        policy_kind=manifest.policy_kind,
        task_registry_sha256=manifest.task_registry_sha256,
        planned_trial_count=manifest.planned_trial_count,
        loaded_artifact_count=len(inventory.artifacts),
        missing_artifact_count=missing_count,
        protocol_invalid_count=invalid_count,
        eligible_trial_count=eligible_count,
        success_count=success_count,
        failure_count=eligible_count - success_count,
        ineligible_count=ineligible_count,
        artifact_completion_rate=len(inventory.artifacts)
        / manifest.planned_trial_count,
        eligible_success_rate=(
            None if eligible_count == 0 else success_count / eligible_count
        ),
        macro_task_success_rate=macro_rate,
        pooled_wilson=(
            None
            if eligible_count == 0
            else wilson_score_interval(
                success_count,
                eligible_count,
                confidence_level=manifest.confidence_level,
            )
        ),
        task_stratified_bootstrap=bootstrap,
        task_summaries=tuple(task_summaries),
        terminal_status_counts=dict(sorted(campaign_statuses.items())),
        missing_trial_manifest_sha256s=tuple(
            item.trial_manifest_sha256 for item in inventory.missing_trials
        ),
        protocol_invalid_trials=inventory.protocol_invalid_trials,
        artifact_root_sha256s=tuple(
            (item.artifact.external_root_sha256 for item in inventory.artifacts)
            if not is_v2
            else (
                item.artifact_root_sha256
                for item in inventory.candidate_provenance
                if item.artifact_root_sha256 is not None
            )
        ),
        attempt_receipt_sha256s=tuple(
            (item.attempt_receipt_sha256 for item in inventory.artifacts)
            if not is_v2
            else (
                cast(str, item.attempt_receipt_sha256)
                for item in inventory.candidate_provenance
                if item.artifact_root_sha256 is not None
            )
        ),
        statistically_complete=(
            valid_outcome_count == target_valid_count
            and missing_required_count == 0
            and invalid_count == 0
            if is_v2
            else missing_count == 0 and invalid_count == 0 and ineligible_count == 0
        ),
        confidence_level=manifest.confidence_level,
        bootstrap_resamples=manifest.bootstrap_resamples,
        bootstrap_seed=manifest.bootstrap_seed,
        semantic_version=manifest.semantic_version,
        target_valid_trial_count=target_valid_count,
        candidate_trial_count=(None if not is_v2 else manifest.planned_trial_count),
        attempted_candidate_count=(
            None if not is_v2 else valid_outcome_count + exception_replacement_count
        ),
        exception_replacement_count=(
            None if not is_v2 else exception_replacement_count
        ),
        unused_reserve_count=(None if not is_v2 else unused_reserve_count),
        valid_outcome_count=(None if not is_v2 else valid_outcome_count),
        missing_required_candidate_count=(
            None if not is_v2 else missing_required_count
        ),
        candidate_provenance=inventory.candidate_provenance,
        exception_attempt_receipt_sha256s=tuple(
            item.attempt_receipt_sha256
            for item in inventory.candidate_provenance
            if item.classification is CleanCandidateClassification.EXCEPTION_REPLACED
            and item.attempt_receipt_sha256 is not None
        ),
    )


def _load_expected_request(
    root: Path,
    manifest: CleanCampaignManifest,
    spec: CleanCampaignTrialSpec,
) -> LoadedLiveUniVTACRun:
    request_path = _member_path(root, spec.request_relpath, "request")
    document, raw = read_canonical_json_file(request_path, "frozen clean request")
    del document
    if sha256_bytes(raw) != spec.request_file_sha256:
        raise CleanCampaignError("frozen clean request hash changed")
    request = load_live_univtac_request(request_path)
    run = load_live_univtac_run(request)
    artifact_path = _member_path(root, spec.artifact_relpath, "artifact")
    checks = (
        request.condition is Condition.CLEAN,
        request.policy_kind is manifest.policy_kind,
        request.output_dir == artifact_path,
        run.fault_manifest is None,
        run.rest_references is None,
        run.trial.task == spec.task,
        run.trial.initial_seed == spec.initial_seed,
        run.trial.exogenous_seed == spec.exogenous_seed,
        run.trial.base_system_id == spec.base_system_id,
        run.trial.dataset_sha256 == spec.dataset_sha256,
        run.trial.base_system_manifest_sha256 == spec.base_system_manifest_sha256,
        run.trial.checkpoint_sha256 == spec.checkpoint_sha256,
        run.trial.config_sha256 == spec.config_sha256,
        run.trial.sha256 == spec.trial_manifest_sha256,
        run.trial.pair_key == spec.pair_key,
        run.run_spec.sha256 == spec.run_spec_sha256,
        run.content_sha256 == spec.run_content_sha256,
        run.run_spec.max_control_cycles == spec.max_control_cycles,
        run.run_spec.max_observation_steps == spec.max_observation_steps,
        run.run_spec.execute_action_steps == spec.execute_action_steps,
    )
    if not all(checks):
        raise CleanCampaignError("frozen clean request no longer matches its manifest")
    return run


def _member_path(root: Path, relative: str, name: str) -> Path:
    path = root / Path(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise CleanCampaignError(f"{name} path cannot traverse a symlink")
    return path


def _real_root(path: Path) -> Path:
    root = Path(path).absolute()
    if root.is_symlink() or not root.is_dir():
        raise CleanCampaignError("deployment_root must be a real existing directory")
    return root.resolve(strict=True)


def _invalid(spec: CleanCampaignTrialSpec, reason: str) -> ProtocolInvalidTrial:
    return ProtocolInvalidTrial(spec.trial_manifest_sha256, reason)


__all__ = [
    "CleanArtifactInventory",
    "CleanBaselineSummary",
    "IncompleteCleanCampaignError",
    "VerifiedCleanArtifact",
    "aggregate_clean_campaign",
    "build_clean_baseline_summary",
    "load_clean_artifact_inventory",
]
