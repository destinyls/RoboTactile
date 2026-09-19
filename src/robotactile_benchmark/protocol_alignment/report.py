"""Aggregate verified Clean artifacts into a bounded N0 protocol comparison."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Optional, Tuple

from robotactile_benchmark.backends.univtac_contracts import (
    FIXED_NATIVE_STEP_CONTRACT,
    N0_DECIMATION,
    N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT,
    N0_PHYSICS_STEPS_PER_ACTION,
    N0_STOCK_EE_ACTION_EXECUTION_CONTRACT,
    N0_STOCK_EE_NATIVE_STEP_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    SIM_HZ,
)
from robotactile_benchmark.clean_baseline.aggregation import (
    CleanArtifactInventory,
    VerifiedCleanArtifact,
)
from robotactile_benchmark.clean_baseline.attempts import CleanAttemptDisposition
from robotactile_benchmark.clean_baseline.contracts import CleanCampaignManifest
from robotactile_benchmark.clean_baseline.summary_values import (
    CleanCandidateClassification,
)
from robotactile_benchmark.closed_loop.capture import TransitionTraceEntry
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.protocol_alignment.contracts import (
    ALIGNMENT_EVIDENCE_LEVEL,
    ALIGNMENT_REPORT_SEMANTIC_VERSION,
    GateStatus,
    N0PaperReference,
    ProtocolGate,
    TrialProtocolDiagnostic,
)
from robotactile_benchmark.trials import TerminalStatus

STOCK_ACTION_EXECUTION = N0_STOCK_EE_ACTION_EXECUTION_CONTRACT
FIXED_ENDPOINT_EXECUTION = N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT
TRAINING_60HZ_ACTION_EXECUTION = N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
TRAINING_ACTION_DELIVERY_HZ = SIM_HZ / N0_PHYSICS_STEPS_PER_ACTION


def _mapping(value: object) -> Optional[Mapping[str, object]]:
    return value if isinstance(value, Mapping) else None


def _optional_bool(mapping: Mapping[str, object], key: str) -> Optional[bool]:
    value = mapping.get(key)
    return value if type(value) is bool else None


def _status_for_flags(values: Sequence[Optional[bool]]) -> GateStatus:
    if not values or any(value is False for value in values):
        return GateStatus.FAIL
    if any(value is None for value in values):
        return GateStatus.UNKNOWN
    return GateStatus.PASS


def _execution_status(
    transitions: Sequence[TransitionTraceEntry], result_status: TerminalStatus
) -> GateStatus:
    if result_status is TerminalStatus.CRASH:
        return GateStatus.FAIL
    flags: list[Optional[bool]] = []
    for transition in transitions:
        flags.extend(
            (
                _optional_bool(transition.diagnostics, "execution_success"),
                _optional_bool(transition.diagnostics, "plan_success"),
            )
        )
    return _status_for_flags(flags)


def _cadence_status(
    initial: Mapping[str, object], transitions: Sequence[TransitionTraceEntry]
) -> GateStatus:
    native_contract = initial.get("native_step_contract")
    deltas = tuple(item.diagnostics.get("physics_step_delta") for item in transitions)
    transition_contracts = tuple(
        item.diagnostics.get("native_step_contract") for item in transitions
    )
    if native_contract == N0_STOCK_EE_NATIVE_STEP_CONTRACT:
        if not deltas or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in deltas
        ):
            return GateStatus.UNKNOWN
        return (
            GateStatus.PASS
            if all(value == native_contract for value in transition_contracts)
            else GateStatus.FAIL
        )
    if native_contract is None:
        return GateStatus.UNKNOWN
    if native_contract != FIXED_NATIVE_STEP_CONTRACT:
        return GateStatus.FAIL
    expected = initial.get("physics_steps_per_action")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        return GateStatus.UNKNOWN
    if not deltas or any(
        isinstance(value, bool) or not isinstance(value, int) for value in deltas
    ):
        return GateStatus.UNKNOWN
    if not all(value == native_contract for value in transition_contracts):
        return GateStatus.FAIL
    if not all(value == expected for value in deltas):
        return GateStatus.FAIL
    action_execution = _action_execution_mode(initial, transitions)
    explicit_action_execution = initial.get("action_execution_contract")
    if explicit_action_execution == TRAINING_60HZ_ACTION_EXECUTION:
        if action_execution != TRAINING_60HZ_ACTION_EXECUTION:
            return GateStatus.FAIL
        sim_hz = initial.get("sim_hz")
        decimation = initial.get("decimation")
        camera_delivery_hz = initial.get("camera_delivery_hz")
        if (
            isinstance(sim_hz, bool)
            or not isinstance(sim_hz, (int, float))
            or isinstance(decimation, bool)
            or not isinstance(decimation, int)
            or isinstance(camera_delivery_hz, bool)
            or not isinstance(camera_delivery_hz, (int, float))
        ):
            return GateStatus.UNKNOWN
        if (
            float(sim_hz) != float(SIM_HZ)
            or decimation != N0_DECIMATION
            or expected != N0_PHYSICS_STEPS_PER_ACTION
            or float(camera_delivery_hz) != TRAINING_ACTION_DELIVERY_HZ
        ):
            return GateStatus.FAIL
    return GateStatus.PASS


def _action_execution_mode(
    initial: Mapping[str, object],
    transitions: Sequence[TransitionTraceEntry],
) -> Optional[str]:
    explicit = initial.get("action_execution_contract")
    transition_values = tuple(
        item.diagnostics.get("action_execution_contract") for item in transitions
    )
    if isinstance(explicit, str) and explicit:
        if all(value == explicit for value in transition_values):
            return explicit
        return None
    if not transitions:
        return None
    values = tuple(
        item.diagnostics.get("n0_fixed_cadence", ...) for item in transitions
    )
    if any(isinstance(value, Mapping) for value in values):
        return FIXED_ENDPOINT_EXECUTION
    if all(value is None for value in values):
        return STOCK_ACTION_EXECUTION
    return None


def _reset_mode(initial: Mapping[str, object]) -> Optional[str]:
    reset = _mapping(initial.get("n0_reset"))
    if reset is None:
        return None
    mode = reset.get("mode_during")
    return mode if isinstance(mode, str) and mode else None


def _detailed_predicate_witness(
    initial: Mapping[str, object], transitions: Sequence[TransitionTraceEntry]
) -> bool:
    candidates: list[object] = [initial.get("task")]
    if transitions:
        candidates.append(transitions[-1].diagnostics.get("task"))
    for value in candidates:
        task = _mapping(value)
        if task is not None and (
            task.get("available") is True or type(task.get("predicate_success")) is bool
        ):
            return True
    return False


def _classification(
    terminal: TerminalStatus,
    transitions: Sequence[TransitionTraceEntry],
    immediate: bool,
) -> str:
    if terminal is TerminalStatus.CRASH:
        return "infrastructure_crash"
    if immediate and terminal is TerminalStatus.SUCCESS:
        return "first_action_success"
    if immediate and terminal is TerminalStatus.EARLY_STOP:
        return "first_action_early_stop"
    if terminal is TerminalStatus.TIMEOUT and transitions:
        last = transitions[-1]
        if last.diagnostics.get("action_horizon") == last.benchmark_step_index:
            return "backend_horizon_timeout"
    return str(terminal.value)


def _trial_diagnostic(verified: VerifiedCleanArtifact) -> TrialProtocolDiagnostic:
    spec = verified.trial_spec
    artifact = verified.artifact
    evidence = artifact.evidence
    result = evidence.result
    transitions = evidence.transition_entries
    executed = sum(item.executed_actions.shape[0] for item in evidence.action_entries)
    transition_status = (
        GateStatus.PASS
        if len(transitions) == result.observation_count - 1 == executed
        else GateStatus.FAIL
    )
    immediate = (
        len(transitions) == 1
        and result.observation_count == 2
        and result.control_cycle_count == 1
        and result.terminal_status
        in {
            TerminalStatus.SUCCESS,
            TerminalStatus.EARLY_STOP,
            TerminalStatus.TASK_FAILURE,
        }
    )
    initial = evidence.initial_diagnostics
    return TrialProtocolDiagnostic(
        ordinal=spec.ordinal,
        task_id=spec.task,
        artifact_root_sha256=artifact.external_root_sha256,
        terminal_status=result.terminal_status.value,
        score_success=result.score_success,
        observation_count=result.observation_count,
        control_cycle_count=result.control_cycle_count,
        transition_count=len(transitions),
        executed_action_count=executed,
        first_signal=None if not transitions else transitions[0].signal.value,
        last_signal=None if not transitions else transitions[-1].signal.value,
        classification=_classification(result.terminal_status, transitions, immediate),
        immediate_terminal=immediate,
        initial_success_check=_optional_bool(initial, "success_check"),
        initial_early_stop=_optional_bool(initial, "early_stop"),
        detailed_predicate_witness=_detailed_predicate_witness(initial, transitions),
        transition_contract_status=transition_status,
        execution_contract_status=_execution_status(
            transitions, result.terminal_status
        ),
        cadence_status=_cadence_status(initial, transitions),
        reset_mode=_reset_mode(initial),
        action_execution_mode=_action_execution_mode(initial, transitions),
        failure_stage=result.failure_stage,
        failure_code=result.failure_code,
    )


def _gate(
    gate_id: str,
    status: GateStatus,
    code: str,
    detail: str,
    tasks: Sequence[str] = (),
    *,
    paper_blocking: bool = True,
) -> ProtocolGate:
    return ProtocolGate(
        gate_id=gate_id,
        status=status,
        code=code,
        detail=detail,
        affected_tasks=tuple(sorted(set(tasks))),
        paper_blocking=paper_blocking,
    )


def _aggregate_status(
    diagnostics: Sequence[TrialProtocolDiagnostic], attribute: str
) -> GateStatus:
    values = tuple(getattr(item, attribute) for item in diagnostics)
    if any(value is GateStatus.FAIL for value in values):
        return GateStatus.FAIL
    if not values or any(value is GateStatus.UNKNOWN for value in values):
        return GateStatus.UNKNOWN
    return GateStatus.PASS


def _initial_predicate_observability(
    diagnostics: Sequence[TrialProtocolDiagnostic],
) -> tuple[GateStatus, Tuple[str, ...]]:
    """Require reset predicates to be observed without filtering their values."""

    missing = tuple(
        item.task_id
        for item in diagnostics
        if item.initial_success_check is None or item.initial_early_stop is None
    )
    status = GateStatus.PASS if diagnostics and not missing else GateStatus.UNKNOWN
    return status, tuple(sorted(set(missing)))


def _build_gates(
    manifest: CleanCampaignManifest,
    inventory: CleanArtifactInventory,
    reference: N0PaperReference,
    diagnostics: Sequence[TrialProtocolDiagnostic],
) -> Tuple[ProtocolGate, ...]:
    tasks = {item.task for item in manifest.trials}
    counts = Counter(item.task for item in manifest.trials)
    expected_tasks = set(reference.task_ids)
    sampling_spec = manifest.sampling
    valid_roots = {
        item.artifact.external_root_sha256
        for item in inventory.artifacts
        if item.disposition is CleanAttemptDisposition.VALID_OUTCOME
    }
    valid_diagnostics = tuple(
        item for item in diagnostics if item.artifact_root_sha256 in valid_roots
    )
    candidate_counts = Counter(
        item.classification for item in inventory.candidate_provenance
    )
    if sampling_spec is None:
        complete = (
            len(inventory.artifacts) == manifest.planned_trial_count
            and not inventory.missing_trials
            and not inventory.protocol_invalid_trials
        )
        sampling = (
            manifest.protocol_id is reference.required_protocol_id
            and tasks == expected_tasks
            and all(
                counts[task] == reference.trials_per_task for task in expected_tasks
            )
        )
        seed_matches = False
        exception_matches = False
        replacement_count = 0
    else:
        target_count = sampling_spec.target_valid_trials_per_task * len(tasks)
        valid_count = candidate_counts[CleanCandidateClassification.VALID_OUTCOME]
        complete = (
            valid_count == target_count
            and not inventory.missing_trials
            and not inventory.protocol_invalid_trials
        )
        sampling = (
            manifest.protocol_id is reference.required_protocol_id
            and tasks == expected_tasks
            and sampling_spec.target_valid_trials_per_task == reference.trials_per_task
            and all(
                counts[task] == sampling_spec.candidate_trials_per_task
                for task in expected_tasks
            )
        )
        seed_matches = (
            sampling_spec.seed_protocol == reference.required_seed_protocol
            and sampling_spec.policy_seed_mode == "same_as_task_seed_v1"
        )
        exception_matches = (
            sampling_spec.exception_handling == reference.required_exception_handling
        )
        replacement_count = candidate_counts[
            CleanCandidateClassification.EXCEPTION_REPLACED
        ]
    reset_modes = tuple(item.reset_mode for item in valid_diagnostics)
    if any(
        mode is not None and mode != reference.required_reset_mode
        for mode in reset_modes
    ):
        reset_status = GateStatus.FAIL
    elif reset_modes and all(
        mode == reference.required_reset_mode for mode in reset_modes
    ):
        reset_status = GateStatus.PASS
    else:
        reset_status = GateStatus.UNKNOWN
    action_modes = tuple(item.action_execution_mode for item in valid_diagnostics)
    if any(
        mode is not None and mode != reference.required_action_execution
        for mode in action_modes
    ):
        action_status = GateStatus.FAIL
    elif action_modes and all(
        mode == reference.required_action_execution for mode in action_modes
    ):
        action_status = GateStatus.PASS
    else:
        action_status = GateStatus.UNKNOWN
    reset_observability_status, missing_initial_predicates = (
        _initial_predicate_observability(valid_diagnostics)
    )
    crashes = tuple(
        item.task_id
        for item in diagnostics
        if item.terminal_status == TerminalStatus.CRASH.value
    )
    missing_predicates = tuple(
        item.task_id
        for item in valid_diagnostics
        if not item.detailed_predicate_witness
    )
    return (
        _gate(
            "artifact_inventory",
            GateStatus.PASS if complete else GateStatus.FAIL,
            "complete" if complete else "incomplete",
            "Every required valid outcome has strict artifact/receipt evidence; unused replacement reserves are exempt.",
        ),
        _gate(
            "policy_family",
            GateStatus.PASS
            if manifest.policy_kind is LivePolicyKind.N0
            else GateStatus.FAIL,
            "n0" if manifest.policy_kind is LivePolicyKind.N0 else "wrong_policy",
            "The campaign must execute N0-TWAM, not ACT.",
        ),
        _gate(
            "task_set",
            GateStatus.PASS if tasks == expected_tasks else GateStatus.FAIL,
            "all_8" if tasks == expected_tasks else "task_set_mismatch",
            "The task set must exactly match the frozen eight-task UniVTAC registry.",
        ),
        _gate(
            "paper_sampling",
            GateStatus.PASS if sampling else GateStatus.FAIL,
            "paper_v1_100_per_task" if sampling else "not_100_per_task",
            "Paper comparison requires exactly 100 valid randomized trials per task.",
        ),
        _gate(
            "seed_protocol",
            GateStatus.PASS if seed_matches else GateStatus.FAIL,
            "official_seed_sequence" if seed_matches else "seed_contract_mismatch",
            "The released evaluator uses one consecutive task-local seed sequence, reused independently by each task.",
        ),
        _gate(
            "reset_mode",
            reset_status,
            "eval_reset"
            if reset_status is GateStatus.PASS
            else "reset_mode_unproven_or_mismatched",
            "The task must remain in eval mode throughout reset.",
            tuple(
                item.task_id
                for item in valid_diagnostics
                if item.reset_mode != reference.required_reset_mode
            ),
        ),
        _gate(
            "initial_predicate_observability",
            reset_observability_status,
            "initial_predicates_recorded"
            if reset_observability_status is GateStatus.PASS
            else "initial_predicate_observability_missing",
            (
                "Reset-time check_success and check_early_stop are observability "
                "witnesses only under official_reproduction; a recorded true value "
                "does not exclude or invalidate the official scored outcome."
            ),
            missing_initial_predicates,
        ),
        _gate(
            "action_execution",
            action_status,
            "n0_training_60hz_ee"
            if action_status is GateStatus.PASS
            else "non_release_or_unknown_action_path",
            (
                "The RoboTactile release comparison requires the source-bound N0 "
                "training-aligned 60 Hz EE endpoint contract; this does not establish "
                "identity with the unpublished paper evaluator."
            ),
            tuple(
                item.task_id
                for item in valid_diagnostics
                if item.action_execution_mode != reference.required_action_execution
            ),
        ),
        _gate(
            "transition_integrity",
            _aggregate_status(valid_diagnostics, "transition_contract_status"),
            "transition_action_link",
            "Transition count, observation count, and executed action count must agree.",
        ),
        _gate(
            "planning_execution",
            _aggregate_status(valid_diagnostics, "execution_contract_status"),
            (
                "planning_execution_observed"
                if _aggregate_status(valid_diagnostics, "execution_contract_status")
                is GateStatus.PASS
                else "planning_execution_diagnostic"
            ),
            (
                "Planning and execution flags remain robust diagnostics; under "
                "official_reproduction they are not an independent paper-comparison "
                "exclusion gate beyond the released evaluator's terminal outcome."
            ),
            tuple(
                item.task_id
                for item in valid_diagnostics
                if item.execution_contract_status is not GateStatus.PASS
            ),
            paper_blocking=False,
        ),
        _gate(
            "native_cadence",
            _aggregate_status(valid_diagnostics, "cadence_status"),
            (
                "training_60hz_cadence"
                if _aggregate_status(valid_diagnostics, "cadence_status")
                is GateStatus.PASS
                else "cadence_mismatch_or_unknown"
            ),
            (
                "The production N0 contract requires 120 Hz simulation, exactly two "
                "native physics steps per EE endpoint, and 60 Hz camera/action delivery."
            ),
        ),
        _gate(
            "task_predicate_witness",
            GateStatus.PASS if not missing_predicates else GateStatus.UNKNOWN,
            "all_predicates_witnessed"
            if not missing_predicates
            else "predicate_state_unavailable",
            "Task-specific state witnesses are required to explain success and early-stop decisions.",
            missing_predicates,
        ),
        _gate(
            "infrastructure_outcomes",
            (GateStatus.PASS if not crashes or exception_matches else GateStatus.FAIL),
            (
                "no_crash"
                if not crashes
                else "exceptions_replaced"
                if exception_matches
                else "crash_present"
            ),
            "Infrastructure exceptions are reported separately and may only be excluded through the released replacement contract.",
            crashes,
        ),
        _gate(
            "exception_denominator",
            GateStatus.PASS if exception_matches else GateStatus.FAIL,
            "official_replacement" if exception_matches else "crash_counted_as_failure",
            (
                "The released evaluator excludes caught exceptions from the denominator "
                f"and replaces them; verified replacements: {replacement_count}."
            ),
        ),
        _gate(
            "task_action_transform",
            GateStatus.UNKNOWN,
            "raw_native_action_equivalence_unproven",
            "Artifacts do not yet prove that every executed and KV-committed native action is unmodified.",
        ),
        _gate(
            "checkpoint_paper_lineage",
            GateStatus.UNKNOWN,
            "paper_checkpoint_revision_unpublished",
            "Released weights are pinned, but the public 84.5% result does not identify an exact checkpoint revision.",
        ),
        _gate(
            "observation_parity",
            GateStatus.UNKNOWN,
            "training_eval_pixel_time_parity_unbound",
            "Existing artifacts do not cryptographically bind paper-eval camera, renderer, tactile, and model-boundary tensors.",
        ),
        _gate(
            "expert_trajectory_alignment",
            GateStatus.UNKNOWN,
            "expert_alignment_not_attached",
            "Expert alignment tools exist, but no per-task expert report is attached to this campaign report.",
        ),
    )


def build_n0_protocol_alignment_report(
    manifest: CleanCampaignManifest,
    inventory: CleanArtifactInventory,
    reference: N0PaperReference,
) -> dict[str, object]:
    """Build a no-false-pass comparison report from already verified evidence."""

    if inventory.campaign_manifest_sha256 != manifest.sha256:
        raise ValueError("inventory belongs to another campaign manifest")
    diagnostics = tuple(_trial_diagnostic(item) for item in inventory.artifacts)
    gates = _build_gates(manifest, inventory, reference, diagnostics)
    blocking_gates = tuple(gate for gate in gates if gate.paper_blocking)
    blockers = tuple(
        gate.code for gate in blocking_gates if gate.status is not GateStatus.PASS
    )
    eligible = tuple(
        item.artifact.evidence.result
        for item in inventory.artifacts
        if item.disposition is CleanAttemptDisposition.VALID_OUTCOME
        and item.artifact.evidence.result.score_eligible
    )
    completed_tasks = {
        item.trial_spec.task
        for item in inventory.artifacts
        if item.disposition is CleanAttemptDisposition.VALID_OUTCOME
    }
    successes = sum(result.score_success is True for result in eligible)
    paper_ready = bool(blocking_gates) and all(
        gate.status is GateStatus.PASS for gate in blocking_gates
    )
    integration_ready = all(
        next(gate for gate in gates if gate.gate_id == gate_id).status
        is GateStatus.PASS
        for gate_id in (
            "artifact_inventory",
            "policy_family",
            "task_set",
            "transition_integrity",
        )
    )
    report: dict[str, object] = {
        "evaluation_semantics": reference.evaluation_semantics.value,
        "reference_id": reference.reference_id,
        "reference_sha256": reference.sha256,
        "reference_report_url": reference.report_url,
        "campaign_id": manifest.campaign_id,
        "campaign_manifest_sha256": manifest.sha256,
        "protocol_id": manifest.protocol_id.value,
        "planned_trial_count": manifest.planned_trial_count,
        "loaded_artifact_count": len(inventory.artifacts),
        "completed_task_count": len(completed_tasks),
        "eligible_trial_count": len(eligible),
        "success_count": successes,
        "observed_success_rate": None if not eligible else successes / len(eligible),
        "reported_macro_success_rate": reference.reported_macro_success_rate,
        "reported_rate_difference": (
            successes / len(eligible) - reference.reported_macro_success_rate
            if paper_ready and eligible
            else None
        ),
        "go_for_integration_diagnostic": integration_ready,
        "go_for_paper_comparison": paper_ready,
        "blocker_codes": list(dict.fromkeys(blockers)),
        "gates": [gate.to_dict() for gate in gates],
        "trials": [item.to_dict() for item in diagnostics],
        "claim_boundaries": [
            "The reported 84.5% is a public reference, not an acceptance threshold.",
            "Only paper_v1 with 100 valid outcomes per task is statistically comparable to the public reference.",
            "The action gate proves RoboTactile release/training-cadence alignment, not identity with the unpublished paper evaluator.",
            "FAIL denotes contradictory evidence; UNKNOWN denotes evidence that is absent or unpublished.",
            "Infrastructure failures are reported separately from model task failures.",
            "Reset-time success and early-stop flags are observability witnesses, not official outcome filters.",
            "Planning/execution flags are robust diagnostics and are non-blocking under official_reproduction.",
        ],
        "evidence_level": ALIGNMENT_EVIDENCE_LEVEL,
        "semantic_version": ALIGNMENT_REPORT_SEMANTIC_VERSION,
    }
    report["report_content_sha256"] = canonical_hash(report)
    return report


__all__ = ["build_n0_protocol_alignment_report"]
