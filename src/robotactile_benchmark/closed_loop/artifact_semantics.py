"""Cross-object chronology and sensor/provenance checks for artifact evidence."""

from __future__ import annotations

from collections.abc import Sequence

from robotactile_benchmark.closed_loop.artifact_contracts import (
    ArtifactValidationError,
)
from robotactile_benchmark.closed_loop.capture import ActionTraceEntry
from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.closed_loop.runner_checks import build_episode_context
from robotactile_benchmark.contracts import EvaluationRecord, array_sha256
from robotactile_benchmark.trials import TrialManifest


def validate_trial_execution_semantics(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    result: ClosedLoopTrialResult,
    finalization: DeliveryFinalization,
    action_entries: Sequence[ActionTraceEntry],
) -> None:
    """Reject self-consistent hashes that violate runner-owned semantics."""

    clean_records = finalization.clean_records
    delivered_records = finalization.delivered_records
    expected_episode_id = build_episode_context(trial, run_spec.prompt).episode_id
    for records, clean in ((clean_records, True), (delivered_records, False)):
        for expected_step, record in enumerate(records):
            _validate_record(
                record,
                expected_step=expected_step,
                expected_episode_id=expected_episode_id,
                expected_task=trial.task,
                expected_seed=trial.initial_seed,
                require_fault_free=clean,
            )

    entries = tuple(action_entries)
    if (
        result.control_cycle_count > run_spec.max_control_cycles
        or result.observation_count > run_spec.max_observation_steps
        or result.control_cycle_count != len(entries)
        or result.observation_count != len(clean_records)
        or len(clean_records) != len(delivered_records)
    ):
        raise ArtifactValidationError("result counts violate the run specification")
    cumulative_executed = 0
    for entry in entries:
        executed_count = entry.executed_actions.shape[0]
        if executed_count > run_spec.execute_action_steps:
            raise ArtifactValidationError("executed action prefix exceeds the run spec")
        if cumulative_executed >= len(delivered_records):
            raise ArtifactValidationError("action source has no preceding observation")
        expected_source_step = delivered_records[
            cumulative_executed
        ].observation.step_index
        if entry.source_step_index != expected_source_step:
            raise ArtifactValidationError(
                "action source step violates runner chronology"
            )
        cumulative_executed += executed_count
    if result.observation_count != cumulative_executed + 1:
        raise ArtifactValidationError("executed prefixes do not explain observations")


def _validate_record(
    record: EvaluationRecord,
    *,
    expected_step: int,
    expected_episode_id: str,
    expected_task: str,
    expected_seed: int,
    require_fault_free: bool,
) -> None:
    observation = record.observation
    if (
        observation.step_index != expected_step
        or observation.episode_id != expected_episode_id
        or observation.task != expected_task
        or observation.seed != expected_seed
    ):
        raise ArtifactValidationError(
            "observation identity or step chronology mismatch"
        )
    sensor_slots = tuple(sensor.slot_id for sensor in observation.tactile)
    provenance_slots = tuple(item.slot_id for item in record.provenance)
    if sensor_slots != provenance_slots:
        raise ArtifactValidationError("sensor and provenance slot ordering mismatch")
    for sensor, provenance in zip(observation.tactile, record.provenance):
        if sensor.delivery_index != expected_step:
            raise ArtifactValidationError("sensor delivery indices must be dense")
        if require_fault_free and provenance.active_fault_ids:
            raise ArtifactValidationError(
                "clean trace contains active fault provenance"
            )
        if sensor.payload_present:
            if sensor.payload is None or provenance.payload_sha256 != array_sha256(
                sensor.payload
            ):
                raise ArtifactValidationError(
                    "present payload disagrees with evaluator provenance"
                )
        elif sensor.payload is not None or provenance.payload_sha256 is not None:
            raise ArtifactValidationError(
                "absent payload retains payload bytes or provenance hash"
            )
