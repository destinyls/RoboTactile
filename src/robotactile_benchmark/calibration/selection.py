"""Deterministic no-contact selection from strict clean live artifacts."""

from __future__ import annotations

from typing import Sequence, Tuple

from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.calibration.contracts import (
    NO_CONTACT_PREDICATE_ID,
    NoContactValidationReceipt,
    RestReferenceArtifactError,
)
from robotactile_benchmark.constants import SENSOR_SLOTS
from robotactile_benchmark.contracts import (
    ContactPhase,
    EvaluationRecord,
    array_sha256,
    canonical_hash,
    delivered_hash,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.rest_references import ReferenceSplit, RestReferenceBundle
from robotactile_benchmark.trials import Condition


def _validate_clean_record(
    record: EvaluationRecord,
    artifact: LoadedLiveUniVTACArtifact,
    expected_calibration_sha256: str,
) -> None:
    observation = record.observation
    if (
        observation.task != artifact.trial.task
        or observation.seed != artifact.trial.initial_seed
        or observation.step_index < 0
    ):
        raise RestReferenceArtifactError("clean record identity mismatch")
    if record.clean_record_sha256 != record.delivered_record_sha256 or (
        delivered_hash(observation, record.provenance) != record.clean_record_sha256
    ):
        raise RestReferenceArtifactError("source clean record hash mismatch")
    for slot in SENSOR_SLOTS:
        sensor = observation.sensor(slot)
        provenance = record.provenance_for(slot)
        if (
            sensor.payload is None
            or not sensor.payload_present
            or provenance.payload_sha256 != array_sha256(sensor.payload)
        ):
            raise RestReferenceArtifactError("calibration requires present tactile RGB")
        if provenance.active_fault_ids:
            raise RestReferenceArtifactError(
                "calibration record contains active faults"
            )
        if (
            sensor.delivery_index != observation.step_index
            or provenance.source_index != observation.step_index
        ):
            raise RestReferenceArtifactError("calibration record is not current-source")
        if provenance.calibration_sha256 != expected_calibration_sha256:
            raise RestReferenceArtifactError("sensor calibration differs from registry")


def _longest_free_run(records: Sequence[EvaluationRecord]) -> Tuple[int, int]:
    best = (0, 0)
    current_start = 0
    for position, record in enumerate(records):
        free = all(
            record.provenance_for(slot).phase is ContactPhase.FREE
            for slot in SENSOR_SLOTS
        )
        if free:
            if position == 0 or not all(
                records[position - 1].provenance_for(slot).phase is ContactPhase.FREE
                for slot in SENSOR_SLOTS
            ):
                current_start = position
            candidate = (current_start, position + 1)
            if candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
    return best


def build_rest_reference_from_live_artifact(
    artifact: LoadedLiveUniVTACArtifact,
    dataset_split: ReferenceSplit,
    minimum_consecutive_free_records: int = 5,
) -> tuple[RestReferenceBundle, NoContactValidationReceipt]:
    """Select and freeze one rest frame from a strict clean live artifact."""

    if not isinstance(artifact, LoadedLiveUniVTACArtifact):
        raise TypeError("artifact must be a strict LoadedLiveUniVTACArtifact")
    if not isinstance(dataset_split, ReferenceSplit):
        dataset_split = ReferenceSplit(dataset_split)
    if dataset_split is ReferenceSplit.SYNTHETIC_DEVELOPMENT:
        raise RestReferenceArtifactError(
            "production calibration cannot use synthetic-development"
        )
    if (
        artifact.trial.condition is not Condition.CLEAN
        or artifact.fault_manifest is not None
        or artifact.rest_references is not None
    ):
        raise RestReferenceArtifactError("calibration source must be a clean run")
    finalization = artifact.evidence.finalization
    if finalization is None or finalization.validation is not None:
        raise RestReferenceArtifactError("clean calibration trace is unavailable")
    records = finalization.clean_records
    if not records:
        raise RestReferenceArtifactError("clean calibration trace is empty")
    config = build_univtac_backend_config(artifact.trial.task)
    expected_calibration = config.aliases.calibration_config_sha256
    episode_ids = {record.observation.episode_id for record in records}
    if len(episode_ids) != 1:
        raise RestReferenceArtifactError("calibration trace spans multiple episodes")
    for expected_step, record in enumerate(records):
        if record.observation.step_index != expected_step:
            raise RestReferenceArtifactError(
                "calibration trace must be dense from zero"
            )
        _validate_clean_record(record, artifact, expected_calibration)
    start, stop = _longest_free_run(records)
    if isinstance(minimum_consecutive_free_records, bool) or not isinstance(
        minimum_consecutive_free_records, int
    ):
        raise TypeError("minimum_consecutive_free_records must be an integer")
    if minimum_consecutive_free_records < 1 or (
        stop - start < minimum_consecutive_free_records
    ):
        raise RestReferenceArtifactError(
            "no consecutive FREE interval satisfies the requested minimum"
        )
    selected_position = start + (stop - start - 1) // 2
    selected = records[selected_position]
    selected_step = selected.observation.step_index
    calibration = {
        slot: selected.provenance_for(slot).calibration_sha256 for slot in SENSOR_SLOTS
    }
    payload_hashes = {
        slot: array_sha256(selected.observation.sensor(slot).payload)
        for slot in SENSOR_SLOTS
    }
    if any(value is None for value in payload_hashes.values()):
        raise RestReferenceArtifactError("selected payload unexpectedly absent")
    record_ids = {
        slot: canonical_hash(
            {
                "namespace": "robotactile.rest-reference-record.v1",
                "source_live_artifact_root_sha256": artifact.root_receipt_sha256,
                "clean_record_sha256": canonical_hash(selected),
                "slot_id": slot,
                "source_index": selected.provenance_for(slot).source_index,
                "payload_sha256": payload_hashes[slot],
            }
        )
        for slot in SENSOR_SLOTS
    }
    validation = NoContactValidationReceipt(
        source_live_artifact_root_sha256=artifact.root_receipt_sha256,
        source_trial_manifest_sha256=artifact.trial.sha256,
        dataset_split=dataset_split,
        split_manifest_sha256=artifact.trial.dataset_sha256,
        task=artifact.trial.task,
        episode_id=next(iter(episode_ids)),
        initial_seed=artifact.trial.initial_seed,
        no_contact_predicate_id=NO_CONTACT_PREDICATE_ID,
        phase_tracker_sha256=canonical_hash(config.phase_tracker),
        minimum_consecutive_free_records=minimum_consecutive_free_records,
        qualified_start_index=records[start].observation.step_index,
        qualified_stop_index=records[stop - 1].observation.step_index + 1,
        selected_step_index=selected_step,
        qualified_clean_record_sha256s=tuple(
            canonical_hash(record) for record in records[start:stop]
        ),
        qualified_record_ids=record_ids,
        calibration_sha256=calibration,
        selected_payload_sha256={
            slot: str(payload_hashes[slot]) for slot in SENSOR_SLOTS
        },
    )
    references = RestReferenceBundle(
        reference_id=(
            f"rest-{artifact.trial.task}-{artifact.root_receipt_sha256[:12]}-"
            f"{selected_step:06d}"
        ),
        dataset_split=dataset_split,
        split_manifest_sha256=artifact.trial.dataset_sha256,
        source_artifact_sha256=artifact.root_receipt_sha256,
        no_contact_predicate_id=NO_CONTACT_PREDICATE_ID,
        no_contact_validation_sha256=validation.sha256,
        no_contact_verified=True,
        qualified_record_ids=record_ids,
        calibration_sha256=calibration,
        payloads={
            slot: selected.observation.sensor(slot).payload for slot in SENSOR_SLOTS
        },
    )
    return references, validation
