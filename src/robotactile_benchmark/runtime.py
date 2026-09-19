"""Deterministic operator execution and trace hashing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Tuple

from robotactile_benchmark.constants import (
    OPTICAL_MARKER_REGISTRY_IDS,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import (
    EvaluationRecord,
    canonical_hash,
    delivered_hash,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operators import get_operator
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.validators import ValidationReport, validate_delivery


@dataclass(frozen=True)
class ReplayResult:
    """Delivered episode, validator report, and content-addressed trace."""

    records: Tuple[EvaluationRecord, ...]
    validation: ValidationReport
    trace_sha256: str


def trace_hash(records: Sequence[EvaluationRecord], manifest: FaultManifest) -> str:
    return canonical_hash(
        {
            "manifest_sha256": manifest.sha256,
            "records": [
                delivered_hash(record.observation, record.provenance)
                for record in records
            ],
        }
    )


def apply_fault(
    clean_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    rest_references: Optional[RestReferenceBundle] = None,
) -> ReplayResult:
    """Apply one atomic registered fault and validate the delivery."""

    clean_records = tuple(clean_records)
    if manifest.stop_index > len(clean_records):
        raise ValueError("fault window exceeds the episode length")
    if operator_requires_rest_reference(
        manifest.operator_id,
        severity_registry=manifest.severity_registry,
    ):
        if rest_references is None:
            raise ValueError("this operator requires a frozen rest-reference bundle")
        if manifest.parameters["rest_reference_sha256"] != rest_references.sha256:
            raise ValueError("rest-reference bundle does not match the fault manifest")
        for slot_id in manifest.sensor_slots:
            calibration_hashes = {
                record.provenance_for(slot_id).calibration_sha256
                for record in clean_records
            }
            if calibration_hashes != {rest_references.calibration_sha256[slot_id]}:
                raise ValueError(
                    "rest-reference calibration does not match the clean episode"
                )
    if manifest.operator_id == "A2_frame_erasure":
        last_erasure = manifest.start_index + int(
            manifest.parameters["erased_offsets"][-1]
        )
        if (
            last_erasure >= len(clean_records) - 1
            and manifest.parameters.get("a2_end_policy") != "episode_censored_v1"
        ):
            raise ValueError("A2 requires a later clean payload to resume")
    if manifest.severity_registry in OPTICAL_MARKER_REGISTRY_IDS:
        from robotactile_benchmark.streaming.session import StreamingFaultSession

        session = StreamingFaultSession(manifest, rest_references)
        records = tuple(session.deliver_one(record) for record in clean_records)
    else:
        operator = get_operator(manifest.operator_id)
        records = operator.apply(clean_records, manifest, rest_references)
    if manifest.parameters.get("a2_end_policy") == "episode_censored_v1":
        from robotactile_benchmark.closed_loop.validation import (
            validate_online_delivery,
        )

        validation = validate_online_delivery(
            clean_records, records, manifest, rest_references
        )
    else:
        validation = validate_delivery(
            clean_records, records, manifest, rest_references=rest_references
        )
    return ReplayResult(records, validation, trace_hash(records, manifest))
