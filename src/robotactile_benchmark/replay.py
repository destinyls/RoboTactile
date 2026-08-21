"""Frozen offline replay artifact writer and loader."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np

from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.fixtures import (
    make_synthetic_episode,
    make_synthetic_rest_references,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operators import list_operator_ids
from robotactile_benchmark.runtime import ReplayResult, apply_fault
from robotactile_benchmark.severity import severity_value

MAX_REPLAY_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_REPLAY_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_REPLAY_COMPRESSION_RATIO = 200.0
EXPECTED_REPLAY_MEMBERS = {
    "tactile_rgb.npy",
    "payload_present.npy",
    "source_index.npy",
}


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_npz_container(path: Path) -> None:
    """Reject oversized or structurally surprising NPZ files before extraction."""

    if path.stat().st_size > MAX_REPLAY_ARTIFACT_BYTES:
        raise ValueError("stored episode artifact exceeds the replay size limit")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if {member.filename for member in members} != EXPECTED_REPLAY_MEMBERS:
            raise ValueError("stored replay contains an unexpected NPZ member set")
        uncompressed = sum(member.file_size for member in members)
        compressed = max(1, sum(member.compress_size for member in members))
        if uncompressed > MAX_REPLAY_UNCOMPRESSED_BYTES:
            raise ValueError("stored replay exceeds the uncompressed size limit")
        if uncompressed / compressed > MAX_REPLAY_COMPRESSION_RATIO:
            raise ValueError("stored replay exceeds the compression-ratio limit")


def run_smoke_replay(output_dir: Path, manifest: FaultManifest) -> ReplayResult:
    """Replay a deterministic fixture and write content-addressed metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    clean = make_synthetic_episode(length=max(12, manifest.stop_index + 2))
    rest_references = make_synthetic_rest_references()
    result = apply_fault(clean, manifest, rest_references=rest_references)
    if not result.validation.passed:
        raise RuntimeError(f"delivery validation failed: {result.validation.failures}")
    _write_json(output_dir / "fault_manifest.json", manifest.to_dict())
    payloads = []
    present = []
    source_indices = []
    for record in result.records:
        row = []
        row_present = []
        row_sources = []
        for slot in ("left", "right"):
            sensor = record.observation.sensor(slot)
            row.append(
                sensor.payload
                if sensor.payload is not None
                else np.zeros_like(clean[0].observation.sensor(slot).payload)
            )
            row_present.append(sensor.payload_present)
            source = record.provenance_for(slot).source_index
            row_sources.append(-1 if source is None else source)
        payloads.append(row)
        present.append(row_present)
        source_indices.append(row_sources)
    episode_path = output_dir / "delivered_episode.npz"
    np.savez_compressed(
        episode_path,
        tactile_rgb=np.asarray(payloads, dtype=np.uint8),
        payload_present=np.asarray(present, dtype=np.bool_),
        source_index=np.asarray(source_indices, dtype=np.int64),
    )
    _write_json(
        output_dir / "validation_report.json",
        {
            "passed": result.validation.passed,
            "failure_codes": list(result.validation.failure_codes),
            "failures": list(result.validation.failures),
            "metrics": dict(result.validation.metrics),
            "manifest_sha256": manifest.sha256,
            "trace_sha256": result.trace_sha256,
            "episode_artifact_sha256": _file_sha256(episode_path),
        },
    )
    return result


def load_replay_artifacts(output_dir: Path) -> Dict[str, Any]:
    """Reconstruct and validate a bounded synthetic smoke replay release."""

    report = json.loads(
        (output_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    if not report.get("passed", False):
        raise ValueError("stored replay did not pass delivery validation")
    manifest_value = json.loads(
        (output_dir / "fault_manifest.json").read_text(encoding="utf-8")
    )
    if canonical_hash(manifest_value) != report.get("manifest_sha256"):
        raise ValueError("stored fault manifest hash does not match the report")
    manifest = FaultManifest.from_dict(manifest_value)
    if manifest.to_dict() != manifest_value:
        raise ValueError("stored fault manifest is not canonical")
    if manifest.sha256 != report.get("manifest_sha256"):
        raise ValueError("stored fault manifest hash does not match the report")
    episode_path = output_dir / "delivered_episode.npz"
    _validate_npz_container(episode_path)
    if _file_sha256(episode_path) != report.get("episode_artifact_sha256"):
        raise ValueError("stored episode artifact hash does not match the report")
    with np.load(episode_path, allow_pickle=False) as episode:
        if set(episode.files) != {"tactile_rgb", "payload_present", "source_index"}:
            raise ValueError("stored replay contains an unexpected NPZ key set")
        tactile_rgb = episode["tactile_rgb"]
        payload_present = episode["payload_present"]
        source_index = episode["source_index"]
        if tactile_rgb.ndim != 5 or tactile_rgb.shape[1] != 2:
            raise ValueError("stored replay tactile array must be [T,S,H,W,C]")
        if tactile_rgb.shape[-1] != 3 or tactile_rgb.dtype != np.uint8:
            raise ValueError("stored replay tactile RGB must be uint8 HWC")
        expected_metadata_shape = tactile_rgb.shape[:2]
        if (
            payload_present.shape != expected_metadata_shape
            or source_index.shape != expected_metadata_shape
        ):
            raise ValueError("stored replay metadata shapes do not match")
        if payload_present.dtype != np.bool_ or source_index.dtype != np.int64:
            raise ValueError("stored replay metadata dtypes do not match the contract")
        stored_rgb = tactile_rgb.copy()
        stored_present = payload_present.copy()
        stored_sources = source_index.copy()

    clean = make_synthetic_episode(length=max(12, manifest.stop_index + 2))
    expected = apply_fault(
        clean,
        manifest,
        rest_references=make_synthetic_rest_references(),
    )
    expected_rgb = []
    expected_present = []
    expected_sources = []
    for record in expected.records:
        rgb_row = []
        present_row = []
        source_row = []
        for slot in ("left", "right"):
            sensor = record.observation.sensor(slot)
            rgb_row.append(
                sensor.payload
                if sensor.payload is not None
                else np.zeros_like(clean[0].observation.sensor(slot).payload)
            )
            present_row.append(sensor.payload_present)
            source = record.provenance_for(slot).source_index
            source_row.append(-1 if source is None else source)
        expected_rgb.append(rgb_row)
        expected_present.append(present_row)
        expected_sources.append(source_row)
    if not np.array_equal(stored_rgb, np.asarray(expected_rgb, dtype=np.uint8)):
        raise ValueError(
            "stored replay tactile payloads fail deterministic reconstruction"
        )
    if not np.array_equal(stored_present, np.asarray(expected_present, dtype=np.bool_)):
        raise ValueError(
            "stored replay presence mask fails deterministic reconstruction"
        )
    if not np.array_equal(stored_sources, np.asarray(expected_sources, dtype=np.int64)):
        raise ValueError("stored replay source map fails deterministic reconstruction")
    expected_report = {
        "passed": expected.validation.passed,
        "failure_codes": list(expected.validation.failure_codes),
        "failures": list(expected.validation.failures),
        "metrics": dict(expected.validation.metrics),
        "manifest_sha256": manifest.sha256,
        "trace_sha256": expected.trace_sha256,
        "episode_artifact_sha256": _file_sha256(episode_path),
    }
    if report != expected_report:
        raise ValueError("stored validation report fails deterministic reconstruction")
    return cast(Dict[str, Any], report)


def _smoke_manifest(operator_id: str, severity_level: int) -> FaultManifest:
    start_index = 17 if operator_id == "T1_fixed_source_delay" else 4
    slots = (
        ("left", "right")
        if operator_id in {"T3_inter_sensor_skew", "C1_sensor_identity_misrouting"}
        else ("left",)
    )
    parameters: Dict[str, Any] = {}
    rest_references = make_synthetic_rest_references()
    if operator_id in REST_REFERENCE_OPERATOR_IDS:
        parameters["rest_reference_sha256"] = rest_references.sha256
    if operator_id == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    return FaultManifest(
        operator_id=operator_id,
        severity_level=severity_level,
        operator_seed=20260814,
        start_index=start_index,
        stop_index=22,
        sensor_slots=slots,
        observability=Observability.BLIND,
        parameters=parameters,
    )


def run_smoke_matrix(output_dir: Path) -> Dict[str, Any]:
    """Validate the complete 14-by-5 synthetic operator matrix."""

    output_dir.mkdir(parents=True, exist_ok=True)
    clean = make_synthetic_episode(length=24)
    rest_references = make_synthetic_rest_references()
    cells = []
    for operator_id in list_operator_ids():
        for severity_level in range(1, 6):
            manifest = _smoke_manifest(operator_id, severity_level)
            result = apply_fault(clean, manifest, rest_references=rest_references)
            if not result.validation.passed:
                raise RuntimeError(
                    f"{operator_id} severity {severity_level} failed: {result.validation.failures}"
                )
            cells.append(
                {
                    "operator_id": operator_id,
                    "severity_level": severity_level,
                    "native_dose": severity_value(operator_id, severity_level),
                    "manifest_sha256": manifest.sha256,
                    "trace_sha256": result.trace_sha256,
                    "validation_passed": True,
                    "validation_metrics": dict(result.validation.metrics),
                }
            )
    summary = {
        "matrix_id": "synthetic_core_14x5_v1",
        "evidence_level": "bounded_synthetic_smoke",
        "cell_count": len(cells),
        "cells": cells,
    }
    _write_json(output_dir / "matrix_summary.json", summary)
    return summary
