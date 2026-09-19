"""Render all fourteen production faults on one verified recorded UniVTAC source.

This is an offline operator replay for mechanism inspection, not a policy rollout.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    SENSOR_SLOTS,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import Array, EvaluationRecord, delivered_hash
from robotactile_benchmark.fault_timing import derive_early_random_onset
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.runtime import ReplayResult, apply_fault
from robotactile_benchmark.visualization.optical14 import (
    build_records,
    file_hash,
    grid,
    load_source,
    tile,
)
from scripts.visualize_early_random_onset import _source_spec, _timeline

REGISTRY = "optical_marker_extreme_v1"
ORDER = tuple(sorted(CORE_OPERATOR_IDS, key=lambda name: ("AFTC".index(name[0]), name)))
SEEDS = (0, 1, 3)
DISPLAY_SEED = 1


def _manifest(
    operator: str,
    onset: int,
    stop: int,
    sample_period_s: float,
    rest_sha256: str,
) -> FaultManifest:
    parameters: dict[str, object] = {"sample_period_s": sample_period_s}
    if operator_requires_rest_reference(operator, severity_registry=REGISTRY):
        parameters["rest_reference_sha256"] = rest_sha256
    if operator.startswith("T"):
        parameters["temporal_schedule"] = "window_to_end_v1"
    if operator.startswith("A2"):
        parameters["a2_end_policy"] = "episode_censored_v1"
    if operator.startswith("C2"):
        parameters["realization"] = "registered_pixels"
    slots = (
        SENSOR_SLOTS
        if operator.startswith(("T1", "T3", "C1"))
        else ("left",)
        if operator.startswith("F6")
        else ("right",)
    )
    return FaultManifest(
        operator_id=operator,
        severity_level=5,
        operator_seed=20260906,
        start_index=onset,
        stop_index=stop,
        sensor_slots=slots,
        observability=Observability.BLIND,
        parameters=parameters,
        severity_registry=REGISTRY,
    )


def _rgb(record: EvaluationRecord, slot: str) -> Array | None:
    return record.observation.sensor(slot).payload


def _witness(
    manifest: FaultManifest, peak: int, local_unloading_index: int
) -> tuple[int, int, str]:
    first, witness, meaning = manifest.start_index, peak, "clean contact peak"
    if manifest.operator_id.startswith("A1") or manifest.operator_id.startswith("C1"):
        first = manifest.start_index + int(manifest.parameters["affected_offsets"][0])
    if manifest.operator_id.startswith("A2"):
        first = manifest.start_index + int(manifest.parameters["erased_offsets"][0])
        erased = set(manifest.parameters["erased_offsets"])
        witness = next(
            index
            for index in range(manifest.start_index, manifest.stop_index)
            if index - manifest.start_index not in erased
        )
        meaning = "first retained frame inside erasure window"
    if manifest.operator_id.startswith("F1"):
        witness, meaning = manifest.stop_index - 1, "end of ramp"
    if manifest.operator_id.startswith("F6"):
        witness, meaning = local_unloading_index, "source-selected local unloading"
    if manifest.operator_id.startswith("T2"):
        witness, meaning = manifest.stop_index - 1, "last held observation"
    return first, witness, meaning


def _difference(
    clean: Array | None, delivered: Array | None
) -> tuple[Array | None, float | None]:
    if clean is None or delivered is None:
        return None, None
    absolute = np.abs(delivered.astype(np.int16) - clean.astype(np.int16))
    return np.clip(absolute * 3, 0, 255).astype(np.uint8), float(absolute.mean())


def _source_label(record: EvaluationRecord, slot: str) -> str:
    provenance = record.provenance_for(slot)
    index = provenance.source_index
    physical = provenance.physical_source_id
    return f"source={physical}:{index}" if index is not None else "source=ABSENT"


def _single_slot_panel(
    clean: Sequence[EvaluationRecord],
    replay: ReplayResult,
    manifest: FaultManifest,
    indices: tuple[int, int, int],
    slot: str,
) -> tuple[Image.Image, list[dict[str, object]]]:
    panels: list[Image.Image] = []
    metrics: list[dict[str, object]] = []
    for index, phase in zip(indices, ("before onset", "first scheduled", "witness")):
        original = _rgb(clean[index], slot)
        delivered = _rgb(replay.records[index], slot)
        difference, mae = _difference(original, delivered)
        panels.extend(
            (
                tile(original, f"{phase} t={index} | Clean {slot}"),
                tile(
                    delivered,
                    f"t={index} | {_source_label(replay.records[index], slot)}",
                ),
                tile(
                    difference,
                    "abs RGB delta x3" if mae is not None else "N/A: payload absent",
                ),
            )
        )
        metrics.append(
            {
                "step_index": index,
                "clean_payload_present": original is not None,
                "delivered_payload_present": delivered is not None,
                "delivered_source_index": replay.records[index]
                .provenance_for(slot)
                .source_index,
                "delivered_physical_source_id": replay.records[index]
                .provenance_for(slot)
                .physical_source_id,
                "mean_absolute_rgb_delta_u8": mae,
            }
        )
    heading = f"{manifest.operator_id} | onset={manifest.start_index} | {slot} tactile"
    return grid(panels, 3, heading), metrics


def _two_slot_panel(
    clean: Sequence[EvaluationRecord],
    replay: ReplayResult,
    manifest: FaultManifest,
    indices: tuple[int, int, int],
) -> tuple[Image.Image, list[dict[str, object]]]:
    panels: list[Image.Image] = []
    metrics: list[dict[str, object]] = []
    for index in indices:
        row: dict[str, object] = {"step_index": index}
        for slot in SENSOR_SLOTS:
            panels.append(tile(_rgb(clean[index], slot), f"t={index} Clean {slot}"))
            panels.append(
                tile(
                    _rgb(replay.records[index], slot),
                    f"t={index} Delivered {_source_label(replay.records[index], slot)}",
                )
            )
            _, mae = _difference(
                _rgb(clean[index], slot), _rgb(replay.records[index], slot)
            )
            row[f"{slot}_mean_absolute_rgb_delta_u8"] = mae
            row[f"{slot}_delivered_source_index"] = (
                replay.records[index].provenance_for(slot).source_index
            )
            row[f"{slot}_delivered_physical_source_id"] = (
                replay.records[index].provenance_for(slot).physical_source_id
            )
        metrics.append(row)
    heading = (
        f"{manifest.operator_id} | onset={manifest.start_index} | both tactile slots"
    )
    return grid(panels, 4, heading), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="pull_out_key")
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    spec, source_receipt_hash = _source_spec(source_root, args.task)
    source_receipt = json.loads(
        (source_root / "source_receipt.json").read_text(encoding="utf-8")
    )
    local_unloading = source_receipt["f6_local_unloading_witness"]
    if local_unloading["task"] != args.task:
        raise ValueError("F6 source-selected local unloading does not match task")
    local_unloading_index = int(local_unloading["indices"][-1])
    arrays = load_source(source_root, spec)
    clean, rest = build_records(arrays, spec, source_receipt_hash)
    length, peak = len(clean), int(spec["contact_peak"])
    starts = {
        seed: derive_early_random_onset(task=args.task, seed=seed, stop=length)
        for seed in SEEDS
    }
    onset = starts[DISPLAY_SEED]
    sample_period_s = float(np.diff(arrays["source_time_s"])[0])
    panels: dict[str, Image.Image] = {}
    manifests: dict[str, FaultManifest] = {}
    results: dict[str, dict[str, object]] = {}
    overview = [tile(_rgb(clean[peak], "right"), f"Clean right | peak t={peak}")]
    for operator in ORDER:
        manifest = _manifest(operator, onset, length, sample_period_s, rest.sha256)
        replay = apply_fault(clean, manifest, rest)
        if not replay.validation.passed:
            raise ValueError(f"{operator}: {replay.validation.failure_codes}")
        if any(
            delivered_hash(record.observation, record.provenance)
            != delivered_hash(
                replay.records[index].observation, replay.records[index].provenance
            )
            for index, record in enumerate(clean[:onset])
        ):
            raise ValueError(f"{operator} altered an observation before onset")
        first, witness, witness_meaning = _witness(
            manifest, peak, local_unloading_index
        )
        if not 0 <= first < length or not 0 <= witness < length:
            raise ValueError(f"{operator} witness lies outside source sequence")
        indices = (onset - 1, first, witness)
        if operator.startswith(("T1", "T3", "C1")):
            panels[operator], metrics = _two_slot_panel(
                clean, replay, manifest, indices
            )
        else:
            slot = manifest.sensor_slots[0]
            panels[operator], metrics = _single_slot_panel(
                clean, replay, manifest, indices, slot
            )
        overview_slot = "left" if operator.startswith("F6") else "right"
        overview.append(
            tile(
                _rgb(replay.records[witness], overview_slot),
                f"{operator.split('_')[0]} {overview_slot} | t={witness}",
            )
        )
        manifests[operator] = manifest
        results[operator] = {
            "manifest_sha256": manifest.sha256,
            "trace_sha256": replay.trace_sha256,
            "validation_passed": replay.validation.passed,
            "pre_onset_identical": True,
            "first_event_index": first,
            "witness_index": witness,
            "witness_selection": witness_meaning,
            "sampled_frames": metrics,
        }
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    _timeline(starts, length, peak).save(output / "onset_timeline.png")
    grid(
        overview,
        3,
        "All 14 production operators | preselected mechanism witnesses | recorded raw55",
    ).save(output / "all14_overview.png")
    manifest_dir = output / "fault_manifests"
    manifest_dir.mkdir()
    for operator, panel in panels.items():
        panel.save(output / f"{operator}.png")
        (manifest_dir / f"{operator}.json").write_text(
            json.dumps(manifests[operator].to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    members = {
        path.relative_to(output).as_posix(): file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    report = {
        "evidence_level": "offline_recorded_univtac_production_operator_replay",
        "closed_loop_policy_run": False,
        "source_rgb_recorded": True,
        "fault_delivery_synthetic": True,
        "generative_model_used": False,
        "source_task": args.task,
        "source_raw_episode_id": spec["raw_episode_id"],
        "source_npz_sha256": spec["artifact_sha256"],
        "source_receipt_sha256": source_receipt_hash,
        "source_observation_count": length,
        "source_observation_hz": round(1 / sample_period_s, 8),
        "contact_peak_index": peak,
        "fault_onset_mode": "early_random_onset_v1",
        "fault_onset_indices": starts,
        "display_seed": DISPLAY_SEED,
        "severity_registry": REGISTRY,
        "severity_level": 5,
        "operators": results,
        "members_sha256": members,
    }
    (output / "replay_receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
