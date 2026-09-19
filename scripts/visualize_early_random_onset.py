"""Show seed-bound early fault onset on verified recorded UniVTAC frames.

This is a production-operator offline replay, not a closed-loop policy result.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from robotactile_benchmark.constants import operator_requires_rest_reference
from robotactile_benchmark.contracts import Array, EvaluationRecord
from robotactile_benchmark.fault_timing import derive_early_random_onset
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.runtime import ReplayResult, apply_fault
from robotactile_benchmark.visualization.optical14 import (
    build_records,
    file_hash,
    font,
    grid,
    load_source,
    tile,
)

REGISTRY = "optical_marker_extreme_v1"
OPERATORS = (
    "F3_persistent_surface_artifact",
    "F2_spatial_sensitivity_loss",
    "T2_held_last_freeze",
)
SEEDS = (0, 1, 3)
DISPLAY_SEED = 1


def _source_spec(root: Path, task: str) -> tuple[dict[str, Any], str]:
    receipt_path = root / "source_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("generated_content") is not False
        or receipt.get("fault_operator_applied") is not False
    ):
        raise ValueError("source receipt must attest untouched recorded RGB")
    matches = [item for item in receipt["sources"] if item["task"] == task]
    if len(matches) != 1:
        raise ValueError(f"expected one verified source for task {task}")
    return matches[0], file_hash(receipt_path)


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
    return FaultManifest(
        operator_id=operator,
        severity_level=5,
        operator_seed=20260906,
        start_index=onset,
        stop_index=stop,
        sensor_slots=("right",),
        observability=Observability.BLIND,
        parameters=parameters,
        severity_registry=REGISTRY,
    )


def _right_rgb(record: EvaluationRecord) -> Array:
    value = record.observation.sensor("right").payload
    if value is None:
        raise ValueError("selected operator unexpectedly removed right tactile RGB")
    return value


def _case_panel(
    clean: Sequence[EvaluationRecord],
    replay: ReplayResult,
    manifest: FaultManifest,
    peak: int,
) -> tuple[Image.Image, dict[str, Any]]:
    indices = (manifest.start_index - 1, manifest.start_index, peak)
    labels = ("before onset", "first active", "contact peak")
    tiles: list[Image.Image] = []
    frame_metrics: list[dict[str, Any]] = []
    for label, index in zip(labels, indices):
        original = _right_rgb(clean[index])
        delivered = _right_rgb(replay.records[index])
        absolute = np.abs(delivered.astype(np.int16) - original.astype(np.int16))
        difference = np.clip(absolute * 3, 0, 255).astype(np.uint8)
        mae = float(absolute.mean())
        source_index = replay.records[index].provenance_for("right").source_index
        tiles.extend(
            (
                tile(original, f"t={index} {label} | Clean"),
                tile(delivered, f"t={index} | Delivered; source={source_index}"),
                tile(difference, f"|Delta| x3 display | MAE={mae:.2f}/255"),
            )
        )
        frame_metrics.append(
            {
                "step_index": index,
                "mean_absolute_rgb_delta_u8": mae,
                "changed_rgb_values": int(np.count_nonzero(absolute)),
                "delivered_source_index": source_index,
            }
        )
    return (
        grid(
            tiles,
            3,
            f"{manifest.operator_id} | recorded UniVTAC raw55 | onset={manifest.start_index}",
        ),
        {"sampled_frames": frame_metrics},
    )


def _timeline(starts: dict[int, int], length: int, peak: int) -> Image.Image:
    width, height = 1170, 525
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    heading, text_font, small = font(25), font(17), font(14)
    ink, muted, fault, clean = "#1b2733", "#566577", "#d16823", "#426a88"
    left, right = 160, width - 70
    draw.text(
        (35, 20),
        "Seed-dependent fault onset on one recorded episode",
        font=heading,
        fill=ink,
    )
    draw.text(
        (35, 58),
        "Same clean raw55 sequence; each seed shares its onset across operators",
        font=text_font,
        fill=muted,
    )

    def x(value: int, horizon: int) -> int:
        return left + round((right - left) * value / horizon)

    full_y = (120, 165, 210)
    draw.text((35, 91), "Full episode (178 observations)", font=text_font, fill=ink)
    peak_x = x(peak, length - 1)
    draw.line((peak_x, 112, peak_x, 245), fill="#427a58", width=2)
    draw.text((peak_x - 65, 247), f"contact peak t={peak}", font=small, fill="#427a58")
    for (seed, onset), y in zip(starts.items(), full_y):
        draw.text((45, y - 10), f"seed {seed}", font=text_font, fill=ink)
        draw.line((x(0, length - 1), y, x(onset, length - 1), y), fill=clean, width=15)
        draw.line(
            (x(onset, length - 1), y, x(length - 1, length - 1), y),
            fill=fault,
            width=15,
        )
        draw.ellipse(
            (x(onset, length - 1) - 4, y - 4, x(onset, length - 1) + 4, y + 4), fill=ink
        )
    draw.text((left, 270), "t=0", font=small, fill=muted)
    draw.text((right - 55, 270), f"t={length - 1}", font=small, fill=muted)
    draw.line((left, 304, left + 30, 304), fill=clean, width=10)
    draw.text((left + 39, 294), "unmodified", font=small, fill=ink)
    draw.line((left + 215, 304, left + 245, 304), fill=fault, width=10)
    draw.text((left + 254, 294), "fault scheduled", font=small, fill=ink)

    draw.text(
        (35, 319),
        "Early-window zoom (observation index 0-12)",
        font=text_font,
        fill=ink,
    )
    for tick in (0, 2, 4, 6, 8, 10, 12):
        tick_x = x(tick, 12)
        draw.line((tick_x, 347, tick_x, 478), fill="#d6dde4", width=1)
        draw.text((tick_x - 7, 480), str(tick), font=small, fill=muted)
    for (seed, onset), y in zip(starts.items(), (365, 410, 455)):
        draw.text((45, y - 10), f"seed {seed}", font=text_font, fill=ink)
        draw.line((x(0, 12), y, x(onset, 12), y), fill=clean, width=15)
        draw.line((x(onset, 12), y, x(12, 12), y), fill=fault, width=15)
        draw.text((x(onset, 12) + 7, y - 20), f"start {onset}", font=small, fill=ink)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="pull_out_key")
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    spec, receipt_hash = _source_spec(source_root, args.task)
    arrays = load_source(source_root, spec)
    clean, rest = build_records(arrays, spec, receipt_hash)
    length, peak = len(clean), int(spec["contact_peak"])
    starts = {
        seed: derive_early_random_onset(task=args.task, seed=seed, stop=length)
        for seed in SEEDS
    }
    onset = starts[DISPLAY_SEED]
    sample_period_s = float(np.diff(arrays["source_time_s"])[0])
    results: dict[str, Any] = {}
    panels: dict[str, Image.Image] = {}
    for operator in OPERATORS:
        manifest = _manifest(operator, onset, length, sample_period_s, rest.sha256)
        replay = apply_fault(clean, manifest, rest)
        if not replay.validation.passed:
            raise ValueError(
                f"{operator} validation failed: {replay.validation.failure_codes}"
            )
        if any(
            not np.array_equal(
                _right_rgb(clean[index]), _right_rgb(replay.records[index])
            )
            for index in range(onset)
        ):
            raise ValueError(f"{operator} changed a pre-onset tactile frame")
        panels[operator], metrics = _case_panel(clean, replay, manifest, peak)
        results[operator] = {
            "manifest_sha256": manifest.sha256,
            "trace_sha256": replay.trace_sha256,
            "validation_passed": replay.validation.passed,
            "fault_start_index": manifest.start_index,
            "fault_stop_index": manifest.stop_index,
            "pre_onset_identical": True,
            **metrics,
        }
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    _timeline(starts, length, peak).save(output / "onset_timeline.png")
    for operator, panel in panels.items():
        panel.save(output / f"{operator}.png")
    report = {
        "evidence_level": "offline_recorded_univtac_production_operator_replay",
        "closed_loop_policy_run": False,
        "source_rgb_recorded": True,
        "fault_delivery_synthetic": True,
        "generative_model_used": False,
        "source_task": args.task,
        "source_raw_episode_id": spec["raw_episode_id"],
        "source_npz_sha256": spec["artifact_sha256"],
        "source_receipt_sha256": receipt_hash,
        "source_observation_count": length,
        "contact_peak_index": peak,
        "fault_onset_mode": "early_random_onset_v1",
        "fault_onset_indices": starts,
        "display_seed": DISPLAY_SEED,
        "severity_registry": REGISTRY,
        "operators": results,
    }
    (output / "replay_receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
