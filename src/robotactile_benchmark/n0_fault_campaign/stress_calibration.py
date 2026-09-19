"""Fit a frozen contact-response mask from predeclared Clean development data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
from robotactile_benchmark.contracts import ContactPhase, array_sha256, canonical_hash
from robotactile_benchmark.execution.live_artifacts import load_live_univtac_artifact
from robotactile_benchmark.optical.stress_templates import calibrate_contact_scars
from robotactile_benchmark.trials import Condition

from .stress_metrics import eligible_terminal


def calibrate_clean_artifacts(
    paths: tuple[Path, ...], *, allowed_seeds: tuple[int, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not paths or len(set(paths)) != len(paths) or not allowed_seeds:
        raise ValueError(
            "unique Clean sources and explicit development seeds are required"
        )
    maps: dict[str, Any] = {}
    counts = {slot: 0 for slot in ("left", "right")}
    sources = []
    for path in paths:
        artifact = load_live_univtac_artifact(path)
        if (
            artifact.trial.condition is not Condition.CLEAN
            or artifact.trial.task != "lift_bottle"
            or artifact.trial.initial_seed not in allowed_seeds
        ):
            raise ValueError(
                "calibration source is not declared lift_bottle Clean development data"
            )
        if not eligible_terminal(result_to_dict(artifact.evidence.result)):
            raise ValueError("calibration source is not a valid completed live rollout")
        if artifact.evidence.finalization is not None:
            records = artifact.evidence.finalization.clean_records
        elif artifact.preview_trace is not None:
            records = artifact.preview_trace.clean_records
        else:
            raise ValueError("metrics-only source lacks contact-response pixels")
        references: dict[str, Any] = {}
        used: dict[str, list[int]] = {slot: [] for slot in counts}
        for slot in counts:
            free = [
                r for r in records if r.provenance_for(slot).phase is ContactPhase.FREE
            ]
            if not free or free[0].observation.sensor(slot).payload is None:
                raise ValueError("source lacks a verified free-contact reference")
            references[slot] = np.asarray(
                free[0].observation.sensor(slot).payload, dtype=np.float64
            )
        for record in records:
            for slot in counts:
                if record.provenance_for(slot).phase not in {
                    ContactPhase.ONSET,
                    ContactPhase.SUSTAINED,
                }:
                    continue
                payload = record.observation.sensor(slot).payload
                if payload is None:
                    raise ValueError("Clean contact frame has no tactile pixels")
                response = np.abs(
                    np.asarray(payload, dtype=np.float64) - references[slot]
                ).mean(axis=2)
                if slot not in maps:
                    maps[slot] = np.zeros_like(response)
                if maps[slot].shape != response.shape:
                    raise ValueError("calibration tactile shapes differ")
                maps[slot] += response
                counts[slot] += 1
                used[slot].append(record.observation.step_index)
        sources.append(
            {
                "artifact_root_sha256": artifact.external_root_sha256,
                "seed": artifact.trial.initial_seed,
                "capture_profile": artifact.capture_profile.value,
                "contact_source_indices": used,
            }
        )
    if any(n == 0 for n in counts.values()):
        raise ValueError("calibration has no contact response in one sensor")
    receipt = {
        "schema": "n0_contact_response_calibration_v1",
        "sources": sources,
        "allowed_development_seeds": list(allowed_seeds),
        "contact_frame_counts": counts,
        "estimator": "mean absolute RGB difference from first verified free frame; contact-only sum",
        "scope": "pixel-response proxy on frozen development subset; not held-out coverage guarantee",
        "response_map_sha256": {slot: array_sha256(maps[slot]) for slot in counts},
    }
    return calibrate_contact_scars(maps, source_sha256=canonical_hash(receipt)), receipt
