"""Predeclared model-targeted stress protocol; never modifies benchmark scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from robotactile_benchmark.constants import (
    DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    OPTICAL_CONTACT_STRESS_REGISTRY_ID,
    SENSOR_FULLFRAME_STRESS_REGISTRY_ID,
)
from robotactile_benchmark.contracts import canonical_hash

PROTOCOL_SCHEMA = "n0_noise_stress_protocol_v2"
DROP_TARGET = 0.20


def calibration_selection_rule() -> dict[str, Any]:
    return {
        "schema": "n0_minimum_observed_dose_v1",
        "dose_order": [1, 3, 5],
        "minimum_drop_pp": 20,
        "statistic": "paired_clean_success_minus_fault_success",
        "selection": "first_qualifying_dose_per_family",
        "required_nonnull_families": 2,
        "no_qualifying_dose": "No-Go",
        "require_complete_fixed_cohort": True,
        "retain_all_doses": True,
    }


FAMILIES = {
    "fast_f1": (OPTICAL_CONTACT_STRESS_REGISTRY_ID, "F1_global_response_drift"),
    "contact_f3": (
        OPTICAL_CONTACT_STRESS_REGISTRY_ID,
        "F3_persistent_surface_artifact",
    ),
    "fullframe": (SENSOR_FULLFRAME_STRESS_REGISTRY_ID, "F4_local_nonresponsive_patch"),
    "delay_t1": (OPTICAL_CONTACT_STRESS_REGISTRY_ID, "T1_fixed_source_delay"),
    "skew_t3": (OPTICAL_CONTACT_STRESS_REGISTRY_ID, "T3_inter_sensor_skew"),
    "null": (DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID, "F1_global_response_drift"),
}


@dataclass(frozen=True)
class StressVariant:
    family: str
    level: int

    def __post_init__(self) -> None:
        if (
            self.family not in FAMILIES
            or type(self.level) is not int
            or self.level not in {1, 3, 5}
        ):
            raise ValueError("unknown stress family or dose")
        if self.family == "null" and self.level != 5:
            raise ValueError("null is a level-5 diagnostic, not a calibrated fault")

    @property
    def label(self) -> str:
        return f"{self.family}-s{self.level}"

    @property
    def registry(self) -> str:
        return FAMILIES[self.family][0]

    @property
    def operator(self) -> str:
        return FAMILIES[self.family][1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "level": self.level,
            "label": self.label,
            "registry": self.registry,
            "operator": self.operator,
        }


def freeze_protocol(
    *,
    stage: str,
    seeds: tuple[int, ...],
    variants: tuple[StressVariant, ...],
    excluded_seeds: tuple[int, ...],
    binding: dict[str, str],
    selection_evidence_sha256: str | None = None,
    spatial_calibration_sha256: str | None = None,
    group_paths: dict[str, str] | None = None,
    calibration_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze a complete cohort before execution; calibration is never a test claim."""
    if stage not in {"screening", "calibration", "confirmation"}:
        raise ValueError("unknown stage")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    if any(type(s) is not int or s < 0 for s in excluded_seeds) or set(seeds) & set(
        excluded_seeds
    ):
        raise ValueError("cohort overlaps excluded/development seeds")
    if not variants or len({v.label for v in variants}) != len(variants):
        raise ValueError("variants must be nonempty and unique")
    if stage == "calibration":
        families = {v.family for v in variants}
        if (
            len(families) != 2
            or "null" in families
            or {(v.family, v.level) for v in variants}
            != {(family, level) for family in families for level in (1, 3, 5)}
        ):
            raise ValueError(
                "calibration requires two nonnull families with all s1/s3/s5 doses"
            )
    required = {
        "dataset_sha256",
        "integration_config_sha256",
        "code_sha256",
        "model_sha256",
    }
    if set(binding) != required:
        raise ValueError("dataset/model/config/code bindings are required")
    for value in [
        *binding.values(),
        selection_evidence_sha256,
        spatial_calibration_sha256,
    ]:
        if value is not None and (
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("provenance must be lowercase SHA-256")
    if stage == "confirmation":
        if selection_evidence_sha256 is None or len(variants) != 2 or len(seeds) < 100:
            raise ValueError(
                "confirmation requires frozen calibration evidence, two faults, >=100 seeds"
            )
        if (
            any(v.family == "null" for v in variants)
            or len({v.family for v in variants}) != 2
        ):
            raise ValueError(
                "confirmation requires two different fault families; null is diagnostic only"
            )
        if calibration_evidence is None or group_paths is None:
            raise ValueError(
                "confirmation requires verified calibration evidence and fixed group paths"
            )
        _validate_calibration_evidence(
            calibration_evidence,
            seeds,
            variants,
            excluded_seeds,
            binding,
            selection_evidence_sha256,
            spatial_calibration_sha256,
        )
    elif calibration_evidence is not None:
        raise ValueError("calibration evidence is only attached to confirmation")
    if group_paths is not None:
        if set(group_paths) != {str(seed) for seed in seeds}:
            raise ValueError("fixed group paths must cover the complete seed cohort")
        paths = list(group_paths.values())
        if len(set(paths)) != len(paths) or any(
            not isinstance(path, str)
            or not PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or str(PurePosixPath(path)) != path
            for path in paths
        ):
            raise ValueError("group paths must be unique normalized absolute paths")
    if (
        any(v.family == "contact_f3" for v in variants)
        and spatial_calibration_sha256 is None
    ):
        raise ValueError("contact F3 requires a frozen calibration template")
    payload: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "task": "lift_bottle",
        "model": "N0-TWAM",
        "stage": stage,
        "seeds": list(seeds),
        "excluded_seeds": sorted(set(excluded_seeds)),
        "variants": [v.to_dict() for v in variants],
        "binding": dict(binding),
        "selection_evidence_sha256": selection_evidence_sha256,
        "spatial_calibration_sha256": spatial_calibration_sha256,
        "target_drop_absolute": DROP_TARGET,
        "target_units": "percentage_points",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260919,
        "confidence_level": 0.95,
        "familywise_alpha": 0.05,
        "multiplicity": "Holm_exact_McNemar_two_sided",
        "planned_live_rollouts": len(seeds) * (1 + len(variants)),
        "selection": "calibration_only; all trials retained; no outcome-driven reruns",
        "paired_order": "Clean first; fault order rotated deterministically by seed",
        "success_rule": "unchanged upstream task success predicate",
        "scope": "model-targeted diagnostic stress; not canonical benchmark severity",
    }
    # Optional fields preserve the hashes of already-running exploratory studies.
    if stage == "calibration":
        payload["selection_rule"] = calibration_selection_rule()
    if group_paths is not None:
        payload["group_paths"] = dict(group_paths)
    if calibration_evidence is not None:
        payload["calibration_evidence"] = calibration_evidence
    return {**payload, "protocol_sha256": canonical_hash(payload)}


def _validate_calibration_evidence(
    evidence: dict[str, Any],
    seeds: tuple[int, ...],
    variants: tuple[StressVariant, ...],
    excluded: tuple[int, ...],
    binding: dict[str, str],
    evidence_hash: str | None,
    spatial_hash: str | None,
) -> None:
    if (
        evidence.get("schema") != "n0_stress_selection_evidence_v1"
        or canonical_hash(evidence) != evidence_hash
    ):
        raise ValueError("selection evidence content/hash mismatch")
    parent = evidence["protocol"]
    if parent.get("stage") != "calibration":
        raise ValueError("selection requires a calibration-stage parent")
    parent = validate_protocol(parent)
    if not parent.get("group_paths") or len(parent["seeds"]) < 20:
        raise ValueError("selection requires >=20 fixed-path calibration seeds")
    development = set(parent["seeds"]) | set(parent["excluded_seeds"])
    if set(seeds) & development or not development <= set(excluded):
        raise ValueError(
            "confirmation overlaps or omits calibration/development exclusions"
        )
    if parent["binding"] != binding:
        raise ValueError("model/data/config/code must remain frozen after calibration")
    if spatial_hash != parent["spatial_calibration_sha256"]:
        raise ValueError("spatial template must remain frozen after calibration")
    if any(v.to_dict() not in parent["variants"] for v in variants):
        raise ValueError("confirmation selected an untested calibration variant")
    selection = select_calibration_doses(parent, evidence["report"])
    if selection["decision"] != "Go":
        raise ValueError(
            "calibration selection is No-Go: a family has no qualifying dose"
        )
    if {v.label for v in variants} != {
        v["label"] for v in selection["selected_variants"]
    }:
        raise ValueError("confirmation must use the frozen minimum qualifying doses")


def select_calibration_doses(
    protocol: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    """Derive selection from all paired counts, never caller-authored choices."""
    parent = validate_protocol(protocol)
    if parent["stage"] != "calibration" or not parent.get("group_paths"):
        raise ValueError("selection requires a fixed-path calibration protocol")
    if len(parent["seeds"]) < 20:
        raise ValueError("selection requires >=20 fixed-path calibration seeds")
    planned = parent["planned_live_rollouts"]
    required = {
        "schema": "n0_noise_stress_results_v2",
        "protocol_sha256": parent["protocol_sha256"],
        "stage": "calibration",
        "planned_live_rollouts": planned,
        "accepted_live_rollouts": planned,
        "eligible_live_attempt_rollouts": planned,
        "accepted_seed_groups": len(parent["seeds"]),
        "missing_accepted_cells": 0,
        "invalid_attempt_rollouts": 0,
        "unsupported_contract": 0,
        "provisional_live_attempt_rollouts": 0,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("selection requires complete validated calibration results")
    conditions = report["conditions"]
    if (
        {row["condition"] for row in conditions}
        != {v["label"] for v in parent["variants"]}
        or len(conditions) != len(parent["variants"])
        or any(row["paired_count"] != len(parent["seeds"]) for row in conditions)
    ):
        raise ValueError("calibration condition coverage mismatch")
    by_label = {row["condition"]: row for row in conditions}
    chosen = []
    family_results = []
    for family in sorted({v["family"] for v in parent["variants"]}):
        doses = []
        selected = None
        for level in parent["selection_rule"]["dose_order"]:
            variant = StressVariant(family, level)
            row = by_label[variant.label]
            n = row["paired_count"]
            clean, fault = row["clean_success"], row["fault_success"]
            losses = row["clean_success_fault_failure"]
            gains = row["clean_failure_fault_success"]
            if (
                any(
                    type(v) is not int or not 0 <= v <= n
                    for v in (clean, fault, losses, gains)
                )
                or losses + gains > n
                or clean - fault != losses - gains
                or losses > clean
                or gains > fault
                or losses > n - fault
                or gains > n - clean
            ):
                raise ValueError("invalid paired calibration counts")
            # Integer comparison avoids rounding away the exact 20pp boundary.
            qualifies = 100 * (losses - gains) >= 20 * n
            doses.append(
                {
                    "variant": variant.to_dict(),
                    "observed_drop_pp": 100 * (losses - gains) / n,
                    "qualifies": qualifies,
                }
            )
            if qualifies and selected is None:
                selected = variant.to_dict()
        family_results.append(
            {"family": family, "doses": doses, "selected_variant": selected}
        )
        if selected is not None:
            chosen.append(selected)
    go = len(chosen) == 2
    return {
        "decision": "Go" if go else "No-Go",
        "selected_variants": chosen if go else [],
        "families": family_results,
        "interpretation": "exploratory observed paired drop; not a confirmation claim",
    }


def validate_protocol(value: dict[str, Any]) -> dict[str, Any]:
    expected = freeze_protocol(
        stage=value["stage"],
        seeds=tuple(value["seeds"]),
        variants=tuple(
            StressVariant(v["family"], v["level"]) for v in value["variants"]
        ),
        excluded_seeds=tuple(value["excluded_seeds"]),
        binding=value["binding"],
        selection_evidence_sha256=value["selection_evidence_sha256"],
        spatial_calibration_sha256=value["spatial_calibration_sha256"],
        group_paths=value.get("group_paths"),
        calibration_evidence=value.get("calibration_evidence"),
    )
    if value != expected:
        raise ValueError("protocol differs from frozen contract/hash")
    return expected
