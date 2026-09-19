"""Validated source prefixes and actual tensor differences for offline replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robotactile_benchmark.closed_loop.artifact_values import result_to_dict
from robotactile_benchmark.contracts import Array, EvaluationRecord
from robotactile_benchmark.execution.live_artifacts import load_live_univtac_artifact
from robotactile_benchmark.execution.loading import load_live_univtac_run
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.trials import Condition

from .io import file_sha256
from .stress_group import load_stress_group
from .stress_metrics import eligible_terminal


def load_replay_inputs(
    group: Path,
    source: Path,
    target: int,
) -> tuple[
    dict[str, Any], dict[str, Any], tuple[EvaluationRecord, ...], dict[str, Any]
]:
    plan, manifest, requests = load_stress_group(group)
    artifact = load_live_univtac_artifact(source)
    trial = artifact.trial
    if (
        trial.condition is not Condition.CLEAN
        or trial.task != plan["task"]
        or trial.initial_seed not in plan["excluded_seeds"]
        or trial.checkpoint_sha256 != plan["binding"]["model_sha256"]
        or trial.dataset_sha256 != plan["binding"]["dataset_sha256"]
    ):
        raise ValueError(
            "replay requires declared development Clean data with matching task/model/data"
        )
    if not eligible_terminal(result_to_dict(artifact.evidence.result)):
        raise ValueError(
            "replay source must be a valid terminal; task failures remain eligible"
        )
    finalization = artifact.evidence.finalization
    if finalization is None or len(finalization.clean_records) <= target:
        raise ValueError("replay requires a full captured prefix, not sparse previews")
    clean = tuple(finalization.clean_records[: target + 1])
    variants, contracts = {}, {}
    for selected, request in zip(manifest["selected"], requests):
        if selected["condition"] == "clean":
            continue
        loaded = load_live_univtac_run(request)
        original = loaded.fault_manifest
        if original is None or original.start_index > target:
            raise ValueError("diagnostic target is before the declared fault onset")
        # Re-materialize only the prefix horizon; all doses/onsets remain frozen.
        projected = original.reparameterized(stop_index=target + 1)
        replay = apply_fault(clean, projected, loaded.rest_references)
        if not replay.validation.passed:
            raise ValueError(f"invalid offline delivery: {selected['condition']}")
        variants[selected["condition"]] = replay.records
        contracts[selected["condition"]] = {
            "original_manifest_sha256": original.sha256,
            "prefix_manifest": projected.to_dict(),
            "prefix_manifest_sha256": projected.sha256,
            "delivery_trace_sha256": replay.trace_sha256,
            "delivery_validation_passed": True,
        }
    provenance = {
        "source_path": str(source.resolve()),
        "source_sha256": artifact.external_root_sha256,
        "source_seed": trial.initial_seed,
        "model_seed": trial.exogenous_seed,
        "source_status": artifact.evidence.result.terminal_status.value,
        "source_selected_independent_of_success": True,
        "group_path": str(group.resolve()),
        "group_sha256": manifest["group_sha256"],
        "contracts": contracts,
    }
    return plan, provenance, clean, variants


def _array(record_path: Path, summary: dict[str, Any]) -> Array:
    descriptor = summary.get("array_file")
    if not isinstance(descriptor, dict):
        raise ValueError("sampled tensor array was not captured")
    relative = Path(descriptor["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("probe array escapes its trace directory")
    path = record_path.parent / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError("probe array is missing or linked")
    path.resolve().relative_to(record_path.parent.resolve())
    if file_sha256(path) != descriptor["file_sha256"]:
        raise ValueError("probe array hash mismatch")
    value = np.load(path, allow_pickle=False)
    if (
        list(value.shape) != descriptor["shape"]
        or value.dtype.str != descriptor["dtype"]
    ):
        raise ValueError("probe array shape/dtype mismatch")
    if not np.isfinite(value).all():
        raise ValueError("probe tensor has nonfinite values")
    return value


def _difference(a: Array, b: Array) -> dict[str, Any]:
    if a.shape != b.shape:
        raise ValueError("compared probe tensor shapes differ")
    left, right = a.astype(np.float64), b.astype(np.float64)
    delta = float(np.linalg.norm(left - right))
    return {
        "l2_difference": delta,
        "relative_l2": delta / max(float(np.linalg.norm(left)), 1e-12),
        "relative_l2_denominator_floor": 1e-12,
        "max_abs_difference": float(np.max(np.abs(left - right))),
        "identical": bool(np.array_equal(a, b)),
    }


def summarize_replay_probes(
    root: Path,
    *,
    source_sha256: str,
    protocol_sha256: str,
    conditions: list[str],
) -> dict[str, Any]:
    expected = [
        ("clean", "reference"),
        ("clean", "repeat"),
        *((c, "fault") for c in conditions),
    ]
    groups: dict[
        tuple[str, str], dict[tuple[Any, ...], tuple[Path, dict[str, Any]]]
    ] = {key: {} for key in expected}
    rng_hashes: set[str] = set()
    missing_rng = False
    for path in sorted(root.glob("*/*.json")):
        record = json.loads(path.read_text())
        context = record.get("diagnostic_context", {})
        if (
            context.get("source_sha256") != source_sha256
            or context.get("protocol_sha256") != protocol_sha256
        ):
            raise ValueError("probe has unrelated or unresolved replay context")
        branch = (context["condition"], context["branch"])
        if branch not in groups:
            raise ValueError("probe has an unexpected replay branch")
        key = (record["encode_call_index"], record["call_mode"], record["frame_st_id"])
        if key in groups[branch]:
            raise ValueError(
                "duplicate replay encode; do not select between reset attempts"
            )
        groups[branch][key] = (path, record)
        digest = record.get("rng_after_reset_sha256")
        if not digest or record.get("rng_after_reset_missing"):
            missing_rng = True
        else:
            rng_hashes.add(digest)
    reference = groups["clean", "reference"]
    complete = bool(reference) and all(
        set(g) == set(reference) for g in groups.values()
    )
    differences = []
    if complete:
        for key, (path, record) in reference.items():
            ref = {
                **record["latents"],
                **{
                    f"tensor-{i}": v
                    for i, v in enumerate(record["tactile_tensor_outputs"])
                },
            }
            for branch, group in groups.items():
                if branch == ("clean", "reference"):
                    continue
                other_path, other = group[key]
                tensors = {
                    **other["latents"],
                    **{
                        f"tensor-{i}": v
                        for i, v in enumerate(other["tactile_tensor_outputs"])
                    },
                }
                if set(ref) != set(tensors):
                    raise ValueError("replay tensor/latent capture coverage differs")
                for name, summary in ref.items():
                    differences.append(
                        {
                            "condition": branch[0],
                            "branch": branch[1],
                            "encode_key": list(key),
                            "tensor": name,
                            **_difference(
                                _array(path, summary), _array(other_path, tensors[name])
                            ),
                        }
                    )
    return {
        "complete_encode_coverage": complete,
        "rng_after_reset_equal_and_complete": complete
        and not missing_rng
        and len(rng_hashes) == 1,
        "branch_encode_counts": {f"{a}/{b}": len(groups[a, b]) for a, b in expected},
        "differences": differences,
        "interpretation": "actual sampled model tensors/latents under fixed-input reset replay; zero effects retained",
    }
