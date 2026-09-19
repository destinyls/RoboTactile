"""Read completed action traces; never interpret a pixel hash as policy sensitivity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from robotactile_benchmark.contracts import Array


def action_distance(clean: Array, faulted: Array) -> dict[str, float | int]:
    """Compare executed EE8 targets at matching source indices, in physical units."""
    if any(
        x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all()
        for x in (clean, faulted)
    ):
        raise ValueError("action distance requires finite (N, 8) arrays")
    n = min(len(clean), len(faulted))
    if not n:
        raise ValueError("empty action overlap")
    a, b = clean[:n].astype(np.float64), faulted[:n].astype(np.float64)
    translation = np.linalg.norm(a[:, :3] - b[:, :3], axis=1) * 1000
    qa, qb = a[:, 3:7], b[:, 3:7]
    na, nb = np.linalg.norm(qa, axis=1), np.linalg.norm(qb, axis=1)
    if np.any(na < 1e-8) or np.any(nb < 1e-8):
        raise ValueError("action quaternion cannot be zero")
    dots = np.abs(np.sum(qa * qb, axis=1) / (na * nb))
    rotation = np.rad2deg(2 * np.arccos(np.clip(dots, 0, 1)))
    gripper = np.abs(a[:, 7] - b[:, 7]) * 1000
    return {
        "overlap_action_count": n,
        "translation_mean_mm": float(translation.mean()),
        "translation_max_mm": float(translation.max()),
        "rotation_mean_deg": float(rotation.mean()),
        "rotation_max_deg": float(rotation.max()),
        "gripper_mean_mm": float(gripper.mean()),
        "gripper_max_mm": float(gripper.max()),
    }


def _read_array(root: Path, descriptor: dict[str, Any]) -> Array:
    path = (root / descriptor["path"]).resolve(strict=True)
    if root.resolve(strict=True) not in path.parents:
        raise ValueError("action array escapes artifact")
    if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["file_sha256"]:
        raise ValueError("action array digest mismatch")
    result = np.load(path, allow_pickle=False)
    if list(result.shape) != descriptor["shape"]:
        raise ValueError("action array shape mismatch")
    return cast(Array, np.asarray(result))


def compare_action_artifacts(clean: Path, faulted: Path) -> dict[str, Any]:
    """Report paired trajectory divergence; later observations need not be equal."""
    c = json.loads((clean / "terminal_result.json").read_text())
    f = json.loads((faulted / "terminal_result.json").read_text())
    manifest = json.loads((faulted / "fault_manifest.json").read_text())
    onset = int(manifest["start_index"])
    ca = json.loads((clean / "action_trace.json").read_text())["entries"]
    fa = json.loads((faulted / "action_trace.json").read_text())["entries"]
    by_step = {e["source_step_index"]: e for e in ca}
    comparisons = []
    for entry in fa:
        step = entry["source_step_index"]
        if step not in by_step:
            continue
        reference = by_step[step]
        comparisons.append(
            {
                "source_step_index": step,
                "post_onset": step >= onset,
                "action_plan_identical": entry["action_plan_sha256"]
                == reference["action_plan_sha256"],
                **action_distance(
                    _read_array(clean, reference["executed_actions"]),
                    _read_array(faulted, entry["executed_actions"]),
                ),
            }
        )
    after = [x for x in comparisons if x["post_onset"]]
    valid_pair = c["initial_state_sha256"] == f["initial_state_sha256"] and all(
        x.get("validation_passed") is True and x.get("score_eligible") is True
        for x in (c, f)
    )
    report: dict[str, Any] = {
        "clean_artifact": str(clean),
        "fault_artifact": str(faulted),
        "operator_id": manifest["operator_id"],
        "onset_index": onset,
        "initial_state_equal": c["initial_state_sha256"] == f["initial_state_sha256"],
        "valid_pair": valid_pair,
        "clean_success": c.get("score_success"),
        "fault_success": f.get("score_success"),
        "fault_terminal_status": f["terminal_status"],
        "clean_success_to_fault_failure": valid_pair
        and c["score_success"] is True
        and f["score_success"] is False,
        "post_onset_replan_count": sum(e["source_step_index"] >= onset for e in fa),
        "first_post_onset_action_comparison": after[0] if after else None,
        "action_comparisons": comparisons,
        "delivery_exposure": delivery_exposure(faulted, fa, onset),
        "interpretation": "closed-loop action divergence, not isolated-input causal sensitivity; quaternion sign is normalized; no learned-feature use is inferred",
    }
    parameters = manifest["parameters"]
    if after and "target_gain_rgb" in parameters:
        rise = int(
            parameters.get("rise_duration_frames", manifest["stop_index"] - onset)
        )
        fraction = min(1, (after[0]["source_step_index"] - onset + 1) / rise)
        report["f1_ramp_fraction_at_first_post_onset_replan"] = fraction
        report["f1_channel_gains_at_first_post_onset_replan"] = [
            1 + fraction * (float(g) - 1) for g in parameters["target_gain_rgb"]
        ]
    return report


def delivery_exposure(
    root: Path, action_entries: list[dict[str, Any]], onset: int
) -> dict[str, Any]:
    """Describe captured policy keyframes, not internal encoder activation."""
    full, preview = root / "delivery_trace.json", root / "preview_trace.json"
    trace = json.loads(full.read_text())["finalization"] if full.exists() else None
    if trace is None and preview.exists():
        trace = json.loads(preview.read_text())
    if trace is None:
        return {"status": "not_captured"}
    source = {r["observation"]["step_index"]: r for r in trace["clean_records"]}
    delivered = {r["observation"]["step_index"]: r for r in trace["delivered_records"]}
    latest_infer = max((int(e["source_step_index"]) for e in action_entries), default=0)
    # Official delta adapter commits every third action observation, and warm
    # inference reads that committed history. After terminal there is no commit.
    selected = list(range(3, latest_infer + 1, 3))
    frames = []
    for i in selected:
        if i not in source or i not in delivered:
            continue
        c, d = source[i]["observation"], delivered[i]["observation"]
        frames.append(
            {
                "source_step_index": i,
                "post_onset": i >= onset,
                "vision_equal": c["vision"] == d["vision"],
                "proprio_equal": c["proprio"] == d["proprio"],
                "raw_tactile_hashes": [
                    {
                        "slot": cs["slot_id"],
                        "clean": cs["payload"]["array_sha256"]
                        if cs["payload"]
                        else None,
                        "delivered": ds["payload"]["array_sha256"]
                        if ds["payload"]
                        else None,
                    }
                    for cs, ds in zip(c["tactile"], d["tactile"])
                ],
            }
        )
    held = source.get(onset - 1)
    return {
        "status": "captured_subset",
        "expected_committed_keyframes": len(selected),
        "captured_committed_keyframes": len(frames),
        "frames": frames,
        "held_source_provenance": held.get("provenance") if held else None,
        "interpretation": "raw model-boundary keyframe hashes; missing preview frames are not evidence of absent delivery; no VAE/cross-attention sensitivity is asserted",
    }


def summarize_campaign(root: Path) -> dict[str, Any]:
    """Read only terminal cells and keep unfinished/invalid cells separate."""
    manifest = json.loads((root / "campaign_manifest.json").read_text())
    clean_by_pair = {
        c["pair_key"]: c for c in manifest["cells"] if c["condition"] == "clean"
    }
    rows: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        path = root / cell["artifact_relpath"] if cell["artifact_relpath"] else None
        terminal = path / "terminal_result.json" if path else None
        row: dict[str, Any] = {
            "task": cell["task"],
            "seed": cell["initial_seed"],
            "condition": cell["operator_id"] or "clean",
            "artifact": str(path) if path else None,
        }
        if terminal is None or not terminal.exists():
            row["status"] = (
                "unsupported"
                if cell["disposition"] == "unsupported_contract"
                else "pending"
            )
        else:
            t = json.loads(terminal.read_text())
            row.update(
                status="terminal",
                **{
                    k: t.get(k)
                    for k in (
                        "validation_passed",
                        "score_eligible",
                        "score_success",
                        "terminal_status",
                        "failure_code",
                        "failure_stage",
                        "observation_count",
                        "control_cycle_count",
                    )
                },
            )
            if (
                cell["operator_id"]
                and path is not None
                and (path / "action_trace.json").is_file()
            ):
                clean = root / clean_by_pair[cell["pair_key"]]["artifact_relpath"]
                if (clean / "terminal_result.json").exists():
                    row["action_effect"] = compare_action_artifacts(clean, path)
        rows.append(row)
    return {
        "schema": "n0_decision_stress_summary_v1",
        "campaign_root": str(root),
        "rows": rows,
        "completed": sum(r["status"] == "terminal" for r in rows),
        "planned": len(rows),
        "held_out_confirmation": False,
    }
