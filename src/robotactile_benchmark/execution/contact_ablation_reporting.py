"""Diagnostic summaries for matched N0 tactile-absence and action replay runs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from robotactile_benchmark.contracts import Array, canonical_hash, thaw_value
from robotactile_benchmark.execution.action_replay import ActionTraceReplayReceipt
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.n0_fault_campaign.io import LoadedN0FaultCampaign

CONTACT_ABLATION_EVIDENCE_LEVEL = "n0_contact_ablation_diagnostic_v1"
CONTACT_ABLATION_SEMANTIC_VERSION = "1.0"


def _artifact_summary(artifact: LoadedLiveUniVTACArtifact) -> dict[str, object]:
    result = artifact.evidence.result
    final_diagnostics: Mapping[str, object] = {}
    if artifact.evidence.transition_entries:
        final_diagnostics = artifact.evidence.transition_entries[-1].diagnostics
    return {
        "artifact_root_sha256": artifact.external_root_sha256,
        "trial_manifest_sha256": artifact.trial.sha256,
        "initial_state_sha256": result.initial_state_sha256,
        "terminal_status": result.terminal_status.value,
        "score_success": result.score_success,
        "observation_count": result.observation_count,
        "control_cycle_count": result.control_cycle_count,
        "action_trace_sha256": result.action_trace_sha256,
        "final_transition_diagnostics": thaw_value(final_diagnostics),
    }


def _flat_actions(artifact: LoadedLiveUniVTACArtifact) -> Array:
    return np.concatenate(
        [entry.executed_actions for entry in artifact.evidence.action_entries], axis=0
    )


def _action_comparison(
    clean: LoadedLiveUniVTACArtifact,
    absence: LoadedLiveUniVTACArtifact,
) -> dict[str, object]:
    clean_actions = _flat_actions(clean)
    absence_actions = _flat_actions(absence)
    aligned_count = min(len(clean_actions), len(absence_actions))
    aligned_clean = clean_actions[:aligned_count]
    aligned_absence = absence_actions[:aligned_count]
    per_step_l2 = np.linalg.norm(aligned_clean - aligned_absence, axis=1)
    clean_plans = clean.evidence.action_entries
    absence_plans = absence.evidence.action_entries
    aligned_cycles = min(len(clean_plans), len(absence_plans))
    equal_plan_count = sum(
        clean_plans[index].action_plan_sha256 == absence_plans[index].action_plan_sha256
        for index in range(aligned_cycles)
    )
    return {
        "aligned_action_count": aligned_count,
        "aligned_control_cycle_count": aligned_cycles,
        "equal_action_plan_hash_count": equal_plan_count,
        "action_trace_equal": (
            clean.evidence.result.action_trace_sha256
            == absence.evidence.result.action_trace_sha256
        ),
        "mean_action_l2": float(per_step_l2.mean()),
        "max_action_l2": float(per_step_l2.max()),
        "action_rmse": float(np.sqrt(np.mean((aligned_clean - aligned_absence) ** 2))),
    }


def build_contact_ablation_pair_summary(
    *,
    pair_key: str,
    initial_seed: int,
    exogenous_seed: int,
    clean: LoadedLiveUniVTACArtifact,
    absence: LoadedLiveUniVTACArtifact,
    replay: ActionTraceReplayReceipt,
    replay_file_sha256: str,
) -> dict[str, object]:
    """Bind one matched three-condition comparison and require reset equality."""

    states = (
        clean.evidence.result.initial_state_sha256,
        absence.evidence.result.initial_state_sha256,
        replay.replay_initial_state_sha256,
    )
    if states[0] is None or len(set(states)) != 1:
        raise RuntimeError("Clean, absence, and replay initial states differ")
    if clean.trial.task != absence.trial.task or clean.trial.task != replay.task:
        raise ValueError("contact ablation conditions changed task identity")
    return {
        "task": clean.trial.task,
        "pair_key": pair_key,
        "initial_seed": initial_seed,
        "exogenous_seed": exogenous_seed,
        "initial_state_exact_match": True,
        "clean": _artifact_summary(clean),
        "observed_tactile_absence": _artifact_summary(absence),
        "clean_action_replay": {
            **replay.to_dict(),
            "receipt_file_sha256": replay_file_sha256,
            "receipt_sha256": replay.sha256,
        },
        "clean_vs_absence_actions": _action_comparison(clean, absence),
    }


def build_contact_ablation_pilot_summary(
    campaign: LoadedN0FaultCampaign,
    rows: list[dict[str, object]],
    *,
    integration_manifest_sha256: str,
    runtime_count: int,
    action_execution_contract: str,
) -> dict[str, object]:
    """Aggregate exact success and behavior diagnostics without a paper claim."""

    if not rows:
        raise ValueError("contact ablation pilot requires at least one result row")

    def successes(key: str) -> int:
        return sum(bool(_mapping(row[key], key)["score_success"]) for row in rows)

    replay_successes = sum(
        bool(_mapping(row["clean_action_replay"], "replay")["task_success"])
        for row in rows
    )
    tasks = {row["task"] for row in rows}
    if len(tasks) != 1:
        raise ValueError("contact ablation rows span multiple tasks")
    document: dict[str, object] = {
        "campaign_id": campaign.manifest.campaign_id,
        "campaign_manifest_sha256": campaign.manifest.sha256,
        "generation_receipt_file_sha256": campaign.receipt_file_sha256,
        "integration_manifest_sha256": integration_manifest_sha256,
        "task": next(iter(tasks)),
        "pair_count": len(rows),
        "isaac_simulation_app_initialization_count": 1,
        "fresh_task_runtime_count": runtime_count,
        "action_execution_contract": action_execution_contract,
        "capture_profile": LiveCaptureProfile.METRICS_ONLY.value,
        "clean_success_count": successes("clean"),
        "absence_success_count": successes("observed_tactile_absence"),
        "action_replay_success_count": replay_successes,
        "all_initial_states_exact_match": all(
            bool(row["initial_state_exact_match"]) for row in rows
        ),
        "pairs": rows,
        "evidence_level": CONTACT_ABLATION_EVIDENCE_LEVEL,
        "simulator_execution_claimed": True,
        "policy_inference_executed_for_action_replay": False,
        "paper_claim": False,
        "semantic_version": CONTACT_ABLATION_SEMANTIC_VERSION,
    }
    document["content_sha256"] = canonical_hash(document)
    return document


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} summary must be a mapping")
    return value


__all__ = [
    "CONTACT_ABLATION_EVIDENCE_LEVEL",
    "CONTACT_ABLATION_SEMANTIC_VERSION",
    "build_contact_ablation_pair_summary",
    "build_contact_ablation_pilot_summary",
]
