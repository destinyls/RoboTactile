"""Offline reset-and-replay diagnostics; never execute predicted actions."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from robotactile_benchmark.contracts import (
    Array,
    EvaluationRecord,
    array_sha256,
    canonical_hash,
)
from robotactile_benchmark.execution.contracts import N0ObservedTactileMode
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
)
from robotactile_benchmark.policies.n0_official import (
    _wire_observation,
    n0_training_prompt,
    native_to_ee8_actions,
)
from robotactile_benchmark.transport.n0_official import OfficialN0RPC

from .action_effects import action_distance


def _wire(record: EvaluationRecord) -> tuple[dict[str, Array], dict[str, Array], Array]:
    return _wire_observation(
        record.observation,
        input_profile=N0_LIVE_UNIVTAC_INPUT_PROFILE,
        observed_tactile_mode=N0ObservedTactileMode.REQUIRED,
    )


def _non_tactile_identity(record: EvaluationRecord) -> str:
    obs = record.observation
    return canonical_hash(
        {
            "episode": obs.episode_id,
            "task": obs.task,
            "seed": obs.seed,
            "step": obs.step_index,
            "vision": {k: array_sha256(v) for k, v in obs.vision.items()},
            "proprio": array_sha256(obs.proprio),
        }
    )


def _native(response: Mapping[str, object]) -> Array:
    if set(response) - {"action", "server_timing"} or "action" not in response:
        raise ValueError("offline infer response lacks native actions")
    value = np.asarray(response["action"], dtype=np.float32)
    if value.shape != (20, 2, 12) or not np.isfinite(value).all():
        raise ValueError("offline replay requires finite official [20,2,12] actions")
    return value.copy()


def _timing_only(response: Mapping[str, object]) -> None:
    if set(response) - {"server_timing"}:
        raise ValueError("reset/grounding response contains unexpected fields")


def _branch(
    *,
    rpc_factory: Callable[[], OfficialN0RPC],
    records: Sequence[EvaluationRecord],
    seed: int,
    context: dict[str, str],
    steps: tuple[int, ...],
    reference_actions: dict[int, Array] | None,
) -> tuple[dict[int, Array], list[dict[str, Any]]]:
    """Fresh connection + reset + every preceding infer/grounding call.

    Grounding always uses frozen Clean-reference actions. Fault predictions are
    measured only, never fed back as actions or executed in a simulator.
    """
    rpc = rpc_factory()
    actions: dict[int, Array] = {}
    calls: list[dict[str, Any]] = []
    prompt = n0_training_prompt(records[0].observation.task)
    try:
        _timing_only(
            rpc.infer(
                {
                    "reset": True,
                    "prompt": prompt,
                    "seed": seed,
                    "_robotactile_diagnostic_context": context,
                }
            )
        )
        for ordinal, step in enumerate(steps):
            vision, tactile, state = _wire(records[step])
            payload: dict[str, object] = {
                "obs": vision,
                "tactile": tactile,
                "current_state": state.tolist(),
                "prompt": prompt,
            }
            calls.append(
                {
                    "mode": "infer",
                    "source_indices": [step],
                    "payload_sha256": canonical_hash(payload),
                }
            )
            actions[step] = _native(rpc.infer(copy.deepcopy(payload)))
            if ordinal == len(steps) - 1:
                continue
            end = steps[ordinal + 1]
            selected = tuple(range(step + 3, end + 1, 3))
            wire = [_wire(records[index]) for index in selected]
            fixed_action = (
                actions[step] if reference_actions is None else reference_actions[step]
            )
            grounding: dict[str, object] = {
                "obs": [item[0] for item in wire],
                "tactile": [item[1] for item in wire],
                "state": fixed_action,
                "current_state": state.tolist(),
                "compute_kv_cache": True,
                "imagine": False,
                "prompt": prompt,
            }
            calls.append(
                {
                    "mode": "grounding",
                    "source_indices": list(selected),
                    "payload_sha256": canonical_hash(grounding),
                    "action_history_sha256": array_sha256(fixed_action),
                }
            )
            _timing_only(rpc.infer(copy.deepcopy(grounding)))
    finally:
        rpc.close()
    return actions, calls


def run_fixed_input_replay(
    *,
    clean_records: Sequence[EvaluationRecord],
    fault_records: Mapping[str, Sequence[EvaluationRecord]],
    rpc_factory: Callable[[], OfficialN0RPC],
    seed: int,
    source_sha256: str,
    protocol_sha256: str,
    target_steps: tuple[int, ...] = (12, 36, 60),
) -> dict[str, Any]:
    """Measure Clean/Clean and Clean/Fault with fixed observations and history.

    This API must receive a separately owned offline server. The public runner
    launches that server itself rather than accepting an online endpoint.
    Complete RNG equality additionally requires server probe evidence.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("replay seed must be a nonnegative integer")
    if (
        not target_steps
        or tuple(sorted(set(target_steps))) != target_steps
        or any(
            type(s) is not int or s < 12 or (s - 12) % 24 or s > 60
            for s in target_steps
        )
    ):
        raise ValueError(
            "diagnostic targets must be ordered unique members of 12,36,60"
        )
    if not fault_records or {"clean-reference", "clean-repeat"} & set(fault_records):
        raise ValueError("unique named fault branches are required")
    for value in (source_sha256, protocol_sha256):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("replay provenance must be SHA256")
    last = max(target_steps)
    clean = tuple(clean_records[: last + 1])
    if len(clean) != last + 1 or any(
        r.observation.step_index != i for i, r in enumerate(clean)
    ):
        raise ValueError(
            "complete contiguous Clean prefix is required, not preview interpolation"
        )
    if (
        len(
            {
                (r.observation.episode_id, r.observation.task, r.observation.seed)
                for r in clean
            }
        )
        != 1
    ):
        raise ValueError("Clean prefix mixes episode identities")
    expected = tuple(_non_tactile_identity(r) for r in clean)
    variants: dict[str, tuple[EvaluationRecord, ...]] = {}
    for label, records in fault_records.items():
        subset = tuple(records[: last + 1])
        if tuple(_non_tactile_identity(r) for r in subset) != expected:
            raise ValueError(
                "fault branch changed RGB/proprio/identity or lacks full prefix"
            )
        variants[label] = subset
    steps = (0, *range(12, last + 1, 24))
    context = {"source_sha256": source_sha256, "protocol_sha256": protocol_sha256}
    reference, reference_calls = _branch(
        rpc_factory=rpc_factory,
        records=clean,
        seed=seed,
        context={**context, "condition": "clean", "branch": "reference"},
        steps=steps,
        reference_actions=None,
    )
    repeat, repeat_calls = _branch(
        rpc_factory=rpc_factory,
        records=clean,
        seed=seed,
        context={**context, "condition": "clean", "branch": "repeat"},
        steps=steps,
        reference_actions=reference,
    )
    if reference_calls != repeat_calls:
        raise ValueError("Clean replay requests changed despite frozen inputs/history")
    branches: dict[str, Any] = {
        "clean-reference": {"calls": reference_calls},
        "clean-repeat": {"calls": repeat_calls},
    }
    comparisons = []
    for label, records in variants.items():
        fault, calls = _branch(
            rpc_factory=rpc_factory,
            records=records,
            seed=seed,
            context={**context, "condition": label, "branch": "fault"},
            steps=steps,
            reference_actions=reference,
        )
        if [c.get("action_history_sha256") for c in calls] != [
            c.get("action_history_sha256") for c in reference_calls
        ]:
            raise ValueError(
                "fault replay action history differs from frozen Clean history"
            )
        branches[label] = {"calls": calls}
        for step in target_steps:
            clean_ee = native_to_ee8_actions(reference[step], cold_chunk=False)
            repeat_ee = native_to_ee8_actions(repeat[step], cold_chunk=False)
            fault_ee = native_to_ee8_actions(fault[step], cold_chunk=False)
            baseline = action_distance(clean_ee, repeat_ee)
            effect = action_distance(clean_ee, fault_ee)
            comparisons.append(
                {
                    "condition": label,
                    "target_step": step,
                    "clean_repeat": baseline,
                    "clean_fault": effect,
                    "clean_native_sha256": array_sha256(reference[step]),
                    "repeat_native_sha256": array_sha256(repeat[step]),
                    "fault_native_sha256": array_sha256(fault[step]),
                    "clean_ee8": clean_ee.tolist(),
                    "repeat_ee8": repeat_ee.tolist(),
                    "fault_ee8": fault_ee.tolist(),
                }
            )
    return {
        "schema": "n0_fixed_input_noise_replay_v1",
        "source_sha256": source_sha256,
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "target_steps": list(target_steps),
        "input_profile": N0_LIVE_UNIVTAC_INPUT_PROFILE.to_dict(),
        "complete_prefix_replayed": True,
        "rgb_proprio_fixed": True,
        "action_history": "frozen Clean-reference predictions; fault predictions never fed back",
        "cache_initialization": "independent reset plus complete prefix per branch",
        "rng_verification": "requires matching server-side reset fingerprints",
        "actions_executed": False,
        "success_rate_claimed": False,
        "branches": branches,
        "comparisons": comparisons,
        "interpretation": "offline intervention on fixed live-observation prefix; not closed-loop SR; zero effects retained",
    }
