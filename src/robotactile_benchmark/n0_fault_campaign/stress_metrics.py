"""Paired seed statistics with infrastructure exclusions and held-out goal gates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .stress_protocol import validate_protocol


def eligible_terminal(value: dict[str, Any]) -> bool:
    return (
        value.get("score_eligible") is True
        and value.get("validation_passed") is True
        and value.get("terminal_status")
        in {"success", "task_failure", "early_stop", "timeout"}
        and type(value.get("observation_count")) is int
        and value["observation_count"] > 0
        and type(value.get("score_success")) is bool
    )


def _mcnemar(losses: int, gains: int) -> float:
    n = losses + gains
    if not n:
        return 1.0
    return float(
        min(
            1.0,
            2.0 * sum(math.comb(n, k) for k in range(min(losses, gains) + 1)) / 2**n,
        )
    )


def paired_statistics(
    clean: list[bool], fault: list[bool], *, random_seed: int, replicates: int
) -> dict[str, Any]:
    if (
        not clean
        or len(clean) != len(fault)
        or any(type(v) is not bool for v in clean + fault)
    ):
        raise ValueError(
            "paired binary outcomes with equal nonzero length are required"
        )
    delta = np.asarray(clean, dtype=np.float64) - np.asarray(fault, dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    boot = np.empty(replicates)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        boot[start : start + count] = delta[
            rng.integers(len(delta), size=(count, len(delta)))
        ].mean(axis=1)
    losses = sum(c and not f for c, f in zip(clean, fault))
    gains = sum(f and not c for c, f in zip(clean, fault))
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    return {
        "paired_count": len(clean),
        "clean_success": sum(clean),
        "fault_success": sum(fault),
        "clean_sr": sum(clean) / len(clean),
        "fault_sr": sum(fault) / len(fault),
        "absolute_drop": float(delta.mean()),
        "drop_pp": float(100 * delta.mean()),
        "paired_bootstrap_95ci": [float(lo), float(hi)],
        "clean_success_fault_failure": losses,
        "clean_failure_fault_success": gains,
        "exact_mcnemar_p": _mcnemar(losses, gains),
    }


def summarize_stress(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Only accepted whole-group attempts enter paired inference; keep other counts."""
    plan = validate_protocol(protocol)
    labels = ["clean", *(v["label"] for v in plan["variants"])]
    expected = {(s, label) for s in plan["seeds"] for label in labels}
    accepted: dict[tuple[int, str], dict[str, Any]] = {}
    live = 0
    infra = 0
    unsupported = 0
    seen: set[tuple[int, str]] = set()
    for row in rows:
        key = (row["seed"], row["condition"])
        if key not in expected:
            raise ValueError("unexpected seed/condition in results")
        if row.get("protocol_sha256") != plan["protocol_sha256"]:
            raise ValueError("result protocol hash mismatch")
        if "group_paths" in plan:
            if row.get("group_root") != plan["group_paths"][str(row["seed"])]:
                raise ValueError("result came from an unplanned group root")
            if key in seen:
                raise ValueError(
                    "fixed-path studies forbid duplicate cells or replacement attempts"
                )
            seen.add(key)
        if row.get("terminal_status") == "unsupported_contract":
            unsupported += 1
            continue
        valid = eligible_terminal(row)
        live += int(valid)
        infra += int(not valid)
        if row.get("group_accepted") is True:
            if not valid or row.get("exact_snapshot_reset_verified") is not True:
                raise ValueError(
                    "accepted group contains invalid terminal/reset evidence"
                )
            if key in accepted:
                raise ValueError(
                    "duplicate accepted seed/condition; do not mix retries"
                )
            accepted[key] = row
    accepted_seeds = []
    for seed in plan["seeds"]:
        group = [accepted.get((seed, label)) for label in labels]
        if not any(g is not None for g in group):
            continue
        if any(g is None for g in group):
            raise ValueError("partially accepted group is forbidden")
        attempts = {g.get("attempt_id") for g in group if g is not None}
        if len(attempts) != 1 or None in attempts:
            raise ValueError("accepted group mixes attempts")
        accepted_seeds.append(seed)
    summaries = []
    for label in labels[1:]:
        if not accepted_seeds:
            summaries.append({"condition": label, "paired_count": 0, "goal_met": False})
            continue
        stats = paired_statistics(
            [accepted[s, "clean"]["score_success"] for s in accepted_seeds],
            [accepted[s, label]["score_success"] for s in accepted_seeds],
            random_seed=plan["bootstrap_seed"],
            replicates=plan["bootstrap_replicates"],
        )
        summaries.append({"condition": label, **stats})
    adjusted = 0.0
    ranked = sorted(
        (s for s in summaries if s["paired_count"]), key=lambda s: s["exact_mcnemar_p"]
    )
    for index, stats in enumerate(ranked):
        adjusted = max(
            adjusted, min(1.0, stats["exact_mcnemar_p"] * (len(ranked) - index))
        )
        stats["holm_adjusted_p"] = adjusted
        stats["goal_met"] = (
            plan["stage"] == "confirmation"
            and len(accepted_seeds) == len(plan["seeds"])
            and stats["absolute_drop"] + 1e-12 >= plan["target_drop_absolute"]
            and stats["paired_bootstrap_95ci"][0] > 0
            and adjusted < plan["familywise_alpha"]
        )
    return {
        "schema": "n0_noise_stress_results_v2",
        "protocol_sha256": plan["protocol_sha256"],
        "stage": plan["stage"],
        "planned_live_rollouts": len(expected),
        "eligible_live_attempt_rollouts": live,
        "accepted_live_rollouts": len(accepted),
        "provisional_live_attempt_rollouts": live - len(accepted),
        "invalid_attempt_rollouts": infra,
        "unsupported_contract": unsupported,
        "accepted_seed_groups": len(accepted_seeds),
        "missing_accepted_cells": len(expected) - len(accepted),
        "conditions": summaries,
        "goal_met": bool(summaries) and all(s["goal_met"] for s in summaries),
        "interpretation": "20 percentage points in observed SR, not proof population drop >=20pp",
    }
