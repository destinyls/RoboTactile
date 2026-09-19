"""Write-once JSON artifacts for recorded tactile-causal diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.recorded.metrics import ActionErrorMetrics
from robotactile_benchmark.recorded.tactile_causal import (
    EVIDENCE_LEVEL,
    METRIC_FIELDS,
    PRIMARY_TASK_IDS,
    RecordedTactileCausalResult,
)


def _metric(value: ActionErrorMetrics) -> dict[str, float | int]:
    return value.to_dict()


def recorded_tactile_causal_episode_document(
    result: RecordedTactileCausalResult,
) -> dict[str, object]:
    selection = result.anchor_selection
    clean = _metric(result.clean_expert_full)
    absence = _metric(result.absence_expert_full)
    return {
        "task_id": result.episode.task_id,
        "episode_id": result.episode.episode_id,
        "source_file": result.episode.source_path.name,
        "source_sha256": result.episode.source_sha256,
        "initial_seed": result.episode.initial_seed,
        "exogenous_seed": result.exogenous_seed,
        "clean_policy_identity_sha256": result.clean_identity.sha256,
        "absence_policy_identity_sha256": result.absence_identity.sha256,
        "checkpoint_sha256": result.clean_identity.checkpoint_sha256,
        "config_sha256": result.clean_identity.config_sha256,
        "anchor_selection": {
            "anchor_index": selection.anchor_index,
            "rest_index": selection.rest_index,
            "rest_mode": selection.rest_mode,
            "contact_mode": selection.contact_mode,
            "left_indentation_mm": selection.left_indentation_mm,
            "right_indentation_mm": selection.right_indentation_mm,
        },
        "expert_actions": result.episode.expert_actions.tolist(),
        "clean_prediction": result.clean_prediction.tolist(),
        "observed_tactile_absence_prediction": result.absence_prediction.tolist(),
        "clean_expert_full": clean,
        "observed_tactile_absence_expert_full": absence,
        "absence_minus_clean_error": {
            field: float(absence[field]) - float(clean[field])
            for field in METRIC_FIELDS
        },
        "clean_expert_future_h1_h11": _metric(result.clean_expert_future),
        "observed_tactile_absence_expert_future_h1_h11": _metric(
            result.absence_expert_future
        ),
        "clean_absence_action_drift": _metric(result.clean_absence_drift_full),
    }


def write_recorded_tactile_causal_artifact(
    path: Path,
    *,
    results: Sequence[RecordedTactileCausalResult],
    summary: object,
    source_commit: str,
    split_manifest_sha256: str,
) -> str:
    """Publish one deterministic, no-clobber cohort document."""

    document = {
        "schema_version": EVIDENCE_LEVEL,
        "source_commit": source_commit,
        "split_manifest_sha256": split_manifest_sha256,
        "summary": summary,
        "artifact_type": "cohort_v1",
        "episodes": [
            recorded_tactile_causal_episode_document(item) for item in results
        ],
    }
    return _publish(path, document)


def write_recorded_tactile_causal_task_artifact(
    path: Path,
    *,
    task_id: str,
    results: Sequence[RecordedTactileCausalResult],
    source_commit: str,
    split_manifest_sha256: str,
) -> str:
    """Publish one five-episode task shard for remote-client execution."""

    items = tuple(results)
    if not task_id or any(item.episode.task_id != task_id for item in items):
        raise ValueError("task shard results do not match task_id")
    document = {
        "schema_version": EVIDENCE_LEVEL,
        "artifact_type": "task_shard_v1",
        "source_commit": source_commit,
        "split_manifest_sha256": split_manifest_sha256,
        "task_id": task_id,
        "episode_count": len(items),
        "episodes": [recorded_tactile_causal_episode_document(item) for item in items],
    }
    return _publish(path, document)


def _publish(path: Path, document: object) -> str:
    payload = canonical_json_bytes(document)
    output = Path(path).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("causal diagnostic output already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def load_recorded_tactile_causal_task_artifact(path: Path) -> dict[str, object]:
    """Strict-load one canonical task shard."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError("causal task artifact must be a non-symlink regular file")
    raw = source.read_bytes()
    document: object = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict) or canonical_json_bytes(document) != raw:
        raise ValueError("causal task artifact is not canonical JSON")
    required = {
        "schema_version",
        "artifact_type",
        "source_commit",
        "split_manifest_sha256",
        "task_id",
        "episode_count",
        "episodes",
    }
    if (
        set(document) != required
        or document["schema_version"] != EVIDENCE_LEVEL
        or document["artifact_type"] != "task_shard_v1"
    ):
        raise ValueError("causal task artifact fields mismatch")
    episodes = document["episodes"]
    if not isinstance(episodes, list) or document["episode_count"] != len(episodes):
        raise ValueError("causal task artifact episode count mismatch")
    return cast(dict[str, object], document)


def _metrics(episode: Mapping[str, object], key: str) -> dict[str, float]:
    value = episode.get(key)
    if not isinstance(value, Mapping) or any(
        field not in value for field in METRIC_FIELDS
    ):
        raise ValueError(f"episode metric is incomplete: {key}")
    normalized: dict[str, float] = {}
    for field in METRIC_FIELDS:
        scalar = value[field]
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise ValueError(f"episode metric is not numeric: {key}.{field}")
        normalized[field] = float(scalar)
        if not math.isfinite(normalized[field]):
            raise ValueError(f"episode metric is not finite: {key}.{field}")
    return normalized


def _mean_metric_documents(
    episodes: Sequence[Mapping[str, object]], key: str
) -> dict[str, float]:
    if not episodes:
        raise ValueError("cannot aggregate an empty episode collection")
    return {
        field: sum(float(_metrics(episode, key)[field]) for episode in episodes)
        / len(episodes)
        for field in METRIC_FIELDS
    }


def _scope_document(episodes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    clean = _mean_metric_documents(episodes, "clean_expert_full")
    absence = _mean_metric_documents(episodes, "observed_tactile_absence_expert_full")
    return {
        "episode_count": len(episodes),
        "clean_expert_error": clean,
        "observed_tactile_absence_expert_error": absence,
        "absence_minus_clean_error": {
            field: absence[field] - clean[field] for field in METRIC_FIELDS
        },
        "absence_worse_episode_fraction": {
            field: sum(
                float(
                    _metrics(episode, "observed_tactile_absence_expert_full")[field]
                    > _metrics(episode, "clean_expert_full")[field]
                )
                for episode in episodes
            )
            / len(episodes)
            for field in METRIC_FIELDS
        },
        "clean_absence_action_drift": _mean_metric_documents(
            episodes, "clean_absence_action_drift"
        ),
    }


def write_recorded_tactile_causal_cohort_from_shards(
    path: Path,
    *,
    shard_paths: Sequence[Path],
    expected_task_ids: Sequence[str],
    episodes_per_task: int,
) -> tuple[str, dict[str, object]]:
    """Merge eight strict task shards into the complete offline cohort."""

    task_ids = tuple(expected_task_ids)
    shards = tuple(
        load_recorded_tactile_causal_task_artifact(item) for item in shard_paths
    )
    if len(set(task_ids)) != len(task_ids) or len(shards) != len(task_ids):
        raise ValueError("cohort shard/task cardinality mismatch")
    by_task: dict[str, list[Mapping[str, object]]] = {}
    source_commits: set[str] = set()
    split_hashes: set[str] = set()
    source_hashes: set[str] = set()
    for shard in shards:
        task = shard["task_id"]
        episodes = shard["episodes"]
        if not isinstance(task, str) or not isinstance(episodes, list):
            raise ValueError("causal task shard task/episodes types mismatch")
        if task in by_task or len(episodes) != episodes_per_task:
            raise ValueError("causal task shard count is not task-balanced")
        typed_episodes: list[Mapping[str, object]] = []
        for episode in episodes:
            if not isinstance(episode, Mapping) or episode.get("task_id") != task:
                raise ValueError("causal episode does not match its task shard")
            source_hash = episode.get("source_sha256")
            if not isinstance(source_hash, str) or source_hash in source_hashes:
                raise ValueError("causal cohort HDF5 source identity is duplicated")
            source_hashes.add(source_hash)
            _metrics(episode, "clean_expert_full")
            _metrics(episode, "observed_tactile_absence_expert_full")
            _metrics(episode, "clean_absence_action_drift")
            typed_episodes.append(episode)
        by_task[task] = typed_episodes
        source_commits.add(cast(str, shard["source_commit"]))
        split_hashes.add(cast(str, shard["split_manifest_sha256"]))
    if (
        set(by_task) != set(task_ids)
        or len(source_commits) != 1
        or len(split_hashes) != 1
    ):
        raise ValueError("causal cohort task/source/split identity mismatch")
    all_episodes = [episode for task in task_ids for episode in by_task[task]]
    primary = [
        episode for episode in all_episodes if episode["task_id"] in PRIMARY_TASK_IDS
    ]
    summary = {
        "schema_version": EVIDENCE_LEVEL,
        "evidence_level": EVIDENCE_LEVEL,
        "sampling_unit": "recorded_hdf5_episode",
        "success_rate_claimed": False,
        "completed_episode_count": len(all_episodes),
        "expected_episode_count": len(task_ids) * episodes_per_task,
        "micro_all_episodes": _scope_document(all_episodes),
        "primary_tasks": {
            "task_ids": list(PRIMARY_TASK_IDS),
            **_scope_document(primary),
        },
        "per_task": {task: _scope_document(by_task[task]) for task in task_ids},
        "claim_boundary": (
            "Paired offline model inference on expert HDF5 observations; predicted "
            "actions were not executed, so this artifact does not report task "
            "success rate or closed-loop robustness."
        ),
    }
    document = {
        "schema_version": EVIDENCE_LEVEL,
        "artifact_type": "cohort_v1",
        "source_commit": next(iter(source_commits)),
        "split_manifest_sha256": next(iter(split_hashes)),
        "summary": summary,
        "episodes": all_episodes,
    }
    return _publish(path, document), summary


__all__ = [
    "load_recorded_tactile_causal_task_artifact",
    "recorded_tactile_causal_episode_document",
    "write_recorded_tactile_causal_artifact",
    "write_recorded_tactile_causal_cohort_from_shards",
    "write_recorded_tactile_causal_task_artifact",
]
