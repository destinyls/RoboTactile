"""Build one train-only mixed-task LeRobot root for N0-VTLA.

The eight certified qpos8 task roots already contain the desired state/action
supervision and encoded videos.  This module only rewrites global index fields
in the small Parquet files and hardlinks the immutable videos under global
episode names.  It never reads or references the frozen validation split.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import TASKS, atomic_json, canonical_json_sha256, file_sha256
from .materialize_qpos8 import verify_task_view

MIXED_VIEW_PROTOCOL = "robotactile.n0_vtla.univtac_qpos8_mixed8_train_only.v1"
MIXED_RECEIPT_NAME = "_robotactile_n0_vtla_qpos8_mixed8_receipt.json"
EXPECTED_EPISODES = 759
EXPECTED_FRAMES = 144_484


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())


def _constant_stats(value: int, count: int) -> dict[str, object]:
    return {
        "min": [value],
        "max": [value],
        "mean": [float(value)],
        "std": [0.0],
        "count": [count],
    }


def _range_stats(start: int, count: int) -> dict[str, object]:
    values = np.arange(start, start + count, dtype=np.float64)
    return {
        "min": [start],
        "max": [start + count - 1],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [count],
    }


def _format_path(
    root: Path, template: str, chunks_size: int, episode: int, **extra: str
) -> Path:
    return root / template.format(
        episode_chunk=episode // chunks_size,
        episode_index=episode,
        **extra,
    )


def _replace_index_columns(
    source: Path,
    destination: Path,
    *,
    local_episode: int,
    local_frame_start: int,
    global_episode: int,
    global_frame_start: int,
    task_index: int,
) -> int:
    pa: Any = importlib.import_module("pyarrow")
    pq: Any = importlib.import_module("pyarrow.parquet")
    table = pq.read_table(source)
    required = {"episode_index", "frame_index", "index", "task_index"}
    if not required.issubset(table.column_names):
        raise ValueError(f"source Parquet index columns changed: {source}")
    count = len(table)
    frame_index = np.asarray(table["frame_index"].to_numpy())
    episode_values = np.asarray(table["episode_index"].to_numpy())
    index_values = np.asarray(table["index"].to_numpy())
    task_values = np.asarray(table["task_index"].to_numpy())
    if (
        count <= 0
        or not np.array_equal(frame_index, np.arange(count))
        or not np.all(episode_values == local_episode)
        or not np.array_equal(
            index_values, np.arange(local_frame_start, local_frame_start + count)
        )
        or not np.all(task_values == 0)
    ):
        raise ValueError(f"source Parquet temporal contract changed: {source}")
    replacements = {
        "episode_index": np.full(count, global_episode, dtype=np.int64),
        "index": np.arange(
            global_frame_start, global_frame_start + count, dtype=np.int64
        ),
        "task_index": np.full(count, task_index, dtype=np.int64),
    }
    for key, values in replacements.items():
        table = table.set_column(
            table.schema.get_field_index(key), key, pa.array(values)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(None), destination, compression="zstd")
    return count


def _validate_episode_metadata(
    row: Mapping[str, Any], *, episode: int, prompt: str
) -> int:
    length = row.get("length")
    actions = row.get("action_config")
    if (
        row.get("episode_index") != episode
        or row.get("tasks") != [prompt]
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        or not isinstance(actions, list)
        or not actions
        or any(
            not isinstance(item, dict) or item.get("action_text") != prompt
            for item in actions
        )
    ):
        raise ValueError(f"source episode metadata mismatch: episode={episode}")
    return length


def _source_contracts(source_root: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    receipts: dict[str, dict[str, Any]] = {}
    manifest_sha: str | None = None
    for task in TASKS:
        task_root = source_root / task
        raw_receipt = _json(task_root / "_robotactile_n0_vtla_qpos8_receipt.json")
        current_sha = raw_receipt.get("source_manifest_sha256")
        if not isinstance(current_sha, str):
            raise ValueError(f"source manifest digest missing: {task}")
        if manifest_sha is None:
            manifest_sha = current_sha
        elif current_sha != manifest_sha:
            raise ValueError("task views do not share one certified source manifest")
        receipt = verify_task_view(task_root, task=task, manifest_sha256=current_sha)
        if receipt.get("validation_data_used") is not False:
            raise ValueError(f"validation data is not excluded: {task}")
        receipts[task] = receipt
    if manifest_sha is None:
        raise ValueError("no task views found")
    return manifest_sha, receipts


def verify_mixed8_view(root: Path) -> dict[str, Any]:
    """Fail closed unless a published mixed root has the complete contract."""

    pq: Any = importlib.import_module("pyarrow.parquet")
    receipt_path = root / MIXED_RECEIPT_NAME
    receipt = _json(receipt_path)
    unsigned = dict(receipt)
    claimed = unsigned.pop("view_receipt_sha256", None)
    if (
        claimed != canonical_json_sha256(unsigned)
        or receipt.get("protocol_id") != MIXED_VIEW_PROTOCOL
        or receipt.get("status") != "complete"
        or receipt.get("source_split") != "train"
        or receipt.get("validation_data_used") is not False
        or receipt.get("episode_count") != EXPECTED_EPISODES
        or receipt.get("frame_count") != EXPECTED_FRAMES
        or receipt.get("tasks") != list(TASKS)
    ):
        raise ValueError(f"mixed8 receipt mismatch: {receipt_path}")
    info = _json(root / "meta" / "info.json")
    if (
        info.get("total_episodes") != EXPECTED_EPISODES
        or info.get("total_frames") != EXPECTED_FRAMES
        or info.get("total_tasks") != len(TASKS)
        or info.get("splits") != {"train": f"0:{EXPECTED_EPISODES}"}
    ):
        raise ValueError("mixed8 info metadata mismatch")
    task_rows = _jsonl(root / "meta" / "tasks.jsonl")
    prompt_map = receipt.get("task_prompts")
    if not isinstance(prompt_map, dict) or task_rows != [
        {"task_index": index, "task": prompt_map[task]}
        for index, task in enumerate(TASKS)
    ]:
        raise ValueError("mixed8 prompt mapping mismatch")
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    video_files = sorted((root / "videos").rglob("*.mp4"))
    if (
        len(parquet_files) != EXPECTED_EPISODES
        or len(video_files) != EXPECTED_EPISODES * 4
    ):
        raise ValueError("mixed8 file count mismatch")
    if any(path.is_symlink() for path in (*parquet_files, *video_files)):
        raise ValueError("mixed8 views may not contain symlinks")
    frame_cursor = 0
    episode_map = receipt.get("episode_map")
    if not isinstance(episode_map, list) or len(episode_map) != EXPECTED_EPISODES:
        raise ValueError("mixed8 episode map is incomplete")
    for episode, (path, mapping) in enumerate(zip(parquet_files, episode_map)):
        table = pq.read_table(path, columns=["episode_index", "index", "task_index"])
        count = len(table)
        task_index = mapping.get("task_index") if isinstance(mapping, dict) else None
        if (
            not np.all(np.asarray(table["episode_index"].to_numpy()) == episode)
            or not np.array_equal(
                np.asarray(table["index"].to_numpy()),
                np.arange(frame_cursor, frame_cursor + count),
            )
            or not np.all(np.asarray(table["task_index"].to_numpy()) == task_index)
        ):
            raise ValueError(f"mixed8 global index mismatch: episode={episode}")
        frame_cursor += count
    if frame_cursor != EXPECTED_FRAMES:
        raise ValueError("mixed8 global frame count mismatch")
    return receipt


def materialize_mixed8(*, source_root: Path, output_root: Path) -> dict[str, Any]:
    """Atomically publish one eight-task train759 root."""

    destination = output_root / "mixed8"
    if destination.exists():
        return verify_mixed8_view(destination)
    manifest_sha, source_receipts = _source_contracts(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mixed8.", dir=output_root))
    try:
        base_info: dict[str, Any] | None = None
        task_prompts: dict[str, str] = {}
        task_episode_counts: dict[str, int] = {}
        task_frame_counts: dict[str, int] = {}
        task_frame_ranges: dict[str, list[int]] = {}
        source_receipt_sha256: dict[str, str] = {}
        episode_map: list[dict[str, Any]] = []
        output_episodes: list[dict[str, Any]] = []
        output_stats: list[dict[str, Any]] = []
        global_episode = 0
        global_frame = 0
        video_count = 0
        for task_index, task in enumerate(TASKS):
            task_root = source_root / task
            info = _json(task_root / "meta" / "info.json")
            if base_info is None:
                base_info = copy.deepcopy(info)
            else:
                stable_keys = (
                    "codebase_version",
                    "fps",
                    "robot_type",
                    "chunks_size",
                    "data_path",
                    "video_path",
                    "features",
                )
                if any(info.get(key) != base_info.get(key) for key in stable_keys):
                    raise ValueError(f"task view schema differs: {task}")
            task_rows = _jsonl(task_root / "meta" / "tasks.jsonl")
            if len(task_rows) != 1 or task_rows[0].get("task_index") != 0:
                raise ValueError(f"task prompt metadata mismatch: {task}")
            prompt = task_rows[0].get("task")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"empty task prompt: {task}")
            task_prompts[task] = prompt
            episode_rows = _jsonl(task_root / "meta" / "episodes.jsonl")
            stats_rows = _jsonl(task_root / "meta" / "episodes_stats.jsonl")
            expected_episodes = source_receipts[task]["episode_count"]
            if (
                len(episode_rows) != expected_episodes
                or len(stats_rows) != expected_episodes
            ):
                raise ValueError(f"task episode metadata count mismatch: {task}")
            task_start = global_frame
            local_frame = 0
            chunks_size = int(info["chunks_size"])
            data_template = str(info["data_path"])
            video_template = str(info["video_path"])
            video_keys = sorted(
                key
                for key, feature in info["features"].items()
                if feature.get("dtype") == "video"
            )
            if len(video_keys) != 4:
                raise ValueError(f"expected four video streams: {task}")
            source_hashes = source_receipts[task].get("episode_source_sha256")
            if (
                not isinstance(source_hashes, list)
                or len(source_hashes) != expected_episodes
            ):
                raise ValueError(f"source episode hashes missing: {task}")
            for local_episode, (episode_row, stats_row) in enumerate(
                zip(episode_rows, stats_rows)
            ):
                expected_length = _validate_episode_metadata(
                    episode_row, episode=local_episode, prompt=prompt
                )
                source_parquet = _format_path(
                    task_root, data_template, chunks_size, local_episode
                )
                destination_parquet = _format_path(
                    staging, data_template, chunks_size, global_episode
                )
                frame_count = _replace_index_columns(
                    source_parquet,
                    destination_parquet,
                    local_episode=local_episode,
                    local_frame_start=local_frame,
                    global_episode=global_episode,
                    global_frame_start=global_frame,
                    task_index=task_index,
                )
                if frame_count != expected_length:
                    raise ValueError(
                        f"Parquet/episode length mismatch: {task}/{local_episode}"
                    )
                output_episode = copy.deepcopy(episode_row)
                output_episode["episode_index"] = global_episode
                output_episodes.append(output_episode)
                if stats_row.get("episode_index") != local_episode or not isinstance(
                    stats_row.get("stats"), dict
                ):
                    raise ValueError(f"episode stats mismatch: {task}/{local_episode}")
                output_stat = copy.deepcopy(stats_row)
                output_stat["episode_index"] = global_episode
                output_stat["stats"]["episode_index"] = _constant_stats(
                    global_episode, frame_count
                )
                output_stat["stats"]["task_index"] = _constant_stats(
                    task_index, frame_count
                )
                output_stat["stats"]["index"] = _range_stats(global_frame, frame_count)
                output_stats.append(output_stat)
                for video_key in video_keys:
                    source_video = _format_path(
                        task_root,
                        video_template,
                        chunks_size,
                        local_episode,
                        video_key=video_key,
                    )
                    destination_video = _format_path(
                        staging,
                        video_template,
                        chunks_size,
                        global_episode,
                        video_key=video_key,
                    )
                    if source_video.is_symlink() or not source_video.is_file():
                        raise ValueError(f"invalid source video: {source_video}")
                    destination_video.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source_video, destination_video)
                    video_count += 1
                episode_map.append(
                    {
                        "frame_start": global_frame,
                        "frame_stop": global_frame + frame_count,
                        "global_episode_index": global_episode,
                        "local_episode_index": local_episode,
                        "source_episode_sha256": source_hashes[local_episode],
                        "task": task,
                        "task_index": task_index,
                    }
                )
                global_episode += 1
                global_frame += frame_count
                local_frame += frame_count
            task_episode_counts[task] = expected_episodes
            task_frame_counts[task] = local_frame
            task_frame_ranges[task] = [task_start, global_frame]
            source_receipt_sha256[task] = file_sha256(
                task_root / "_robotactile_n0_vtla_qpos8_receipt.json"
            )
        if (
            base_info is None
            or global_episode != EXPECTED_EPISODES
            or global_frame != EXPECTED_FRAMES
        ):
            raise ValueError(
                f"mixed8 aggregate mismatch: episodes={global_episode}, frames={global_frame}"
            )
        if video_count != EXPECTED_EPISODES * 4:
            raise ValueError(f"mixed8 video count mismatch: {video_count}")
        base_info.update(
            {
                "splits": {"train": f"0:{EXPECTED_EPISODES}"},
                "total_chunks": math.ceil(
                    EXPECTED_EPISODES / int(base_info["chunks_size"])
                ),
                "total_episodes": EXPECTED_EPISODES,
                "total_frames": EXPECTED_FRAMES,
                "total_tasks": len(TASKS),
                "total_videos": video_count,
            }
        )
        if "data_files_size_in_mb" in base_info:
            base_info["data_files_size_in_mb"] = (
                sum(
                    path.stat().st_size
                    for path in (staging / "data").rglob("*.parquet")
                )
                / 1_000_000
            )
        if "video_files_size_in_mb" in base_info:
            base_info["video_files_size_in_mb"] = (
                sum(path.stat().st_size for path in (staging / "videos").rglob("*.mp4"))
                / 1_000_000
            )
        atomic_json(staging / "meta" / "info.json", base_info)
        _write_jsonl(
            staging / "meta" / "tasks.jsonl",
            [
                {"task_index": index, "task": task_prompts[task]}
                for index, task in enumerate(TASKS)
            ],
        )
        _write_jsonl(staging / "meta" / "episodes.jsonl", output_episodes)
        _write_jsonl(staging / "meta" / "episodes_stats.jsonl", output_stats)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "protocol_id": MIXED_VIEW_PROTOCOL,
            "status": "complete",
            "tasks": list(TASKS),
            "task_prompts": task_prompts,
            "task_episode_counts": task_episode_counts,
            "task_frame_counts": task_frame_counts,
            "task_frame_ranges": task_frame_ranges,
            "episode_count": EXPECTED_EPISODES,
            "frame_count": EXPECTED_FRAMES,
            "video_file_count": video_count,
            "episode_map": episode_map,
            "source_split": "train",
            "validation_data_used": False,
            "frozen_episode_count": 0,
            "source_manifest_sha256": manifest_sha,
            "source_view_receipt_sha256": source_receipt_sha256,
            "state_schema": "joint8_current_step",
            "action_schema": "joint8_absolute_next_step",
            "source_fps": 10,
            "parquet_storage": "rewritten_global_episode_index_index_task_index",
            "video_storage": "hardlink_to_certified_task_qpos8_views",
        }
        receipt["view_receipt_sha256"] = canonical_json_sha256(receipt)
        atomic_json(staging / MIXED_RECEIPT_NAME, receipt)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_mixed8_view(destination)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = materialize_mixed8(
        source_root=args.source_root, output_root=args.output_root
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
