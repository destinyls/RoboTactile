"""Materialize train-only N0-VTLA qpos8 views without re-encoding video.

The certified N0-TWAM train759 repositories already contain source-aligned,
validated AV1 video.  This tool hardlinks those immutable video files and only
rewrites the small state/action Parquet columns and corresponding metadata.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .contracts import (
    TASKS,
    SourceRecord,
    atomic_json,
    canonical_json_sha256,
    file_sha256,
    load_certified_manifest,
    select_train_records,
)
from .qpos8 import load_episode_qpos8, vector_stats

VIEW_PROTOCOL = "robotactile-n0-vtla-univtac-qpos8-train-only-v1"
QPOS_NAMES = [f"panda_joint{index}" for index in range(1, 8)] + ["panda_finger_joint1"]


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _verify_source_receipt(
    source_repo: Path,
    *,
    task: str,
    manifest_sha256: str,
    records: Sequence[SourceRecord],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = source_repo / "_robotactile_task_receipt.json"
    receipt = _json(path)
    unsigned = dict(receipt)
    claimed = unsigned.pop("task_receipt_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise ValueError(f"source task receipt digest mismatch: {task}")
    episode_map = receipt.get("episode_map")
    if (
        receipt.get("status") != "complete"
        or receipt.get("task") != task
        or receipt.get("source_split") != "train"
        or receipt.get("source_manifest_sha256") != manifest_sha256
        or receipt.get("episode_count") != len(records)
        or not isinstance(episode_map, list)
        or len(episode_map) != len(records)
    ):
        raise ValueError(f"source task receipt contract mismatch: {task}")
    for index, (item, record) in enumerate(zip(episode_map, records)):
        expected = {
            "lerobot_episode_index": index,
            "source_episode_id": record.episode_id,
            "source_relative_path": record.relative_path,
            "source_sha256": record.sha256,
            "usable_source_range": (
                None
                if record.usable_source_range is None
                else list(record.usable_source_range)
            ),
        }
        if not isinstance(item, dict) or any(
            item.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"source episode map mismatch: {task}/{index}")
        frame_count = item.get("converted_frame_count")
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count <= 0
        ):
            raise ValueError(f"invalid converted frame count: {task}/{index}")
    return receipt, episode_map


def _fixed_qpos8(values: npt.NDArray[np.float32]) -> Any:
    pa: Any = importlib.import_module("pyarrow")

    flat = pa.array(np.ascontiguousarray(values, dtype=np.float32).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, 8)


def _rewrite_parquet(
    source: Path,
    destination: Path,
    *,
    state: npt.NDArray[np.float32],
    action: npt.NDArray[np.float32],
    episode_index: int,
) -> None:
    pq: Any = importlib.import_module("pyarrow.parquet")

    table = pq.read_table(source)
    if len(table) != len(state) or state.shape != action.shape:
        raise ValueError(f"source Parquet/HDF5 length mismatch: {source}")
    required = {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    if set(table.column_names) != required:
        raise ValueError(f"source Parquet columns changed: {source}")
    frame_index = np.asarray(table["frame_index"].to_numpy())
    episode_values = np.asarray(table["episode_index"].to_numpy())
    task_values = np.asarray(table["task_index"].to_numpy())
    timestamps = np.asarray(table["timestamp"].to_numpy())
    if (
        not np.array_equal(frame_index, np.arange(len(state)))
        or not np.all(episode_values == episode_index)
        or not np.all(task_values == 0)
        or not np.allclose(timestamps, np.arange(len(state)) / 10.0, atol=1e-5)
    ):
        raise ValueError(f"source Parquet temporal contract changed: {source}")
    table = table.set_column(
        table.schema.get_field_index("observation.state"),
        "observation.state",
        _fixed_qpos8(state),
    )
    table = table.set_column(
        table.schema.get_field_index("action"), "action", _fixed_qpos8(action)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(None), destination, compression="zstd")


def _link_video_tree(source: Path, destination: Path) -> int:
    if not source.is_dir():
        raise ValueError(f"source video directory missing: {source}")
    count = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            raise ValueError(f"video source may not be a symlink: {path}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            raise ValueError(f"unsupported video source entry: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        count += 1
    return count


def _rewrite_info(source_repo: Path, staging: Path) -> dict[str, Any]:
    info = _json(source_repo / "meta" / "info.json")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("source info has no feature mapping")
    for key in ("observation.state", "action"):
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("shape") != [20]:
            raise ValueError(f"unexpected source feature schema: {key}")
        feature["shape"] = [8]
        feature["names"] = QPOS_NAMES
    atomic_json(staging / "meta" / "info.json", info)
    return info


def _rewrite_episode_stats(
    source_repo: Path,
    staging: Path,
    qpos_by_episode: Mapping[
        int, tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    ],
) -> None:
    source = source_repo / "meta" / "episodes_stats.jsonl"
    destination = staging / "meta" / "episodes_stats.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open(encoding="utf-8") as input_stream,
        destination.open("x", encoding="utf-8") as output_stream,
    ):
        seen: set[int] = set()
        for line in input_stream:
            row = json.loads(line)
            episode_index = row.get("episode_index")
            stats = row.get("stats")
            if episode_index not in qpos_by_episode or not isinstance(stats, dict):
                raise ValueError("source episode stats do not match episode map")
            state, action = qpos_by_episode[int(episode_index)]
            stats["observation.state"] = vector_stats(state)
            stats["action"] = vector_stats(action)
            output_stream.write(
                json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            )
            seen.add(int(episode_index))
        if seen != set(qpos_by_episode):
            raise ValueError("source episode stats are incomplete")


def _source_parquet(
    source_repo: Path, info: Mapping[str, Any], episode_index: int
) -> Path:
    template = info.get("data_path")
    chunks_size = info.get("chunks_size", 1000)
    if not isinstance(template, str) or not isinstance(chunks_size, int):
        raise ValueError("source data path template is invalid")
    return source_repo / template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def verify_task_view(root: Path, *, task: str, manifest_sha256: str) -> dict[str, Any]:
    """Validate the published qpos8 view consumed by official N0-VTLA."""

    pq: Any = importlib.import_module("pyarrow.parquet")

    receipt = _json(root / "_robotactile_n0_vtla_qpos8_receipt.json")
    unsigned = dict(receipt)
    claimed = unsigned.pop("view_receipt_sha256", None)
    if (
        claimed != canonical_json_sha256(unsigned)
        or receipt.get("status") != "complete"
        or receipt.get("protocol_id") != VIEW_PROTOCOL
        or receipt.get("task") != task
        or receipt.get("source_manifest_sha256") != manifest_sha256
        or receipt.get("source_split") != "train"
    ):
        raise ValueError(f"qpos8 view receipt mismatch: {root}")
    info = _json(root / "meta" / "info.json")
    for key in ("observation.state", "action"):
        if info["features"][key].get("shape") != [8]:
            raise ValueError(f"qpos8 info feature mismatch: {key}")
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    episode_count = receipt.get("episode_count")
    if len(parquet_files) != episode_count:
        raise ValueError("qpos8 Parquet episode count mismatch")
    for path in parquet_files:
        schema = pq.read_schema(path)
        for key in ("observation.state", "action"):
            if schema.field(key).type.list_size != 8:
                raise ValueError(f"qpos8 Parquet schema mismatch: {path}")
    video_files = sorted((root / "videos").rglob("*.mp4"))
    if len(video_files) != int(episode_count) * 4:
        raise ValueError("qpos8 view must contain four videos per train episode")
    return receipt


def materialize_task(
    *,
    task: str,
    records: Sequence[SourceRecord],
    manifest_sha256: str,
    source_train_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    destination = output_root / task
    if destination.exists():
        return verify_task_view(destination, task=task, manifest_sha256=manifest_sha256)
    source_repo = source_train_root / task
    source_receipt, episode_map = _verify_source_receipt(
        source_repo,
        task=task,
        manifest_sha256=manifest_sha256,
        records=records,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{task}.", dir=output_root))
    try:
        info = _rewrite_info(source_repo, staging)
        for name in ("tasks.jsonl", "episodes.jsonl"):
            shutil.copy2(source_repo / "meta" / name, staging / "meta" / name)
        video_count = _link_video_tree(source_repo / "videos", staging / "videos")
        qpos_by_episode: dict[
            int, tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        ] = {}
        total_frames = 0
        for item, record in zip(episode_map, records):
            episode_index = int(item["lerobot_episode_index"])
            state, action = load_episode_qpos8(record)
            expected_frames = int(item["converted_frame_count"])
            if len(state) != expected_frames:
                raise ValueError(f"qpos8/source frame mismatch: {record.relative_path}")
            source_parquet = _source_parquet(source_repo, info, episode_index)
            destination_parquet = staging / source_parquet.relative_to(source_repo)
            _rewrite_parquet(
                source_parquet,
                destination_parquet,
                state=state,
                action=action,
                episode_index=episode_index,
            )
            qpos_by_episode[episode_index] = (state, action)
            total_frames += len(state)
        _rewrite_episode_stats(source_repo, staging, qpos_by_episode)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "protocol_id": VIEW_PROTOCOL,
            "status": "complete",
            "task": task,
            "source_split": "train",
            "validation_data_used": False,
            "source_manifest_sha256": manifest_sha256,
            "source_task_receipt_sha256": file_sha256(
                source_repo / "_robotactile_task_receipt.json"
            ),
            "source_repo": str(source_repo.resolve(strict=True)),
            "episode_count": len(records),
            "frame_count": total_frames,
            "state_schema": "joint8_current_step",
            "action_schema": "joint8_absolute_next_step",
            "joint_projection": "embodiment/joint[:8]",
            "source_fps": 10,
            "video_storage": "hardlink_to_certified_train759",
            "video_file_count": video_count,
            "episode_source_sha256": [record.sha256 for record in records],
            "source_task_protocol": source_receipt["protocol_id"],
        }
        receipt["view_receipt_sha256"] = canonical_json_sha256(receipt)
        atomic_json(staging / "_robotactile_n0_vtla_qpos8_receipt.json", receipt)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_task_view(destination, task=task, manifest_sha256=manifest_sha256)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-train-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_sha, all_records = load_certified_manifest(
        args.source_manifest, raw_root=args.raw_root
    )
    results = {
        task: materialize_task(
            task=task,
            records=select_train_records(all_records, task=task),
            manifest_sha256=manifest_sha,
            source_train_root=args.source_train_root,
            output_root=args.output_root,
        )
        for task in args.tasks
    }
    print(json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
