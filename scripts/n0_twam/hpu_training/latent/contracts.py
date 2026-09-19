"""Train759 identity and 16-worker latent preprocessing contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

TARGET_FPS = 10
EXPECTED_EPISODES = 759
ACTION_SCHEMA = "ee20_absolute_next_step"
USED_ACTION_CHANNEL_IDS = tuple(range(10))
OFFICIAL_VIDEO_ENCODER_SHA256 = (
    "c731103d95059b5ad769a0f28d2b68ba664c02f3b95c4557dbc632d27a6d14ff"
)
OFFICIAL_TACTILE_ENCODER_SHA256 = (
    "bd0975bf9330f60f6cd16adb9fbd28b6f96b577d09cbf06f0654ed3348eb7b79"
)
VIDEO_KEYS = (
    "observation.images.top",
    "observation.images.wrist_l",
)
TACTILE_KEYS = (
    "observation.images.tactile_a",
    "observation.images.tactile_b",
)
OBSERVATION_KEYS = VIDEO_KEYS + TACTILE_KEYS
TASKS = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)
Kind = Literal["vision", "tactile"]


@dataclass(frozen=True)
class LatentSpec:
    dataset_root: Path
    source_manifest_path: Path
    conversion_receipt_path: Path
    official_repo: Path
    model_path: Path


@dataclass(frozen=True)
class EpisodeRecord:
    task: str
    repo_path: Path
    lerobot_episode_index: int
    source_episode_id: int
    source_relative_path: str
    source_sha256: str
    length: int


@dataclass(frozen=True)
class WorkUnit:
    worker_id: int
    node: str
    local_device: int
    kind: Kind
    task: str
    repo_path: Path
    episodes: tuple[EpisodeRecord, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "node": self.node,
            "local_device": self.local_device,
            "kind": self.kind,
            "task": self.task,
            "repo_path": str(self.repo_path),
            "episode_ids": [episode.lerobot_episode_index for episode in self.episodes],
            "source_episode_ids": [
                episode.source_episode_id for episode in self.episodes
            ],
            "episodes": [
                {
                    "lerobot_episode_index": episode.lerobot_episode_index,
                    "source_episode_id": episode.source_episode_id,
                    "source_relative_path": episode.source_relative_path,
                    "source_sha256": episode.source_sha256,
                    "length": episode.length,
                }
                for episode in self.episodes
            ],
        }


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _absolute_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute: {value!r}")
    return path


def load_latent_spec(path: Path) -> LatentSpec:
    payload = _load_object(path, label="latent spec")
    allowed = {
        "dataset_root",
        "source_manifest_path",
        "conversion_receipt_path",
        "official_repo",
        "model_path",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported latent spec keys: {unknown}")
    dataset_root = _absolute_path(payload.get("dataset_root"), field="dataset_root")
    if dataset_root.name != "train759" or "frozen40" in dataset_root.parts:
        raise ValueError("dataset_root must point exactly at train759")
    return LatentSpec(
        dataset_root=dataset_root,
        source_manifest_path=_absolute_path(
            payload.get("source_manifest_path"), field="source_manifest_path"
        ),
        conversion_receipt_path=_absolute_path(
            payload.get("conversion_receipt_path"), field="conversion_receipt_path"
        ),
        official_repo=_absolute_path(
            payload.get("official_repo"), field="official_repo"
        ),
        model_path=_absolute_path(payload.get("model_path"), field="model_path"),
    )


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_encoders(repo: Path) -> dict[str, str]:
    scripts = {
        "vision": repo / "script" / "encode_lerobot_n0_latents.py",
        "tactile": repo / "script" / "encode_tactile_latent.py",
    }
    expected = {
        "vision": OFFICIAL_VIDEO_ENCODER_SHA256,
        "tactile": OFFICIAL_TACTILE_ENCODER_SHA256,
    }
    actual: dict[str, str] = {}
    for kind, script in scripts.items():
        if not script.is_file():
            raise FileNotFoundError(script)
        actual[kind] = sha256_file(script)
        if actual[kind] != expected[kind]:
            raise ValueError(f"official {kind} encoder digest mismatch: {actual[kind]}")
    return actual


def _validate_manifest(
    path: Path,
) -> tuple[dict[tuple[str, int], dict[str, object]], str]:
    manifest = _load_object(path, label="source split manifest")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 800:
        raise ValueError("source manifest must contain exactly 800 episodes")
    recorded_sha = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    actual_sha = canonical_sha256(unhashed)
    if recorded_sha != actual_sha:
        raise ValueError("source manifest SHA256 mismatch")
    if manifest.get("source_fps") != TARGET_FPS:
        raise ValueError("source manifest must certify source_fps=10")
    if manifest.get("action_per_frame") != 4:
        raise ValueError("source manifest must certify action_per_frame=4")
    expected_contract: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": "univtac_train759_absee20_v1",
        "action_schema": ACTION_SCHEMA,
        "quaternion_order": "wxyz",
        "gripper_source": "embodiment/joint[:,7]",
        "used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
        "split_counts": {"train": 759, "frozen": 40, "quarantine": 1},
        "materialized_splits": ["train"],
    }
    if any(manifest.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("source manifest protocol contract mismatch")

    lookup: dict[tuple[str, int], dict[str, object]] = {}
    split_counts = {"train": 0, "frozen": 0, "quarantine": 0}
    for raw in episodes:
        if not isinstance(raw, dict):
            raise ValueError("source manifest episode entries must be objects")
        task = raw.get("task")
        episode_id = raw.get("episode_id")
        split = raw.get("split")
        if (
            task not in TASKS
            or isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
        ):
            raise ValueError("invalid source episode identity")
        if split not in split_counts:
            raise ValueError(f"invalid source split: {split!r}")
        key = (str(task), episode_id)
        if key in lookup:
            raise ValueError(f"duplicate source episode: {key}")
        lookup[key] = raw
        split_counts[str(split)] += 1
    if split_counts != {"train": 759, "frozen": 40, "quarantine": 1}:
        raise ValueError(f"source split counts are invalid: {split_counts}")
    return lookup, actual_sha


def _load_episode_lengths(repo_path: Path) -> dict[int, int]:
    path = repo_path / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    lengths: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"invalid episode metadata entry: {path}")
        episode_id = raw.get("episode_index")
        length = raw.get("length")
        if (
            isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
        ):
            raise ValueError(f"invalid episode index/length in {path}")
        if episode_id in lengths:
            raise ValueError(f"duplicate LeRobot episode index in {path}: {episode_id}")
        lengths[episode_id] = length
    return lengths


def _verify_signed(payload: Mapping[str, object], *, field: str, label: str) -> str:
    unsigned = dict(payload)
    claimed = unsigned.pop(field, None)
    actual = canonical_sha256(unsigned)
    if claimed != actual:
        raise ValueError(f"{label} SHA256 mismatch")
    return actual


def _numeric_value(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def load_certified_train759(spec: LatentSpec) -> tuple[EpisodeRecord, ...]:
    """Bind every dense LeRobot episode to one source-manifest train identity."""
    for path in (
        spec.dataset_root,
        spec.source_manifest_path,
        spec.conversion_receipt_path,
        spec.official_repo,
        spec.model_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    output_root = spec.dataset_root.parent.resolve(strict=True)
    if (
        spec.source_manifest_path.resolve(strict=True)
        != output_root / "source_split_manifest.json"
    ):
        raise ValueError("source manifest must be the conversion root manifest")
    if (
        spec.conversion_receipt_path.resolve(strict=True)
        != output_root / "train759_receipt.json"
    ):
        raise ValueError("conversion receipt must be the conversion root receipt")
    source_lookup, manifest_sha = _validate_manifest(spec.source_manifest_path)
    conversion = _load_object(spec.conversion_receipt_path, label="conversion receipt")
    _verify_signed(conversion, field="receipt_sha256", label="conversion receipt")
    expected_conversion: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": "univtac_train759_absee20_v1",
        "dataset_path": str(spec.dataset_root.resolve(strict=True)),
        "source_manifest_relative_path": "source_split_manifest.json",
        "source_manifest_sha256": manifest_sha,
        "source_split_counts": {"train": 759, "frozen": 40, "quarantine": 1},
        "materialized_split": "train",
        "materialized_episode_count": EXPECTED_EPISODES,
        "excluded_episode_count": 41,
        "completed_task_count": len(TASKS),
        "source_fps": TARGET_FPS,
        "action_per_frame": 4,
        "action_schema": ACTION_SCHEMA,
        "quaternion_order": "wxyz",
        "gripper_source": "embodiment/joint[:,7]",
        "used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
        "validation_dataset_path": None,
        "materialized_splits": ["train"],
    }
    if any(conversion.get(key) != value for key, value in expected_conversion.items()):
        raise ValueError("conversion receipt contract mismatch")
    normalization = conversion.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("conversion normalization receipt is missing")
    for name in ("norm_stat_path", "per_repo_norm_stat_path", "raw_report_path"):
        relative = normalization.get(name)
        expected_sha = normalization.get(f"{name}_sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected_sha, str)
        ):
            raise ValueError(f"invalid conversion normalization artifact: {name}")
        artifact = output_root / relative
        if not artifact.is_file() or sha256_file(artifact) != expected_sha:
            raise ValueError(f"conversion normalization artifact mismatch: {name}")
    raw_repos = conversion.get("task_repos")
    if not isinstance(raw_repos, list) or len(raw_repos) != len(TASKS):
        raise ValueError("conversion receipt task_repos is incomplete")
    repo_receipts: dict[str, dict[str, object]] = {}
    for raw in raw_repos:
        if not isinstance(raw, dict) or raw.get("task") not in TASKS:
            raise ValueError("conversion receipt task repo entry is invalid")
        task_name = str(raw["task"])
        if task_name in repo_receipts:
            raise ValueError(f"duplicate task repo receipt: {task_name}")
        repo_receipts[task_name] = raw

    records: list[EpisodeRecord] = []
    seen_sources: set[tuple[str, int]] = set()
    actual_directories = {
        path.name
        for path in spec.dataset_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    if actual_directories != set(TASKS):
        raise ValueError(
            f"train759 task repositories are invalid: {actual_directories}"
        )
    for task in TASKS:
        repo_path = spec.dataset_root / task
        info = _load_object(repo_path / "meta" / "info.json", label=f"{task} info")
        if (
            int(round(_numeric_value(info.get("fps"), label=f"{task} fps")))
            != TARGET_FPS
        ):
            raise ValueError(f"{task} must preserve real 10 Hz metadata")
        lengths = _load_episode_lengths(repo_path)
        task_receipt = _load_object(
            repo_path / "_robotactile_task_receipt.json",
            label=f"{task} conversion receipt",
        )
        task_digest = _verify_signed(
            task_receipt, field="task_receipt_sha256", label=f"{task} task receipt"
        )
        expected_repo_path = f"train759/{task}"
        expected_task: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "protocol_id": "univtac_train759_absee20_task_v1",
            "task": task,
            "repo_relative_path": expected_repo_path,
            "physical_repo_basename": task,
            "per_repo_norm_key": task,
            "source_manifest_sha256": manifest_sha,
            "source_split": "train",
            "source_fps": TARGET_FPS,
            "action_per_frame": 4,
            "action_schema": ACTION_SCHEMA,
            "quaternion_order": "wxyz",
            "gripper_source": "embodiment/joint[:,7]",
            "used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
            "observation_keys": list(OBSERVATION_KEYS),
        }
        if any(task_receipt.get(key) != value for key, value in expected_task.items()):
            raise ValueError(f"task receipt contract mismatch for {task}")
        global_repo = repo_receipts[task]
        expected_global_repo: dict[str, object] = {
            "task": task,
            "repo_relative_path": expected_repo_path,
            "physical_repo_basename": task,
            "per_repo_norm_key": task,
            "task_receipt_relative_path": (
                f"{expected_repo_path}/_robotactile_task_receipt.json"
            ),
            "task_receipt_sha256": task_digest,
            "episode_count": len(lengths),
            "frame_count": sum(lengths.values()),
        }
        if any(
            global_repo.get(key) != value for key, value in expected_global_repo.items()
        ):
            raise ValueError(f"global task repo contract mismatch for {task}")
        if task_receipt.get("episode_count") != len(lengths):
            raise ValueError(f"task episode count mismatch for {task}")
        if task_receipt.get("frame_count") != sum(lengths.values()):
            raise ValueError(f"task frame count mismatch for {task}")
        episode_map = task_receipt.get("episode_map")
        if not isinstance(episode_map, list):
            raise ValueError(f"task receipt has no episode_map: {task}")
        local_ids: list[int] = []
        for raw in episode_map:
            if not isinstance(raw, dict):
                raise ValueError(f"invalid episode_map entry for {task}")
            local_id = raw.get("lerobot_episode_index")
            source_id = raw.get("source_episode_id")
            if (
                isinstance(local_id, bool)
                or not isinstance(local_id, int)
                or isinstance(source_id, bool)
                or not isinstance(source_id, int)
            ):
                raise ValueError(f"invalid mapped episode identity for {task}")
            source = source_lookup.get((task, source_id))
            if source is None or source.get("split") != "train":
                raise ValueError(
                    f"refusing frozen/quarantine episode: {task}/{source_id}"
                )
            if raw.get("source_relative_path") != source.get("relative_path"):
                raise ValueError(f"source path mismatch for {task}/{source_id}")
            if raw.get("source_sha256") != source.get("sha256"):
                raise ValueError(f"source SHA mismatch for {task}/{source_id}")
            if raw.get("usable_source_range") != source.get("usable_source_range"):
                raise ValueError(f"source range mismatch for {task}/{source_id}")
            if local_id not in lengths:
                raise ValueError(f"missing episode metadata for {task}/{local_id}")
            if raw.get("converted_frame_count") != lengths[local_id]:
                raise ValueError(
                    f"converted frame count mismatch for {task}/{local_id}"
                )
            source_key = (task, source_id)
            if source_key in seen_sources:
                raise ValueError(f"duplicate materialized source: {source_key}")
            seen_sources.add(source_key)
            local_ids.append(local_id)
            records.append(
                EpisodeRecord(
                    task=task,
                    repo_path=repo_path,
                    lerobot_episode_index=local_id,
                    source_episode_id=source_id,
                    source_relative_path=str(source["relative_path"]),
                    source_sha256=str(source["sha256"]),
                    length=lengths[local_id],
                )
            )
        if sorted(local_ids) != list(range(len(local_ids))):
            raise ValueError(f"LeRobot episode IDs are not dense for {task}")
        if set(local_ids) != set(lengths):
            raise ValueError(f"episode_map does not exactly cover {task} metadata")

    expected_train = {
        key for key, value in source_lookup.items() if value.get("split") == "train"
    }
    if len(records) != EXPECTED_EPISODES or seen_sources != expected_train:
        raise ValueError("train759 does not exactly cover all 759 certified sources")
    if conversion.get("frame_count") != sum(record.length for record in records):
        raise ValueError("global converted frame count mismatch")
    return tuple(
        sorted(
            records,
            key=lambda item: (TASKS.index(item.task), item.lerobot_episode_index),
        )
    )


def build_work_units(
    records: Sequence[EpisodeRecord], nodes: Sequence[str]
) -> tuple[WorkUnit, ...]:
    """Create 8 task x 2 modality units, balanced 4+4 per formal node."""
    if len(nodes) != 2 or len(set(nodes)) != 2 or any(not node for node in nodes):
        raise ValueError("latent preprocessing requires two explicit unique nodes")
    by_task = {
        task: tuple(record for record in records if record.task == task)
        for task in TASKS
    }
    if any(not episodes for episodes in by_task.values()):
        raise ValueError("every task must contain certified episodes")
    raw_units: list[tuple[int, Kind, str, tuple[EpisodeRecord, ...]]] = []
    for task_index, task in enumerate(TASKS):
        raw_units.append((task_index % 2, "vision", task, by_task[task]))
        raw_units.append(((task_index + 1) % 2, "tactile", task, by_task[task]))
    units: list[WorkUnit] = []
    for node_index, node in enumerate(nodes):
        node_units = [unit for unit in raw_units if unit[0] == node_index]
        if len(node_units) != 8:
            raise RuntimeError("formal node must receive exactly eight workers")
        for local_device, (_, kind, task, episodes) in enumerate(node_units):
            units.append(
                WorkUnit(
                    worker_id=node_index * 8 + local_device,
                    node=node,
                    local_device=local_device,
                    kind=kind,
                    task=task,
                    repo_path=episodes[0].repo_path,
                    episodes=episodes,
                )
            )
    return tuple(units)
