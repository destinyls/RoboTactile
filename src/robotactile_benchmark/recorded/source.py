"""Strict read-only loading of recorded UniVTAC HDF5 episodes."""

from __future__ import annotations

import hashlib
import importlib
import io
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.adapters.univtac import ContactPhaseState
from robotactile_benchmark.backends.univtac_conversion import convert_raw_observation
from robotactile_benchmark.backends.univtac_registry import build_config
from robotactile_benchmark.contracts import (
    Array,
    ContactPhase,
    EvaluationRecord,
    canonical_hash,
    freeze_array,
)
from robotactile_benchmark.rest_references import (
    ReferenceSplit,
    RestReferenceBundle,
)

ACTION_HORIZON = 12
DEFAULT_CONTEXT_AFTER_ANCHOR = 32

_DATASETS = (
    "step",
    "embodiment/ee",
    "embodiment/joint",
    "observation/head/rgb",
    "observation/wrist/rgb",
    "tactile/left_gsmini/rgb_marker",
    "tactile/left_gsmini/depth",
    "tactile/right_gsmini/rgb_marker",
    "tactile/right_gsmini/depth",
)


class RecordedSourceError(ValueError):
    """A recorded episode violates the source or benchmark contract."""


@dataclass(frozen=True)
class RecordedEpisode:
    """Canonical records and expert EE8 targets from one HDF5 prefix."""

    source_path: Path
    source_sha256: str
    task_id: str
    episode_id: str
    initial_seed: int
    anchor_index: int
    rest_index: int
    total_record_count: int
    records: Tuple[EvaluationRecord, ...]
    native_steps: Tuple[int, ...]
    expert_actions: Array

    def __post_init__(self) -> None:
        actions = freeze_array(self.expert_actions, np.float32)
        if actions.shape != (ACTION_HORIZON, 8) or not np.isfinite(actions).all():
            raise RecordedSourceError("expert actions must be finite float32 [12,8]")
        if len(self.records) != len(self.native_steps):
            raise RecordedSourceError("record/native-step lengths do not match")
        if not 0 <= self.rest_index < len(self.records):
            raise RecordedSourceError("rest index is outside loaded records")
        if not 0 <= self.anchor_index < len(self.records):
            raise RecordedSourceError("anchor index is outside loaded records")
        object.__setattr__(self, "source_path", Path(self.source_path).absolute())
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "native_steps", tuple(self.native_steps))
        object.__setattr__(self, "expert_actions", actions)

    @property
    def anchor_record(self) -> EvaluationRecord:
        return self.records[self.anchor_index]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ImportError as error:
        raise RecordedSourceError(
            "recorded UniVTAC loading requires h5py in the selected runtime"
        ) from error


def _decode_rgb(value: object, name: str) -> Array:
    if isinstance(value, np.ndarray) and value.ndim == 3:
        decoded = np.asarray(value)
        if decoded.dtype != np.uint8 or decoded.shape[-1] != 3:
            raise RecordedSourceError(f"{name} raw image must be uint8 HWC RGB")
        return cast(Array, np.ascontiguousarray(decoded))
    if isinstance(value, np.ndarray) and value.ndim == 1:
        if value.dtype != np.uint8:
            raise RecordedSourceError(f"{name} encoded array must use uint8")
        raw = value.tobytes()
    elif isinstance(value, (np.bytes_, bytes, bytearray)):
        raw = bytes(value)
    else:
        raise RecordedSourceError(f"{name} must contain encoded image bytes")
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        try:
            image_module = importlib.import_module("PIL.Image")
            with image_module.open(io.BytesIO(raw)) as image:
                decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., ::-1]
        except (ImportError, OSError, ValueError) as error:
            raise RecordedSourceError(f"cannot decode {name}") from error
    else:
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RecordedSourceError(f"cannot decode {name}")
    if decoded.ndim != 3 or decoded.shape[-1] != 3 or decoded.dtype != np.uint8:
        raise RecordedSourceError(f"decoded {name} must be uint8 HWC")
    return cast(Array, np.ascontiguousarray(decoded))


def _dataset(root: Any, name: str) -> Any:
    try:
        return root[name]
    except KeyError as error:
        raise RecordedSourceError(
            f"required HDF5 dataset is missing: {name}"
        ) from error


def _validate_schema(root: Any) -> int:
    datasets = {name: _dataset(root, name) for name in _DATASETS}
    lengths = {int(value.shape[0]) for value in datasets.values()}
    if len(lengths) != 1:
        raise RecordedSourceError("HDF5 datasets do not share one record count")
    count = lengths.pop()
    if count < ACTION_HORIZON:
        raise RecordedSourceError("HDF5 episode is shorter than the N0 horizon")
    expected = {
        "embodiment/ee": (count, 7),
        "embodiment/joint": (count, 9),
        "tactile/left_gsmini/depth": (count, 240, 320),
        "tactile/right_gsmini/depth": (count, 240, 320),
    }
    for name, shape in expected.items():
        if tuple(datasets[name].shape) != shape:
            raise RecordedSourceError(f"HDF5 dataset shape mismatch: {name}")
    if datasets["step"].dtype != np.dtype(np.int64):
        raise RecordedSourceError("HDF5 step must use int64")
    for name in (
        "embodiment/ee",
        "embodiment/joint",
        "tactile/left_gsmini/depth",
        "tactile/right_gsmini/depth",
    ):
        if datasets[name].dtype != np.dtype(np.float32):
            raise RecordedSourceError(f"HDF5 dataset must use float32: {name}")
    for name in (
        "observation/head/rgb",
        "observation/wrist/rgb",
        "tactile/left_gsmini/rgb_marker",
        "tactile/right_gsmini/rgb_marker",
    ):
        dataset = datasets[name]
        encoded = dataset.dtype.kind in {"O", "S"}
        raw_hwc = (
            dataset.dtype == np.dtype(np.uint8)
            and len(dataset.shape) == 4
            and dataset.shape[-1] == 3
        )
        if not encoded and not raw_hwc:
            raise RecordedSourceError(f"HDF5 image dataset must contain bytes: {name}")
    return count


def _raw_frame(root: Any, index: int) -> Mapping[str, object]:
    return {
        "step": int(root["step"][index]),
        "observation": {
            "head": {"rgb": _decode_rgb(root["observation/head/rgb"][index], "head")},
            "wrist": {
                "rgb": _decode_rgb(root["observation/wrist/rgb"][index], "wrist")
            },
        },
        "tactile": {
            "left_tactile": {
                "rgb": _decode_rgb(
                    root["tactile/left_gsmini/rgb_marker"][index], "left tactile"
                ),
                "depth": np.asarray(
                    root["tactile/left_gsmini/depth"][index], dtype=np.float32
                ),
            },
            "right_tactile": {
                "rgb": _decode_rgb(
                    root["tactile/right_gsmini/rgb_marker"][index], "right tactile"
                ),
                "depth": np.asarray(
                    root["tactile/right_gsmini/depth"][index], dtype=np.float32
                ),
            },
        },
        "embodiment": {
            "ee": np.asarray(root["embodiment/ee"][index], dtype=np.float32),
            "joint": np.asarray(root["embodiment/joint"][index], dtype=np.float32),
        },
    }


def load_univtac_hdf5_episode(
    path: Path,
    *,
    task_id: str,
    episode_id: str,
    initial_seed: int,
    anchor_index: int,
    rest_index: int,
    stop_index: Optional[int] = None,
) -> RecordedEpisode:
    """Load a source-hashed HDF5 prefix through the canonical UniVTAC converter."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise RecordedSourceError("HDF5 source must be a non-symlink regular file")
    if initial_seed < 0 or anchor_index < 0 or rest_index < 0:
        raise RecordedSourceError("seed and indices must be non-negative")
    config = build_config(task_id, action_spec=EE8_ACTION_SPEC)
    handshake = config.expected_handshake(config.canonical_joint_names)
    h5py = _h5py()
    source_before = source.stat()
    records: list[EvaluationRecord] = []
    native_steps: list[int] = []
    expert: Array
    with h5py.File(source, "r") as root:
        total_count = _validate_schema(root)
        minimum_stop = anchor_index + ACTION_HORIZON
        selected_stop = stop_index or min(
            total_count, anchor_index + DEFAULT_CONTEXT_AFTER_ANCHOR
        )
        if (
            minimum_stop > total_count
            or selected_stop < minimum_stop
            or selected_stop > total_count
            or rest_index >= selected_stop
        ):
            raise RecordedSourceError("requested HDF5 range is outside the episode")
        states = {
            "left": ContactPhaseState(False),
            "right": ContactPhaseState(False),
        }
        for index in range(selected_stop):
            converted = convert_raw_observation(
                _raw_frame(root, index),
                config=config,
                handshake=handshake,
                phase_states=states,
                episode_id=episode_id,
                task_id=task_id,
                initial_seed=initial_seed,
                benchmark_step=index,
            )
            records.append(converted.record)
            native_steps.append(converted.native_step_id)
            states = converted.phase_state_mapping()
        ee = np.asarray(
            root["embodiment/ee"][anchor_index:minimum_stop], dtype=np.float32
        )
        joint = np.asarray(
            root["embodiment/joint"][anchor_index:minimum_stop], dtype=np.float32
        )
        expert = np.concatenate((ee, joint[:, -2:-1]), axis=1).astype(
            np.float32, copy=False
        )
    source_sha256 = _file_sha256(source)
    source_after = source.stat()
    if (
        source_before.st_dev,
        source_before.st_ino,
        source_before.st_size,
        source_before.st_mtime_ns,
    ) != (
        source_after.st_dev,
        source_after.st_ino,
        source_after.st_size,
        source_after.st_mtime_ns,
    ):
        raise RecordedSourceError("HDF5 source changed while it was being loaded")
    return RecordedEpisode(
        source_path=source,
        source_sha256=source_sha256,
        task_id=task_id,
        episode_id=episode_id,
        initial_seed=initial_seed,
        anchor_index=anchor_index,
        rest_index=rest_index,
        total_record_count=total_count,
        records=tuple(records),
        native_steps=tuple(native_steps),
        expert_actions=expert,
    )


def build_recorded_rest_references(episode: RecordedEpisode) -> RestReferenceBundle:
    """Build development-only rest inputs from an explicitly free source frame."""

    record = episode.records[episode.rest_index]
    if any(
        record.provenance_for(slot).phase is not ContactPhase.FREE
        for slot in ("left", "right")
    ):
        raise RecordedSourceError("selected rest frame is not free for both sensors")
    source_identity = {
        "source_sha256": episode.source_sha256,
        "episode_id": episode.episode_id,
        "rest_index": episode.rest_index,
        "native_step": episode.native_steps[episode.rest_index],
    }
    source_hash = canonical_hash(source_identity)
    return RestReferenceBundle(
        reference_id=f"recorded-rest-{source_hash[:16]}",
        dataset_split=ReferenceSplit.DEVELOPMENT,
        split_manifest_sha256=canonical_hash(
            {"source_sha256": episode.source_sha256, "split": "development"}
        ),
        source_artifact_sha256=episode.source_sha256,
        no_contact_predicate_id="univtac-depth-hysteresis-both-free-v1",
        no_contact_validation_sha256=canonical_hash(
            {**source_identity, "phase": "free", "verified": True}
        ),
        no_contact_verified=True,
        qualified_record_ids={
            slot: f"{episode.episode_id}:{episode.rest_index}:{slot}"
            for slot in ("left", "right")
        },
        calibration_sha256={
            slot: record.provenance_for(slot).calibration_sha256
            for slot in ("left", "right")
        },
        payloads={
            slot: record.observation.sensor(slot).payload for slot in ("left", "right")
        },
    )


def retarget_recorded_episode(
    episode: RecordedEpisode, anchor_index: int
) -> RecordedEpisode:
    """Reuse one loaded source prefix at another full-horizon EE8 anchor."""

    stop = anchor_index + ACTION_HORIZON
    if anchor_index < 0 or stop > len(episode.records):
        raise RecordedSourceError("retargeted anchor lacks a complete 12-step horizon")
    actions = np.stack(
        [record.observation.proprio for record in episode.records[anchor_index:stop]]
    ).astype(np.float32, copy=False)
    return replace(
        episode,
        anchor_index=anchor_index,
        expert_actions=actions,
    )
