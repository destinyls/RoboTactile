"""Verify that parsed UniVTAC camera and tactile arrays remain canonical RGB."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TypeAlias, cast

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

UInt8Array: TypeAlias = npt.NDArray[np.uint8]


def _load_modules(ftp1_root: Path) -> tuple[Any, ModuleType]:
    data_root = (ftp1_root / "data_processing").resolve(strict=True)
    sys.path.insert(0, str(data_root))
    from common.replay_buffer import ReplayBuffer  # type: ignore[import-not-found]
    from parse_data_module import parse_data_univtac  # type: ignore[import-not-found]

    return ReplayBuffer, parse_data_univtac


def _numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 0, path.name


def _sha256(array: UInt8Array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _raw_node(file: Any, options: tuple[str, ...]) -> Any:
    for option in options:
        node = file.get(option)
        if isinstance(node, h5py.Dataset):
            return node
    raise KeyError(f"none of the source keys exist: {options}")


def _reference(
    upstream: ModuleType,
    node: Any,
    frame_idx: int,
    image_size: int,
) -> UInt8Array:
    stream_to_img = cast(
        Callable[..., UInt8Array],
        upstream._stream_to_img,
    )
    return cast(
        UInt8Array,
        stream_to_img(
            node[frame_idx : frame_idx + 1],
            (image_size, image_size),
        )[0],
    )


def _compare(
    *,
    reference: UInt8Array,
    actual: UInt8Array,
    episode_idx: int,
    frame_idx: int,
    source: Path,
    stream: str,
) -> dict[str, object]:
    actual = cast(UInt8Array, np.asarray(actual))
    if reference.dtype != np.uint8 or actual.dtype != np.uint8:
        raise ValueError(f"{stream} is not uint8")
    if reference.shape != actual.shape:
        raise ValueError(
            f"{stream} shape mismatch: source={reference.shape}, zarr={actual.shape}"
        )
    delta = np.abs(reference.astype(np.int16) - actual.astype(np.int16))
    maximum = int(delta.max(initial=0))
    mean = float(delta.mean())
    if maximum != 0:
        raise ValueError(
            f"{stream} is not RGB-identical for episode={episode_idx}, "
            f"frame={frame_idx}: max_abs={maximum}, mean_abs={mean:.6f}"
        )
    return {
        "episode_index": episode_idx,
        "frame_index": frame_idx,
        "max_abs_diff": maximum,
        "mean_abs_diff": mean,
        "source_hdf5": str(source),
        "source_rgb_sha256": _sha256(reference),
        "stream": stream,
        "zarr_rgb_sha256": _sha256(actual),
    }


def verify(
    *,
    ftp1_root: Path,
    staging_root: Path,
    zarr_path: Path,
    task: str,
    image_size: int,
    use_wrist: bool,
) -> dict[str, object]:
    replay_buffer_type, upstream = _load_modules(ftp1_root)
    sources = sorted(
        (staging_root / task / "demo" / "hdf5").glob("*.hdf5"),
        key=_numeric_key,
    )
    replay = replay_buffer_type.create_from_path(str(zarr_path), mode="r")
    if len(sources) != replay.n_episodes:
        raise ValueError(
            f"source/Zarr episode mismatch: {len(sources)} != {replay.n_episodes}"
        )
    episode_indices = sorted({0, len(sources) // 2, len(sources) - 1})
    comparisons: list[dict[str, object]] = []
    for episode_idx in episode_indices:
        source = sources[episode_idx]
        episode = replay.get_episode(episode_idx)
        frame_count = int(episode["camera_ego_rgb"].shape[0])
        frame_indices = sorted({0, frame_count // 2, frame_count - 1})
        with h5py.File(source, "r") as file:
            nodes = {
                "camera_ego_rgb": _raw_node(file, ("observation/head/rgb",)),
                "left_tactile": _raw_node(
                    file,
                    (
                        "tactile/left_gsmini/rgb_marker",
                        "tactile/left_tactile/rgb_marker",
                    ),
                ),
                "right_tactile": _raw_node(
                    file,
                    (
                        "tactile/right_gsmini/rgb_marker",
                        "tactile/right_tactile/rgb_marker",
                    ),
                ),
            }
            if use_wrist:
                nodes["right_wrist_camera_rgb"] = _raw_node(
                    file, ("observation/wrist/rgb",)
                )
            for frame_idx in frame_indices:
                comparisons.append(
                    _compare(
                        reference=_reference(
                            upstream, nodes["camera_ego_rgb"], frame_idx, image_size
                        ),
                        actual=episode["camera_ego_rgb"][frame_idx],
                        episode_idx=episode_idx,
                        frame_idx=frame_idx,
                        source=source,
                        stream="camera_ego_rgb",
                    )
                )
                for tactile_idx, name in enumerate(("left_tactile", "right_tactile")):
                    comparisons.append(
                        _compare(
                            reference=_reference(
                                upstream, nodes[name], frame_idx, image_size
                            ),
                            actual=episode["right_tactile_data_gripper"][
                                frame_idx, tactile_idx
                            ],
                            episode_idx=episode_idx,
                            frame_idx=frame_idx,
                            source=source,
                            stream=name,
                        )
                    )
                if use_wrist:
                    comparisons.append(
                        _compare(
                            reference=_reference(
                                upstream,
                                nodes["right_wrist_camera_rgb"],
                                frame_idx,
                                image_size,
                            ),
                            actual=episode["right_wrist_camera_rgb"][frame_idx],
                            episode_idx=episode_idx,
                            frame_idx=frame_idx,
                            source=source,
                            stream="right_wrist_camera_rgb",
                        )
                    )
    return {
        "color_contract": "canonical_rgb_exact_v1",
        "comparison_count": len(comparisons),
        "episode_count": int(replay.n_episodes),
        "sampled_episode_indices": episode_indices,
        "status": "passed",
        "task_id": task,
        "comparisons": comparisons,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ftp1-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--zarr-path", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-wrist", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(
        json.dumps(
            verify(
                ftp1_root=args.ftp1_root,
                staging_root=args.staging_root,
                zarr_path=args.zarr_path,
                task=args.task,
                image_size=args.image_size,
                use_wrist=args.use_wrist,
            ),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
