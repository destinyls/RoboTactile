"""Compare eager and lazy Franka loaders on one real episode per UniVTAC task."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import torch  # type: ignore[import-not-found]

TASKS: Final[tuple[str, ...]] = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)
VIDEO_KEYS: Final[tuple[str, ...]] = (
    "images",
    "wrist_images",
    "tactile_left_images",
    "tactile_right_images",
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_equal(left: Any, right: Any, location: str) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(left, right)
        ):
            raise AssertionError(f"tensor mismatch: {location}")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left, right)
        ):
            raise AssertionError(f"array mismatch: {location}")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise AssertionError(f"mapping keys mismatch: {location}")
        for key in left:
            _assert_equal(left[key], right[key], f"{location}.{key}")
        return
    if type(left) is not type(right) or left != right:
        raise AssertionError(f"value mismatch: {location}: {left!r} != {right!r}")


def _episode_for_task(dataset_root: Path, task: str) -> Path:
    matches = sorted((dataset_root / "train" / task).glob("episode_*/*.hdf5"))
    if not matches:
        raise FileNotFoundError(f"no train episode found for {task}")
    return matches[0]


def _make_dataset(
    module: ModuleType,
    *,
    dataset_root: Path,
    episode_path: Path,
    t5_path: Path,
    use_image_aug: bool,
) -> Any:
    module.get_hdf5_files = lambda *_args, **_kwargs: [str(episode_path)]  # type: ignore[attr-defined]
    return module.FrankaDataset(
        data_dir=str(dataset_root),
        is_train=True,
        chunk_size=8,
        final_image_size=224,
        t5_text_embeddings_path=str(t5_path),
        normalize_images=False,
        normalize_actions=True,
        normalize_proprio=True,
        use_image_aug=use_image_aug,
        use_stronger_image_aug=True,
        use_wrist_images=True,
        use_third_person_images=True,
        use_proprio=True,
        num_duplicates_per_image=4,
        rollout_data_dir="",
        return_value_function_returns=True,
        use_tactile=True,
        use_tactile_image_aug=True,
    )


def _indices(num_steps: int, chunk_size: int) -> tuple[int, ...]:
    return tuple(
        sorted({0, num_steps // 2, max(0, num_steps - chunk_size), num_steps - 1})
    )


def _compare_task(
    *,
    task: str,
    dataset_root: Path,
    eager_module: ModuleType,
    lazy_module: ModuleType,
) -> dict[str, object]:
    episode_path = _episode_for_task(dataset_root, task)
    t5_path = dataset_root / "t5_embeddings.pkl"
    eager = _make_dataset(
        eager_module,
        dataset_root=dataset_root,
        episode_path=episode_path,
        t5_path=t5_path,
        use_image_aug=False,
    )
    lazy = _make_dataset(
        lazy_module,
        dataset_root=dataset_root,
        episode_path=episode_path,
        t5_path=t5_path,
        use_image_aug=False,
    )
    try:
        scalar_fields: Sequence[str] = ("num_episodes", "num_steps", "use_tactile")
        for field in scalar_fields:
            _assert_equal(getattr(eager, field), getattr(lazy, field), field)
        _assert_equal(eager.dataset_stats, lazy.dataset_stats, "dataset_stats")
        _assert_equal(
            eager.dataset_stats_post_norm,
            lazy.dataset_stats_post_norm,
            "dataset_stats_post_norm",
        )
        episode_eager = eager.data[0]
        episode_lazy = lazy.data[0]
        _assert_equal(episode_eager["actions"], episode_lazy["actions"], "actions")
        _assert_equal(episode_eager["proprio"], episode_lazy["proprio"], "proprio")
        indices = _indices(eager.num_steps, eager.chunk_size)
        for index in indices:
            for key in VIDEO_KEYS:
                _assert_equal(
                    episode_eager[key][index],
                    episode_lazy[key][index],
                    f"{task}.{key}[{index}]",
                )
            _assert_equal(eager[index], lazy[index], f"{task}.sample[{index}].aug_off")

        augmented_index = indices[len(indices) // 2]
        eager.use_image_aug = True
        lazy.use_image_aug = True
        torch.manual_seed(20260902)
        eager_augmented = eager[augmented_index]
        torch.manual_seed(20260902)
        lazy_augmented = lazy[augmented_index]
        _assert_equal(
            eager_augmented,
            lazy_augmented,
            f"{task}.sample[{augmented_index}].aug_on",
        )
        return {
            "task": task,
            "episode": str(episode_path),
            "num_steps": eager.num_steps,
            "indices": list(indices),
            "augmented_index": augmented_index,
            "status": "passed",
        }
    finally:
        frame_pool = getattr(lazy, "_frame_pool", None)
        close_pool = getattr(frame_pool, "close", None)
        if callable(close_pool):
            close_pool()
        del eager
        del lazy
        gc.collect()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--eager-source", required=True, type=Path)
    parser.add_argument("--lazy-source", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    eager_module = _load_module("robotactile_eager_franka", args.eager_source)
    lazy_module = _load_module("robotactile_lazy_franka", args.lazy_source)
    results = [
        _compare_task(
            task=task,
            dataset_root=args.dataset_root,
            eager_module=eager_module,
            lazy_module=lazy_module,
        )
        for task in TASKS
    ]
    print(
        json.dumps(
            {
                "status": "passed",
                "protocol_id": "dream_tac_franka_eager_lazy_parity_v1",
                "task_count": len(results),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
