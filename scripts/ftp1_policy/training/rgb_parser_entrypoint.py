"""Run the pinned UniVTAC parser with an RGB-correct output boundary."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable, TypeAlias, cast

import numpy as np
import numpy.typing as npt

Array: TypeAlias = npt.NDArray[np.generic]

_RGB_OUTPUT_KEYS = (
    "camera_ego_rgb",
    "right_wrist_camera_rgb",
    "right_tactile_data_gripper",
)


def _load_upstream(ftp1_root: Path) -> ModuleType:
    source = (
        ftp1_root / "data_processing" / "parse_data_module" / "parse_data_univtac.py"
    ).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(
        "robotactile_pinned_univtac_parser", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned UniVTAC parser: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_rgb_boundary(upstream: ModuleType) -> None:
    original = cast(
        Callable[..., dict[str, Array] | None],
        upstream._load_univtac_episode,
    )

    def rgb_load(*args: object, **kwargs: object) -> dict[str, Array] | None:
        episode = original(*args, **kwargs)
        if episode is None:
            return None
        corrected = dict(episode)
        for key in _RGB_OUTPUT_KEYS:
            value = corrected.get(key)
            if value is not None:
                corrected[key] = cast(Array, np.ascontiguousarray(value[..., ::-1]))
        return corrected

    upstream._load_univtac_episode = rgb_load  # type: ignore[attr-defined]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ftp1-root", type=Path, required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--task-list")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    return parser


def main() -> None:
    args = _parser().parse_args()
    upstream = _load_upstream(args.ftp1_root)
    _install_rgb_boundary(upstream)
    task_list = None
    if args.task_list:
        task_list = [item.strip() for item in args.task_list.split(",") if item.strip()]
    run: Callable[..., None] = upstream._run
    run(
        base_dir=args.base_dir,
        save_dir=args.save_dir,
        episodes_per_task=args.episodes_per_task,
        task_list=task_list,
        downsample=args.downsample,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
