# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Memory-safe frame-lazy replacement for Dream-Tac's Franka dataset."""

from __future__ import annotations

import os
import pickle
from collections import OrderedDict
from typing import Any, Optional

import cv2
import h5py
import numpy as np
import torch
from cosmos_policy.datasets.dataset_common import (
    compute_monte_carlo_returns,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    get_hdf5_files,
    preprocess_image,
    rescale_data,
    resize_images,
)
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.utils.tactile_self_attn_gate import (
    mean_abs_diff_uint8_pair,
    scalar_gate_from_raw,
)
from cosmos_policy.utils.utils import duplicate_array
from tqdm import tqdm


class _FrameReaderPool:
    def __init__(self, max_open_videos: int, max_cached_frames: int) -> None:
        if max_open_videos < 1 or max_cached_frames < 1:
            raise ValueError("video and frame cache sizes must be at least 1")
        self._max_open_videos = max_open_videos
        self._max_cached_frames = max_cached_frames
        self._captures: OrderedDict[str, Any] = OrderedDict()
        self._frames: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()

    def _open_capture(self, path: str) -> Any:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video file: {path}")
        return capture

    def _capture(self, path: str) -> Any:
        capture = self._captures.pop(path, None)
        if capture is None:
            capture = self._open_capture(path)
        self._captures[path] = capture
        while len(self._captures) > self._max_open_videos:
            _, evicted = self._captures.popitem(last=False)
            evicted.release()
        return capture

    def _drop_capture(self, path: str) -> None:
        capture = self._captures.pop(path, None)
        if capture is not None:
            capture.release()

    def read(self, path: str, frame_idx: int, resize_size: int) -> np.ndarray:
        key = (path, frame_idx, resize_size)
        cached = self._frames.pop(key, None)
        if cached is not None:
            self._frames[key] = cached
            return cached

        capture = self._capture(path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = capture.read()
        if not ok:
            self._drop_capture(path)
            capture = self._capture(path)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = capture.read()
        if not ok:
            raise ValueError(f"Could not decode frame {frame_idx} from: {path}")
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (resize_size, resize_size):
            frame = resize_images(np.expand_dims(frame, axis=0), resize_size)[0]
        frame = np.asarray(frame, dtype=np.uint8)
        self._frames[key] = frame
        while len(self._frames) > self._max_cached_frames:
            self._frames.popitem(last=False)
        return frame


def _normalize_index(index: int, length: int) -> int:
    normalized = index + length if index < 0 else index
    if normalized < 0 or normalized >= length:
        raise IndexError(index)
    return normalized


class _VideoFrameSequence:
    def __init__(
        self, pool: _FrameReaderPool, path: str, length: int, image_size: int
    ) -> None:
        self._pool = pool
        self._path = path
        self._length = length
        self._image_size = image_size

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> np.ndarray:
        return self._pool.read(
            self._path, _normalize_index(index, self._length), self._image_size
        )


class _PaddedVideoFrameSequence(_VideoFrameSequence):
    def __init__(
        self,
        pool: _FrameReaderPool,
        path: str,
        valid_length: int,
        total_length: int,
        image_size: int,
    ) -> None:
        super().__init__(pool, path, total_length, image_size)
        self._valid_length = valid_length
        self._zero = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    def __getitem__(self, index: int) -> np.ndarray:
        normalized = _normalize_index(index, len(self))
        if normalized >= self._valid_length:
            return self._zero
        return super().__getitem__(normalized)


class _ZeroFrameSequence:
    def __init__(self, length: int, image_size: int) -> None:
        self._length = length
        self._zero = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> np.ndarray:
        _normalize_index(index, self._length)
        return self._zero


def _read_path(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _video_frame_count(path: str) -> int:
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video while indexing: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count < 1:
        raise ValueError(f"Video reports no frames while indexing: {path}")
    return count


class FrankaDataset(LIBERODataset):
    def __init__(
        self,
        data_dir: str,
        is_train: bool = True,
        chunk_size: int = 8,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        treat_success_rollouts_as_demos: bool = False,
        return_value_function_returns: bool = True,
        gamma: float = 0.99,
        use_tactile: Optional[bool] = None,
        use_tactile_image_aug: bool = True,
        max_open_videos: int = 32,
        max_cached_frames: int = 64,
    ) -> None:
        if not is_train:
            raise ValueError("The memory-safe Dream-Tac overlay is train-only")
        if not (use_wrist_images or use_third_person_images):
            raise ValueError("Must use at least one wrist or third-person image")
        max_open_videos = int(
            os.environ.get("DREAM_TAC_FRANKA_MAX_OPEN_VIDEOS", max_open_videos)
        )
        max_cached_frames = int(
            os.environ.get("DREAM_TAC_FRANKA_FRAME_CACHE_SIZE", max_cached_frames)
        )

        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.rollout_data_dir = rollout_data_dir
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob
        self.treat_success_rollouts_as_demos = treat_success_rollouts_as_demos
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self._use_tactile_mode = use_tactile
        self.use_tactile_image_aug = use_tactile_image_aug
        self._frame_pool = _FrameReaderPool(max_open_videos, max_cached_frames)
        files = get_hdf5_files(data_dir, is_train=True)
        if os.environ.get("DEBUGGING", "False").lower() == "true":
            files = files[:1]
        self.data: dict[int, dict[str, Any]] = {}
        self.rollout_episode_metadata: dict[int, dict[str, Any]] = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.rollout_num_episodes = 0
        self.rollout_num_steps = 0
        self.unique_commands: set[str] = set()
        self._suite_to_step_indices: dict[str, list[int]] = {}
        self.use_tactile = False
        for file_path in tqdm(files, desc="Indexing Franka episodes"):
            self._index_episode(file_path)
        self._build_step_index_mapping()
        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as file:
                self.t5_text_embeddings = pickle.load(file)
        else:
            self.t5_text_embeddings = {}
        self.dataset_stats = load_or_compute_dataset_statistics(
            data_dir=self.data_dir,
            data=self.data,
            calculate_dataset_statistics_func=calculate_dataset_statistics,
            stats_filename="dataset_statistics_franka.json",
        )
        if self.normalize_actions:
            self.data = rescale_data(self.data, self.dataset_stats, "actions")
        if self.normalize_proprio:
            self.data = rescale_data(self.data, self.dataset_stats, "proprio")
        if self.normalize_actions or self.normalize_proprio:
            self.dataset_stats_post_norm = (
                load_or_compute_post_normalization_statistics(
                    data_dir=self.data_dir,
                    data=self.data,
                    calculate_dataset_statistics_func=calculate_dataset_statistics,
                    stats_filename="dataset_statistics_post_norm_franka.json",
                )
            )
        self._build_rollout_step_index_mapping()
        self._calculate_epoch_structure()

    def _index_episode(self, file_path: str) -> None:
        with h5py.File(file_path, "r") as file:
            observations = file["observations"]
            video_paths = observations.get("video_paths")
            if (
                video_paths is None
                or "cam_front" not in video_paths
                or "cam_high" not in video_paths
            ):
                raise ValueError(f"Missing required Franka video paths: {file_path}")
            file_dir = os.path.dirname(file_path)
            front_path = os.path.join(file_dir, _read_path(video_paths["cam_front"]))
            high_path = os.path.join(file_dir, _read_path(video_paths["cam_high"]))
            proprio = np.asarray(file["observations/qpos"][:], dtype=np.float32)
            actions = np.asarray(file["action"][:], dtype=np.float32)
            num_steps = min(
                _video_frame_count(front_path),
                _video_frame_count(high_path),
                len(proprio),
                len(actions),
            )
            proprio = proprio[:num_steps]
            actions = actions[:num_steps]
            images = _VideoFrameSequence(
                self._frame_pool, front_path, num_steps, self.final_image_size
            )
            wrist_images = _VideoFrameSequence(
                self._frame_pool, high_path, num_steps, self.final_image_size
            )

            tactile_left: Any = _ZeroFrameSequence(num_steps, self.final_image_size)
            tactile_right: Any = _ZeroFrameSequence(num_steps, self.final_image_size)
            if (
                self._use_tactile_mode is not False
                and "tactile_rectify_left" in video_paths
                and "tactile_rectify_right" in video_paths
            ):
                left_path = os.path.join(
                    file_dir, _read_path(video_paths["tactile_rectify_left"])
                )
                right_path = os.path.join(
                    file_dir, _read_path(video_paths["tactile_rectify_right"])
                )
                tactile_steps = min(
                    _video_frame_count(left_path),
                    _video_frame_count(right_path),
                    num_steps,
                )
                tactile_left = _PaddedVideoFrameSequence(
                    self._frame_pool,
                    left_path,
                    tactile_steps,
                    num_steps,
                    self.final_image_size,
                )
                tactile_right = _PaddedVideoFrameSequence(
                    self._frame_pool,
                    right_path,
                    tactile_steps,
                    num_steps,
                    self.final_image_size,
                )
                self.use_tactile = True

            command = file.attrs.get("task_name", "unknown")
            if isinstance(command, bytes):
                command = command.decode("utf-8")
            command = str(command)
            self.unique_commands.add(command)
            returns = (
                compute_monte_carlo_returns(
                    num_steps, terminal_reward=1.0, gamma=self.gamma
                )
                if self.return_value_function_returns
                else None
            )
            self.data[self.num_episodes] = {
                "images": images,
                "wrist_images": wrist_images,
                "tactile_left_images": tactile_left,
                "tactile_right_images": tactile_right,
                "proprio": proprio,
                "actions": actions,
                "command": command,
                "num_steps": num_steps,
                "suite": "franka",
                "returns": returns.copy() if returns is not None else None,
            }
            self.num_episodes += 1
            self.num_steps += num_steps

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)
        if not self.use_tactile:
            return sample
        global_step_idx = idx % self.num_steps
        episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
        episode_data = self.data[episode_idx]
        future_idx = min(
            relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1
        )

        def tactile_block(step_idx: int) -> np.ndarray:
            left = duplicate_array(
                episode_data["tactile_left_images"][step_idx],
                total_num_copies=self.num_duplicates_per_image,
            )
            right = duplicate_array(
                episode_data["tactile_right_images"][step_idx],
                total_num_copies=self.num_duplicates_per_image,
            )
            return np.concatenate([left, right], axis=0)

        tactile_use_aug = self.use_image_aug and self.use_tactile_image_aug
        tactile_current = preprocess_image(
            tactile_block(relative_step_idx),
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=tactile_use_aug,
            stronger_image_aug=self.use_stronger_image_aug
            if tactile_use_aug
            else False,
        )
        tactile_future = preprocess_image(
            tactile_block(future_idx),
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=tactile_use_aug,
            stronger_image_aug=self.use_stronger_image_aug
            if tactile_use_aug
            else False,
        )
        video = sample["video"]
        sample["video"] = torch.cat(
            [video[:, :13], tactile_current, video[:, 13:29], tactile_future], dim=1
        )
        sample["action_latent_idx"] = sample["action_latent_idx"] + 2
        for key in (
            "future_proprio_latent_idx",
            "future_wrist_image_latent_idx",
            "future_image_latent_idx",
        ):
            if sample.get(key, -1) >= 0:
                sample[key] = sample[key] + 2
        sample["future_tactile_left_latent_idx"] = torch.tensor(10, dtype=torch.long)
        sample["future_tactile_right_latent_idx"] = torch.tensor(11, dtype=torch.long)

        left = episode_data["tactile_left_images"]
        right = episode_data["tactile_right_images"]
        raw_gate = 0.0
        if relative_step_idx > 0:
            raw_gate = mean_abs_diff_uint8_pair(
                left[relative_step_idx],
                right[relative_step_idx],
                left[relative_step_idx - 1],
                right[relative_step_idx - 1],
            )
        sample["tactile_self_attn_gate"] = torch.tensor(
            scalar_gate_from_raw(raw_gate), dtype=torch.float32
        )
        return sample
