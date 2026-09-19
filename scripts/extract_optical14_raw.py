"""Decode two fixed recorded UniVTAC witnesses without estimating contact labels.

The recorded writer passes RGB arrays to cv2.imencode: do not add a BGR/RGB
conversion after imdecode. The old reference indices are comparison witnesses,
never qualified no-contact frames. Optical reference preparation is separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from robotactile_benchmark.visualization.optical14 import file_hash, source_array_hash

# Frozen clean-only display choices, selected before any fault is applied.
SOURCES = (
    ("pull_out_key", 55, (6, 7), 121),
    ("insert_HDMI", 90, (4, 5), 49),
)


def decode_dataset(dataset: Any, shape: tuple[int, int, int]) -> np.ndarray[Any, Any]:
    import cv2

    frames = []
    for encoded in dataset:
        if isinstance(encoded, np.ndarray):
            if encoded.ndim != 1 or encoded.dtype != np.uint8:
                raise ValueError("compressed image must be a uint8 byte vector")
            buffer = encoded
        else:
            buffer = np.frombuffer(bytes(encoded), dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None or frame.dtype != np.uint8 or frame.shape != shape:
            raise ValueError(f"invalid decoded image in {dataset.name}")
        frames.append(frame)
    return cast(np.ndarray[Any, Any], np.stack(frames))


def extract(raw_root: Path, output_root: Path) -> None:
    import h5py

    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    sources = []
    for task, episode, references, witness in SOURCES:
        relative = Path(task) / "clean" / f"{episode}.hdf5"
        raw = (raw_root / relative).resolve(strict=True)
        if not raw.is_relative_to(raw_root.resolve()):
            raise ValueError("raw episode escapes input root")
        initial_hash = file_hash(raw)
        with h5py.File(raw, "r") as handle:
            tactile = np.stack(
                [
                    decode_dataset(
                        handle[f"tactile/{slot}_gsmini/rgb_marker"], (240, 320, 3)
                    )
                    for slot in ("left", "right")
                ],
                axis=1,
            )
            wrist = decode_dataset(handle["observation/wrist/rgb"], (270, 480, 3))
            steps = np.asarray(handle["step"][:], dtype=np.int64)
            proprio = np.asarray(handle["embodiment/joint"][:, :8], dtype=np.float32)
        length = len(tactile)
        if (
            len(wrist) != length
            or steps.shape != (length,)
            or proprio.shape != (length, 8)
            or not np.isfinite(proprio).all()
            or not np.all(np.diff(steps) == 2)
            or witness >= length
        ):
            raise ValueError("recorded stream shape, cadence, or witness mismatch")
        times = (steps - steps[0]).astype(np.float64) / 120.0
        arrays = {
            "tactile_rgb": tactile,
            "wrist_rgb": wrist,
            "proprio": proprio,
            "source_step": steps,
            "source_time_s": times,
            "delivery_time_s": times.copy(),
            "reference_indices": np.asarray(references, dtype=np.int64),
        }
        artifact = output_root / f"{task}_raw{episode}.npz"
        with artifact.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        if initial_hash != file_hash(raw):
            raise RuntimeError("raw source changed during extraction")
        sources.append(
            {
                "task": task,
                "raw_episode_id": episode,
                "role": "fixed_clean_visual_witness_not_policy_evaluation",
                "reference_indices": references,
                "contact_peak": witness,
                "selection_rule": "frozen clean-only display witness; not a force maximum",
                "release_start": None,
                "release_confirmation": None,
                "raw_relative_path": relative.as_posix(),
                "raw_hdf5_sha256": initial_hash,
                "artifact": artifact.name,
                "artifact_sha256": file_hash(artifact),
                "arrays": {
                    key: {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "sha256": source_array_hash(value),
                    }
                    for key, value in arrays.items()
                },
            }
        )
    receipt = {
        "schema_id": "robotactile.raw_fault_visualization_sources",
        "schema_version": "1.0",
        "pixel_contract": "opencv_decode_only_uint8_rgb",
        "source_encoding_contract": "opencv_imencode_rgb_input_v1",
        "fault_operator_applied": False,
        "normalization_applied": False,
        "enhancement_applied": False,
        "generated_content": False,
        "contact_labels_are_force_ground_truth": False,
        "proxy_id": "none_no_image_based_contact_labels",
        "reference_scope": "historical comparison indices only, not no-contact references",
        "clocks": {
            "observation_hz": 60,
            "physics_hz": 120,
            "physics_steps_per_observation_verified": 2,
        },
        "extractor_sha256": file_hash(Path(__file__)),
        "sources": sources,
    }
    with (output_root / "source_receipt.json").open("x") as stream:
        stream.write(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    extract(args.raw_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
