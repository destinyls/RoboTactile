#!/usr/bin/env python3
"""One GPU-bound source-pinned encoder worker with a no-clobber receipt."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence, cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hpu_training.contract import verify_official_checkout  # noqa: E402
from hpu_training.latent.contracts import (  # noqa: E402
    OFFICIAL_TACTILE_ENCODER_SHA256,
    OFFICIAL_VIDEO_ENCODER_SHA256,
    TACTILE_KEYS,
    TARGET_FPS,
    VIDEO_KEYS,
    Kind,
    sha256_file,
    verify_official_encoders,
)
from hpu_training.latent.file_inventory import (  # noqa: E402
    inventory_episode,
    load_repo_info,
    missing_episode_ids,
)

HCU_DEVICE_ORDER = (0, 1, 5, 4, 2, 3, 7, 6)
VISION_ACCEL_WRAPPER = Path(__file__).resolve().parent / "vision_encoder_accel.py"


def _decode_payload(encoded: str) -> dict[str, object]:
    try:
        payload = json.loads(base64.b64decode(encoded, validate=True))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid worker payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("worker payload must be an object")
    return payload


def _required_string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"worker payload {field} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"worker payload {field} must be a non-negative integer")
    return value


def build_encoder_command(
    *,
    python_path: Path,
    official_repo: Path,
    model_path: Path,
    repo_path: Path,
    kind: Kind,
    episode_ids: Sequence[int],
) -> list[str]:
    """Build one source-pinned encoder invocation for an explicit ID list."""
    if not episode_ids:
        raise ValueError("encoder command requires at least one missing episode")
    if kind == "vision":
        return [
            str(python_path),
            str(VISION_ACCEL_WRAPPER),
            "--official-script",
            str(official_repo / "script" / "encode_lerobot_n0_latents.py"),
            "--dataset-root",
            str(repo_path),
            "--model-path",
            str(model_path),
            "--target-fps",
            str(TARGET_FPS),
            "--height",
            "256",
            "--width",
            "256",
            "--episodes",
            *[str(value) for value in episode_ids],
            "--video-keys",
            *VIDEO_KEYS,
            "--device",
            "cuda:0",
            "--dtype",
            "bf16",
        ]
    return [
        str(python_path),
        str(official_repo / "script" / "encode_tactile_latent.py"),
        "--dataset-root",
        str(repo_path),
        "--model-path",
        str(model_path),
        "--target-fps",
        str(TARGET_FPS),
        "--height",
        "128",
        "--width",
        "128",
        "--episodes",
        *[str(value) for value in episode_ids],
        "--tactile-keys",
        *TACTILE_KEYS,
        "--device",
        "cuda:0",
        "--dtype",
        "bf16",
        "--mode",
        "both",
        "--local-mode",
        "current",
    ]


def encoder_provenance(*, kind: Kind, official_repo: Path) -> dict[str, object]:
    """Describe the exact executable boundary recorded in worker receipts."""
    if kind == "vision":
        return {
            "execution": "robotactile_official_main_device_wrapper",
            "official_script": str(
                official_repo / "script" / "encode_lerobot_n0_latents.py"
            ),
            "official_script_sha256": OFFICIAL_VIDEO_ENCODER_SHA256,
            "wrapper": str(VISION_ACCEL_WRAPPER),
            "wrapper_sha256": sha256_file(VISION_ACCEL_WRAPPER),
            "override": {
                "symbol": "load_text_encoder",
                "argument": "torch_device",
                "official_value": "cpu",
                "effective_value": "cuda:0",
            },
        }
    return {
        "execution": "pinned_official_cli",
        "official_script": str(official_repo / "script" / "encode_tactile_latent.py"),
        "official_script_sha256": OFFICIAL_TACTILE_ENCODER_SHA256,
    }


def _write_receipt(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"worker receipt is no-clobber: {path}")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


def _verify_runtime(payload: Mapping[str, object], *, node_preflight: bool) -> None:
    official_repo = Path(_required_string(payload, "official_repo"))
    model_path = Path(_required_string(payload, "model_path"))
    dataset_root = Path(_required_string(payload, "dataset_root"))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    official_paths = f"{official_repo}:{official_repo / 'n0_twam'}"
    os.environ["PYTHONPATH"] = (
        f"{official_paths}:{existing_pythonpath}"
        if existing_pythonpath
        else official_paths
    )
    verify_official_checkout(official_repo)
    verify_official_encoders(official_repo)
    if not VISION_ACCEL_WRAPPER.is_file():
        raise FileNotFoundError(VISION_ACCEL_WRAPPER)
    if dataset_root.name != "train759" or not dataset_root.is_dir():
        raise ValueError(f"worker dataset is not train759: {dataset_root}")
    for relative in ("vae", "tokenizer", "text_encoder"):
        if not (model_path / relative).is_dir():
            raise FileNotFoundError(model_path / relative)

    import torch  # type: ignore[import-not-found]

    expected_devices = 8 if node_preflight else 1
    actual_devices = torch.cuda.device_count()
    if not torch.cuda.is_available() or actual_devices != expected_devices:
        raise RuntimeError(
            f"CUDA visibility mismatch: expected={expected_devices} actual={actual_devices}"
        )


def _episodes(payload: Mapping[str, object]) -> list[dict[str, object]]:
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("worker must receive an explicit non-empty episode list")
    parsed: list[dict[str, object]] = []
    ids: set[int] = set()
    for raw in raw_episodes:
        if not isinstance(raw, dict):
            raise ValueError("worker episode entry must be an object")
        episode_id = raw.get("lerobot_episode_index")
        length = raw.get("length")
        if (
            isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or episode_id < 0
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
        ):
            raise ValueError("worker episode ID/length is invalid")
        if episode_id in ids:
            raise ValueError(f"duplicate worker episode ID: {episode_id}")
        ids.add(episode_id)
        parsed.append(dict(raw))
    parsed_ids = [_required_int(item, "lerobot_episode_index") for item in parsed]
    if parsed_ids != sorted(ids):
        raise ValueError("worker episode IDs must be sorted")
    return parsed


def _run_worker(payload: dict[str, object]) -> int:
    mode = _required_string(payload, "mode")
    if mode == "node_preflight":
        _verify_runtime(payload, node_preflight=True)
        print("LATENT_NODE_PREFLIGHT_OK devices=8 target_fps=10")
        return 0
    if mode not in {"encode", "inventory"}:
        raise ValueError(f"unsupported worker mode: {mode}")

    local_device = _required_int(payload, "local_device")
    if local_device >= 8:
        raise ValueError("local_device must be below 8")
    physical_device = HCU_DEVICE_ORDER[local_device]
    os.environ["HIP_VISIBLE_DEVICES"] = str(physical_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    _verify_runtime(payload, node_preflight=False)
    worker_id = _required_int(payload, "worker_id")
    task = _required_string(payload, "task")
    kind_value = _required_string(payload, "kind")
    if kind_value not in {"vision", "tactile"}:
        raise ValueError(f"invalid latent kind: {kind_value}")
    kind = cast(Kind, kind_value)
    official_repo = Path(_required_string(payload, "official_repo"))
    repo_path = Path(_required_string(payload, "repo_path"))
    if repo_path.name != task or repo_path.parent.name != "train759":
        raise ValueError("worker task repo is outside the certified train759 layout")
    load_repo_info(repo_path)
    episodes = _episodes(payload)
    episode_pairs = [
        (
            _required_int(item, "lerobot_episode_index"),
            _required_int(item, "length"),
        )
        for item in episodes
    ]
    receipt_path = Path(_required_string(payload, "receipt_path"))
    if receipt_path.exists():
        raise FileExistsError(receipt_path)

    started = time.time()
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "worker_id": worker_id,
        "node": _required_string(payload, "node"),
        "local_device": local_device,
        "task": task,
        "kind": kind,
        "target_fps": TARGET_FPS,
        "repo_path": str(repo_path),
        "episode_ids": [episode_id for episode_id, _ in episode_pairs],
        "episodes": episodes,
        "commands": [],
        "encoder_provenance": encoder_provenance(
            kind=kind,
            official_repo=official_repo,
        ),
    }
    try:
        missing = missing_episode_ids(
            repo_path=repo_path, episodes=episode_pairs, kind=kind
        )
        receipt["initial_missing_episode_ids"] = missing
        if mode == "encode" and missing:
            command = build_encoder_command(
                python_path=Path(_required_string(payload, "python_path")),
                official_repo=official_repo,
                model_path=Path(_required_string(payload, "model_path")),
                repo_path=repo_path,
                kind=kind,
                episode_ids=missing,
            )
            receipt["commands"] = [command]
            result = subprocess.run(
                command,
                cwd=official_repo,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"official {kind} encoder exited {result.returncode}"
                )
        if mode == "inventory" and missing:
            raise FileNotFoundError(
                f"inventory-only worker has missing {kind} episodes: {missing}"
            )
        inventory = [
            {
                **inventory_episode(
                    repo_path=repo_path,
                    episode_id=episode_id,
                    length=length,
                    kind=kind,
                ),
                "source_episode_id": _required_int(source, "source_episode_id"),
                "source_relative_path": str(source["source_relative_path"]),
                "source_sha256": str(source["source_sha256"]),
            }
            for (episode_id, length), source in zip(episode_pairs, episodes)
        ]
        receipt["inventory"] = inventory
        receipt["final_missing_episode_ids"] = []
        receipt["status"] = "complete"
        receipt["elapsed_seconds"] = time.time() - started
        _write_receipt(receipt_path, receipt)
        print(
            f"LATENT_WORKER_COMPLETE worker={worker_id} task={task} kind={kind} "
            f"episodes={len(episodes)} encoded={len(missing)}"
        )
        return 0
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["elapsed_seconds"] = time.time() - started
        _write_receipt(receipt_path, receipt)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return _run_worker(_decode_payload(args.payload))


if __name__ == "__main__":
    sys.exit(main())
