"""Load one source-bound frozen UniVTAC task configuration after Isaac starts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_REGISTRY_SHA256 = "6f8d58b8efce09f1c8d3f1a97a9780ba722b748b0d23086b7051a8bf75272084"
_TASK_IDS = frozenset(
    {
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    }
)


def _source_root() -> Path:
    raw = os.environ.get("ROBOTACTILE_UNIVTAC_ROOT", "")
    source_root = Path(raw)
    if not source_root.is_absolute() or not source_root.is_dir():
        raise RuntimeError("ROBOTACTILE_UNIVTAC_ROOT must be an existing absolute path")
    return source_root.resolve(strict=True)


def _result_path() -> Path:
    raw = os.environ.get("ROBOTACTILE_TASK_SMOKE_RESULT", "")
    result_path = Path(raw)
    if not result_path.is_absolute() or not result_path.parent.is_dir():
        raise RuntimeError(
            "ROBOTACTILE_TASK_SMOKE_RESULT must have an existing absolute parent"
        )
    if result_path.exists() or result_path.is_symlink():
        raise RuntimeError("task smoke result path must not exist")
    return result_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_contract(task_id: str) -> tuple[str, str]:
    registry_path = (
        Path(__file__).resolve().parents[2] / "configs/univtac/tasks_v1.json"
    )
    if _sha256(registry_path) != _REGISTRY_SHA256:
        raise RuntimeError("frozen UniVTAC task registry hash mismatch")
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    tasks = document.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 8
        or {item.get("task_id") for item in tasks if isinstance(item, dict)}
        != _TASK_IDS
    ):
        raise RuntimeError("frozen UniVTAC task registry membership mismatch")
    matches = [item for item in tasks if item.get("task_id") == task_id]
    if len(matches) != 1:
        raise RuntimeError("task is not registered in the frozen UniVTAC registry")
    module_name = matches[0].get("module_name")
    source_sha256 = matches[0].get("task_source_sha256")
    if module_name != f"envs.{task_id}":
        raise RuntimeError("frozen task module does not match its task ID")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise RuntimeError("frozen task source hash is invalid")
    return module_name, source_sha256


def _publish_result(path: Path, payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return serialized


def _enable_headless_task_extensions() -> tuple[str, ...]:
    import omni.kit.app  # type: ignore[import-not-found]

    extension_ids = ("omni.ui",)
    manager = omni.kit.app.get_app().get_extension_manager()
    for extension_id in extension_ids:
        manager.set_extension_enabled_immediate(extension_id, True)
    return extension_ids


def _task_payload(
    task_id: str,
    task_module: Any,
    source_root: Path,
    expected_source_sha256: str,
) -> dict[str, object]:
    config_class = task_module.TaskCfg
    task_class = task_module.Task
    config = config_class()
    module_name = f"envs.{task_id}"
    expected_path = (source_root / Path(*module_name.split("."))).with_suffix(".py")
    module_path = Path(task_module.__file__).resolve(strict=True)
    if module_path != expected_path.resolve(strict=True):
        raise RuntimeError("imported task module came from another checkout")
    source_sha256 = _sha256(module_path)
    if source_sha256 != expected_source_sha256:
        raise RuntimeError("imported task source hash mismatch")
    return {
        "config_class": config_class.__name__,
        "num_envs": int(config.scene.num_envs),
        "registry_resource_sha256": _REGISTRY_SHA256,
        "sim_dt": float(config.sim.dt),
        "status": "passed",
        "tactile_sensor_type": str(config.tactile_sensor_type),
        "task_class": task_class.__name__,
        "task_id": task_id,
        "task_instantiated": False,
        "task_module": module_name,
        "task_source_sha256": source_sha256,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="pull_out_key")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Launch Isaac and load task code without constructing the environment."""
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    task_id = _argument_parser().parse_args(argv).task
    task_module_name, task_source_sha256 = _task_contract(task_id)
    source_root = _source_root()
    result_path = _result_path()
    sys.path.insert(0, os.fspath(source_root))
    launcher = AppLauncher(headless=True)
    simulation_app = launcher.app
    try:
        try:
            extension_ids = _enable_headless_task_extensions()
            task_module = importlib.import_module(task_module_name)
            payload = {
                **_task_payload(task_id, task_module, source_root, task_source_sha256),
                "headless_extensions": list(extension_ids),
                "isaac_app_running": simulation_app.is_running(),
            }
        except Exception as error:
            payload = {
                "error_message": str(error),
                "error_type": type(error).__name__,
                "headless_extensions": ["omni.ui"],
                "status": "failed",
                "task_id": task_id,
                "task_instantiated": False,
                "task_module": task_module_name,
                "task_source_sha256": task_source_sha256,
            }
            serialized = _publish_result(result_path, payload)
            print(serialized, file=sys.stderr, flush=True)
            return 1
        serialized = _publish_result(result_path, payload)
        print(serialized, flush=True)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
