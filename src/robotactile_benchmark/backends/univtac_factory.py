"""Lazy AppLauncher-first construction of live UniVTAC task runtimes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import numpy as np

from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACBackendConfig,
    UniVTACContractError,
    validate_packaged_univtac_config,
)
from robotactile_benchmark.backends.univtac_isaac import UniVTACTaskRuntime
from robotactile_benchmark.contracts import Array


def _git_output(root: Path, arguments: Tuple[str, ...]) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise UniVTACContractError(
            f"unable to verify UniVTAC checkout: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _task_source_path(root: Path, config: UniVTACBackendConfig) -> Path:
    relative = Path(*config.task.module_name.split(".")).with_suffix(".py")
    source = (root / relative).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as error:
        raise UniVTACContractError("task module escapes the upstream root") from error
    return source


def _verify_checkout(root: Path, config: UniVTACBackendConfig) -> None:
    validate_packaged_univtac_config(config)
    root = root.resolve()
    if not root.is_dir():
        raise UniVTACContractError("UniVTAC upstream root does not exist")
    if _git_output(root, ("rev-parse", "HEAD")) != config.upstream_commit:
        raise UniVTACContractError("UniVTAC checkout commit mismatch")
    if _git_output(root, ("status", "--porcelain")):
        raise UniVTACContractError("UniVTAC frozen checkout is dirty")
    source = _task_source_path(root, config)
    if not source.is_file():
        raise UniVTACContractError("UniVTAC task source is missing")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != config.task.task_source_sha256
    ):
        raise UniVTACContractError("UniVTAC task source hash mismatch")


def _configure_task_cfg(
    cfg: Any,
    config: UniVTACBackendConfig,
    runtime_dir: Path,
    device: Optional[str],
) -> None:
    if getattr(cfg, "step_lim", None) != config.task.action_horizon:
        raise UniVTACContractError("upstream task action horizon mismatch")
    cfg.decimation = config.decimation
    cfg.sim.dt = 1.0 / float(config.sim_hz)
    if device is not None:
        cfg.sim.device = device
    cfg.scene.num_envs = 1
    cfg.obs_data_type = {
        "camera": ["rgb"],
        "tactile": [config.aliases.tactile_payload, "depth"],
        "embodiment": ["joint"],
    }
    cfg.save_frequency = 0
    cfg.video_frequency = 0
    cfg.render_frequency = 0
    cfg.random_texture = False
    cfg.tactile_sensor_type = config.aliases.sensor_type
    cfg.save_dir = runtime_dir


def _live_joint_names(task: Any) -> Tuple[str, ...]:
    try:
        names = tuple(task._robot_manager.robot.joint_names)
    except AttributeError as error:
        raise UniVTACContractError(
            "live UniVTAC joint names are unavailable"
        ) from error
    if not all(isinstance(name, str) for name in names):
        raise UniVTACContractError("live UniVTAC joint names must be strings")
    return cast(Tuple[str, ...], names)


def launch_univtac_runtime(
    config: UniVTACBackendConfig,
    *,
    upstream_root: Path,
    runtime_dir: Path,
    launcher_args: Optional[Mapping[str, Any]] = None,
    device: Optional[str] = None,
) -> UniVTACTaskRuntime:
    """Launch Isaac first, then import and construct one verified UniVTAC task."""

    upstream_root = upstream_root.resolve()
    runtime_dir = runtime_dir.resolve()
    _verify_checkout(upstream_root, config)
    isaac_app = importlib.import_module("isaaclab.app")
    app_launcher_type = getattr(isaac_app, "AppLauncher", None)
    if not callable(app_launcher_type):
        raise UniVTACContractError("isaaclab.app.AppLauncher is unavailable")
    arguments = {"headless": True}
    if launcher_args is not None:
        arguments.update(dict(launcher_args))
    launcher = app_launcher_type(argparse.Namespace(**arguments))
    simulation_app = getattr(launcher, "app", None)
    if simulation_app is None or not callable(getattr(simulation_app, "close", None)):
        raise UniVTACContractError("AppLauncher did not expose a closeable app")
    task: Any = None
    try:
        if str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))
        task_module = importlib.import_module(config.task.module_name)
        module_file = getattr(task_module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != _task_source_path(
            upstream_root, config
        ):
            raise UniVTACContractError(
                "imported task module came from another checkout"
            )
        task_cfg_type = getattr(task_module, "TaskCfg", None)
        task_type = getattr(task_module, config.task.class_name, None)
        if not callable(task_cfg_type) or not callable(task_type):
            raise UniVTACContractError("task module lacks TaskCfg/Task constructors")
        cfg = task_cfg_type()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        _configure_task_cfg(cfg, config, runtime_dir, device)
        task = task_type(cfg, mode="eval")
        names = _live_joint_names(task)
        handshake = config.expected_handshake(names)
        config.validate_handshake(handshake)
        torch = importlib.import_module("torch")
        as_tensor = getattr(torch, "as_tensor", None)
        float32 = getattr(torch, "float32", None)
        if not callable(as_tensor) or float32 is None:
            raise UniVTACContractError("torch float32 tensor conversion is unavailable")

        def encode_action(row: Array) -> Any:
            contiguous = np.ascontiguousarray(row, dtype=np.float32)
            return as_tensor(contiguous, dtype=float32, device=task.device)

        def close_runtime() -> None:
            try:
                task.close()
            finally:
                simulation_app.close()

        return UniVTACTaskRuntime(
            task=task,
            handshake=handshake,
            encode_action=encode_action,
            close_runtime=close_runtime,
        )
    except Exception:
        try:
            if task is not None and callable(getattr(task, "close", None)):
                task.close()
        finally:
            simulation_app.close()
        raise
