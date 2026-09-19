"""Lazy AppLauncher-first construction of live UniVTAC task runtimes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC, QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT,
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    UniVTACBackendConfig,
    UniVTACContractError,
    validate_n0_ee_action_execution_contract,
    validate_packaged_univtac_config,
)
from robotactile_benchmark.backends.univtac_diagnostics import (
    install_planner_failure_diagnostics,
)
from robotactile_benchmark.backends.univtac_grasp_initialization import (
    install_grasp_initialization_compatibility,
)
from robotactile_benchmark.backends.univtac_host import UniVTACSimulationAppHost
from robotactile_benchmark.backends.univtac_isaac import UniVTACTaskRuntime
from robotactile_benchmark.backends.univtac_lifecycle import (
    install_runtime_signal_tracing,
    require_hang_detector_disabled,
)
from robotactile_benchmark.backends.univtac_n0_cadence import (
    install_n0_action_execution,
    install_n0_evaluation_reset,
)
from robotactile_benchmark.backends.univtac_placement_compatibility import (
    install_constrained_placement_compatibility as _install_constrained_placement_compatibility,
)
from robotactile_benchmark.backends.univtac_planner_compatibility import (
    install_local_ik_fallback,
)
from robotactile_benchmark.backends.univtac_reset_trajectory import (
    UniVTACPreMoveTrajectory,
)
from robotactile_benchmark.backends.univtac_reset_trajectory_runtime import (
    install_pre_move_trajectory_replay,
)
from robotactile_benchmark.backends.univtac_reuse import (
    prepare_univtac_task_teardown,
    reconstruct_univtac_stage,
)
from robotactile_benchmark.backends.univtac_snapshot import (
    build_live_snapshot_callbacks,
)
from robotactile_benchmark.backends.univtac_tactile_attachment import (
    install_gsmini_attachment_constructor_compatibility,
    repair_gsmini_tactile_attachments,
)
from robotactile_benchmark.contracts import Array

_UPSTREAM_HEADLESS_EXTENSION_IDS = ("omni.ui",)
_ZERO_DISTANCE_GRASP_APPROACH_TASKS = frozenset({"insert_hole", "insert_tube"})
_GRASP_APPROACH_DISTANCE_M = 0.05
_ANTIALIASING_MODES = frozenset({"Off", "FXAA", "DLSS", "TAA", "DLAA"})
_RENDERING_MODES = frozenset({"balanced", "performance", "quality"})
UNIVTAC_ANTIALIASING_MODE = "TAA"
UNIVTAC_RENDERING_MODE = "balanced"
# Backward-compatible name used by the N0-specific diagnostic scripts.
N0_UNIVTAC_ANTIALIASING_MODE = UNIVTAC_ANTIALIASING_MODE
_N0_TRAINING_CADENCE_ENV = "ROBOTACTILE_N0_TRAINING_CADENCE_DIAGNOSTIC"
_RESET_TIME_LIMIT_ENV = "ROBOTACTILE_UNIVTAC_RESET_TIME_LIMIT_S"


def _resolved_reset_time_limit_s(upstream_limit: object) -> float:
    """Resolve an infrastructure-only reset watchdog without shortening upstream."""

    if isinstance(upstream_limit, bool) or not isinstance(upstream_limit, (int, float)):
        raise UniVTACContractError("UniVTAC reset time limit must be numeric")
    baseline = float(upstream_limit)
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise UniVTACContractError(
            "UniVTAC reset time limit must be positive and finite"
        )
    raw_override = os.environ.get(_RESET_TIME_LIMIT_ENV)
    if raw_override is None:
        return baseline
    try:
        override = float(raw_override)
    except ValueError as error:
        raise UniVTACContractError(
            f"{_RESET_TIME_LIMIT_ENV} must be a positive finite number"
        ) from error
    if not math.isfinite(override) or override <= 0.0:
        raise UniVTACContractError(
            f"{_RESET_TIME_LIMIT_ENV} must be a positive finite number"
        )
    if override < baseline:
        raise UniVTACContractError(
            f"{_RESET_TIME_LIMIT_ENV} cannot shorten the upstream watchdog"
        )
    return override


@dataclass(frozen=True)
class _UniVTACApplicationResources:
    """Imports and application state that are safe to reuse across tasks."""

    simulation_app: Any
    torch: Any
    task_cfg_type: Callable[..., Any]
    task_type: Callable[..., Any]


@dataclass(frozen=True)
class _EncodedEEAction:
    """Adapt canonical EE8 values to the pinned UniVTAC slice contract."""

    _values: Array

    def __post_init__(self) -> None:
        values = np.array(self._values, dtype=np.float32, order="C", copy=True)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise UniVTACContractError("EE8 action must be finite float32 [8]")
        values.setflags(write=False)
        object.__setattr__(self, "_values", values)

    def __getitem__(self, key: int | slice) -> Any:
        if key == slice(7, None, None):
            return float(self._values[7])
        selected = self._values[key]
        if isinstance(selected, np.ndarray):
            return np.array(selected, dtype=np.float32, order="C", copy=True)
        return float(selected)


def _close_runtime_component(close: Callable[[], None], name: str) -> None:
    """Treat Isaac's clean ``SystemExit`` as a completed close operation."""

    try:
        close()
    except SystemExit as error:
        if error.code is not None and error.code != 0:
            raise UniVTACContractError(
                f"{name} close raised non-zero SystemExit: {error.code}"
            ) from error


def _emit_runtime_construction_failure(
    stage: str,
    error: BaseException,
) -> None:
    """Flush the Python failure before native Isaac cleanup can exit."""

    payload = {
        "error_message": str(error),
        "error_type": type(error).__name__,
        "event": "robotactile_univtac_runtime_construction_failure",
        "stage": stage,
    }
    sys.stderr.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()


def _record_lifecycle_stage(
    observer: Optional[Callable[[str], None]], stage: str
) -> None:
    if observer is not None:
        observer(stage)


def _validated_initial_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UniVTACContractError("initial seed must be an integer")
    if value < 0:
        raise UniVTACContractError("initial seed must be non-negative")
    return value


def _validated_antialiasing_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _ANTIALIASING_MODES:
        raise UniVTACContractError("unsupported UniVTAC antialiasing mode")
    return value


def _n0_training_cadence_diagnostic_requested() -> bool:
    """Return the legacy fixed-endpoint diagnostic alias opt-in."""

    value = os.environ.get(_N0_TRAINING_CADENCE_ENV, "0")
    if value not in {"0", "1"}:
        raise UniVTACContractError(f"{_N0_TRAINING_CADENCE_ENV} must be either 0 or 1")
    return value == "1"


def _resolve_n0_action_execution_contract(requested: Optional[str]) -> tuple[str, bool]:
    """Resolve production training cadence or an explicit diagnostic path."""

    legacy_diagnostic = _n0_training_cadence_diagnostic_requested()
    if requested is None:
        return (
            N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT
            if legacy_diagnostic
            else N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
            legacy_diagnostic,
        )
    selected = validate_n0_ee_action_execution_contract(requested)
    if legacy_diagnostic and selected != N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT:
        raise UniVTACContractError(
            "explicit N0 action execution conflicts with the diagnostic cadence env"
        )
    return selected, selected == N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT


def _resolved_antialiasing_mode(
    _config: UniVTACBackendConfig,
    value: Optional[str],
) -> Optional[str]:
    """Use the empirically matched renderer for every live policy runtime."""

    validated = _validated_antialiasing_mode(value)
    if validated is not None:
        return validated
    return UNIVTAC_ANTIALIASING_MODE


def _resolved_launcher_args(
    launcher_args: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the N0-TWAM renderer preset before AppLauncher starts."""

    arguments: dict[str, Any] = {
        "headless": True,
        "rendering_mode": UNIVTAC_RENDERING_MODE,
    }
    if launcher_args is not None:
        arguments.update(dict(launcher_args))
    rendering_mode = arguments.get("rendering_mode")
    if not isinstance(rendering_mode, str) or rendering_mode not in _RENDERING_MODES:
        raise UniVTACContractError("unsupported UniVTAC rendering mode")
    return arguments


def _prepare_process_determinism(initial_seed: int) -> None:
    seed = _validated_initial_seed(initial_seed)
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {None, ":4096:8"}:
        raise UniVTACContractError("CUBLAS_WORKSPACE_CONFIG is incompatible")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _seed_torch_process(torch: Any, initial_seed: int) -> None:
    seed = _validated_initial_seed(initial_seed)
    manual_seed = getattr(torch, "manual_seed", None)
    cuda = getattr(torch, "cuda", None)
    manual_seed_all = getattr(cuda, "manual_seed_all", None)
    deterministic = getattr(torch, "use_deterministic_algorithms", None)
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if (
        not callable(manual_seed)
        or not callable(manual_seed_all)
        or not callable(deterministic)
        or cudnn is None
    ):
        raise UniVTACContractError("torch deterministic seed contract is incomplete")
    manual_seed(seed)
    manual_seed_all(seed)
    deterministic(True, warn_only=True)
    cudnn.benchmark = False
    cudnn.deterministic = True


def _install_task_seed_hook(task: Any, torch: Any, construction_seed: int) -> None:
    seed = _validated_initial_seed(construction_seed)
    upstream_seed = getattr(task, "seed", None)
    if not callable(upstream_seed):
        raise UniVTACContractError("upstream task seed method is unavailable")

    def deterministic_seed(requested_seed: int = -1) -> int:
        if requested_seed != seed:
            raise UniVTACContractError(
                "upstream reset seed differs from the construction seed"
            )
        resolved = upstream_seed(requested_seed)
        configured = getattr(getattr(task, "cfg", None), "seed", None)
        if resolved not in {None, seed} or configured != seed:
            raise UniVTACContractError("upstream task changed the requested seed")
        _prepare_process_determinism(seed)
        _seed_torch_process(torch, seed)
        return seed

    task.seed = deterministic_seed


def _install_grasp_approach_compatibility(task: Any, task_id: str) -> bool:
    """Enable cuRobo's approach metric without changing the final grasp pose."""

    if task_id not in _ZERO_DISTANCE_GRASP_APPROACH_TASKS:
        return False
    atom = getattr(task, "atom", None)
    if atom is None:
        raise UniVTACContractError("upstream task atom is unavailable")
    upstream_grasp = getattr(atom, "grasp_actor", None)
    if not callable(upstream_grasp):
        raise UniVTACContractError("upstream grasp_actor method is unavailable")

    def compatible_grasp_actor(
        actor: Any,
        pre_dis: float = 0.1,
        dis: float = 0.0,
        gripper_pos: float = 0.0,
        contact_point_id: Any = None,
        is_close: bool = True,
    ) -> Any:
        approach_distance = (
            _GRASP_APPROACH_DISTANCE_M if pre_dis == 0.0 and dis == 0.0 else pre_dis
        )
        return upstream_grasp(
            actor,
            pre_dis=approach_distance,
            dis=dis,
            gripper_pos=gripper_pos,
            contact_point_id=contact_point_id,
            is_close=is_close,
        )

    atom.grasp_actor = compatible_grasp_actor
    return True


def _install_grasp_initialization_for_action_spec(
    task: Any,
    config: UniVTACBackendConfig,
) -> bool:
    """Keep the N0 grasp preload out of official qpos-policy evaluation.

    The preload intentionally changes the post-reset gripper position to make
    EE endpoint execution stable.  Official ACT checkpoints were trained on
    UniVTAC's released qpos reset and must observe that reset unchanged.
    """

    if config.action_spec == QPOS8_ACTION_SPEC:
        return False
    return install_grasp_initialization_compatibility(task, config.task.task_id)


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
    initial_seed: int,
) -> None:
    initial_seed = _validated_initial_seed(initial_seed)
    if getattr(cfg, "step_lim", None) != config.task.action_horizon:
        raise UniVTACContractError("upstream task action horizon mismatch")
    cfg.seed = initial_seed
    cfg.decimation = config.decimation
    cfg.sim.dt = 1.0 / float(config.sim_hz)
    cfg.sim.render_interval = config.decimation
    if device is not None:
        cfg.sim.device = device
    cfg.scene.num_envs = 1
    embodiment = ["joint"]
    if config.action_spec == EE8_ACTION_SPEC:
        embodiment.append("ee")
    cfg.obs_data_type = {
        "camera": ["rgb"],
        "tactile": [config.aliases.tactile_payload, "depth"],
        "embodiment": embodiment,
    }
    cfg.save_frequency = 0
    cfg.video_frequency = 0
    cfg.render_frequency = 0
    cfg.random_texture = False
    cfg.tactile_sensor_type = config.aliases.sensor_type
    cfg.save_dir = runtime_dir
    if not hasattr(cfg, "reset_time_limit"):
        raise UniVTACContractError("upstream task config lacks reset_time_limit")
    cfg.reset_time_limit = _resolved_reset_time_limit_s(cfg.reset_time_limit)


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


def _enable_upstream_headless_extensions() -> Tuple[str, ...]:
    kit_app = importlib.import_module("omni.kit.app")
    get_app = getattr(kit_app, "get_app", None)
    if not callable(get_app):
        raise UniVTACContractError("omni.kit.app.get_app is unavailable")
    app = get_app()
    get_manager = getattr(app, "get_extension_manager", None)
    if not callable(get_manager):
        raise UniVTACContractError("Isaac extension manager is unavailable")
    manager = get_manager()
    enable = getattr(manager, "set_extension_enabled_immediate", None)
    is_enabled = getattr(manager, "is_extension_enabled", None)
    if not callable(enable) or not callable(is_enabled):
        raise UniVTACContractError("Isaac extension manager contract is incomplete")
    for extension_id in _UPSTREAM_HEADLESS_EXTENSION_IDS:
        enable(extension_id, True)
        if not is_enabled(extension_id):
            raise UniVTACContractError(
                f"required Isaac extension did not enable: {extension_id}"
            )
    return _UPSTREAM_HEADLESS_EXTENSION_IDS


def _build_action_encoder(
    action_spec: str, task: Any, torch: Any
) -> Callable[[Array], Any]:
    """Build the exact host/device representation required by UniVTAC."""

    if action_spec == EE8_ACTION_SPEC:

        def encode_ee_action(row: Array) -> _EncodedEEAction:
            return _EncodedEEAction(row)

        return encode_ee_action
    if action_spec != QPOS8_ACTION_SPEC:
        raise UniVTACContractError("unsupported action spec for UniVTAC encoding")

    as_tensor = getattr(torch, "as_tensor", None)
    float32 = getattr(torch, "float32", None)
    if not callable(as_tensor) or float32 is None:
        raise UniVTACContractError("torch float32 tensor conversion is unavailable")

    def encode_qpos_action(row: Array) -> Any:
        contiguous = np.array(row, dtype=np.float32, order="C", copy=True)
        return as_tensor(contiguous, dtype=float32, device=task.device)

    return encode_qpos_action


def _launch_univtac_application(
    config: UniVTACBackendConfig,
    *,
    upstream_root: Path,
    launcher_args: Optional[Mapping[str, Any]] = None,
    stage_observer: Optional[Callable[[str], None]] = None,
) -> _UniVTACApplicationResources:
    """Launch and prepare reusable application-scoped Isaac resources."""

    _record_lifecycle_stage(stage_observer, "app_launcher_import")
    isaac_app = importlib.import_module("isaaclab.app")
    app_launcher_type = getattr(isaac_app, "AppLauncher", None)
    if not callable(app_launcher_type):
        raise UniVTACContractError("isaaclab.app.AppLauncher is unavailable")
    arguments = _resolved_launcher_args(launcher_args)
    _record_lifecycle_stage(stage_observer, "app_launcher")
    launcher = app_launcher_type(argparse.Namespace(**arguments))
    simulation_app = getattr(launcher, "app", None)
    if simulation_app is None or not callable(getattr(simulation_app, "close", None)):
        raise UniVTACContractError("AppLauncher did not expose a closeable app")
    construction_stage = "runtime_preparation"
    try:
        _record_lifecycle_stage(stage_observer, construction_stage)
        require_hang_detector_disabled()
        install_runtime_signal_tracing()
        _enable_upstream_headless_extensions()
        torch = importlib.import_module("torch")
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
        return _UniVTACApplicationResources(
            simulation_app=simulation_app,
            torch=torch,
            task_cfg_type=task_cfg_type,
            task_type=task_type,
        )
    except BaseException as error:
        _emit_runtime_construction_failure(construction_stage, error)
        with suppress(Exception):
            _close_runtime_component(simulation_app.close, "Isaac application")
        if isinstance(error, SystemExit):
            raise UniVTACContractError(
                "UniVTAC runtime construction aborted via SystemExit during "
                f"{construction_stage}: {error.code!r}"
            ) from error
        raise


def _construct_univtac_task_runtime(
    config: UniVTACBackendConfig,
    resources: _UniVTACApplicationResources,
    *,
    runtime_dir: Path,
    initial_seed: int,
    device: Optional[str],
    antialiasing_mode: Optional[str],
    n0_action_execution_contract: Optional[str],
    reset_trajectory: Optional[UniVTACPreMoveTrajectory],
    stage_observer: Optional[Callable[[str], None]],
) -> UniVTACTaskRuntime:
    """Construct one seed-bound task whose close callback never closes the app."""

    initial_seed = _validated_initial_seed(initial_seed)
    runtime_dir = runtime_dir.resolve()
    task: Any = None
    construction_stage = "runtime_preparation"
    try:
        _prepare_process_determinism(initial_seed)
        _seed_torch_process(resources.torch, initial_seed)
        cfg = resources.task_cfg_type()
        if antialiasing_mode is not None:
            sim_cfg = getattr(cfg, "sim", None)
            render_cfg = getattr(sim_cfg, "render", None)
            if render_cfg is None or not hasattr(render_cfg, "antialiasing_mode"):
                raise UniVTACContractError(
                    "UniVTAC task config lacks render.antialiasing_mode"
                )
            render_cfg.antialiasing_mode = antialiasing_mode
        runtime_dir.mkdir(parents=True, exist_ok=True)
        _configure_task_cfg(cfg, config, runtime_dir, device, initial_seed)
        construction_stage = "tactile_constructor_hook"
        _record_lifecycle_stage(stage_observer, construction_stage)
        install_gsmini_attachment_constructor_compatibility(config.task.task_id)
        construction_stage = "univtac_task_construction"
        _record_lifecycle_stage(stage_observer, construction_stage)
        task = resources.task_type(cfg, mode="eval")
        construction_stage = "tactile_attachment_validation"
        _record_lifecycle_stage(stage_observer, construction_stage)
        repair_gsmini_tactile_attachments(task, config.task.task_id)
        construction_stage = "runtime_compatibility_installation"
        _record_lifecycle_stage(stage_observer, construction_stage)
        _install_task_seed_hook(task, resources.torch, initial_seed)
        _install_grasp_initialization_for_action_spec(task, config)
        _install_grasp_approach_compatibility(task, config.task.task_id)
        _install_constrained_placement_compatibility(task, config.task.task_id)
        install_local_ik_fallback(task, config.task.task_id)
        install_planner_failure_diagnostics(task, config.task.task_id)
        if reset_trajectory is not None:
            if config.action_spec != QPOS8_ACTION_SPEC:
                raise UniVTACContractError(
                    "dense reset trajectory replay is only valid for QPOS8"
                )
            install_pre_move_trajectory_replay(task, reset_trajectory)
        install_n0_evaluation_reset(task, config)
        if (
            config.action_spec == QPOS8_ACTION_SPEC
            and config.physics_steps_per_action != 1
        ):
            from robotactile_benchmark.backends.univtac_qpos_cadence import (
                install_retrained_qpos_cadence,
            )

            install_retrained_qpos_cadence(task, config)
        if config.action_spec == EE8_ACTION_SPEC:
            execution_contract, diagnostic_only = _resolve_n0_action_execution_contract(
                n0_action_execution_contract
                or (
                    config.action_execution_contract
                    if config.action_execution_contract
                    == N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT
                    else None
                )
            )
            install_n0_action_execution(
                task,
                config,
                execution_contract=execution_contract,
                diagnostic_only=diagnostic_only,
            )
        names = _live_joint_names(task)
        handshake = config.expected_handshake(names)
        config.validate_handshake(handshake)
        encode_action = _build_action_encoder(config.action_spec, task, resources.torch)

        def prepare_reset() -> None:
            _prepare_process_determinism(initial_seed)
            _seed_torch_process(resources.torch, initial_seed)

        def close_runtime() -> None:
            _close_runtime_component(task.close, "UniVTAC task")

        capture_state, restore_state, snapshot_state_sha256 = (
            build_live_snapshot_callbacks(task)
        )
        construction_stage = "runtime_ready"
        _record_lifecycle_stage(stage_observer, construction_stage)
        return UniVTACTaskRuntime(
            task=task,
            handshake=handshake,
            construction_seed=initial_seed,
            encode_action=encode_action,
            prepare_reset=prepare_reset,
            close_runtime=close_runtime,
            capture_state=capture_state,
            restore_state=restore_state,
            snapshot_state_sha256=snapshot_state_sha256,
        )
    except BaseException as error:
        _emit_runtime_construction_failure(construction_stage, error)
        with suppress(Exception):
            if task is not None and callable(getattr(task, "close", None)):
                _close_runtime_component(task.close, "UniVTAC task")
        if isinstance(error, SystemExit):
            raise UniVTACContractError(
                "UniVTAC runtime construction aborted via SystemExit during "
                f"{construction_stage}: {error.code!r}"
            ) from error
        raise


def launch_univtac_app_host(
    config: UniVTACBackendConfig,
    *,
    upstream_root: Path,
    initial_seed: int,
    launcher_args: Optional[Mapping[str, Any]] = None,
    device: Optional[str] = None,
    antialiasing_mode: Optional[str] = None,
    n0_action_execution_contract: Optional[str] = None,
    reset_trajectory: Optional[UniVTACPreMoveTrajectory] = None,
    stage_observer: Optional[Callable[[str], None]] = None,
) -> UniVTACSimulationAppHost:
    """Launch one app host that creates a fresh task runtime per episode."""

    upstream_root = upstream_root.resolve()
    initial_seed = _validated_initial_seed(initial_seed)
    resolved_antialiasing = _resolved_antialiasing_mode(config, antialiasing_mode)
    _verify_checkout(upstream_root, config)
    _prepare_process_determinism(initial_seed)
    resources = _launch_univtac_application(
        config,
        upstream_root=upstream_root,
        launcher_args=launcher_args,
        stage_observer=stage_observer,
    )

    def create_task_runtime(
        runtime_dir: Path,
        seed: int,
        runtime_stage_observer: Optional[Callable[[str], None]],
    ) -> UniVTACTaskRuntime:
        return _construct_univtac_task_runtime(
            config,
            resources,
            runtime_dir=runtime_dir,
            initial_seed=seed,
            device=device,
            antialiasing_mode=resolved_antialiasing,
            n0_action_execution_contract=n0_action_execution_contract,
            reset_trajectory=reset_trajectory,
            stage_observer=runtime_stage_observer,
        )

    def close_application() -> None:
        _close_runtime_component(resources.simulation_app.close, "Isaac application")

    def finalize_task_runtime(
        runtime: UniVTACTaskRuntime,
        runtime_stage_observer: Optional[Callable[[str], None]],
    ) -> None:
        prepare_univtac_task_teardown(
            runtime.task,
            stage_observer=runtime_stage_observer,
        )

    def reconstruct_task_stage(
        runtime_stage_observer: Optional[Callable[[str], None]],
    ) -> None:
        reconstruct_univtac_stage(
            resources.simulation_app,
            stage_observer=runtime_stage_observer,
        )

    return UniVTACSimulationAppHost(
        task_runtime_factory=create_task_runtime,
        close_application=close_application,
        task_runtime_finalizer=finalize_task_runtime,
        task_reconstruction_barrier=reconstruct_task_stage,
    )


def launch_univtac_runtime(
    config: UniVTACBackendConfig,
    *,
    upstream_root: Path,
    runtime_dir: Path,
    initial_seed: int,
    launcher_args: Optional[Mapping[str, Any]] = None,
    device: Optional[str] = None,
    antialiasing_mode: Optional[str] = None,
    n0_action_execution_contract: Optional[str] = None,
    reset_trajectory: Optional[UniVTACPreMoveTrajectory] = None,
    stage_observer: Optional[Callable[[str], None]] = None,
) -> UniVTACTaskRuntime:
    """Launch a one-shot task runtime with the legacy task-plus-app close API."""

    host = launch_univtac_app_host(
        config,
        upstream_root=upstream_root,
        initial_seed=initial_seed,
        launcher_args=launcher_args,
        device=device,
        antialiasing_mode=antialiasing_mode,
        n0_action_execution_contract=n0_action_execution_contract,
        reset_trajectory=reset_trajectory,
        stage_observer=stage_observer,
    )
    runtime = host.create_runtime(
        runtime_dir=runtime_dir,
        initial_seed=initial_seed,
        stage_observer=stage_observer,
    )
    close_task_runtime = runtime.close_runtime
    closed = False

    def close_runtime() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            close_task_runtime()
        finally:
            host.close()

    return replace(runtime, close_runtime=close_runtime)
