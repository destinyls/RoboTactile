"""Fail-closed Isaac task teardown and USD reconstruction barriers."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Callable
from typing import Optional, cast

from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACContractError,
)

StageObserver = Callable[[str], None]
ModuleImporter = Callable[[str], object]
GarbageCollector = Callable[[], int]

DEFAULT_STAGE_UPDATE_LIMIT = 4
_REPLICATOR_RENDER_PRODUCT = "Replicator"
_ENV_ROOT_PATH = "/World/envs"


def _notify(observer: Optional[StageObserver], stage: str) -> None:
    if observer is None:
        return
    try:
        observer(stage)
    except Exception as error:
        raise UniVTACContractError(
            f"UniVTAC lifecycle observer failed at {stage}"
        ) from error


def _require_callable(owner: object, name: str, label: str) -> Callable[..., object]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise UniVTACContractError(f"{label} is unavailable")
    return cast(Callable[..., object], value)


def _invoke(
    callback: Callable[..., object],
    *args: object,
    label: str,
) -> object:
    try:
        return callback(*args)
    except Exception as error:
        raise UniVTACContractError(f"{label} failed") from error


def _import_module(import_module: ModuleImporter, name: str) -> object:
    try:
        module = import_module(name)
    except Exception as error:
        raise UniVTACContractError(f"unable to import {name}") from error
    if module is None:
        raise UniVTACContractError(f"importer returned no module for {name}")
    return module


def _root_layer_identifier(stage: object) -> str:
    get_root_layer = _require_callable(
        stage,
        "GetRootLayer",
        "USD stage GetRootLayer",
    )
    root_layer = _invoke(
        get_root_layer,
        label="USD stage root-layer query",
    )
    identifier = getattr(root_layer, "identifier", None)
    if not isinstance(identifier, str) or not identifier:
        raise UniVTACContractError("USD root layer lacks a stable identifier")
    return identifier


def _require_no_simulation_context(instance: Callable[..., object], stage: str) -> None:
    current = _invoke(instance, label="SimulationContext.instance query")
    if current is not None:
        raise UniVTACContractError(f"SimulationContext must be cleared before {stage}")


def prepare_univtac_task_teardown(
    task: object,
    stage_observer: Optional[StageObserver] = None,
    *,
    import_module: ModuleImporter = importlib.import_module,
) -> None:
    """Release task-native renderer state, then stop its timeline.

    ``task.close()`` remains owned by the app host and is deliberately not called
    here.  All required runtime capabilities are resolved before teardown starts,
    so an invalid task fails without claiming any lifecycle progress.
    """

    if stage_observer is not None and not callable(stage_observer):
        raise TypeError("stage_observer must be callable")
    if not callable(import_module):
        raise TypeError("import_module must be callable")

    replicator = _import_module(import_module, "omni.replicator.core")
    viewport_manager = getattr(replicator, "vp_manager", None)
    destroy_hydra_textures = _require_callable(
        viewport_manager,
        "destroy_hydra_textures",
        "omni.replicator.core.vp_manager.destroy_hydra_textures",
    )
    simulation = getattr(task, "sim", None)
    timeline = getattr(simulation, "_timeline", None)
    stop_timeline = _require_callable(
        timeline,
        "stop",
        "UniVTAC task timeline stop",
    )

    _notify(stage_observer, "task_teardown_start")
    _invoke(
        destroy_hydra_textures,
        _REPLICATOR_RENDER_PRODUCT,
        label="Replicator hydra-texture destruction",
    )
    _notify(stage_observer, "renderer_released")
    _invoke(stop_timeline, label="UniVTAC task timeline stop")
    _notify(stage_observer, "timeline_stopped")


def reconstruct_univtac_stage(
    simulation_app: object,
    stage_observer: Optional[StageObserver] = None,
    *,
    import_module: ModuleImporter = importlib.import_module,
    collector: GarbageCollector = gc.collect,
    update_limit: int = DEFAULT_STAGE_UPDATE_LIMIT,
) -> None:
    """Reconstruct an empty USD stage before constructing the next task.

    The barrier proves that the prior ``SimulationContext`` is gone, installs a
    distinct root layer, flushes a bounded number of Kit updates, and rejects any
    stage that still exposes the prior task's ``/World/envs`` hierarchy.
    """

    if stage_observer is not None and not callable(stage_observer):
        raise TypeError("stage_observer must be callable")
    if not callable(import_module):
        raise TypeError("import_module must be callable")
    if not callable(collector):
        raise TypeError("collector must be callable")
    if isinstance(update_limit, bool) or not isinstance(update_limit, int):
        raise TypeError("update_limit must be an integer")
    if update_limit <= 0:
        raise ValueError("update_limit must be positive")

    stage_utils = _import_module(import_module, "isaacsim.core.utils.stage")
    simulation_module = _import_module(import_module, "isaaclab.sim")
    simulation_context = getattr(simulation_module, "SimulationContext", None)
    context_instance = _require_callable(
        simulation_context,
        "instance",
        "isaaclab.sim.SimulationContext.instance",
    )
    get_current_stage = _require_callable(
        stage_utils,
        "get_current_stage",
        "isaacsim.core.utils.stage.get_current_stage",
    )
    create_new_stage = _require_callable(
        stage_utils,
        "create_new_stage",
        "isaacsim.core.utils.stage.create_new_stage",
    )
    update_stage = _require_callable(
        stage_utils,
        "update_stage",
        "isaacsim.core.utils.stage.update_stage",
    )
    update_application = _require_callable(
        simulation_app,
        "update",
        "Isaac simulation application update",
    )
    prior_stage = _invoke(get_current_stage, label="current USD stage query")
    prior_root_identifier = _root_layer_identifier(prior_stage)

    _notify(stage_observer, "task_reconstruction_start")
    _invoke(collector, label="Python garbage collection")
    _notify(stage_observer, "gc_collected")
    _require_no_simulation_context(context_instance, "USD reconstruction")
    _invoke(create_new_stage, label="USD stage creation")
    _invoke(update_stage, label="USD stage update")
    _notify(stage_observer, "usd_stage_recreated")

    for _ in range(update_limit):
        _invoke(update_application, label="Isaac application update flush")

    current_stage = _invoke(get_current_stage, label="reconstructed USD stage query")
    current_root_identifier = _root_layer_identifier(current_stage)
    if current_root_identifier == prior_root_identifier:
        raise UniVTACContractError(
            "USD reconstruction retained the prior root-layer identity"
        )
    get_prim_at_path = _require_callable(
        current_stage,
        "GetPrimAtPath",
        "USD stage GetPrimAtPath",
    )
    environments_prim = _invoke(
        get_prim_at_path,
        _ENV_ROOT_PATH,
        label="USD environment-root query",
    )
    is_valid = _require_callable(
        environments_prim,
        "IsValid",
        "USD environment-root validity query",
    )
    if bool(_invoke(is_valid, label="USD environment-root validity query")):
        raise UniVTACContractError("reconstructed USD stage still contains /World/envs")
    _require_no_simulation_context(context_instance, "task reconstruction readiness")
    _notify(stage_observer, "task_reconstruction_ready")


__all__ = [
    "DEFAULT_STAGE_UPDATE_LIMIT",
    "prepare_univtac_task_teardown",
    "reconstruct_univtac_stage",
]
