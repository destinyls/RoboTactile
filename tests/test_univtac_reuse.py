from __future__ import annotations

import types
from collections.abc import Callable

import pytest

from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACContractError,
)
from robotactile_benchmark.backends.univtac_reuse import (
    prepare_univtac_task_teardown,
    reconstruct_univtac_stage,
)


class _RootLayer:
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier


class _Prim:
    def __init__(self, valid: bool) -> None:
        self._valid = valid

    def IsValid(self) -> bool:
        return self._valid


class _Stage:
    def __init__(self, identifier: str, *, envs_valid: bool = False) -> None:
        self._root_layer = _RootLayer(identifier)
        self._envs_valid = envs_valid
        self.queried_paths: list[str] = []

    def GetRootLayer(self) -> _RootLayer:
        return self._root_layer

    def GetPrimAtPath(self, path: str) -> _Prim:
        self.queried_paths.append(path)
        return _Prim(self._envs_valid)


def _reconstruction_importer(
    *,
    old_stage: _Stage,
    new_stage: _Stage,
    events: list[str],
    context_values: list[object | None],
) -> Callable[[str], object]:
    state = {"stage": old_stage}

    def create_new_stage() -> None:
        events.append("create_new_stage")
        state["stage"] = new_stage

    def get_current_stage() -> _Stage:
        return state["stage"]

    def context_instance() -> object | None:
        events.append("context_instance")
        if context_values:
            return context_values.pop(0)
        return None

    modules = {
        "isaacsim.core.utils.stage": types.SimpleNamespace(
            create_new_stage=create_new_stage,
            get_current_stage=get_current_stage,
            update_stage=lambda: events.append("update_stage"),
        ),
        "isaaclab.sim": types.SimpleNamespace(
            SimulationContext=types.SimpleNamespace(instance=context_instance)
        ),
    }

    def import_module(name: str) -> object:
        events.append(f"import:{name}")
        return modules[name]

    return import_module


def test_prepare_teardown_releases_renderer_then_stops_timeline() -> None:
    events: list[str] = []

    class Timeline:
        def stop(self) -> None:
            events.append("timeline_stop")

    class Task:
        sim = types.SimpleNamespace(_timeline=Timeline())

        def close(self) -> None:
            raise AssertionError("helper must not own task.close")

    replicator = types.SimpleNamespace(
        vp_manager=types.SimpleNamespace(
            destroy_hydra_textures=lambda name: events.append(f"destroy:{name}")
        )
    )

    prepare_univtac_task_teardown(
        Task(),
        lambda stage: events.append(f"stage:{stage}"),
        import_module=lambda name: (
            replicator
            if name == "omni.replicator.core"
            else pytest.fail(f"unexpected import: {name}")
        ),
    )

    assert events == [
        "stage:task_teardown_start",
        "destroy:Replicator",
        "stage:renderer_released",
        "timeline_stop",
        "stage:timeline_stopped",
    ]


def test_prepare_teardown_validates_native_contract_before_mutation() -> None:
    destroy_calls: list[str] = []
    replicator = types.SimpleNamespace(
        vp_manager=types.SimpleNamespace(destroy_hydra_textures=destroy_calls.append)
    )

    with pytest.raises(UniVTACContractError, match="timeline stop"):
        prepare_univtac_task_teardown(
            types.SimpleNamespace(sim=types.SimpleNamespace()),
            import_module=lambda _name: replicator,
        )

    assert destroy_calls == []


def test_prepare_teardown_stops_after_renderer_failure() -> None:
    timeline_stops = 0

    def fail_renderer(_name: str) -> None:
        raise RuntimeError("renderer busy")

    def stop_timeline() -> None:
        nonlocal timeline_stops
        timeline_stops += 1

    task = types.SimpleNamespace(
        sim=types.SimpleNamespace(_timeline=types.SimpleNamespace(stop=stop_timeline))
    )
    replicator = types.SimpleNamespace(
        vp_manager=types.SimpleNamespace(destroy_hydra_textures=fail_renderer)
    )

    with pytest.raises(UniVTACContractError, match="hydra-texture"):
        prepare_univtac_task_teardown(
            task,
            import_module=lambda _name: replicator,
        )

    assert timeline_stops == 0


def test_reconstruct_stage_enforces_barrier_and_bounded_flush() -> None:
    events: list[str] = []
    old_stage = _Stage("anon:old:World.usd", envs_valid=True)
    new_stage = _Stage("anon:new:World.usd")
    importer = _reconstruction_importer(
        old_stage=old_stage,
        new_stage=new_stage,
        events=events,
        context_values=[None, None],
    )
    simulation_app = types.SimpleNamespace(update=lambda: events.append("app_update"))

    reconstruct_univtac_stage(
        simulation_app,
        lambda stage: events.append(f"stage:{stage}"),
        import_module=importer,
        collector=lambda: events.append("gc_collect") or 0,
        update_limit=3,
    )

    runtime_events = [event for event in events if not event.startswith("import:")]
    assert runtime_events == [
        "stage:task_reconstruction_start",
        "gc_collect",
        "stage:gc_collected",
        "context_instance",
        "create_new_stage",
        "update_stage",
        "stage:usd_stage_recreated",
        "app_update",
        "app_update",
        "app_update",
        "context_instance",
        "stage:task_reconstruction_ready",
    ]
    assert new_stage.queried_paths == ["/World/envs"]


def test_reconstruct_stage_rejects_uncleared_context_before_recreation() -> None:
    events: list[str] = []
    importer = _reconstruction_importer(
        old_stage=_Stage("old"),
        new_stage=_Stage("new"),
        events=events,
        context_values=[object()],
    )

    with pytest.raises(UniVTACContractError, match="must be cleared"):
        reconstruct_univtac_stage(
            types.SimpleNamespace(update=lambda: events.append("app_update")),
            import_module=importer,
            collector=lambda: events.append("gc_collect") or 0,
            update_limit=2,
        )

    assert "create_new_stage" not in events
    assert "app_update" not in events


@pytest.mark.parametrize(
    ("new_stage", "context_values", "message"),
    [
        (_Stage("old"), [None, None], "root-layer identity"),
        (_Stage("new", envs_valid=True), [None, None], "/World/envs"),
        (_Stage("new"), [None, object()], "must be cleared"),
    ],
)
def test_reconstruct_stage_rejects_residual_runtime_state(
    new_stage: _Stage,
    context_values: list[object | None],
    message: str,
) -> None:
    events: list[str] = []
    importer = _reconstruction_importer(
        old_stage=_Stage("old"),
        new_stage=new_stage,
        events=events,
        context_values=context_values,
    )

    with pytest.raises(UniVTACContractError, match=message):
        reconstruct_univtac_stage(
            types.SimpleNamespace(update=lambda: events.append("app_update")),
            import_module=importer,
            collector=lambda: 0,
            update_limit=1,
        )


@pytest.mark.parametrize("update_limit", [0, -1, True, 1.5])
def test_reconstruct_stage_rejects_invalid_update_limit(
    update_limit: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="update_limit"):
        reconstruct_univtac_stage(
            types.SimpleNamespace(update=lambda: None),
            import_module=lambda _name: pytest.fail("must validate first"),
            update_limit=update_limit,  # type: ignore[arg-type]
        )
