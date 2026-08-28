"""Reusable Isaac application ownership for isolated UniVTAC task runtimes."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Optional

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError
from robotactile_benchmark.backends.univtac_isaac import UniVTACTaskRuntime

StageObserver = Callable[[str], None]
TaskRuntimeFactory = Callable[
    [Path, int, Optional[StageObserver]],
    UniVTACTaskRuntime,
]
TaskRuntimeFinalizer = Callable[
    [UniVTACTaskRuntime, Optional[StageObserver]],
    None,
]
TaskReconstructionBarrier = Callable[[Optional[StageObserver]], None]


class UniVTACSimulationAppHost:
    """Own one SimulationApp while leasing one isolated task runtime at a time.

    The host deliberately does not reuse a task.  Closing a leased runtime closes
    only that task and releases the lease; closing the host closes the application.
    A construction failure poisons and closes the host because a partially created
    Isaac task cannot be proven safe for a later episode.
    """

    def __init__(
        self,
        *,
        task_runtime_factory: TaskRuntimeFactory,
        close_application: Callable[[], None],
        task_runtime_finalizer: Optional[TaskRuntimeFinalizer] = None,
        task_reconstruction_barrier: Optional[TaskReconstructionBarrier] = None,
    ) -> None:
        if not callable(task_runtime_factory) or not callable(close_application):
            raise TypeError("host callbacks must be callable")
        for callback, name in (
            (task_runtime_finalizer, "task_runtime_finalizer"),
            (task_reconstruction_barrier, "task_reconstruction_barrier"),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable")
        self._task_runtime_factory = task_runtime_factory
        self._close_application = close_application
        self._task_runtime_finalizer = task_runtime_finalizer
        self._task_reconstruction_barrier = task_reconstruction_barrier
        self._active_token: Optional[object] = None
        self._active_close: Optional[Callable[[], None]] = None
        self._runtime_count = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active(self) -> bool:
        return self._active_token is not None

    @property
    def runtime_count(self) -> int:
        return self._runtime_count

    def create_runtime(
        self,
        *,
        runtime_dir: Path,
        initial_seed: int,
        stage_observer: Optional[StageObserver] = None,
    ) -> UniVTACTaskRuntime:
        """Create one task-local runtime without transferring app ownership."""

        if self._closed:
            raise UniVTACContractError(
                "closed UniVTAC app host cannot create a runtime"
            )
        if self._active_token is not None:
            raise UniVTACContractError(
                "UniVTAC app host already has an active task runtime"
            )
        if not isinstance(runtime_dir, Path):
            raise TypeError("runtime_dir must be a pathlib.Path")
        if isinstance(initial_seed, bool) or not isinstance(initial_seed, int):
            raise UniVTACContractError("initial seed must be an integer")
        if initial_seed < 0:
            raise UniVTACContractError("initial seed must be non-negative")
        if stage_observer is not None and not callable(stage_observer):
            raise TypeError("stage_observer must be callable")

        try:
            if (
                self._runtime_count > 0
                and self._task_reconstruction_barrier is not None
            ):
                self._task_reconstruction_barrier(stage_observer)
            runtime = self._task_runtime_factory(
                runtime_dir.resolve(),
                initial_seed,
                stage_observer,
            )
        except BaseException:
            with suppress(BaseException):
                self.close()
            raise
        if not isinstance(runtime, UniVTACTaskRuntime):
            with suppress(BaseException):
                self.close()
            raise TypeError("task runtime factory returned an invalid runtime")

        task_close = runtime.close_runtime
        token = object()
        lease_closed = False

        def close_task_runtime() -> None:
            nonlocal lease_closed
            if lease_closed:
                return
            lease_closed = True
            task_error: Optional[BaseException] = None
            try:
                finalizer = self._task_runtime_finalizer
                if finalizer is not None:
                    try:
                        finalizer(runtime, stage_observer)
                    except BaseException as error:
                        task_error = error
                try:
                    task_close()
                    if task_error is None and stage_observer is not None:
                        stage_observer("task_closed")
                except BaseException as error:
                    if task_error is None:
                        task_error = error
            finally:
                if self._active_token is token:
                    self._active_token = None
                    self._active_close = None
            if task_error is not None:
                with suppress(BaseException):
                    self._poison_application()
                raise task_error

        hosted_runtime = replace(runtime, close_runtime=close_task_runtime)
        self._active_token = token
        self._active_close = close_task_runtime
        self._runtime_count += 1
        return hosted_runtime

    def close(self) -> None:
        """Close an active task first, then close the shared application once."""

        if self._closed:
            return
        self._closed = True
        task_error: Optional[BaseException] = None
        active_close = self._active_close
        if active_close is not None:
            try:
                active_close()
            except BaseException as error:
                task_error = error
        try:
            self._close_application()
        except BaseException:
            if task_error is None:
                raise
        if task_error is not None:
            raise task_error

    def _poison_application(self) -> None:
        """Close the application once after an unsafe task lifecycle failure."""

        if self._closed:
            return
        self._closed = True
        self._close_application()
