"""Behavioral checks shared by the CPU UniVTAC qualification runner."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from robotactile_benchmark.backends.qualification_fakes import (
    FakeUpstreamScenario,
    make_fake_runtime,
)
from robotactile_benchmark.backends.qualification_receipt import (
    expected_terminal_priority_results,
)
from robotactile_benchmark.backends.univtac_contracts import UniVTACBackendConfig
from robotactile_benchmark.backends.univtac_conversion import (
    UniVTACConversionError,
)
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.backends.univtac_signals import resolve_backend_signal
from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    BackendSignal,
    PolicyEpisodeContext,
)
from robotactile_benchmark.contracts import Array, canonical_hash


class UniVTACQualificationCheckError(ValueError):
    """A stable CPU behavioral check failed to demonstrate its contract."""


def qualification_context(config: UniVTACBackendConfig) -> PolicyEpisodeContext:
    """Build the fixed policy/simulator seed separation used by CPU checks."""

    return PolicyEpisodeContext(
        episode_id=canonical_hash(
            {
                "namespace": "robotactile.univtac.cpu-qualification.v1",
                "task_id": config.task.task_id,
                "initial_seed": 11,
            }
        ),
        task=config.task.task_id,
        initial_seed=11,
        exogenous_seed=999,
        instruction=config.task.prompt,
        action_spec=ACTION_SPEC,
    )


def qualification_action() -> Array:
    """Return one bounded action that deterministically changes fake state."""

    action = np.zeros((1, 8), dtype=np.float32)
    action[0, 0] = 0.1
    action[0, 3] = -1.0
    action[0, 7] = 0.02
    return action


def run_terminal_priority_checks(config: UniVTACBackendConfig) -> Dict[str, str]:
    """Exercise pure precedence and every task-applicable backend branch."""

    pure_cases = (
        (
            BackendSignal.SUCCESS,
            dict(
                success=True,
                execution_success=False,
                plan_success=False,
                early_stop=True,
                horizon_reached=True,
            ),
        ),
        (
            BackendSignal.TASK_FAILURE,
            dict(
                success=False,
                execution_success=True,
                plan_success=False,
                early_stop=True,
                horizon_reached=True,
            ),
        ),
        (
            BackendSignal.EARLY_STOP,
            dict(
                success=False,
                execution_success=True,
                plan_success=True,
                early_stop=True,
                horizon_reached=True,
            ),
        ),
        (
            BackendSignal.TIMEOUT,
            dict(
                success=False,
                execution_success=True,
                plan_success=True,
                early_stop=False,
                horizon_reached=True,
            ),
        ),
    )
    for expected, arguments in pure_cases:
        if resolve_backend_signal(**arguments) is not expected:
            raise UniVTACQualificationCheckError(
                "terminal priority resolver qualification failed"
            )

    cases = [
        (
            "success_over_failure",
            FakeUpstreamScenario(
                success_steps=(1,),
                execution_failure_steps=(1,),
                plan_failure_steps=(1,),
                early_stop_steps=(1,),
            ),
            BackendSignal.SUCCESS,
        ),
        (
            "failure_over_early",
            FakeUpstreamScenario(plan_failure_steps=(1,), early_stop_steps=(1,)),
            BackendSignal.TASK_FAILURE,
        ),
    ]
    if config.task.early_stop_capable:
        cases.append(
            (
                "early_over_timeout",
                FakeUpstreamScenario(early_stop_steps=(1,)),
                BackendSignal.EARLY_STOP,
            )
        )
    results = expected_terminal_priority_results(config.task.early_stop_capable)
    for name, scenario, expected in cases:
        runtime, task = make_fake_runtime(config, scenario=scenario)
        backend = UniVTACIsaacBackend(config, runtime)
        try:
            backend.reset(qualification_context(config))
            backend.observe()
            actual = backend.execute(qualification_action()).transitions[-1].signal
            if actual is not expected:
                raise UniVTACQualificationCheckError(
                    "terminal priority qualification failed"
                )
            if results[name] != actual.value:
                raise UniVTACQualificationCheckError("terminal result receipt mismatch")
        finally:
            backend.close()
        if task.close_count != 1:
            raise UniVTACQualificationCheckError("terminal runtime cleanup failed")
    return results


def _expect_conversion_error(
    operation: Callable[[], object], *, code: str, message: str
) -> None:
    try:
        operation()
    except UniVTACConversionError as error:
        if error.code != code or str(error) != message:
            raise UniVTACQualificationCheckError(
                f"conversion gate mismatch: expected {code} / {message}"
            ) from error
    else:
        raise UniVTACQualificationCheckError(f"conversion gate did not fail: {code}")


def run_failure_gate_checks(config: UniVTACBackendConfig) -> None:
    """Exercise exact malformed-input/action/native gates plus cleanup."""

    missing_runtime, missing_task = make_fake_runtime(
        config, scenario=FakeUpstreamScenario(missing_field="left_tactile")
    )
    missing_backend = UniVTACIsaacBackend(config, missing_runtime)
    try:
        _expect_conversion_error(
            lambda: missing_backend.reset(qualification_context(config)),
            code="missing_required_field",
            message="required UniVTAC field is missing: tactile/left_tactile",
        )
    finally:
        missing_backend.close()

    malformed_runtime, malformed_task = make_fake_runtime(
        config, scenario=FakeUpstreamScenario(malformed_field="head_rgb")
    )
    malformed_backend = UniVTACIsaacBackend(config, malformed_runtime)
    try:
        _expect_conversion_error(
            lambda: malformed_backend.reset(qualification_context(config)),
            code="field_shape",
            message=(
                "head RGB shape mismatch: expected "
                f"{config.head_shape}, got (10, 10, 3)"
            ),
        )
    finally:
        malformed_backend.close()

    nan_runtime, nan_task = make_fake_runtime(
        config, scenario=FakeUpstreamScenario(nan_field="right_depth")
    )
    nan_backend = UniVTACIsaacBackend(config, nan_runtime)
    try:
        _expect_conversion_error(
            lambda: nan_backend.reset(qualification_context(config)),
            code="field_nonfinite",
            message="right tactile depth must be finite",
        )
    finally:
        nan_backend.close()

    invalid_runtime, invalid_task = make_fake_runtime(config)
    invalid_backend = UniVTACIsaacBackend(config, invalid_runtime)
    try:
        invalid_backend.reset(qualification_context(config))
        invalid_backend.observe()
        invalid_action = qualification_action().copy()
        invalid_action[0, 0] = np.nan
        _expect_conversion_error(
            lambda: invalid_backend.execute(invalid_action),
            code="action_nonfinite",
            message="actions must be finite",
        )
        if invalid_task.take_action_calls:
            raise UniVTACQualificationCheckError(
                "invalid action caused an upstream side effect"
            )
    finally:
        invalid_backend.close()

    no_progress_runtime, no_progress_task = make_fake_runtime(
        config, scenario=FakeUpstreamScenario(native_step_increment=0)
    )
    no_progress_backend = UniVTACIsaacBackend(config, no_progress_runtime)
    try:
        no_progress_backend.reset(qualification_context(config))
        no_progress_backend.observe()
        _expect_conversion_error(
            lambda: no_progress_backend.execute(qualification_action()),
            code="native_step_continuity",
            message="native step continuity mismatch after take_action",
        )
    finally:
        no_progress_backend.close()
    tasks = (
        missing_task,
        malformed_task,
        nan_task,
        invalid_task,
        no_progress_task,
    )
    if any(task.close_count != 1 for task in tasks):
        raise UniVTACQualificationCheckError("failure-gate cleanup failed")
