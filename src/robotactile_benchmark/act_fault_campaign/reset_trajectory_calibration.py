"""One-reset ACT trajectory calibration without policy construction."""

from __future__ import annotations

import math
import sys
import traceback
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

from robotactile_benchmark.backends.univtac_factory import launch_univtac_runtime
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.backends.univtac_reference_ik import (
    install_reference_guided_pre_move_ik,
)
from robotactile_benchmark.backends.univtac_reset_trajectory import (
    UniVTACPreMoveTrajectory,
)
from robotactile_benchmark.backends.univtac_reset_trajectory_runtime import (
    install_pre_move_trajectory_capture,
)
from robotactile_benchmark.backends.univtac_reset_witness import (
    UniVTACResetReference,
)
from robotactile_benchmark.closed_loop.runner_checks import build_episode_context
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.trials import Condition

from .contracts import ACTFaultCampaignError
from .reset_reference import load_act_reset_reference
from .reset_trajectory import write_act_reset_trajectory


def capture_act_reset_trajectory(
    *,
    clean_request_path: Path,
    reset_reference_path: Path,
    output_path: Path,
) -> tuple[str, UniVTACPreMoveTrajectory, str]:
    """Capture and publish one qualified dense ``pre_move`` trajectory.

    Exactly one upstream reset is executed.  No ACT policy object is loaded,
    reset, inferred, or allowed to act.
    """

    request = load_live_univtac_request(clean_request_path)
    loaded = load_live_univtac_run(request)
    reference = load_act_reset_reference(reset_reference_path)
    _validate_sources(loaded, reference)

    runtime = launch_univtac_runtime(
        loaded.backend_config,
        upstream_root=request.upstream_root,
        runtime_dir=request.runtime_dir,
        initial_seed=loaded.trial.initial_seed,
        launcher_args=request.launcher_args,
        device=request.simulator_device,
    )
    backend: UniVTACIsaacBackend | None = None
    trajectory: UniVTACPreMoveTrajectory | None = None
    reset_receipt_sha256: str | None = None
    created: bool | None = None
    try:
        install_reference_guided_pre_move_ik(
            runtime.task,
            tuple(reference.expected_qpos8[:7]),
        )
        capture = install_pre_move_trajectory_capture(
            runtime.task,
            expected_native_step=reference.expected_native_step,
        )
        backend = UniVTACIsaacBackend(
            loaded.backend_config,
            runtime,
        )
        context = build_episode_context(loaded.trial, loaded.run_spec.prompt)
        receipt = backend.reset(context)
        reset_receipt_sha256 = receipt.sha256
        actual_native_step, actual_qpos8, qpos8_max_abs_error = (
            _require_source_proximate_reset(receipt.diagnostics, reference)
        )
        trajectory = capture.finish(
            task_id=loaded.trial.task,
            action_spec=loaded.trial.action_spec,
            initial_seed=loaded.trial.initial_seed,
            exogenous_seed=loaded.trial.exogenous_seed,
            pair_key=loaded.trial.pair_key,
            dataset_sha256=loaded.trial.dataset_sha256,
            checkpoint_sha256=loaded.trial.checkpoint_sha256,
            config_sha256=loaded.trial.config_sha256,
            source_run_content_sha256=loaded.content_sha256,
            reset_reference_sha256=reference.sha256,
            capture_reset_receipt_sha256=receipt.sha256,
            capture_simulator_state_sha256=receipt.simulator_state_sha256,
            capture_native_step=actual_native_step,
            capture_qpos8=actual_qpos8,
            source_qpos8_max_abs_error=qpos8_max_abs_error,
            upstream_commit=loaded.backend_config.upstream_commit,
            task_source_sha256=loaded.backend_config.task.task_source_sha256,
        )
        # Isaac's application close may terminate the interpreter with a clean
        # process exit.  Publish the qualified artifact before entering native
        # teardown so an exit-0 close cannot discard a completed calibration.
        created = write_act_reset_trajectory(output_path, trajectory)
    except BaseException as error:
        _emit_calibration_failure(error)
        raise
    finally:
        if backend is not None:
            backend.close()
        else:
            runtime.close_runtime()
    if trajectory is None or reset_receipt_sha256 is None or created is None:
        raise ACTFaultCampaignError("reset trajectory capture produced no artifact")
    return (
        "created" if created else "already_present",
        trajectory,
        reset_receipt_sha256,
    )


def _emit_calibration_failure(error: BaseException) -> None:
    """Flush a reset failure before native Isaac teardown can exit cleanly."""

    sys.stderr.write(
        "robotactile_reset_trajectory_calibration_failure:"
        f"{type(error).__name__}:{error}\n"
    )
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()


def _validate_sources(
    loaded: LoadedLiveUniVTACRun,
    reference: UniVTACResetReference,
) -> None:
    request = loaded.request
    trial = loaded.trial
    if (
        request.policy_kind is not LivePolicyKind.ACT
        or request.condition is not Condition.CLEAN
        or trial.condition is not Condition.CLEAN
        or trial.task != "grasp_classify"
    ):
        raise ACTFaultCampaignError(
            "trajectory calibration requires ACT grasp_classify Clean"
        )
    if (
        reference.task_id != trial.task
        or reference.initial_seed != trial.initial_seed
        or reference.exogenous_seed != trial.exogenous_seed
        or reference.pair_key != trial.pair_key
        or reference.dataset_sha256 != trial.dataset_sha256
        or reference.checkpoint_sha256 != trial.checkpoint_sha256
        or reference.config_sha256 != trial.config_sha256
        or reference.source_run_content_sha256 != loaded.content_sha256
    ):
        raise ACTFaultCampaignError(
            "reset reference identity differs from trajectory calibration Clean"
        )


def _require_source_proximate_reset(
    diagnostics: Mapping[str, object],
    reference: UniVTACResetReference,
) -> tuple[int, tuple[float, ...], float]:
    """Require source-proximate robot state without exact pixel-state equality."""

    if diagnostics.get("plan_success") is not True:
        raise ACTFaultCampaignError(
            "reference-guided calibration reset plan failed: "
            f"plan_success={diagnostics.get('plan_success')!r}"
        )
    task = diagnostics.get("task")
    if not isinstance(task, Mapping):
        raise ACTFaultCampaignError("calibration reset lacks task diagnostics")
    witness = task.get("reset_witness")
    if not isinstance(witness, Mapping):
        raise ACTFaultCampaignError("calibration reset lacks reset witness")

    native_step = witness.get("native_step")
    if isinstance(native_step, bool) or not isinstance(native_step, Integral):
        raise ACTFaultCampaignError(
            f"calibration reset witness native_step is not an integer: {native_step!r}"
        )
    actual_native_step = int(native_step)
    if actual_native_step != reference.expected_native_step:
        raise ACTFaultCampaignError(
            "reference-guided calibration native step differs from source: "
            f"actual={actual_native_step}, "
            f"expected={reference.expected_native_step}"
        )

    raw_qpos8 = witness.get("qpos8")
    if (
        isinstance(raw_qpos8, (str, bytes))
        or not isinstance(raw_qpos8, (list, tuple))
        or len(raw_qpos8) != 8
    ):
        raise ACTFaultCampaignError(
            "calibration reset witness qpos8 must contain exactly 8 values: "
            f"{raw_qpos8!r}"
        )
    normalized_qpos8: list[float] = []
    for index, value in enumerate(raw_qpos8):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ACTFaultCampaignError(
                "calibration reset witness qpos8 is non-numeric: "
                f"index={index}, value={value!r}"
            )
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ACTFaultCampaignError(
                "calibration reset witness qpos8 is non-finite: "
                f"index={index}, value={value!r}"
            )
        normalized_qpos8.append(normalized)
    actual_qpos8 = tuple(normalized_qpos8)
    max_abs_error = max(
        abs(actual - expected)
        for actual, expected in zip(actual_qpos8, reference.expected_qpos8)
    )
    if max_abs_error > 0.002:
        raise ACTFaultCampaignError(
            "reference-guided calibration qpos8 differs from source: "
            f"max_abs_error={max_abs_error!r}, threshold=0.002, "
            f"actual={list(actual_qpos8)!r}, "
            f"expected={list(reference.expected_qpos8)!r}"
        )
    return actual_native_step, actual_qpos8, max_abs_error


__all__ = ["capture_act_reset_trajectory"]
