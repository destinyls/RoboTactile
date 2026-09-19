"""Explicit retrained 10 Hz client and executor contracts; no robot claims."""

from dataclasses import replace

import numpy as np
import pytest
from test_n0_official_policy import _identity, _native, _record
from test_univtac_n0_cadence import _task

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.qualification_fakes import (
    FakeUpstreamScenario,
    make_fake_runtime,
)
from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    UniVTACContractError,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_conversion import UniVTACConversionError
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.backends.univtac_n0_cadence import (
    install_n0_action_execution,
)
from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    PolicyEpisodeContext,
    PolicyExecution,
)
from robotactile_benchmark.integrations.n0_twam.dynamic_contract import cadence_gate
from robotactile_benchmark.policies.n0_input_profile import (
    N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
)
from robotactile_benchmark.policies.n0_official import (
    OfficialN0Policy,
    native_to_ee8_actions,
)
from robotactile_benchmark.transport.n0_official import OfficialN0Client


class RPC:
    def __init__(self):
        self.requests = []

    def get_server_metadata(self):
        return {}

    def infer(self, payload):
        self.requests.append(payload)
        if payload.get("reset") or payload.get("compute_kv_cache"):
            return {}
        result = _native()[:, :, :4].copy()
        result[0] = np.arange(8).reshape(2, 4) / 10
        return {"action": result}

    def close(self):
        pass


def test_retrained_cold_four_warm_eight_and_every_action_grounded():
    rpc = RPC()
    client = OfficialN0Client(rpc, action_per_frame=4)
    prompt = "Exact retrained task prompt"
    policy = OfficialN0Policy(
        _identity(),
        lambda: client,
        input_profile=N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
        action_per_frame=4,
        prompt_override=prompt,
    )
    policy.reset(
        PolicyEpisodeContext(
            episode_id=_record(0).episode_id,
            task="pull_out_key",
            initial_seed=17,
            exogenous_seed=29,
            instruction=prompt,
            action_spec=EE8_ACTION_SPEC,
        )
    )
    for start, length in ((0, 4), (4, 8)):
        plan = policy.infer(_record(start))
        assert plan.actions.shape == (length, 8)
        expected = np.arange(4, 8) / 10 if start == 0 else np.arange(8) / 10
        np.testing.assert_allclose(plan.actions[:, 0], expected)
        delivered = tuple(_record(i) for i in range(start + 1, start + length + 1))
        policy.commit(
            PolicyExecution(
                action_plan_sha256=plan.sha256,
                executed_actions=plan.actions,
                delivered_observations=delivered,
                terminal_signal=BackendSignal.RUNNING,
            )
        )
        commit = rpc.requests[-1]
        assert len(commit["obs"]) == length
        assert commit["state"].shape == (20, 2, 4)
        assert commit["prompt"] == prompt
        for received, record in zip(commit["obs"], delivered):
            np.testing.assert_array_equal(
                received["observation.images.top"], record.vision["top"]
            )
    policy.close()


def test_wrong_frame_count_and_degenerate_cold_conditioning():
    native = _native()[:, :, :4].copy()
    native[:, 0] = 0
    assert native_to_ee8_actions(native, cold_chunk=True, action_per_frame=4).shape == (
        4,
        8,
    )
    with pytest.raises(ValueError):
        native_to_ee8_actions(native, cold_chunk=False, action_per_frame=4)
    with pytest.raises(ValueError):
        native_to_ee8_actions(native, cold_chunk=True)
    for value in (True, 0, 8):
        with pytest.raises(ValueError):
            OfficialN0Client(RPC(), action_per_frame=value)


@pytest.mark.parametrize("actual_ticks", [12, 2])
def test_backend_10hz_checks_real_transition_delta(actual_ticks):
    config = build_univtac_backend_config(
        "pull_out_key",
        EE8_ACTION_SPEC,
        n0_action_execution_contract=N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    )
    assert (
        config.action_execution_contract == N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT
    )
    assert config.physics_steps_per_action == 12 and config.decimation == 1
    runtime, task = make_fake_runtime(
        config, scenario=FakeUpstreamScenario(native_step_increment=actual_ticks)
    )
    task._robotactile_n0_fixed_cadence_enabled = True
    task._robotactile_n0_action_execution_contract = config.action_execution_contract
    backend = UniVTACIsaacBackend(config, runtime)
    receipt = backend.reset(
        PolicyEpisodeContext(
            episode_id="episode-1",
            task="pull_out_key",
            initial_seed=11,
            exogenous_seed=999,
            instruction=config.task.prompt,
            action_spec=EE8_ACTION_SPEC,
        )
    )
    assert receipt.diagnostics["camera_delivery_hz"] == 10
    backend.observe()
    action = np.asarray([[0.1, 0, 0, 1, 0, 0, 0, 0.02]], dtype=np.float32)
    if actual_ticks == 12:
        assert (
            backend.execute(action).transitions[0].diagnostics["physics_step_delta"]
            == 12
        )
    else:
        with pytest.raises(UniVTACConversionError, match="native step"):
            backend.execute(action)


def test_legacy_defaults_and_unregistered_cadence_rejected():
    config = build_univtac_backend_config("pull_out_key", EE8_ACTION_SPEC)
    assert (
        config.action_execution_contract == N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
    )
    assert config.physics_steps_per_action == 2
    with pytest.raises(UniVTACContractError):
        replace(config, physics_steps_per_action=6)


def test_dynamic_gate_uses_supplied_10hz_and_feedback_cadence():
    diagnostics = {
        "native_step_delta": 12,
        "physics_step_delta": 12,
        "n0_fixed_cadence": {
            "stock_move_loop_used": False,
            "status": "Success",
            "render_contract": "one_endpoint_render_no_intermediate_render_v1",
        },
    }
    result = cadence_gate(
        diagnostics,
        sim_hz=120,
        decimation=1,
        physics_steps_per_action=12,
        action_rows_per_keyframe=1,
    )
    assert result["passed"]
    assert result["required"]["action_endpoint_hz"] == 10
    assert result["required"]["feedback_keyframe_hz"] == 10


def test_installed_10hz_executor_advances_twelve_ticks_and_one_render():
    config = build_univtac_backend_config(
        "lift_bottle",
        EE8_ACTION_SPEC,
        n0_action_execution_contract=N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    )
    task = _task()
    install_n0_action_execution(
        task,
        config,
        execution_contract=N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
        diagnostic_only=False,
    )
    action = np.asarray([0.4, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02], dtype=np.float32)
    assert task.take_action(action, action_type="ee", force=True) == (True, False)
    assert task.step_count == 29
    assert task.step_modes == ["eval_test"] * 12
    assert task.render_count == 1 and task.mode == "eval"
    with pytest.raises(UniVTACContractError, match="does not match"):
        install_n0_action_execution(
            _task(),
            config,
            execution_contract=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
            diagnostic_only=False,
        )
