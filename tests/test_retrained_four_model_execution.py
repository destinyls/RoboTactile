"""Retrained protocols remain opt-in and preserve actual simulation cadence."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_n0_retrained_live import retrained_request

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_qpos_cadence import (
    install_retrained_qpos_cadence,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.live_artifacts_values import live_request_identity
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.integrations.ftp1_policy.transport import (
    FTP1PolicyTransportError,
    OfficialFTP1PolicyClient,
)


def test_source_scripts_are_not_shadowed_by_external_runtime(tmp_path):
    foreign = tmp_path / "foreign/scripts"
    foreign.mkdir(parents=True)
    (foreign / "__init__.py").write_text("EXTERNAL_PACKAGE = True\n")
    root = Path(__file__).resolve().parents[1]
    checked = subprocess.run(
        [sys.executable, "-c", "import scripts; print(scripts.__file__)"],
        env=dict(os.environ, PYTHONPATH=f"{root}{os.pathsep}{foreign.parent}"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(checked.stdout.strip()) == root / "scripts/__init__.py"


@pytest.mark.parametrize(
    "kind,chunk",
    [
        (LivePolicyKind.N0_VTLA, 50),
        (LivePolicyKind.FTP1_POLICY, 1),
        (LivePolicyKind.DREAM_TAC, 20),
    ],
)
def test_training_request_roundtrip(tmp_path, kind, chunk):
    request = replace(
        retrained_request(tmp_path),
        policy_kind=kind,
        n0_action_per_frame=12,
        n0_prompt_override=None,
        retrained_prompt="exact training prompt",
        retrained_control_hz=10,
        retrained_tactile_payload="rgb_marker",
        execute_action_steps=chunk,
        max_observation_steps=101,
    )
    path = tmp_path / "request.json"
    path.write_text(json.dumps(live_univtac_request_to_dict(request)))
    loaded = load_live_univtac_run(load_live_univtac_request(path))
    assert loaded.request == request
    assert loaded.backend_config.physics_steps_per_action == 12
    assert loaded.backend_config.aliases.tactile_payload == "rgb_marker"
    assert loaded.run_spec.prompt == "exact training prompt"
    assert live_request_identity(loaded)["retrained_control_hz"] == 10
    from robotactile_benchmark.closed_loop.fakes import (
        DeterministicFakeBackend,
        DeterministicFakePolicy,
    )
    from robotactile_benchmark.closed_loop.runner import (
        run_closed_loop_trial_with_evidence,
    )
    from robotactile_benchmark.execution.live_artifacts import (
        load_live_univtac_artifact,
        write_live_univtac_artifact,
    )
    from robotactile_benchmark.fixtures import make_synthetic_episode

    backend = DeterministicFakeBackend(
        make_synthetic_episode(length=101),
        success_predicate_id=loaded.run_spec.success_predicate_id,
    )
    backend.action_spec = loaded.trial.action_spec
    evidence = run_closed_loop_trial_with_evidence(
        loaded.trial,
        loaded.run_spec,
        backend,
        DeterministicFakePolicy.for_trial(loaded.trial),
    )
    target = tmp_path / "artifact"
    write_live_univtac_artifact(
        target, loaded, evidence, capture_profile=LiveCaptureProfile.PREVIEW
    )
    assert load_live_univtac_artifact(target).evidence.result == evidence.result


@pytest.mark.parametrize("hz,ticks", [(10, 12), (60, 2)])
def test_qpos_rate_advances_real_ticks_and_checks_official_success(hz, ticks):
    config = build_univtac_backend_config(
        "lift_can", action_spec=QPOS8_ACTION_SPEC, control_hz=hz
    )
    arm, gripper, renders = [], [], []
    task = SimpleNamespace(
        mode="eval",
        cfg=SimpleNamespace(decimation=1, step_lim=500),
        take_action_cnt=0,
        eval_success=False,
        step_count=7,
    )
    task.take_action = lambda *args, **kwargs: (False, False)
    task._robot_manager = SimpleNamespace(
        set_arm=lambda value, **kwargs: arm.append(value),
        set_gripper=lambda value, **kwargs: gripper.append(value),
    )
    task._step = lambda **kwargs: setattr(task, "step_count", task.step_count + 1)
    task._update_render = lambda: renders.append(task.step_count)
    task.check_success = lambda: task.step_count == 7 + ticks
    assert install_retrained_qpos_cadence(task, config)
    assert task.take_action(list(range(8))) == (True, True)
    assert task.step_count == 7 + ticks and task.mode == "eval"
    assert len(arm) == len(gripper) == len(renders) == 1
    assert task.take_action_cnt == 1


def test_legacy_qpos_cadence_unchanged():
    config = build_univtac_backend_config("lift_can", action_spec=QPOS8_ACTION_SPEC)
    assert config.physics_steps_per_action == 1
    assert not install_retrained_qpos_cadence(SimpleNamespace(), config)


def test_ftp_retrained_requires_explicit_training_contract():
    from scripts.ftp1_policy.serve_official import FTP1PolicyEngine, TaskContract

    wrapper = SimpleNamespace(
        get_state_dim=lambda: 120,
        get_action_dim=lambda: 120,
        get_action_horizon=lambda: 32,
        model_config=SimpleNamespace(use_tactile_input=True),
    )
    engine = FTP1PolicyEngine(
        wrapper,
        task_id="grasp_classify",
        source_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        serve_bundle_sha256="c" * 64,
        task_contract=TaskContract("training prompt", ("top",)),
    )
    with pytest.raises(FTP1PolicyTransportError, match="unsupported"):
        OfficialFTP1PolicyClient._validate_metadata_shape(engine.metadata)
    OfficialFTP1PolicyClient._validate_metadata_shape(
        engine.metadata, ("grasp_classify", "training prompt", False)
    )
    with pytest.raises(FTP1PolicyTransportError, match="prompt"):
        OfficialFTP1PolicyClient._validate_metadata_shape(
            engine.metadata, ("grasp_classify", "wrong prompt", False)
        )


def test_group_fault_windows_cross_inference_boundary(tmp_path):
    from scripts.retrained_evaluation.group import prepare_group

    binding = {
        "model": "n0_vtla",
        "training_step": 160000,
        "deployment_root": str(tmp_path),
        "dataset_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "binding_sha256": "c" * 64,
        "normalizer_sha256": "d" * 64,
        "source_commit": "e" * 40,
        "control_hz": 10,
        "tactile_payload": "rgb",
        "tasks": {"lift_can": {"prompt": "trained prompt"}},
        "rest_references": {},
    }
    output = tmp_path / "group"
    path = prepare_group(binding, "lift_can", output, 0)
    plan = json.loads(path.read_text())
    assert len(plan["ordered_requests"]) > 1
    assert any(
        item["status"] == "not_run_missing_calibration" for item in plan["excluded"]
    )
    freeze = json.loads(
        (output / "faults/optical_marker_v1/T2_held_last_freeze.json").read_text()
    )
    assert freeze["start_index"] == 50
    with pytest.raises(FileExistsError):
        prepare_group(binding, "lift_can", output, 0)


@pytest.mark.parametrize(
    "operator", ["T1_fixed_source_delay", "T2_held_last_freeze", "T3_inter_sensor_skew"]
)
def test_live_retrained_source_clock_passes_temporal_delivery(operator):
    import numpy as np

    from robotactile_benchmark.backends.qualification_fakes import (
        FakeUpstreamScenario,
        make_fake_runtime,
    )
    from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
    from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
    from robotactile_benchmark.manifests import FaultManifest, Observability
    from robotactile_benchmark.streaming import StreamingFaultSession

    config = build_univtac_backend_config("pull_out_key", control_hz=10)
    runtime, _ = make_fake_runtime(
        config,
        construction_seed=0,
        scenario=FakeUpstreamScenario(native_step_increment=12),
    )
    backend = UniVTACIsaacBackend(config, runtime)
    backend.reset(
        PolicyEpisodeContext(
            episode_id="clock-test",
            task="pull_out_key",
            initial_seed=0,
            exogenous_seed=0,
            instruction=config.task.prompt,
            action_spec=config.action_spec,
        )
    )
    manifest = FaultManifest(
        operator_id=operator,
        severity_level=5,
        operator_seed=0,
        start_index=20,
        stop_index=50,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        parameters={"sample_period_s": 0.1},
        severity_registry="optical_marker_v1",
    )
    stream = StreamingFaultSession(manifest)
    assert (
        stream.deliver_one(backend.observe()).provenance_for("left").source_time_s == 0
    )
    action = np.zeros((1, 8), dtype=np.float32)
    action[:, 3], action[:, 7] = -1, 0.02
    for step in range(1, 23):
        batch = backend.execute(action)
        record = batch.transitions[0].clean_record
        assert record.provenance_for("left").source_time_s == pytest.approx(step / 10)
        delivered = stream.deliver_one(record)
        if step == 22 and operator == "T1_fixed_source_delay":
            assert delivered.provenance_for("left").source_time_s == pytest.approx(1.4)
    backend.close()
