"""Dream-Tac first-class adapter and content-bound artifact contracts."""

from __future__ import annotations

import json
import math
from base64 import b64decode
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.cli import main
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import ObservationRecord, SensorObservation
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.integrations.doctor import diagnose_model_integration
from robotactile_benchmark.integrations.dream_tac import (
    DreamTacArtifactManifest,
    DreamTacPolicyAdapter,
    OfficialDreamTacClient,
    build_dream_tac_artifact_manifest,
    load_dream_tac_adapter,
    load_dream_tac_artifact_manifest,
    validate_dream_tac_artifact,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_dream_tac_runtime_artifacts,
)
from robotactile_benchmark.policies.dream_tac import (
    ACTION_HORIZON,
    CLOSED_GRIPPER_QPOS,
    OPEN_GRIPPER_QPOS,
    DreamTacGripperMapping,
    OfficialDreamTacPolicy,
    dream_tac_actions_to_ee8,
    dream_tac_tactile_self_attn_gate,
    dream_tac_wire_observation,
    euler_xyz_to_quaternion_wxyz,
    unwrap_euler_xyz,
)
from scripts.retrained_evaluation.serve_dream import _install_tactile_gate_bridge

SOURCE_COMMIT = "14bab51d6862fd07124745c55cd395ea5caa9fd3"
TASK_ID = "pick_baguette"
INSTRUCTION = "pick up the baguette"
EXPERIMENT_CONFIG = "franka_pick_baguette"


class FakeClient:
    """Dependency-free witness for the official Dream-Tac client protocol."""

    def __init__(self) -> None:
        self.reset_calls: list[tuple[str, int, bool]] = []
        self.observations: list[Mapping[str, object]] = []
        self.closed = False

    def reset(
        self, *, experiment_config: str, action_horizon: int, use_tactile: bool
    ) -> None:
        self.reset_calls.append((experiment_config, action_horizon, use_tactile))

    def infer(self, observation: Mapping[str, object]) -> np.ndarray:
        self.observations.append(observation)
        actions = np.zeros((ACTION_HORIZON, 7), dtype=np.float32)
        actions[:, :3] = np.asarray((0.4, -0.2, 0.7), dtype=np.float32)
        actions[:, 5] = np.float32(math.pi / 2.0)
        actions[1::2, 6] = 1.0
        return actions

    def close(self) -> None:
        self.closed = True


def _sensor(slot_id: str, value: int) -> SensorObservation:
    return SensorObservation(
        slot_id=slot_id,
        payload=np.full((6, 8, 3), value, dtype=np.uint8),
        payload_present=True,
        declared_validity=True,
        delivery_index=0,
        delivery_time_s=0.0,
        visible_source_time_s=0.0,
        frame_id=f"{slot_id}-0",
        calibration_id=f"{slot_id}-calibration",
    )


def _observation() -> ObservationRecord:
    return ObservationRecord(
        episode_id="episode-0",
        task=TASK_ID,
        seed=3,
        step_index=0,
        tactile=(_sensor("left", 17), _sensor("right", 29)),
        vision={
            "top": np.full((10, 12, 3), 41, dtype=np.uint8),
            "wrist_l": np.full((8, 9, 3), 53, dtype=np.uint8),
        },
        proprio=np.asarray((0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.04), dtype=np.float32),
    )


def _context() -> PolicyEpisodeContext:
    return PolicyEpisodeContext(
        episode_id="episode-0",
        task=TASK_ID,
        initial_seed=3,
        exogenous_seed=7,
        instruction=INSTRUCTION,
        action_spec=EE8_ACTION_SPEC,
    )


def _write_artifact_bundle(root: Path) -> tuple[Path, Path, Path, Path]:
    bundle = (root / "dream-tac").absolute()
    checkpoint = bundle / "checkpoint"
    (checkpoint / "transformer").mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        '{"action_horizon":20,"use_tactile":true}\n', encoding="utf-8"
    )
    (checkpoint / "transformer/model.safetensors").write_bytes(b"dream-tac-weights")
    stats = bundle / "dataset_statistics_franka.json"
    stats.write_text('{"state":{"mean":[0.0]}}\n', encoding="utf-8")
    embeddings = bundle / "t5_embeddings.pkl"
    embeddings.write_bytes(b"task-embeddings")
    return bundle, checkpoint, stats, embeddings


def _manifest(
    root: Path,
    *,
    gripper_mapping: DreamTacGripperMapping = DreamTacGripperMapping.GREATER_IS_CLOSED,
    gripper_qpos_min: float | None = None,
    gripper_qpos_max: float | None = None,
) -> DreamTacArtifactManifest:
    bundle, checkpoint, stats, embeddings = _write_artifact_bundle(root)
    return build_dream_tac_artifact_manifest(
        bundle_root=bundle,
        checkpoint_root=checkpoint,
        dataset_stats_path=stats,
        t5_embeddings_path=embeddings,
        task_id=TASK_ID,
        instruction=INSTRUCTION,
        experiment_config=EXPERIMENT_CONFIG,
        control_hz=20.0,
        gripper_mapping=gripper_mapping,
        gripper_threshold=0.5,
        gripper_qpos_min=gripper_qpos_min,
        gripper_qpos_max=gripper_qpos_max,
    )


def _identity(manifest: DreamTacArtifactManifest) -> PolicyIdentity:
    return PolicyIdentity(
        system_id="dream-tac-pick-baguette",
        checkpoint_sha256=manifest.checkpoint_sha256,
        config_sha256=manifest.config_sha256,
        action_spec=EE8_ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )


def test_wire_observation_maps_two_rgb_two_tactile_and_ee8_pose() -> None:
    wire = dream_tac_wire_observation(_observation(), instruction=INSTRUCTION)

    assert set(wire) == {
        "cam_front",
        "cam_high",
        "instruction",
        "state",
        "tactile_left",
        "tactile_right",
        "tactile_self_attn_gate",
    }
    assert wire["instruction"] == INSTRUCTION
    assert np.unique(wire["cam_front"]).tolist() == [41]
    assert np.unique(wire["cam_high"]).tolist() == [53]
    assert np.unique(wire["tactile_left"]).tolist() == [17]
    assert np.unique(wire["tactile_right"]).tolist() == [29]
    np.testing.assert_allclose(
        wire["state"], np.asarray((0.1, 0.2, 0.3, 0.0, 0.0, 0.0), np.float32)
    )
    assert wire["tactile_self_attn_gate"] == pytest.approx(
        dream_tac_tactile_self_attn_gate(
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
            None,
            None,
        )
    )


def test_tactile_gate_matches_training_formula_and_uses_larger_sensor_delta() -> None:
    zeros = np.zeros((4, 5, 3), dtype=np.uint8)
    left = np.full_like(zeros, 1)
    right = np.full_like(zeros, 51)

    gate = dream_tac_tactile_self_attn_gate(left, right, zeros, zeros)
    raw = 51.0 / 255.0
    z = 4.0 * (raw - 0.002) / (0.001 + 1e-6)
    expected = 0.15 + 0.85 / (1.0 + math.exp(-min(30.0, z)))

    assert gate == pytest.approx(expected)


def test_online_euler_unwraps_across_pi_branch() -> None:
    previous = np.asarray((0.0, 0.0, math.radians(179.0)), dtype=np.float32)
    current = np.asarray((0.0, 0.0, math.radians(-179.0)), dtype=np.float32)

    continued = unwrap_euler_xyz(current, previous)

    assert continued[2] == pytest.approx(math.radians(181.0), abs=1e-6)


def test_missing_tactile_payload_fails_closed() -> None:
    observation = _observation()
    absent = observation.sensor("right").without_payload(declared=True)

    with pytest.raises(ValueError, match="both tactile payloads"):
        dream_tac_wire_observation(
            observation.replace_sensor(absent), instruction=INSTRUCTION
        )


def test_euler_xyz_is_converted_to_wxyz_quaternion() -> None:
    quaternion = euler_xyz_to_quaternion_wxyz(
        np.asarray((0.0, 0.0, math.pi / 2.0), dtype=np.float32)
    )

    np.testing.assert_allclose(
        quaternion,
        np.asarray((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), np.float32),
        atol=1e-6,
    )
    assert quaternion.dtype == np.float32
    assert quaternion.flags.writeable is False


@pytest.mark.parametrize(
    ("mapping", "expected"),
    (
        (
            DreamTacGripperMapping.GREATER_IS_CLOSED,
            (OPEN_GRIPPER_QPOS, CLOSED_GRIPPER_QPOS),
        ),
        (
            DreamTacGripperMapping.GREATER_IS_OPEN,
            (CLOSED_GRIPPER_QPOS, OPEN_GRIPPER_QPOS),
        ),
    ),
)
def test_gripper_mapping_is_explicit_in_both_supported_directions(
    mapping: DreamTacGripperMapping, expected: tuple[float, float]
) -> None:
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 6] = (0.0, 1.0)

    converted = dream_tac_actions_to_ee8(
        actions, gripper_mapping=mapping, gripper_threshold=0.5
    )

    np.testing.assert_allclose(converted[:, 7], np.asarray(expected, np.float32))


def test_direct_qpos_preserves_continuous_values_and_clips_to_physical_range() -> None:
    actions = np.zeros((7, 7), dtype=np.float32)
    actions[:, 6] = np.asarray(
        (-0.01, 0.0, 0.0005, 0.02, 0.039, 0.04, 0.05), dtype=np.float32
    )

    converted = dream_tac_actions_to_ee8(
        actions,
        gripper_mapping=DreamTacGripperMapping.DIRECT_QPOS,
        gripper_qpos_min=0.0001,
        gripper_qpos_max=0.0394,
    )

    np.testing.assert_allclose(
        converted[:, 7],
        np.asarray((0.0, 0.0, 0.0005, 0.02, 0.039, 0.04, 0.04), np.float32),
    )


def test_fake_client_policy_lifecycle_returns_full_twenty_step_chunk(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    client = FakeClient()
    policy = load_dream_tac_adapter(_identity(manifest), manifest, lambda: client)
    policy.reset(_context())

    plan = policy.infer(_observation())

    assert DreamTacPolicyAdapter is OfficialDreamTacPolicy
    assert client.reset_calls == [(EXPERIMENT_CONFIG, ACTION_HORIZON, True)]
    assert len(client.observations) == 1
    assert client.observations[0]["tactile_self_attn_gate"] == pytest.approx(
        dream_tac_tactile_self_attn_gate(
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
            None,
            None,
        )
    )
    assert plan.action_spec == EE8_ACTION_SPEC
    assert plan.actions.shape == (20, 8)
    np.testing.assert_allclose(
        plan.actions[:, :3],
        np.tile(np.asarray((0.4, -0.2, 0.7), np.float32), (20, 1)),
    )
    np.testing.assert_allclose(
        plan.actions[0, 3:7],
        (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        plan.actions[:2, 7], (OPEN_GRIPPER_QPOS, CLOSED_GRIPPER_QPOS)
    )
    policy.commit(
        PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=tuple(_observation() for _ in range(20)),
            terminal_signal=BackendSignal.RUNNING,
        )
    )
    policy.close()
    assert client.closed is True


def test_policy_carries_adjacent_tactile_gate_and_continuous_euler_across_chunk(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    client = FakeClient()
    policy = load_dream_tac_adapter(_identity(manifest), manifest, lambda: client)
    policy.reset(_context())
    initial = replace(
        _observation(),
        proprio=np.concatenate(
            (
                _observation().proprio[:3],
                euler_xyz_to_quaternion_wxyz(
                    np.asarray((0.0, 0.0, math.radians(179.0)), dtype=np.float32)
                ),
                _observation().proprio[7:],
            )
        ).astype(np.float32),
    )
    plan = policy.infer(initial)
    delivered = []
    for step in range(1, 21):
        tactile_value = 79 if step == 20 else 29
        delivered.append(
            replace(
                initial,
                step_index=step,
                tactile=(_sensor("left", 17), _sensor("right", tactile_value)),
                proprio=np.concatenate(
                    (
                        initial.proprio[:3],
                        euler_xyz_to_quaternion_wxyz(
                            np.asarray(
                                (0.0, 0.0, math.radians(-179.0)),
                                dtype=np.float32,
                            )
                        ),
                        initial.proprio[7:],
                    )
                ).astype(np.float32),
            )
        )
    policy.commit(
        PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=tuple(delivered),
            terminal_signal=BackendSignal.RUNNING,
        )
    )

    policy.infer(delivered[-1])

    second_wire = client.observations[1]
    expected_gate = dream_tac_tactile_self_attn_gate(
        np.full((6, 8, 3), 17, dtype=np.uint8),
        np.full((6, 8, 3), 79, dtype=np.uint8),
        np.full((6, 8, 3), 17, dtype=np.uint8),
        np.full((6, 8, 3), 29, dtype=np.uint8),
    )
    assert second_wire["tactile_self_attn_gate"] == pytest.approx(expected_gate)
    assert second_wire["state"][5] == pytest.approx(math.radians(181.0), abs=1e-5)


def test_direct_qpos_factory_lifecycle_preserves_continuous_gripper(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        gripper_mapping=DreamTacGripperMapping.DIRECT_QPOS,
        gripper_qpos_min=0.0,
        gripper_qpos_max=1.0,
    )
    assert manifest.to_dict()["gripper_qpos_min"] == 0.0
    assert manifest.to_dict()["gripper_qpos_max"] == 1.0
    assert DreamTacArtifactManifest.from_dict(manifest.to_dict()) == manifest
    client = FakeClient()
    policy = load_dream_tac_adapter(_identity(manifest), manifest, lambda: client)
    policy.reset(_context())

    plan = policy.infer(_observation())

    assert np.array_equal(
        plan.actions[:, 7], np.asarray(([0.0, 0.04] * 10), dtype=np.float32)
    )
    policy.commit(
        PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=tuple(_observation() for _ in range(20)),
            terminal_signal=BackendSignal.RUNNING,
        )
    )
    policy.close()
    assert client.closed is True


def test_artifact_build_load_hash_and_drift_detection(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = manifest.bundle_root / "artifact_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest.to_dict()))

    loaded = load_dream_tac_artifact_manifest(manifest_path)

    assert loaded == manifest
    assert loaded.external_commit == SOURCE_COMMIT
    assert loaded.action_horizon == 20
    assert loaded.raw_action_dim == 7
    assert loaded.benchmark_action_dim == 8
    assert loaded.control_hz == 20.0
    assert "gripper_qpos_min" not in loaded.to_dict()
    assert "gripper_qpos_max" not in loaded.to_dict()
    assert tuple(item.path for item in loaded.checkpoint_files) == (
        "config.json",
        "transformer/model.safetensors",
    )
    validate_dream_tac_artifact(loaded)

    (loaded.checkpoint_root / "transformer/model.safetensors").write_bytes(b"drift")
    with pytest.raises(ValueError, match="checkpoint file SHA256"):
        validate_dream_tac_artifact(loaded)


def test_factory_rejects_identity_mismatch_before_client_allocation(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    allocations = 0

    def client_factory() -> FakeClient:
        nonlocal allocations
        allocations += 1
        return FakeClient()

    invalid = replace(_identity(manifest), config_sha256="f" * 64)
    with pytest.raises(ValueError, match="configuration identity mismatch"):
        load_dream_tac_adapter(invalid, manifest, client_factory)

    assert allocations == 0


def test_official_http_client_validates_handshake_and_encodes_png(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OfficialDreamTacClient("http://127.0.0.1:5001")
    requests: list[tuple[str, object]] = []

    def request(path: str, *, payload: object = None) -> Mapping[str, object]:
        requests.append((path, payload))
        if path == "/info":
            return {
                "chunk_size": 20,
                "config": EXPERIMENT_CONFIG,
                "service_name": "Cosmos Policy Franka API",
                "use_tactile": True,
            }
        return {
            "actions": np.zeros((20, 7), dtype=np.float32).tolist(),
            "success": True,
        }

    monkeypatch.setattr(client, "_request", request)
    client.reset(
        experiment_config=EXPERIMENT_CONFIG,
        action_horizon=20,
        use_tactile=True,
    )
    actions = client.infer(
        dream_tac_wire_observation(_observation(), instruction=INSTRUCTION)
    )

    assert actions.shape == (20, 7)
    assert actions.dtype == np.float32
    assert actions.flags.writeable is False
    assert [path for path, _ in requests] == ["/info", "/infer"]
    payload = requests[1][1]
    assert isinstance(payload, dict)
    assert payload["tactile_self_attn_gate"] == pytest.approx(
        dream_tac_tactile_self_attn_gate(
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
            None,
            None,
        )
    )
    images = payload["images"]
    assert isinstance(images, dict)
    assert set(images) == {
        "cam_front",
        "cam_high",
        "tactile_left",
        "tactile_right",
    }
    for encoded in images.values():
        assert isinstance(encoded, str)
        assert b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


def test_server_bridge_injects_float32_gate_for_every_batch_member() -> None:
    captured: dict[str, object] = {}

    class Video:
        shape = (2, 3, 4, 5, 6)
        device = "cuda:0"

    class Model:
        def generate_samples_from_batch(
            self, data_batch: dict[str, object], *args: object, **kwargs: object
        ) -> str:
            captured.update(data_batch)
            return "generated"

    model = Model()

    class Request:
        @staticmethod
        def get_json(*, silent: bool) -> dict[str, float]:
            assert silent is True
            return {"tactile_self_attn_gate": 0.625}

    class Server:
        request = Request()

        @staticmethod
        def get_action(*args: object, **kwargs: object) -> str:
            return model.generate_samples_from_batch({"video": Video()})

    class Torch:
        float32 = np.float32

        @staticmethod
        def full(
            shape: tuple[int, ...],
            value: float,
            *,
            dtype: object,
            device: object,
        ) -> np.ndarray:
            assert device == "cuda:0"
            return np.full(shape, value, dtype=dtype)

    server = Server()
    _install_tactile_gate_bridge(server, model, Torch())

    assert server.get_action() == "generated"
    gate = captured["tactile_self_attn_gate"]
    assert isinstance(gate, np.ndarray)
    assert gate.dtype == np.float32
    assert gate.tolist() == [0.625, 0.625]


def test_official_http_client_rejects_non_origin_endpoint() -> None:
    with pytest.raises(ValueError, match="origin URL"):
        OfficialDreamTacClient("http://127.0.0.1:5001/infer")


def test_configure_cli_publishes_reloadable_dream_tac_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deployment = tmp_path / "deployment"
    bundle, checkpoint, stats, embeddings = _write_artifact_bundle(tmp_path)
    command = [
        "integrations",
        "configure",
        "dream-tac",
        "--root",
        str(deployment),
        "--bundle-root",
        str(bundle),
        "--checkpoint-root",
        str(checkpoint),
        "--dataset-stats",
        str(stats),
        "--t5-embeddings",
        str(embeddings),
        "--task",
        TASK_ID,
        "--instruction",
        INSTRUCTION,
        "--experiment-config",
        EXPERIMENT_CONFIG,
        "--control-hz",
        "20",
        "--gripper-mapping",
        DreamTacGripperMapping.DIRECT_QPOS.value,
        "--gripper-qpos-min",
        "0.0",
        "--gripper-qpos-max",
        "0.04",
    ]

    assert main(command) == 0
    payload = json.loads(capsys.readouterr().out)
    config_path = Path(payload["integration_config"])
    runtime = resolve_dream_tac_runtime_artifacts(config_path)

    assert runtime.manifest.task_id == TASK_ID
    assert runtime.manifest.instruction == INSTRUCTION
    assert runtime.manifest.control_hz == 20.0
    assert runtime.manifest.gripper_mapping == DreamTacGripperMapping.DIRECT_QPOS.value
    assert runtime.manifest.gripper_qpos_min == 0.0
    assert runtime.manifest.gripper_qpos_max == 0.04
    assert config_path == (
        deployment
        / "artifacts/models/dream_tac/configs"
        / TASK_ID
        / "integration_config.json"
    )
    doctor = diagnose_model_integration(
        integration_id="dream_tac",
        layout=DeploymentLayout(deployment),
        config_path=config_path,
        task_id=TASK_ID,
    )
    checks = {check.check_id: check for check in doctor.checks}
    assert checks["artifact_manifest"].passed is True
    assert checks["transport"].passed is True
    assert checks["official_checkpoint_release"].passed is False
    assert checks["paper_inference_parity"].passed is False
    assert checks["univtac_task_alignment"].passed is False
    assert doctor.passed is False
