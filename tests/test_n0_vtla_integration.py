"""Official N0-VTLA first-class integration contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.cli import main
from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import ObservationRecord, SensorObservation
from robotactile_benchmark.integrations.n0_vtla import (
    N0VTLAArtifactManifest,
    N0VTLAPolicyAdapter,
    build_n0_vtla_artifact_manifest,
    load_n0_vtla_adapter,
    validate_n0_vtla_artifact,
)
from robotactile_benchmark.integrations.n0_vtla.artifacts import (
    ASSET_ID,
    CHECKPOINT_REVISION,
)
from robotactile_benchmark.integrations.registry import load_model_integration_config
from robotactile_benchmark.policies.n0_vtla import OfficialN0VTLAPolicy

SOURCE_COMMIT = "03a0ce4d7091ca2354864796770715aa212601b7"


class FakeClient:
    def __init__(self) -> None:
        self.reset_count = 0
        self.requests: list[dict[str, object]] = []
        self.closed = False

    def reset(self) -> None:
        self.reset_count += 1

    def infer(self, observation: object) -> np.ndarray:
        assert isinstance(observation, dict)
        self.requests.append(observation)
        result = np.zeros((50, 32), dtype=np.float32)
        result[:, :8] = np.arange(400, dtype=np.float32).reshape(50, 8)
        return result

    def close(self) -> None:
        self.closed = True


def _identity(checkpoint: str = "a" * 64, config: str = "b" * 64) -> PolicyIdentity:
    return PolicyIdentity(
        system_id="n0-vtla-insert-hole",
        checkpoint_sha256=checkpoint,
        config_sha256=config,
        action_spec=QPOS8_ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )


def _context() -> PolicyEpisodeContext:
    return PolicyEpisodeContext(
        episode_id="episode-0",
        task="insert_hole",
        initial_seed=3,
        exogenous_seed=7,
        instruction="insert hole",
        action_spec=QPOS8_ACTION_SPEC,
    )


def _sensor(slot: str, value: int) -> SensorObservation:
    return SensorObservation(
        slot_id=slot,
        payload=np.full((6, 8, 3), value, dtype=np.uint8),
        payload_present=True,
        declared_validity=True,
        delivery_index=0,
        delivery_time_s=0.0,
        visible_source_time_s=0.0,
        frame_id=f"{slot}-0",
        calibration_id=f"{slot}-calibration",
    )


def _observation() -> ObservationRecord:
    return ObservationRecord(
        episode_id="episode-0",
        task="insert_hole",
        seed=3,
        step_index=0,
        tactile=(_sensor("left", 17), _sensor("right", 29)),
        vision={
            "top": np.full((10, 12, 3), 41, dtype=np.uint8),
            "wrist_l": np.full((8, 9, 3), 53, dtype=np.uint8),
        },
        proprio=np.arange(8, dtype=np.float32),
    )


def _manifest(root: Path) -> N0VTLAArtifactManifest:
    checkpoint_root = root / "checkpoint"
    return N0VTLAArtifactManifest(
        bundle_root=root,
        checkpoint_root=checkpoint_root,
        checkpoint_path=checkpoint_root / "model.safetensors",
        checkpoint_sha256="a" * 64,
        config_path=checkpoint_root / "config.json",
        config_sha256="b" * 64,
        normalizer_path=(checkpoint_root / "assets" / ASSET_ID / "norm_stats.json"),
        normalizer_sha256="c" * 64,
        external_commit=SOURCE_COMMIT,
    )


def test_adapter_delivers_real_tactile_and_full_official_chunk() -> None:
    client = FakeClient()
    policy = OfficialN0VTLAPolicy(_identity(), lambda: client)
    policy.reset(_context())

    plan = policy.infer(_observation())

    assert client.reset_count == 1
    assert plan.actions.shape == (50, 8)
    assert np.array_equal(plan.actions, np.arange(400, dtype=np.float32).reshape(50, 8))
    request = client.requests[0]
    assert request["prompt"] == "insert hole"
    assert np.unique(request["observation/left_tactile"]).tolist() == [17]
    assert np.unique(request["observation/right_tactile"]).tolist() == [29]
    policy.commit(
        PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=tuple(_observation() for _ in range(50)),
            terminal_signal=BackendSignal.RUNNING,
        )
    )


def test_running_commit_rejects_truncated_chunk() -> None:
    policy = OfficialN0VTLAPolicy(_identity(), FakeClient)
    policy.reset(_context())
    plan = policy.infer(_observation())

    with pytest.raises(RuntimeError, match="full 50-step"):
        policy.commit(
            PolicyExecution(
                action_plan_sha256=plan.sha256,
                executed_actions=plan.actions[:6],
                delivered_observations=tuple(_observation() for _ in range(6)),
                terminal_signal=BackendSignal.RUNNING,
            )
        )


def test_factory_binds_hashes_before_lazy_client(tmp_path: Path) -> None:
    effects = 0

    def client_factory() -> FakeClient:
        nonlocal effects
        effects += 1
        return FakeClient()

    policy = load_n0_vtla_adapter(
        _identity(), _manifest(tmp_path.absolute()), client_factory
    )

    assert isinstance(policy, OfficialN0VTLAPolicy)
    assert N0VTLAPolicyAdapter is OfficialN0VTLAPolicy
    assert effects == 0


def test_artifact_builder_hashes_released_layout(tmp_path: Path) -> None:
    bundle = (tmp_path / "bundle").absolute()
    checkpoint = bundle / "checkpoint"
    normalizer = checkpoint / "assets" / ASSET_ID / "norm_stats.json"
    normalizer.parent.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "config.json").write_text(
        '{"config":"sim_single_arm_tactile"}\n', encoding="utf-8"
    )
    normalizer.write_text('{"norm":true}\n', encoding="utf-8")

    manifest = build_n0_vtla_artifact_manifest(
        bundle_root=bundle,
        checkpoint_root=checkpoint,
    )

    validate_n0_vtla_artifact(manifest)
    assert manifest.checkpoint_revision == CHECKPOINT_REVISION
    assert manifest.external_commit == SOURCE_COMMIT


def test_configure_cli_materializes_hash_bound_n0_vtla_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deployment = (tmp_path / "deployment").absolute()
    bundle = deployment / "artifacts/models/n0_vtla"
    checkpoint = bundle / "checkpoint"
    normalizer = checkpoint / "assets" / ASSET_ID / "norm_stats.json"
    normalizer.parent.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "config.json").write_text(
        '{"config":"sim_single_arm_tactile"}\n', encoding="utf-8"
    )
    normalizer.write_text('{"norm":true}\n', encoding="utf-8")

    assert (
        main(
            [
                "integrations",
                "configure",
                "n0-vtla",
                "--root",
                str(deployment),
            ]
        )
        == 0
    )
    capsys.readouterr()
    config_path = bundle / "configs/insert_hole/integration_config.json"
    config = load_model_integration_config("n0_vtla", config_path)

    assert config.transport == "official_zmq"
    assert Path(config.artifact_manifest).is_file()
