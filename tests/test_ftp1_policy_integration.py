"""FTP-1 first-class integration and content-binding contracts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import ObservationRecord, SensorObservation
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.integrations.ftp1_policy import (
    FTP1PolicyAdapter,
    build_ftp1_policy_artifact_manifest,
    build_official_ftp1_policy_request,
    load_ftp1_policy_adapter,
    validate_ftp1_policy_artifact,
)
from robotactile_benchmark.integrations.ftp1_policy.artifacts import (
    CHECKPOINT_REVISION,
    TASK_RELEASES,
    FTP1PolicyArtifactManifest,
)
from robotactile_benchmark.integrations.ftp1_policy.transport import (
    decode_wire_value,
    encode_wire_value,
)
from robotactile_benchmark.integrations.registry import (
    load_model_integration_config,
)
from robotactile_benchmark.policies.ftp1_policy import OfficialFTP1Policy
from robotactile_benchmark.trials import Condition


class FakeClient:
    def __init__(self) -> None:
        self.reset_calls: list[tuple[str, str, int]] = []
        self.observations: list[Mapping[str, object]] = []
        self.closed = False

    def reset(self, *, task_id: str, prompt: str, seed: int) -> object:
        self.reset_calls.append((task_id, prompt, seed))
        return {"status": "ok"}

    def infer(self, observation: Mapping[str, object]) -> np.ndarray:
        self.observations.append(observation)
        return np.arange(8, dtype=np.float32)

    def close(self) -> None:
        self.closed = True


def _write_bundle(root: Path, task_id: str = "insert_hole") -> Path:
    release = TASK_RELEASES[task_id]
    checkpoint = root / release.checkpoint_name / "19999"
    tokenizer = checkpoint / "hpt_tokenizer"
    normalization = checkpoint / "normalization"
    domain = normalization / release.domain_name
    tokenizer.mkdir(parents=True)
    domain.mkdir(parents=True)
    for name, payload in (
        ("model.safetensors", b"official-model"),
        ("model_config.json", b'{"action_dim":120}\n'),
        ("train_config.json", b'{"action_joint_rep":"mix"}\n'),
        ("tactile_input_config_file.json", b'{"use_tactile":true}\n'),
    ):
        (checkpoint / name).write_bytes(payload)
    (tokenizer / "GelSightMini_image_224_224_3.safetensors").write_bytes(
        b"gelsight-tokenizer"
    )
    (tokenizer / "shared_image_chunk_encoder.safetensors").write_bytes(
        b"shared-tokenizer"
    )
    for name in (
        "action_group_frequency_stats_train.json",
        "dataset_stats.json",
        "norm_params_snapshot.json",
        "share_norm_stats_all_t0_zscore.json",
    ):
        (normalization / name).write_text(f'{{"file":"{name}"}}\n', encoding="utf-8")
    for name in (
        "contact_detection_thresholds.json",
        "independent_norm_stats_all_t0_zscore.json",
        "train_val_split.json",
    ):
        (domain / name).write_text(f'{{"file":"{name}"}}\n', encoding="utf-8")
    return checkpoint


def _manifest(
    tmp_path: Path, task_id: str = "insert_hole"
) -> FTP1PolicyArtifactManifest:
    bundle = (tmp_path / "ftp1").absolute()
    checkpoint = _write_bundle(bundle, task_id)
    return build_ftp1_policy_artifact_manifest(
        bundle_root=bundle,
        checkpoint_root=checkpoint,
        task_id=task_id,
    )


def _identity(manifest: FTP1PolicyArtifactManifest) -> PolicyIdentity:
    return PolicyIdentity(
        system_id=f"ftp1-{manifest.task_id}",
        checkpoint_sha256=manifest.checkpoint_sha256,
        config_sha256=manifest.config_sha256,
        action_spec=QPOS8_ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
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


def _observation(task_id: str = "insert_hole") -> ObservationRecord:
    return ObservationRecord(
        episode_id="episode-0",
        task=task_id,
        seed=3,
        step_index=0,
        tactile=(_sensor("left", 17), _sensor("right", 29)),
        vision={
            "top": np.full((10, 12, 3), 41, dtype=np.uint8),
            "wrist_l": np.full((8, 9, 3), 53, dtype=np.uint8),
        },
        proprio=np.arange(8, dtype=np.float32),
    )


def test_artifact_hashes_complete_released_serve_tree(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    validate_ftp1_policy_artifact(manifest)

    assert manifest.checkpoint_revision == CHECKPOINT_REVISION
    assert manifest.checkpoint_sha256 == manifest.model_sha256
    assert manifest.config_sha256 == manifest.serve_bundle_sha256
    assert manifest.normalizer_sha256 == manifest.normalization_tree_sha256
    assert len(manifest.normalization_files) == 7
    assert manifest.action_dim == 120
    assert manifest.action_horizon == 32
    assert manifest.action_rep == "mix"
    assert manifest.chunk_index_offset == 1
    assert manifest.chunk_first_n == 20
    assert manifest.use_tactile is True
    assert manifest.weight_license_status == "upstream_unspecified"


def test_artifact_validation_rejects_normalization_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    target = manifest.normalization_root / "dataset_stats.json"
    target.write_text('{"drift":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="normalization file SHA256"):
        validate_ftp1_policy_artifact(manifest)


def test_adapter_forwards_delivered_tactile_and_returns_one_qpos8(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    client = FakeClient()
    policy = load_ftp1_policy_adapter(_identity(manifest), manifest, lambda: client)
    context = PolicyEpisodeContext(
        episode_id="episode-0",
        task="insert_hole",
        initial_seed=3,
        exogenous_seed=7,
        instruction=manifest.prompt,
        action_spec=QPOS8_ACTION_SPEC,
    )
    policy.reset(context)

    plan = policy.infer(_observation())

    assert FTP1PolicyAdapter is OfficialFTP1Policy
    assert client.reset_calls == [("insert_hole", manifest.prompt, 7)]
    assert plan.actions.shape == (1, 8)
    assert plan.actions.tolist() == [list(range(8))]
    wire = client.observations[0]
    assert wire["task"] == "insert_hole"
    assert wire["prompt"] == manifest.prompt
    assert np.unique(wire["left"]).tolist() == [17]
    assert np.unique(wire["right"]).tolist() == [29]
    policy.commit(
        PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=(_observation(),),
            terminal_signal=BackendSignal.RUNNING,
        )
    )


def test_factory_rejects_partial_config_identity(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    bad_identity = PolicyIdentity(
        system_id="ftp1-insert-hole",
        checkpoint_sha256=manifest.checkpoint_sha256,
        config_sha256=manifest.model_config_sha256,
        action_spec=QPOS8_ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )

    with pytest.raises(ValueError, match="serving bundle identity"):
        load_ftp1_policy_adapter(bad_identity, manifest, FakeClient)


def test_msgpack_array_descriptor_is_binary_and_round_trips() -> None:
    source = np.arange(24, dtype=np.float32).reshape(3, 8)
    encoded = encode_wire_value({"action": source})
    assert isinstance(encoded, dict)
    descriptor = encoded["action"]
    assert isinstance(descriptor, dict)
    assert descriptor["dtype"] == "float32"
    assert isinstance(descriptor["data"], bytes)

    decoded = decode_wire_value(encoded)
    assert isinstance(decoded, dict)
    assert np.array_equal(decoded["action"], source)
    assert decoded["action"].flags.writeable is False


def test_live_request_binds_full_bundle_and_executes_one_step(tmp_path: Path) -> None:
    deployment = (tmp_path / "deployment").absolute()
    manifest = _manifest(deployment)
    request = build_official_ftp1_policy_request(
        manifest=manifest,
        layout=DeploymentLayout(deployment),
        condition=Condition.CLEAN,
        dataset_sha256="d" * 64,
        initial_seed=3,
        exogenous_seed=7,
        max_control_cycles=6,
        max_observation_steps=300,
        wall_timeout_s=1800.0,
        simulator_device="cuda:0",
        live_output_dir=deployment / "outputs/ftp1/clean",
    )

    assert request.policy_kind is LivePolicyKind.FTP1_POLICY
    assert request.execute_action_steps == 1
    assert request.checkpoint_sha256 == manifest.model_sha256
    assert request.config_sha256 == manifest.serve_bundle_sha256
    assert request.n0_normalizer_sha256 == manifest.normalization_tree_sha256


def test_checked_in_integration_config_is_canonical() -> None:
    config = load_model_integration_config("ftp1_policy")

    assert config.transport == "official_zmq"
    assert config.device == "cuda:0"
