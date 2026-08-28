"""First-class official N0-TWAM integration facade tests."""

from pathlib import Path

import pytest

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.integrations import get_model_integration
from robotactile_benchmark.integrations.n0_twam import (
    N0TWAMArtifactManifest,
    N0TWAMPolicyAdapter,
    load_n0_twam_adapter,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
)
from robotactile_benchmark.policies.n0_official import OfficialN0Policy

_OFFICIAL_COMMIT = "c43a2160dd31c449d92b28eab52c0e2f09e4738a"


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        system_id="n0-twam-test",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        action_spec=EE8_ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )


def _manifest(tmp_path: Path) -> N0TWAMArtifactManifest:
    root = tmp_path.absolute()
    return N0TWAMArtifactManifest(
        bundle_root=root,
        base_root=root / "base",
        checkpoint_root=root / "univtac-delta",
        serve_bundle_root=root / "serve-bundle",
        serve_pool_root=root / "serve-pools/pull_out_key",
        checkpoint_path=root / "univtac-delta/transformer/model.safetensors",
        checkpoint_sha256="a" * 64,
        config_path=root / "univtac-delta/transformer/config.json",
        config_sha256="b" * 64,
        train_meta_path=root / "univtac-delta/train_meta.json",
        train_meta_sha256="c" * 64,
        normalizer_path=root / "serve-pools/pull_out_key/normalizer.json",
        normalizer_sha256="c" * 64,
        prompt_manifest_path=root / "univtac-delta/norm/PROMPTS.json",
        prompt_manifest_sha256="d" * 64,
        serve_bundle_manifest_path=root / "serve_bundle_manifest.json",
        serve_bundle_sha256="e" * 64,
        serve_info_path=root / "serve-pools/pull_out_key/meta/info.json",
        serve_info_sha256="f" * 64,
        serve_tasks_path=root / "serve-pools/pull_out_key/meta/tasks.jsonl",
        serve_tasks_sha256="1" * 64,
        task_id="pull_out_key",
        serve_task_id="univtac_pull_out_key_rot6d_current",
        external_commit=_OFFICIAL_COMMIT,
    )


def test_n0_facade_reuses_validated_policy() -> None:
    assert N0TWAMPolicyAdapter is OfficialN0Policy
    spec = get_model_integration("n0_twam")
    assert spec.factory_path.endswith(":load_n0_twam_adapter")
    assert spec.capabilities.stateful_commit is True
    assert spec.capabilities.matched_no_touch is False


def test_n0_factory_binds_artifact_identity_before_client_effect(
    tmp_path: Path,
) -> None:
    client_effects = 0

    def client_factory() -> object:
        nonlocal client_effects
        client_effects += 1
        raise AssertionError("client must remain lazy")

    policy = load_n0_twam_adapter(
        _identity(),
        _manifest(tmp_path),
        client_factory,
        input_profile=N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
    )

    assert isinstance(policy, OfficialN0Policy)
    assert policy.input_profile is N0_RECORDED_CHECKPOINT_INPUT_PROFILE
    assert client_effects == 0


def test_n0_factory_rejects_artifact_mismatch_before_client_effect(
    tmp_path: Path,
) -> None:
    payload = _manifest(tmp_path).to_dict()
    payload["checkpoint_sha256"] = "d" * 64
    mismatched = N0TWAMArtifactManifest.from_dict(payload)
    client_effects = 0

    def client_factory() -> object:
        nonlocal client_effects
        client_effects += 1
        raise AssertionError("client must not be constructed")

    with pytest.raises(ValueError, match="checkpoint"):
        load_n0_twam_adapter(
            _identity(),
            mismatched,
            client_factory,
            input_profile=N0_RECORDED_CHECKPOINT_INPUT_PROFILE,
        )
    assert client_effects == 0
