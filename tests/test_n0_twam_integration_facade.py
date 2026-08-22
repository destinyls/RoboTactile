"""First-class N0-TWAM integration facade tests."""

from pathlib import Path

import pytest

from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC, PolicyIdentity
from robotactile_benchmark.integrations import get_model_integration
from robotactile_benchmark.integrations.n0_twam import (
    N0TWAMArtifactManifest,
    N0TWAMPolicyAdapter,
    load_n0_twam_adapter,
)
from robotactile_benchmark.policies.n0 import N0Policy


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        system_id="n0-twam-test",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        action_spec=ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )


def _manifest(tmp_path: Path) -> N0TWAMArtifactManifest:
    return N0TWAMArtifactManifest(
        bundle_root=tmp_path.absolute(),
        checkpoint_path=(tmp_path / "checkpoint.pt").absolute(),
        checkpoint_sha256="a" * 64,
        config_path=(tmp_path / "config.json").absolute(),
        config_sha256="b" * 64,
        normalizer_path=(tmp_path / "normalizer.json").absolute(),
        normalizer_sha256="c" * 64,
        external_commit="9036c130409f8cf5494b12489fea339f7213b9d6",
    )


def test_n0_facade_reuses_validated_policy() -> None:
    assert N0TWAMPolicyAdapter is N0Policy
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

    policy = load_n0_twam_adapter(_identity(), _manifest(tmp_path), client_factory)

    assert isinstance(policy, N0Policy)
    assert client_effects == 0


def test_n0_factory_rejects_artifact_mismatch_before_client_effect(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    mismatched = N0TWAMArtifactManifest(
        bundle_root=manifest.bundle_root,
        checkpoint_path=manifest.checkpoint_path,
        checkpoint_sha256="d" * 64,
        config_path=manifest.config_path,
        config_sha256=manifest.config_sha256,
        normalizer_path=manifest.normalizer_path,
        normalizer_sha256=manifest.normalizer_sha256,
        external_commit=manifest.external_commit,
    )
    client_effects = 0

    def client_factory() -> object:
        nonlocal client_effects
        client_effects += 1
        raise AssertionError("client must not be constructed")

    with pytest.raises(ValueError, match="checkpoint"):
        load_n0_twam_adapter(_identity(), mismatched, client_factory)
    assert client_effects == 0
