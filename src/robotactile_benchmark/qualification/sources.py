"""Frozen source identities used by CPU policy protocol qualification."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from robotactile_benchmark.contracts import canonical_hash

ACT_RUNTIME_SOURCE_PATH = (
    "../visual-tactile_world_model_pipeline/deployment/ACTStrict/deploy_policy.py"
)
ACT_RUNTIME_SOURCE_SHA256 = (
    "c4ea8999df836a4defe5f9236795e5c1fe3b0f6187fc4294a2bb5b64e7e95e49"
)

TASK6A_SOURCE_HASHES: Mapping[str, str] = MappingProxyType(
    {
        "RoboTactile/src/robotactile_benchmark/policies/__init__.py": "1939e6012f610ac10372518a3dd242c31adeb3ecb9946b7de54d0712813e5d81",
        "RoboTactile/src/robotactile_benchmark/policies/act.py": "940605aec715351d7d765197e438c77ad49fabb4f318e550dfc1418119923443",
        "RoboTactile/src/robotactile_benchmark/policies/act_loading.py": "1ee5ed1b565eb8027ef143969f80e1e5d86476d25f43d92015006a369b77aae2",
        "RoboTactile/src/robotactile_benchmark/policies/n0.py": "61a7445acff538ea044918503a70cb3630f111f423563476af3b6f7842956e2e",
        "RoboTactile/src/robotactile_benchmark/transport/__init__.py": "9da4553d12af86ee70502609648f6c5e96b909490026bf927307b10b2c7b6513",
        "RoboTactile/src/robotactile_benchmark/transport/n0_binding.py": "00ffff81f52778492897d9778e43b3a1a66796a3b928bc9a5322d97c49112c11",
        "RoboTactile/src/robotactile_benchmark/transport/n0_client.py": "22a45f9f5961f3731927278d3c5b3e559eb9e77eb600e9aacc6b0778700fc78e",
        "RoboTactile/src/robotactile_benchmark/transport/n0_codec.py": "5d4e125c5a138b8dd84fec8f7f477f93288036b22df55e6fcfceb28216b764be",
        "RoboTactile/src/robotactile_benchmark/transport/n0_commit.py": "20e76c18e421c9e96ebddff82a7a0457748bf9c7cf79dcc8452da7614d296ab5",
        "RoboTactile/src/robotactile_benchmark/transport/n0_contracts.py": "bd018902c5c36801ffd6b108c63b017553cdaf07e1347da891b223d0c170e5b4",
        "RoboTactile/src/robotactile_benchmark/transport/n0_modalities.py": "cb726391220886029fb0c8f7da744764c6a19b058a2cc6c65b19a42036ec8392",
    }
)

TASK6B_SOURCE_HASHES: Mapping[str, str] = MappingProxyType(
    {
        "N0-TWAM/n0_twam/transport/robotactile_protocol.py": "d1d6b76dae8706598f787c141fe735255b437753ddc9cab220ffe7c1d9207eb5",
        "N0-TWAM/n0_twam/transport/robotactile_grounding.py": "97efdf1828a6b5de8306e397760bfef56b47817310c4e00f5cb6337f45a80cc9",
        "N0-TWAM/n0_twam/transport/robotactile_transactions.py": "4620e536491100a6d89f8892a6487ea399dfec643a82dbe9f0ca16294eb4ff02",
        "N0-TWAM/n0_twam/transport/robotactile_gateway.py": "b240bf943ef027d64925c7172fea5cc5bd1a5831d7974eacee21ef146233f862",
        "N0-TWAM/n0_twam/transport/robotactile_artifact_validation.py": "0554457ea2beb6235460e1dd49c4698559c6a0ed336c71ba2cba736a3f1de4e8",
        "N0-TWAM/n0_twam/transport/robotactile_artifacts.py": "79dad98c41b12ab555f04cff8f4d51de64e82102e9d835cb1e8ab0fc75bba9ad",
        "N0-TWAM/n0_twam/configs/twam_track31_univtac_server_cfg.py": "c1084dbd5a6ae7e5b3e3c015a8e7255a24db20e7cc650f450cc26690c0fab149",
        "N0-TWAM/n0_twam/configs/__init__.py": "f279b230ed1a567c2b6c6e2911030bac2ef7ff441d8c4159520c8e9fee1b3faa",
    }
)


class SourceIdentityError(ValueError):
    """A reviewed source file is missing, aliased, or changed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(workspace_root: Path, relative_path: str) -> Path:
    root = workspace_root.resolve(strict=True)
    candidate = root / relative_path
    if candidate.is_symlink() or not candidate.is_file():
        raise SourceIdentityError(f"reviewed source is unavailable: {relative_path}")
    resolved = candidate.resolve(strict=True)
    expected = (root / relative_path).resolve(strict=True)
    if resolved != expected:
        raise SourceIdentityError(f"reviewed source path drift: {relative_path}")
    return resolved


def collect_reviewed_sources(
    workspace_root: Path, expected: Mapping[str, str]
) -> Mapping[str, str]:
    """Rehash an exact reviewed source set and reject all drift."""

    actual = {}
    for relative_path, expected_sha256 in expected.items():
        digest = _sha256_file(_verified_file(workspace_root, relative_path))
        if digest != expected_sha256:
            raise SourceIdentityError(f"reviewed source hash drift: {relative_path}")
        actual[relative_path] = digest
    return MappingProxyType(actual)


def collect_act_runtime_source(workspace_root: Path) -> tuple[str, str]:
    path = _verified_file(workspace_root, ACT_RUNTIME_SOURCE_PATH)
    digest = _sha256_file(path)
    if digest != ACT_RUNTIME_SOURCE_SHA256:
        raise SourceIdentityError("ACT allowed runtime source hash drift")
    return ACT_RUNTIME_SOURCE_PATH, digest


def source_manifest_sha256(value: Mapping[str, str]) -> str:
    return canonical_hash(value)
