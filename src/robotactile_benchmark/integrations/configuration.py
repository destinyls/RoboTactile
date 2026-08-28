"""Generate runnable model integration manifests from real local artifacts."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    sha256_bytes,
)
from robotactile_benchmark.integrations.act.artifacts import (
    act_artifact_manifest_to_dict,
    build_act_artifact_manifest,
)
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    build_n0_twam_artifact_manifest,
    validate_n0_twam_artifact,
)
from robotactile_benchmark.integrations.registry import ModelIntegrationConfig
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile


@dataclass(frozen=True)
class GeneratedIntegrationConfiguration:
    integration_id: str
    artifact_manifest_path: Path
    integration_config_path: Path
    artifact_manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_manifest": str(self.artifact_manifest_path),
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "evidence_level": "local_artifact_configuration_only_v1",
            "integration_config": str(self.integration_config_path),
            "integration_id": self.integration_id,
            "live_inference_claimed": False,
        }


def _publish_canonical(path: Path, value: object) -> bytes:
    target = Path(path).absolute()
    if target.is_symlink():
        raise ValueError("configuration output cannot be a symlink")
    payload = canonical_json_bytes(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different output: {target}")
        return payload
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _assert_publishable(path: Path, value: object) -> None:
    target = Path(path).absolute()
    payload = canonical_json_bytes(value)
    if target.is_symlink():
        raise ValueError("configuration output cannot be a symlink")
    if target.exists() and (not target.is_file() or target.read_bytes() != payload):
        raise FileExistsError(f"refusing to replace different output: {target}")


def _write_pair(
    *,
    integration_id: str,
    manifest: object,
    manifest_path: Path,
    config_path: Path,
    device: str,
) -> GeneratedIntegrationConfiguration:
    transport = "in_process" if integration_id == "act" else "official_websocket"
    config = ModelIntegrationConfig(
        schema_version="robotactile-model-integration-config-v1",
        integration_id=integration_id,
        artifact_manifest=str(Path(manifest_path).absolute()),
        device=device,
        transport=transport,
    )
    _assert_publishable(manifest_path, manifest)
    _assert_publishable(config_path, config.to_dict())
    manifest_payload = _publish_canonical(manifest_path, manifest)
    _publish_canonical(config_path, config.to_dict())
    return GeneratedIntegrationConfiguration(
        integration_id=integration_id,
        artifact_manifest_path=Path(manifest_path).absolute(),
        integration_config_path=Path(config_path).absolute(),
        artifact_manifest_sha256=sha256_bytes(manifest_payload),
    )


def configure_act_integration(
    *,
    task_id: str,
    profile: OfficialACTProfile,
    artifact_root: Path,
    upstream_root: Path,
    manifest_path: Path,
    config_path: Path,
    device: str,
) -> GeneratedIntegrationConfiguration:
    manifest = build_act_artifact_manifest(
        task_id=task_id,
        profile=profile,
        artifact_root=artifact_root,
        upstream_root=upstream_root,
    )
    return _write_pair(
        integration_id="act",
        manifest=act_artifact_manifest_to_dict(manifest),
        manifest_path=manifest_path,
        config_path=config_path,
        device=device,
    )


def configure_n0_twam_integration(
    *,
    bundle_root: Path,
    task_id: str,
    base_root: Path,
    checkpoint_root: Path,
    serve_bundle_root: Path,
    serve_pool_root: Path,
    checkpoint_path: Path,
    model_config_path: Path,
    train_meta_path: Path,
    normalizer_path: Path,
    prompt_manifest_path: Path,
    serve_bundle_manifest_path: Path,
    serve_info_path: Path,
    serve_tasks_path: Path,
    manifest_path: Path,
    config_path: Path,
    device: str,
) -> GeneratedIntegrationConfiguration:
    manifest = build_n0_twam_artifact_manifest(
        bundle_root=bundle_root,
        task_id=task_id,
        base_root=base_root,
        checkpoint_root=checkpoint_root,
        serve_bundle_root=serve_bundle_root,
        serve_pool_root=serve_pool_root,
        checkpoint_path=checkpoint_path,
        config_path=model_config_path,
        train_meta_path=train_meta_path,
        normalizer_path=normalizer_path,
        prompt_manifest_path=prompt_manifest_path,
        serve_bundle_manifest_path=serve_bundle_manifest_path,
        serve_info_path=serve_info_path,
        serve_tasks_path=serve_tasks_path,
    )
    validate_n0_twam_artifact(manifest)
    return _write_pair(
        integration_id="n0_twam",
        manifest=manifest.to_dict(),
        manifest_path=manifest_path,
        config_path=config_path,
        device=device,
    )


__all__ = [
    "GeneratedIntegrationConfiguration",
    "configure_act_integration",
    "configure_n0_twam_integration",
]
