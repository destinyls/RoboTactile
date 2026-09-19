"""Fail-closed source, artifact, and transport readiness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    discover_repository_root,
)
from robotactile_benchmark.integrations.act.artifacts import (
    load_act_artifact_manifest,
)
from robotactile_benchmark.integrations.dream_tac.artifacts import (
    load_dream_tac_artifact_manifest,
    validate_dream_tac_artifact,
)
from robotactile_benchmark.integrations.ftp1_policy.artifacts import (
    load_ftp1_policy_artifact_manifest,
    validate_ftp1_policy_artifact,
)
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    load_n0_twam_artifact_manifest,
    validate_n0_twam_artifact,
)
from robotactile_benchmark.integrations.n0_vtla.artifacts import (
    load_n0_vtla_artifact_manifest,
    validate_n0_vtla_artifact,
)
from robotactile_benchmark.integrations.provenance import (
    load_integration_lock,
    verify_external_checkout,
)
from robotactile_benchmark.integrations.registry import (
    load_model_integration_config,
)
from robotactile_benchmark.policies.univtac_official_act_loading import (
    validate_official_univtac_act_artifact,
)


@dataclass(frozen=True)
class IntegrationDoctorCheck:
    check_id: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "detail": self.detail,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class IntegrationDoctorResult:
    integration_id: str
    checks: tuple[IntegrationDoctorCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "checks": [check.to_dict() for check in self.checks],
            "evidence_level": "integration_readiness_diagnostic_only_v1",
            "integration_id": self.integration_id,
            "live_inference_claimed": False,
            "passed": self.passed,
        }


def _attempt(check_id: str, action: Callable[[], object]) -> IntegrationDoctorCheck:
    try:
        action()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return IntegrationDoctorCheck(check_id, False, str(error))
    return IntegrationDoctorCheck(check_id, True, "verified")


def _manifest_path(raw: str) -> Path:
    selected = Path(raw)
    if selected.is_absolute():
        return selected
    return discover_repository_root() / selected


def diagnose_model_integration(
    *,
    integration_id: str,
    layout: DeploymentLayout,
    config_path: Optional[Path] = None,
    task_id: str = "pull_out_key",
    profile_id: str = "univtac",
) -> IntegrationDoctorResult:
    """Verify source pins and artifacts without starting a model."""

    model_dir = integration_id
    if config_path is None:
        selected_config = layout.root / (
            f"artifacts/models/{model_dir}/configs/{task_id}/{profile_id}/integration_config.json"
            if integration_id == "act"
            else f"artifacts/models/{model_dir}/configs/{task_id}/integration_config.json"
        )
        legacy = layout.root / "artifacts/models/act/integration_config.json"
        if (
            integration_id == "act"
            and legacy.is_file()
            and not selected_config.exists()
        ):
            selected_config = legacy
    else:
        selected_config = Path(config_path)
    checks: list[IntegrationDoctorCheck] = []
    loaded_config = None
    try:
        loaded_config = load_model_integration_config(integration_id, selected_config)
    except (OSError, TypeError, ValueError) as error:
        checks.append(IntegrationDoctorCheck("integration_config", False, str(error)))
    else:
        checks.append(IntegrationDoctorCheck("integration_config", True, "verified"))

    lock = load_integration_lock()
    source_ids = {
        "act": ("univtac",),
        "dream_tac": ("dream_tac", "univtac"),
        "ftp1_policy": ("ftp1_policy", "univtac"),
        "n0_twam": ("n0_twam", "univtac"),
        "n0_vtla": ("n0_vtla", "univtac"),
    }[integration_id]
    unreleased = tuple(
        source_id for source_id in source_ids if not lock.by_id(source_id).release_ready
    )
    checks.append(
        IntegrationDoctorCheck(
            "external_release_readiness",
            not unreleased,
            "verified"
            if not unreleased
            else f"release_ready=false: {','.join(unreleased)}",
        )
    )
    for source_id in source_ids:
        pin = lock.by_id(source_id)
        source_root = layout.sources / pin.source_directory
        checks.append(
            _attempt(
                f"source_{source_id}",
                partial(verify_external_checkout, pin, source_root),
            )
        )

    if loaded_config is not None:
        manifest_path = _manifest_path(loaded_config.artifact_manifest)
        if integration_id == "act":
            checks.append(
                _attempt(
                    "artifact_manifest",
                    lambda: validate_official_univtac_act_artifact(
                        load_act_artifact_manifest(manifest_path)
                    ),
                )
            )
        elif integration_id == "dream_tac":
            checks.append(
                _attempt(
                    "artifact_manifest",
                    lambda: validate_dream_tac_artifact(
                        load_dream_tac_artifact_manifest(manifest_path)
                    ),
                )
            )
        elif integration_id == "ftp1_policy":
            checks.append(
                _attempt(
                    "artifact_manifest",
                    lambda: validate_ftp1_policy_artifact(
                        load_ftp1_policy_artifact_manifest(manifest_path)
                    ),
                )
            )
        elif integration_id == "n0_twam":
            checks.append(
                _attempt(
                    "artifact_manifest",
                    lambda: validate_n0_twam_artifact(
                        load_n0_twam_artifact_manifest(manifest_path)
                    ),
                )
            )
        else:
            checks.append(
                _attempt(
                    "artifact_manifest",
                    lambda: validate_n0_vtla_artifact(
                        load_n0_vtla_artifact_manifest(manifest_path)
                    ),
                )
            )
    checks.append(
        IntegrationDoctorCheck(
            "transport",
            True,
            (
                {
                    "act": "in_process",
                    "dream_tac": (
                        "official_http_contract_registered_endpoint_not_probed"
                    ),
                    "ftp1_policy": "official_zmq_contract_registered_endpoint_not_probed",
                    "n0_twam": (
                        "official_websocket_contract_registered_endpoint_not_probed"
                    ),
                    "n0_vtla": "official_zmq_contract_registered_endpoint_not_probed",
                }[integration_id]
            ),
        )
    )
    if integration_id == "dream_tac":
        checks.extend(
            (
                IntegrationDoctorCheck(
                    "official_checkpoint_release",
                    False,
                    (
                        "the pinned upstream documentation exposes only local "
                        "checkpoint placeholders; provide and provenance-bind "
                        "task-aligned weights"
                    ),
                ),
                IntegrationDoctorCheck(
                    "paper_inference_parity",
                    False,
                    (
                        "the pinned upstream HTTP inference path does not apply "
                        "the paper CASA contact gate"
                    ),
                ),
                IntegrationDoctorCheck(
                    "univtac_task_alignment",
                    False,
                    (
                        "no public Dream-Tac checkpoint bound to the UniVTAC "
                        "task, camera, tactile, and timing contracts was discovered"
                    ),
                ),
            )
        )
    return IntegrationDoctorResult(integration_id, tuple(checks))


__all__ = [
    "IntegrationDoctorCheck",
    "IntegrationDoctorResult",
    "diagnose_model_integration",
]
