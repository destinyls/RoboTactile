"""Read-only deployment profile diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from robotactile_benchmark.deployment.layout import DeploymentLayout

DEPLOYMENT_PROFILES = ("core", "act-univtac", "n0-univtac")


@dataclass(frozen=True)
class DeploymentDoctorCheck:
    check_id: str
    relative_path: str
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class DeploymentDoctorResult:
    profile: str
    root: Path
    checks: Tuple[DeploymentDoctorCheck, ...]

    def __post_init__(self) -> None:
        if self.profile not in DEPLOYMENT_PROFILES:
            raise ValueError("unknown deployment doctor profile")
        if not self.checks:
            raise ValueError("deployment doctor requires checks")

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "checks": [item.to_dict() for item in self.checks],
            "evidence_level": "deployment_readiness_diagnostic_only_v1",
            "passed": self.passed,
            "profile": self.profile,
            "root": str(self.root),
            "simulator_execution_claimed": False,
        }


def _requirements(profile: str) -> tuple[tuple[str, str, str], ...]:
    core = (
        ("layout_receipt", "artifacts/deployment/layout_receipt.json", "file"),
        ("sources_directory", "sources", "directory"),
        ("runtime_directory", "runtime", "directory"),
        ("artifacts_directory", "artifacts", "directory"),
        ("requests_directory", "requests", "directory"),
        ("outputs_directory", "outputs", "directory"),
        ("logs_directory", "logs", "directory"),
    )
    if profile == "core":
        return core
    if profile == "act-univtac":
        return core + (
            ("univtac_source", "sources/UniVTAC", "directory"),
            ("act_source", "sources/WorldArena", "directory"),
            ("isaac_python", "runtime/isaac-sim-4.5.0/python.sh", "file"),
            ("act_artifacts", "artifacts/models/act", "directory"),
        )
    if profile == "n0-univtac":
        return core + (
            ("univtac_source", "sources/UniVTAC", "directory"),
            ("n0_source", "sources/N0-TWAM", "directory"),
            ("isaac_python", "runtime/isaac-sim-4.5.0/python.sh", "file"),
            ("n0_artifacts", "artifacts/models/n0_twam", "directory"),
        )
    raise ValueError("unknown deployment doctor profile")


def diagnose_deployment(
    layout: DeploymentLayout, profile: str
) -> DeploymentDoctorResult:
    """Inspect expected paths without importing runtimes or writing files."""

    checks = []
    for check_id, relative, kind in _requirements(profile):
        path = layout.root / relative
        passed = not path.is_symlink() and (
            path.is_file() if kind == "file" else path.is_dir()
        )
        checks.append(DeploymentDoctorCheck(check_id, relative, passed))
    return DeploymentDoctorResult(profile, layout.root, tuple(checks))


__all__ = [
    "DEPLOYMENT_PROFILES",
    "DeploymentDoctorCheck",
    "DeploymentDoctorResult",
    "diagnose_deployment",
]
