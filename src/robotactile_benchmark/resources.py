"""Load versioned machine-readable benchmark resources."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, cast

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID,
    DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
    DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    OPERATOR_IMPLEMENTATION_VERSION,
    OPTICAL_DECISION_STRESS_REGISTRY_ID,
    OPTICAL_MARKER_EXTREME_REGISTRY_ID,
    OPTICAL_MARKER_STRESS_REGISTRY_ID,
    SEVERITY_REGISTRY_ID,
)
from robotactile_benchmark.severity import (
    DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_VALUES,
    DIAGNOSTIC_STRESS_MAX_VALUES,
    DIAGNOSTIC_TACTILE_NULL_VALUES,
    OPTICAL_DECISION_STRESS_VALUES,
    OPTICAL_MARKER_EXTREME_VALUES,
    OPTICAL_MARKER_STRESS_VALUES,
    SEVERITY_PATHS,
    THREE_DOSE_STRESS_PATHS,
)

_OPERATOR_RESOURCE_SHA256 = (
    "3a0fafaa065cd59d2462edb8a1ea5e0ba5ad6dfb68224ada63b8a058dfb5b86d"
)
_SEVERITY_RESOURCE_SHA256 = (
    "d86598e229e6f17ca5c78cfa37852a6f89bbfe2e814d8f8f1ba00f22d0838ef4"
)
_DIAGNOSTIC_STRESS_MAX_RESOURCE_SHA256 = (
    "02cd4e77737ba5f298c794fa225fb3e57d386dd076fd4eef7e3f670be257c9f5"
)
_DIAGNOSTIC_TACTILE_NULL_RESOURCE_SHA256 = (
    "a60b4f4b241b96f02384f81be2ec2b2e8fd922e188005fe5e2e12cf969a51219"
)
_DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_RESOURCE_SHA256 = (
    "a1c51d42ca7b446b85d814fcdec3bd932ede2e1188e4ff7342b115c14cbcf325"
)
_OPTICAL_MARKER_STRESS_RESOURCE_SHA256 = (
    "b04937a82a7d95865b3bb41ee497adada23f1a2bb69e8068f1a1cb15aff8cbf5"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(relative_path: str, expected_sha256: str) -> Dict[str, Any]:
    package_resource = resources.files("robotactile_benchmark").joinpath(relative_path)
    if package_resource.is_file():
        payload = package_resource.read_bytes()
    else:
        payload = (_project_root() / relative_path).read_bytes()
    import hashlib

    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"packaged benchmark resource digest drift: {relative_path}")
    return cast(Dict[str, Any], json.loads(payload.decode("utf-8")))


def load_operator_registry() -> Dict[str, Any]:
    """Load the frozen 2+7+3+2 registry."""

    registry = _load_json("configs/operators/core_v2.json", _OPERATOR_RESOURCE_SHA256)
    if set(registry) != {
        "registry_id",
        "semantic_version",
        "implementation_version",
        "scope",
        "complete_root_cause_taxonomy",
        "operators",
    }:
        raise ValueError("operator resource top-level fields drift")
    if registry.get("registry_id") != "robotactile_core_2_7_3_2_v2":
        raise ValueError("operator resource registry ID drift")
    if registry.get("semantic_version") != "2.0":
        raise ValueError("operator resource semantic version drift")
    if registry.get("implementation_version") != OPERATOR_IMPLEMENTATION_VERSION:
        raise ValueError("operator resource implementation version drift")
    operators = registry.get("operators")
    if not isinstance(operators, list) or {
        entry.get("operator_id") for entry in operators if isinstance(entry, dict)
    } != set(CORE_OPERATOR_IDS):
        raise ValueError("operator resource does not match the runtime registry")
    return registry


def load_severity_registry(
    registry_id: str = SEVERITY_REGISTRY_ID,
) -> Dict[str, Any]:
    """Load one registered paper or non-paper diagnostic severity resource."""

    if registry_id in THREE_DOSE_STRESS_PATHS:
        digests = {
            "optical_contact_stress_v2": "0baad38684cb5f41869ad0dc4e9dfc05fd0daaebad2c76eeb73c3fc658a828fc",
            "sensor_fullframe_stress_v1": "6929e006a6921adcfac759fab19d15a439686f4fec33123f7488ae994650b7dc",
        }
        registry = _load_json(
            f"configs/severity/{registry_id}.json", digests[registry_id]
        )
        expected_paths = {
            key: list(value)
            for key, value in THREE_DOSE_STRESS_PATHS[registry_id].items()
        }
        if (
            registry["registry_id"] != registry_id
            or registry["levels"] != [1, 3, 5]
            or registry["paths"] != expected_paths
        ):
            raise ValueError("three-dose stress resource drift")
        return registry
    diagnostic_resources = {
        OPTICAL_DECISION_STRESS_REGISTRY_ID: (
            "configs/severity/optical_decision_stress_v1.json",
            "f9b2b06373ff845f3d63af830b15f60e5a3b040fe43417d8ae178d1609d56622",
            "1.0",
            OPTICAL_DECISION_STRESS_VALUES,
        ),
        OPTICAL_MARKER_EXTREME_REGISTRY_ID: (
            "configs/severity/optical_marker_extreme_v1.json",
            "9044b7b4242c4da217bbd0b2eedc6e5c5f7aafcdadcc63eb29ca260c69f5e594",
            "1.0",
            OPTICAL_MARKER_EXTREME_VALUES,
        ),
        OPTICAL_MARKER_STRESS_REGISTRY_ID: (
            "configs/severity/optical_marker_stress_v1.json",
            _OPTICAL_MARKER_STRESS_RESOURCE_SHA256,
            "1.0",
            OPTICAL_MARKER_STRESS_VALUES,
        ),
        DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_REGISTRY_ID: (
            "configs/severity/diagnostic_observed_tactile_absence_v1.json",
            _DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_RESOURCE_SHA256,
            "1.0",
            DIAGNOSTIC_OBSERVED_TACTILE_ABSENCE_VALUES,
        ),
        DIAGNOSTIC_STRESS_MAX_REGISTRY_ID: (
            "configs/severity/diagnostic_stress_max_v1.json",
            _DIAGNOSTIC_STRESS_MAX_RESOURCE_SHA256,
            "1.0",
            DIAGNOSTIC_STRESS_MAX_VALUES,
        ),
        DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID: (
            "configs/severity/diagnostic_tactile_null_black_v1.json",
            _DIAGNOSTIC_TACTILE_NULL_RESOURCE_SHA256,
            "1.0",
            DIAGNOSTIC_TACTILE_NULL_VALUES,
        ),
    }
    if registry_id in diagnostic_resources:
        relative_path, digest, semantic_version, registered_values = (
            diagnostic_resources[registry_id]
        )
        registry = _load_json(
            relative_path,
            digest,
        )
        if set(registry) != {
            "registry_id",
            "semantic_version",
            "calibration_status",
            "cross_operator_comparable",
            "level",
            "paths",
        }:
            raise ValueError("diagnostic stress resource fields drift")
        if (
            registry.get("registry_id") != registry_id
            or registry.get("semantic_version") != semantic_version
            or registry.get("level") != 5
        ):
            raise ValueError("diagnostic stress resource identity drift")
        paths = registry.get("paths")
        if not isinstance(paths, dict) or set(paths) != set(registered_values):
            raise ValueError("diagnostic stress operator set drift")
        for operator_id, value in registered_values.items():
            if paths[operator_id].get("value") != value:
                raise ValueError(f"diagnostic stress dose drift: {operator_id}")
        return registry
    if registry_id != SEVERITY_REGISTRY_ID:
        raise ValueError(f"unsupported severity registry: {registry_id}")

    registry = _load_json(
        "configs/severity/provisional_v2.json", _SEVERITY_RESOURCE_SHA256
    )
    if set(registry) != {
        "registry_id",
        "semantic_version",
        "calibration_status",
        "cross_operator_comparable",
        "paths",
    }:
        raise ValueError("severity resource top-level fields drift")
    if registry.get("semantic_version") != "2.0":
        raise ValueError("severity resource semantic version drift")
    if registry.get("registry_id") != SEVERITY_REGISTRY_ID:
        raise ValueError("severity resource registry ID drift")
    paths = registry.get("paths")
    if not isinstance(paths, dict) or set(paths) != set(SEVERITY_PATHS):
        raise ValueError("severity resource does not match the runtime registry")
    for operator_id, values in SEVERITY_PATHS.items():
        if tuple(paths[operator_id].get("values", ())) != values:
            raise ValueError(f"severity resource path drift: {operator_id}")
    return registry


def load_source_manifest() -> str:
    """Load the generated per-file SHA256 release manifest."""

    package_resource = resources.files("robotactile_benchmark").joinpath(
        "source_manifest.sha256"
    )
    if package_resource.is_file():
        return package_resource.read_text(encoding="utf-8")
    path = _project_root() / "release" / "source_manifest.sha256"
    return path.read_text(encoding="utf-8")
