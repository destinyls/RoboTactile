"""Load versioned machine-readable benchmark resources."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, cast

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    OPERATOR_IMPLEMENTATION_VERSION,
    SEVERITY_REGISTRY_ID,
)
from robotactile_benchmark.severity import SEVERITY_PATHS

_OPERATOR_RESOURCE_SHA256 = (
    "3a0fafaa065cd59d2462edb8a1ea5e0ba5ad6dfb68224ada63b8a058dfb5b86d"
)
_SEVERITY_RESOURCE_SHA256 = (
    "d86598e229e6f17ca5c78cfa37852a6f89bbfe2e814d8f8f1ba00f22d0838ef4"
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


def load_severity_registry() -> Dict[str, Any]:
    """Load provisional within-operator severity paths."""

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
