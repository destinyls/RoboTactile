"""Rest-reference binding helpers for pull-out-key request generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from robotactile_benchmark.calibration import (
    LoadedRestReferenceArtifact,
    load_rest_reference_artifact,
)
from robotactile_benchmark.calibration.contracts import (
    REST_REFERENCE_PATH,
    ROOT_RECEIPT_PATH,
    VALIDATION_PATH,
)
from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS


@dataclass(frozen=True)
class RestReferenceBinding:
    """Strict artifact plus its source directory for deterministic copying."""

    source: Path
    loaded: LoadedRestReferenceArtifact


def load_rest_reference_binding(
    source: Optional[Path], operator_id: str, task_id: str
) -> Optional[RestReferenceBinding]:
    requires_rest = operator_id in REST_REFERENCE_OPERATOR_IDS
    if requires_rest and source is None:
        raise ValueError(
            f"{operator_id} requires --rest-reference-artifact; the generator "
            "does not synthesize or substitute one"
        )
    if not requires_rest and source is not None:
        raise ValueError("rest-reference artifact is only valid for required operators")
    if source is None:
        return None
    absolute = source.expanduser().absolute()
    loaded = load_rest_reference_artifact(absolute)
    if loaded.validation.task != task_id:
        raise ValueError(f"rest-reference artifact task must be {task_id}")
    return RestReferenceBinding(absolute, loaded)


def fault_parameters(
    operator_id: str, binding: Optional[RestReferenceBinding]
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    if binding is not None:
        parameters["rest_reference_sha256"] = binding.loaded.references.sha256
    if operator_id == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    return parameters


def copy_rest_reference(staging: Path, binding: RestReferenceBinding) -> None:
    target = staging / "rest_references"
    target.mkdir()
    for name in (REST_REFERENCE_PATH, VALIDATION_PATH, ROOT_RECEIPT_PATH):
        (target / name).write_bytes((binding.source / name).read_bytes())


def request_rest_reference_path(
    faulted: bool, binding: Optional[RestReferenceBinding]
) -> Optional[str]:
    if not faulted or binding is None:
        return None
    return f"../rest_references/{REST_REFERENCE_PATH}"


def rest_reference_receipt(
    binding: Optional[RestReferenceBinding],
) -> Optional[dict[str, Any]]:
    if binding is None:
        return None
    loaded = binding.loaded
    return {
        "path": "rest_references",
        "artifact_root_sha256": loaded.root_receipt_sha256,
        "rest_reference_sha256": loaded.references.sha256,
        "validation_sha256": loaded.validation.sha256,
        "source_live_artifact_root_sha256": (
            loaded.validation.source_live_artifact_root_sha256
        ),
    }
