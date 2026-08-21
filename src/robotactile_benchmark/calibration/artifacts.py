"""Canonical persistence for one rest-reference calibration bundle."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.calibration.contracts import (
    CALIBRATION_EVIDENCE_LEVEL,
    REST_REFERENCE_PATH,
    ROOT_RECEIPT_PATH,
    VALIDATION_PATH,
    LoadedRestReferenceArtifact,
    NoContactValidationReceipt,
    RestReferenceArtifactError,
    RestReferenceArtifactMember,
    RestReferenceArtifactRootReceipt,
)
from robotactile_benchmark.closed_loop.artifact_contracts import (
    ArtifactValidationError,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    sha256_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.constants import SENSOR_SLOTS
from robotactile_benchmark.contracts import array_sha256, canonical_hash
from robotactile_benchmark.rest_references import (
    FrozenPayload,
    ReferenceSplit,
    RestReferenceBundle,
)

_EXPECTED_FILES = frozenset({REST_REFERENCE_PATH, VALIDATION_PATH, ROOT_RECEIPT_PATH})
_REST_FIELDS = frozenset(RestReferenceBundle.__dataclass_fields__)
_MAX_MEMBER_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class RestReferenceArtifactExportReceipt:
    artifact_root_sha256: str
    rest_reference_sha256: str
    no_contact_validation_sha256: str
    evidence_level: str = CALIBRATION_EVIDENCE_LEVEL


def _bundle_to_dict(references: RestReferenceBundle) -> dict[str, object]:
    payloads: dict[str, object] = {}
    for slot in SENSOR_SLOTS:
        frozen = references.payloads[slot]
        if not isinstance(frozen, FrozenPayload):
            raise RestReferenceArtifactError("rest payload is not frozen")
        payloads[slot] = {
            "dtype": frozen.dtype,
            "shape": list(frozen.shape),
            "data_hex": frozen.data_hex,
        }
    return {
        "reference_id": references.reference_id,
        "dataset_split": references.dataset_split.value,
        "split_manifest_sha256": references.split_manifest_sha256,
        "source_artifact_sha256": references.source_artifact_sha256,
        "no_contact_predicate_id": references.no_contact_predicate_id,
        "no_contact_validation_sha256": references.no_contact_validation_sha256,
        "no_contact_verified": references.no_contact_verified,
        "qualified_record_ids": dict(references.qualified_record_ids),
        "calibration_sha256": dict(references.calibration_sha256),
        "payloads": payloads,
    }


def _string_mapping(value: object, name: str) -> Mapping[str, str]:
    if not isinstance(value, dict) or set(value) != set(SENSOR_SLOTS):
        raise RestReferenceArtifactError(f"{name} must cover left and right")
    if any(not isinstance(item, str) for item in value.values()):
        raise RestReferenceArtifactError(f"{name} values must be strings")
    return cast(Mapping[str, str], value)


def _payload(value: object, slot: str) -> FrozenPayload:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_hex"}:
        raise RestReferenceArtifactError(f"{slot} payload fields mismatch")
    shape = value["shape"]
    if not isinstance(shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in shape
    ):
        raise RestReferenceArtifactError(f"{slot} payload shape is invalid")
    if not isinstance(value["dtype"], str) or not isinstance(value["data_hex"], str):
        raise RestReferenceArtifactError(f"{slot} payload encoding is invalid")
    return FrozenPayload(value["dtype"], tuple(shape), value["data_hex"])


def _bundle_from_dict(value: object) -> RestReferenceBundle:
    if not isinstance(value, dict) or set(value) != _REST_FIELDS:
        raise RestReferenceArtifactError("rest-reference fields mismatch")
    payloads = value["payloads"]
    if not isinstance(payloads, dict) or set(payloads) != set(SENSOR_SLOTS):
        raise RestReferenceArtifactError("rest payloads must cover left and right")
    return RestReferenceBundle(
        reference_id=value["reference_id"],
        dataset_split=ReferenceSplit(value["dataset_split"]),
        split_manifest_sha256=value["split_manifest_sha256"],
        source_artifact_sha256=value["source_artifact_sha256"],
        no_contact_predicate_id=value["no_contact_predicate_id"],
        no_contact_validation_sha256=value["no_contact_validation_sha256"],
        no_contact_verified=value["no_contact_verified"],
        qualified_record_ids=_string_mapping(
            value["qualified_record_ids"], "qualified_record_ids"
        ),
        calibration_sha256=_string_mapping(
            value["calibration_sha256"], "calibration_sha256"
        ),
        payloads={slot: _payload(payloads[slot], slot) for slot in SENSOR_SLOTS},
    )


def _strict_document(raw: bytes, name: str) -> object:
    if len(raw) > _MAX_MEMBER_BYTES:
        raise RestReferenceArtifactError(f"{name} exceeds calibration size cap")
    try:
        return strict_json_bytes(raw, name)
    except ArtifactValidationError as error:
        raise RestReferenceArtifactError(str(error)) from error


def _scan(root: Path) -> dict[str, bytes]:
    try:
        root_stat = root.lstat()
    except OSError as error:
        raise RestReferenceArtifactError(
            "calibration artifact root is absent"
        ) from error
    if root.is_symlink() or not root.is_dir() or not root_stat:
        raise RestReferenceArtifactError(
            "calibration artifact root must be a directory"
        )
    files: dict[str, bytes] = {}
    for entry in sorted(os.scandir(root), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise RestReferenceArtifactError("calibration artifact forbids links/dirs")
        stat = entry.stat(follow_symlinks=False)
        if not 1 <= stat.st_size <= _MAX_MEMBER_BYTES:
            raise RestReferenceArtifactError("calibration member size is invalid")
        raw = Path(entry.path).read_bytes()
        if len(raw) != stat.st_size:
            raise RestReferenceArtifactError("calibration member changed while reading")
        files[entry.name] = raw
    if set(files) != _EXPECTED_FILES:
        raise RestReferenceArtifactError("calibration artifact inventory mismatch")
    return files


def _validate_links(
    references: RestReferenceBundle,
    validation: NoContactValidationReceipt,
    root: RestReferenceArtifactRootReceipt,
) -> None:
    if references.sha256 != root.rest_reference_sha256:
        raise RestReferenceArtifactError("root does not bind rest-reference content")
    if validation.sha256 != root.no_contact_validation_sha256:
        raise RestReferenceArtifactError("root does not bind validation content")
    if references.no_contact_validation_sha256 != validation.sha256:
        raise RestReferenceArtifactError("rest reference does not bind validation")
    if references.no_contact_predicate_id != validation.no_contact_predicate_id:
        raise RestReferenceArtifactError("no-contact predicate binding mismatch")
    if references.source_artifact_sha256 != validation.source_live_artifact_root_sha256:
        raise RestReferenceArtifactError("source live artifact binding mismatch")
    if (
        root.source_live_artifact_root_sha256
        != validation.source_live_artifact_root_sha256
    ):
        raise RestReferenceArtifactError("root source binding mismatch")
    if references.dataset_split is not validation.dataset_split or (
        references.split_manifest_sha256 != validation.split_manifest_sha256
    ):
        raise RestReferenceArtifactError("calibration split binding mismatch")
    config = build_univtac_backend_config(validation.task)
    if canonical_hash(config.phase_tracker) != validation.phase_tracker_sha256:
        raise RestReferenceArtifactError("phase tracker differs from packaged registry")
    selected_offset = validation.selected_step_index - validation.qualified_start_index
    selected_record_sha256 = validation.qualified_clean_record_sha256s[selected_offset]
    for slot in SENSOR_SLOTS:
        if (
            references.qualified_record_ids[slot]
            != validation.qualified_record_ids[slot]
        ):
            raise RestReferenceArtifactError("qualified record identity mismatch")
        if references.calibration_sha256[slot] != validation.calibration_sha256[slot]:
            raise RestReferenceArtifactError("calibration hash mismatch")
        if (
            array_sha256(references.payload_for(slot))
            != validation.selected_payload_sha256[slot]
        ):
            raise RestReferenceArtifactError("selected payload hash mismatch")
        if references.payload_for(slot).shape != config.tactile_rgb_shape:
            raise RestReferenceArtifactError("rest payload shape differs from registry")
        if validation.calibration_sha256[slot] != (
            config.aliases.calibration_config_sha256
        ):
            raise RestReferenceArtifactError("calibration differs from registry")
        expected_record_id = canonical_hash(
            {
                "namespace": "robotactile.rest-reference-record.v1",
                "source_live_artifact_root_sha256": (
                    validation.source_live_artifact_root_sha256
                ),
                "clean_record_sha256": selected_record_sha256,
                "slot_id": slot,
                "source_index": validation.selected_step_index,
                "payload_sha256": validation.selected_payload_sha256[slot],
            }
        )
        if validation.qualified_record_ids[slot] != expected_record_id:
            raise RestReferenceArtifactError(
                "qualified record ID is not reconstructable"
            )


def load_rest_reference_artifact(output: Path) -> LoadedRestReferenceArtifact:
    """Strictly load and cross-validate one calibration artifact directory."""

    try:
        files = _scan(Path(output))
        root = RestReferenceArtifactRootReceipt.from_dict(
            _strict_document(files[ROOT_RECEIPT_PATH], ROOT_RECEIPT_PATH)
        )
        by_path = {item.path: item for item in root.members}
        for path in (VALIDATION_PATH, REST_REFERENCE_PATH):
            member = by_path[path]
            if (
                len(files[path]) != member.size_bytes
                or sha256_bytes(files[path]) != member.sha256
            ):
                raise RestReferenceArtifactError(f"calibration member mismatch: {path}")
        references = _bundle_from_dict(
            _strict_document(files[REST_REFERENCE_PATH], REST_REFERENCE_PATH)
        )
        validation = NoContactValidationReceipt.from_dict(
            _strict_document(files[VALIDATION_PATH], VALIDATION_PATH)
        )
        _validate_links(references, validation, root)
        return LoadedRestReferenceArtifact(
            references=references,
            validation=validation,
            root_receipt=root,
            root_receipt_sha256=sha256_bytes(files[ROOT_RECEIPT_PATH]),
        )
    except RestReferenceArtifactError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise RestReferenceArtifactError(
            "rest-reference artifact failed typed validation"
        ) from error


def _write_staging(
    staging: Path,
    references: RestReferenceBundle,
    validation: NoContactValidationReceipt,
) -> None:
    documents = {
        REST_REFERENCE_PATH: canonical_json_bytes(_bundle_to_dict(references)),
        VALIDATION_PATH: canonical_json_bytes(validation.to_dict()),
    }
    for path, raw in documents.items():
        if len(raw) > _MAX_MEMBER_BYTES:
            raise RestReferenceArtifactError("calibration document exceeds size cap")
        (staging / path).write_bytes(raw)
    members = tuple(
        RestReferenceArtifactMember(path, sha256_bytes(raw), len(raw))
        for path, raw in sorted(documents.items())
    )
    root = RestReferenceArtifactRootReceipt(
        rest_reference_sha256=references.sha256,
        no_contact_validation_sha256=validation.sha256,
        source_live_artifact_root_sha256=validation.source_live_artifact_root_sha256,
        members=members,
    )
    (staging / ROOT_RECEIPT_PATH).write_bytes(canonical_json_bytes(root.to_dict()))


def write_rest_reference_artifact(
    output: Path,
    references: RestReferenceBundle,
    validation: NoContactValidationReceipt,
) -> RestReferenceArtifactExportReceipt:
    """Stage, reload, and atomically publish one calibration artifact."""

    _validate_links(
        references,
        validation,
        RestReferenceArtifactRootReceipt(
            references.sha256,
            validation.sha256,
            validation.source_live_artifact_root_sha256,
            (
                RestReferenceArtifactMember(VALIDATION_PATH, "0" * 64, 1),
                RestReferenceArtifactMember(REST_REFERENCE_PATH, "0" * 64, 1),
            ),
        ),
    )
    output = Path(output)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise FileExistsError("rest-reference target is not a real directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".rest-reference-staging-", dir=output.parent)
    )
    try:
        _write_staging(staging, references, validation)
        staged = load_rest_reference_artifact(staging)
        if output.exists() and any(output.iterdir()):
            existing = load_rest_reference_artifact(output)
            if existing.root_receipt_sha256 != staged.root_receipt_sha256:
                raise FileExistsError("non-empty target contains another calibration")
            loaded = existing
        else:
            if output.exists():
                output.rmdir()
            os.replace(staging, output)
            loaded = load_rest_reference_artifact(output)
        return RestReferenceArtifactExportReceipt(
            loaded.root_receipt_sha256,
            loaded.references.sha256,
            loaded.validation.sha256,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
