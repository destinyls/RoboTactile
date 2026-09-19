"""Strict canonical persistence for generated N0 fault campaigns."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Optional, cast

from robotactile_benchmark.calibration import load_rest_reference_artifact
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.n0_fault_campaign.contracts import (
    N0_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION,
    N0_FAULT_GENERATION_EVIDENCE_LEVEL,
    N0FaultCampaignCellSpec,
    N0FaultCampaignError,
    N0FaultCampaignManifest,
    N0FaultCellDisposition,
    N0UnsupportedContractSpec,
    require_identifier,
    require_integer,
    require_sha256,
)
from robotactile_benchmark.trials import Condition

CAMPAIGN_MANIFEST_PATH = "campaign_manifest.json"
GENERATION_RECEIPT_PATH = "generation_receipt.json"
MAX_CAMPAIGN_JSON_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class N0FaultCampaignGenerationReceipt:
    """Hash inventory proving deterministic request generation only."""

    campaign_id: str
    campaign_manifest_sha256: str
    generation_contract_sha256: str
    pair_count: int
    cell_count: int
    live_request_count: int
    unsupported_contract_count: int
    fault_manifest_count: int
    members: Mapping[str, str]
    evidence_level: str = N0_FAULT_GENERATION_EVIDENCE_LEVEL
    simulator_execution_claimed: bool = False
    task_success_claimed: bool = False
    semantic_version: str = N0_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", require_identifier(self.campaign_id, "campaign_id")
        )
        for name in ("campaign_manifest_sha256", "generation_contract_sha256"):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        for name in (
            "pair_count",
            "cell_count",
            "live_request_count",
            "fault_manifest_count",
        ):
            object.__setattr__(
                self, name, require_integer(getattr(self, name), name, 1)
            )
        object.__setattr__(
            self,
            "unsupported_contract_count",
            require_integer(
                self.unsupported_contract_count, "unsupported_contract_count"
            ),
        )
        members: dict[str, str] = {}
        for path, digest in self.members.items():
            candidate = PurePosixPath(path)
            if (
                not path
                or path == GENERATION_RECEIPT_PATH
                or candidate.is_absolute()
                or ".." in candidate.parts
                or "\\" in path
            ):
                raise N0FaultCampaignError("receipt member path is unsafe")
            members[path] = require_sha256(digest, f"member {path}")
        if not members or tuple(members) != tuple(sorted(members)):
            raise N0FaultCampaignError("receipt members must be non-empty and sorted")
        object.__setattr__(self, "members", MappingProxyType(members))
        if (
            self.evidence_level != N0_FAULT_GENERATION_EVIDENCE_LEVEL
            or self.simulator_execution_claimed
            or self.task_success_claimed
        ):
            raise N0FaultCampaignError("generation receipt cannot claim execution")
        if self.semantic_version != N0_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION:
            raise N0FaultCampaignError("generation receipt version mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            name: dict(getattr(self, name))
            if name == "members"
            else getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> N0FaultCampaignGenerationReceipt:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise N0FaultCampaignError("generation receipt fields mismatch")
        return cls(**cast(Any, dict(value)))


@dataclass(frozen=True)
class LoadedN0FaultCampaign:
    """Strictly reloaded N0 fault campaign bundle."""

    root: Path
    manifest: N0FaultCampaignManifest
    receipt: N0FaultCampaignGenerationReceipt
    receipt_file_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.root, Path)
            or type(self.manifest) is not N0FaultCampaignManifest
            or type(self.receipt) is not N0FaultCampaignGenerationReceipt
        ):
            raise TypeError("loaded campaign fields must use exact contract types")
        object.__setattr__(self, "root", self.root.absolute())
        object.__setattr__(
            self,
            "receipt_file_sha256",
            require_sha256(self.receipt_file_sha256, "receipt_file_sha256"),
        )


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise N0FaultCampaignError("campaign member must be a regular file")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> Mapping[str, object]:
    """Read a bounded exact canonical JSON object without TOCTOU drift."""

    candidate = Path(path).absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise N0FaultCampaignError(f"{label} must be a regular file")
    before = candidate.stat()
    if not 1 <= before.st_size <= MAX_CAMPAIGN_JSON_BYTES:
        raise N0FaultCampaignError(f"{label} size is invalid")
    raw = candidate.read_bytes()
    after = candidate.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise N0FaultCampaignError(f"{label} changed while reading")
    try:
        value = strict_json_bytes(raw, label)
    except (TypeError, ValueError) as error:
        raise N0FaultCampaignError(f"{label} is not strict canonical JSON") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise N0FaultCampaignError(f"{label} must be a canonical JSON object")
    return cast(Mapping[str, object], value)


def write_json(path: Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def scan_files(
    root: Path,
    *,
    exclude: Optional[str] = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> dict[str, str]:
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise N0FaultCampaignError("campaign root must be a real directory")
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise N0FaultCampaignError("campaign bundle cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise N0FaultCampaignError("campaign members must be regular files")
        dynamic = any(
            relative == prefix or relative.startswith(f"{prefix}/")
            for prefix in exclude_prefixes
        )
        if relative != exclude and not dynamic:
            result[relative] = file_sha256(path)
    return result


def copy_regular_file(source: Path, destination: Path) -> None:
    origin = Path(source)
    if origin.is_symlink() or not origin.is_file():
        raise N0FaultCampaignError("resource source must be a regular file")
    if not 1 <= origin.stat().st_size <= MAX_CAMPAIGN_JSON_BYTES:
        raise N0FaultCampaignError("resource source size is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(origin.read_bytes())


def staging_directory(output: Path) -> Path:
    target = Path(output).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))


def discard_staging(path: Path) -> None:
    candidate = Path(path)
    if candidate.exists():
        shutil.rmtree(candidate)


def publish_staging(staging: Path, output: Path) -> str:
    source = Path(staging)
    target = Path(output).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise N0FaultCampaignError("another campaign publication is active") from error
    try:
        if target.exists() or target.is_symlink():
            if (
                not target.is_symlink()
                and target.is_dir()
                and scan_files(target) == scan_files(source)
            ):
                return "already_present"
            raise FileExistsError(f"refusing to replace existing output: {target}")
        os.rename(source, target)
        return "created"
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def load_campaign_manifest(path: Path) -> N0FaultCampaignManifest:
    return N0FaultCampaignManifest.from_dict(read_json(path, "campaign manifest"))


def load_generation_receipt(path: Path) -> N0FaultCampaignGenerationReceipt:
    return N0FaultCampaignGenerationReceipt.from_dict(
        read_json(path, "generation receipt")
    )


def load_unsupported_contract(path: Path) -> N0UnsupportedContractSpec:
    return N0UnsupportedContractSpec.from_dict(
        read_json(path, "unsupported contract receipt")
    )


def load_n0_fault_campaign_bundle(root: Path) -> LoadedN0FaultCampaign:
    """Fail closed on inventory, hashes, request links, or receipts."""

    directory = Path(root).expanduser().absolute()
    receipt = load_generation_receipt(directory / GENERATION_RECEIPT_PATH)
    if scan_files(
        directory,
        exclude=GENERATION_RECEIPT_PATH,
        exclude_prefixes=("artifacts", "executions"),
    ) != dict(receipt.members):
        raise N0FaultCampaignError("campaign member inventory mismatch")
    manifest = load_campaign_manifest(directory / CAMPAIGN_MANIFEST_PATH)
    if (
        manifest.sha256 != receipt.campaign_manifest_sha256
        or manifest.campaign_id != receipt.campaign_id
    ):
        raise N0FaultCampaignError("generation receipt does not bind the manifest")
    _validate_loaded_counts(manifest, receipt)
    _validate_rest_bindings(directory, manifest)
    clean_pairs: set[str] = set()
    for cell in manifest.cells:
        fault = _load_cell_fault(directory, cell)
        if cell.disposition is N0FaultCellDisposition.UNSUPPORTED_CONTRACT:
            _validate_unsupported_cell(directory, manifest.campaign_id, cell)
            continue
        request_path = directory / str(cell.request_relpath)
        if file_sha256(request_path) != cell.request_file_sha256:
            raise N0FaultCampaignError("request file hash differs from campaign cell")
        request = load_live_univtac_request(request_path)
        loaded = load_live_univtac_run(request)
        expected_output = (directory / str(cell.artifact_relpath)).resolve()
        if (
            request.policy_kind is not LivePolicyKind.N0
            or loaded.trial.sha256 != cell.trial_manifest_sha256
            or loaded.trial.pair_key != cell.pair_key
            or loaded.fault_manifest != fault
            or request.output_dir is None
            or request.output_dir.resolve() != expected_output
        ):
            raise N0FaultCampaignError("live request differs from campaign cell")
        if cell.condition is Condition.CLEAN:
            clean_pairs.add(cell.pair_key)
    if clean_pairs != {cell.pair_key for cell in manifest.cells}:
        raise N0FaultCampaignError("campaign Clean pairing inventory mismatch")
    return LoadedN0FaultCampaign(
        root=directory,
        manifest=manifest,
        receipt=receipt,
        receipt_file_sha256=file_sha256(directory / GENERATION_RECEIPT_PATH),
    )


def _validate_rest_bindings(directory: Path, manifest: N0FaultCampaignManifest) -> None:
    for task, binding in manifest.rest_reference_bindings.items():
        loaded = load_rest_reference_artifact(directory / binding["artifact_relpath"])
        if (
            loaded.validation.task != task
            or loaded.root_receipt_sha256 != binding["artifact_root_sha256"]
            or loaded.references.sha256 != binding["rest_reference_sha256"]
        ):
            raise N0FaultCampaignError("rest-reference binding drifted")


def _load_cell_fault(
    directory: Path, cell: N0FaultCampaignCellSpec
) -> Optional[FaultManifest]:
    if cell.fault_manifest_relpath is None:
        return None
    fault = FaultManifest.from_dict(
        read_json(directory / cell.fault_manifest_relpath, "fault manifest")
    )
    if (
        fault.sha256 != cell.fault_manifest_sha256
        or fault.operator_seed != cell.operator_template_seed
        or fault.operator_id != cell.operator_id
        or fault.severity_level != cell.severity_level
    ):
        raise N0FaultCampaignError("fault manifest differs from campaign cell")
    return fault


def _validate_unsupported_cell(
    directory: Path, campaign_id: str, cell: N0FaultCampaignCellSpec
) -> None:
    path = directory / str(cell.unsupported_receipt_relpath)
    unsupported = load_unsupported_contract(path)
    if (
        file_sha256(path) != cell.unsupported_receipt_sha256
        or unsupported.campaign_id != campaign_id
        or unsupported.task != cell.task
        or unsupported.initial_seed != cell.initial_seed
        or unsupported.exogenous_seed != cell.exogenous_seed
        or unsupported.pair_key != cell.pair_key
        or unsupported.operator_id != cell.operator_id
        or unsupported.severity_level != cell.severity_level
        or unsupported.operator_template_seed != cell.operator_template_seed
        or unsupported.fault_manifest_sha256 != cell.fault_manifest_sha256
        or unsupported.trial_manifest_sha256 != cell.trial_manifest_sha256
    ):
        raise N0FaultCampaignError("unsupported receipt differs from campaign cell")


def _validate_loaded_counts(
    manifest: N0FaultCampaignManifest,
    receipt: N0FaultCampaignGenerationReceipt,
) -> None:
    if (
        receipt.pair_count != manifest.pair_count
        or receipt.cell_count != manifest.cell_count
        or receipt.live_request_count != manifest.live_request_count
        or receipt.unsupported_contract_count != manifest.unsupported_contract_count
        or receipt.fault_manifest_count != manifest.cell_count - manifest.pair_count
    ):
        raise N0FaultCampaignError("generation receipt counts disagree")
