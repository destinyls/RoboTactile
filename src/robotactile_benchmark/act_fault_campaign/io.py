"""Canonical no-clobber persistence for official ACT fault campaigns."""

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
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.trials import Condition

from .contracts import (
    ACT_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION,
    ACT_FAULT_GENERATION_EVIDENCE_LEVEL,
    ACTFaultCampaignCellSpec,
    ACTFaultCampaignError,
    ACTFaultCampaignManifest,
    ACTFaultCellDisposition,
    ACTUnsupportedContractSpec,
    _identifier,
    _integer,
    _sha256,
)
from .reset_reference import (
    act_reset_reference_relpath,
    load_act_reset_reference,
)
from .reset_trajectory import (
    act_reset_trajectory_relpath,
    load_act_reset_trajectory,
)

CAMPAIGN_MANIFEST_PATH = "campaign_manifest.json"
GENERATION_RECEIPT_PATH = "generation_receipt.json"
MAX_CAMPAIGN_JSON_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class ACTFaultCampaignGenerationReceipt:
    """Hash inventory proving request generation, never simulator execution."""

    campaign_id: str
    campaign_manifest_sha256: str
    generation_contract_sha256: str
    pair_count: int
    cell_count: int
    live_request_count: int
    unsupported_contract_count: int
    fault_manifest_count: int
    members: Mapping[str, str]
    evidence_level: str = ACT_FAULT_GENERATION_EVIDENCE_LEVEL
    simulator_execution_claimed: bool = False
    task_success_claimed: bool = False
    semantic_version: str = ACT_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", _identifier(self.campaign_id, "campaign_id")
        )
        for name in ("campaign_manifest_sha256", "generation_contract_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in (
            "pair_count",
            "cell_count",
            "live_request_count",
            "fault_manifest_count",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        object.__setattr__(
            self,
            "unsupported_contract_count",
            _integer(self.unsupported_contract_count, "unsupported_contract_count"),
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
                raise ACTFaultCampaignError("receipt member path is unsafe")
            members[path] = _sha256(digest, f"member {path}")
        if not members or tuple(members) != tuple(sorted(members)):
            raise ACTFaultCampaignError("receipt members must be non-empty and sorted")
        object.__setattr__(self, "members", MappingProxyType(members))
        if (
            self.evidence_level != ACT_FAULT_GENERATION_EVIDENCE_LEVEL
            or self.simulator_execution_claimed
            or self.task_success_claimed
            or self.semantic_version != ACT_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION
        ):
            raise ACTFaultCampaignError("generation receipt cannot claim execution")

    def to_dict(self) -> dict[str, object]:
        return {
            name: dict(getattr(self, name))
            if name == "members"
            else getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> ACTFaultCampaignGenerationReceipt:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ACTFaultCampaignError("generation receipt fields mismatch")
        return cls(**cast(Any, dict(value)))


@dataclass(frozen=True)
class LoadedACTFaultCampaign:
    """Strictly reloaded generated ACT campaign bundle."""

    root: Path
    manifest: ACTFaultCampaignManifest
    receipt: ACTFaultCampaignGenerationReceipt
    receipt_file_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.root, Path)
            or type(self.manifest) is not ACTFaultCampaignManifest
            or type(self.receipt) is not ACTFaultCampaignGenerationReceipt
        ):
            raise TypeError("loaded campaign fields must use exact contract types")
        object.__setattr__(self, "root", self.root.absolute())
        object.__setattr__(
            self,
            "receipt_file_sha256",
            _sha256(self.receipt_file_sha256, "receipt_file_sha256"),
        )


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ACTFaultCampaignError("campaign member must be a regular file")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> Mapping[str, object]:
    """Read one bounded canonical JSON object without TOCTOU drift."""

    candidate = Path(path).absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise ACTFaultCampaignError(f"{label} must be a regular file")
    before = candidate.stat()
    if not 1 <= before.st_size <= MAX_CAMPAIGN_JSON_BYTES:
        raise ACTFaultCampaignError(f"{label} size is invalid")
    raw = candidate.read_bytes()
    after = candidate.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ACTFaultCampaignError(f"{label} changed while reading")
    try:
        value = strict_json_bytes(raw, label)
    except (TypeError, ValueError) as error:
        raise ACTFaultCampaignError(f"{label} is not strict canonical JSON") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise ACTFaultCampaignError(f"{label} must be a canonical JSON object")
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
        raise ACTFaultCampaignError("campaign root must be a real directory")
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise ACTFaultCampaignError("campaign bundle cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ACTFaultCampaignError("campaign members must be regular files")
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
        raise ACTFaultCampaignError("resource source must be a regular file")
    if not 1 <= origin.stat().st_size <= MAX_CAMPAIGN_JSON_BYTES:
        raise ACTFaultCampaignError("resource source size is invalid")
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
        raise ACTFaultCampaignError("another campaign publication is active") from error
    try:
        if target.exists() or target.is_symlink():
            if (
                not target.is_symlink()
                and target.is_dir()
                and scan_files(
                    target,
                    exclude=GENERATION_RECEIPT_PATH,
                    exclude_prefixes=("artifacts", "executions"),
                )
                == scan_files(
                    source,
                    exclude=GENERATION_RECEIPT_PATH,
                    exclude_prefixes=("artifacts", "executions"),
                )
                and file_sha256(target / GENERATION_RECEIPT_PATH)
                == file_sha256(source / GENERATION_RECEIPT_PATH)
            ):
                return "already_present"
            raise FileExistsError(f"refusing to replace existing output: {target}")
        os.rename(source, target)
        return "created"
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def load_campaign_manifest(path: Path) -> ACTFaultCampaignManifest:
    return ACTFaultCampaignManifest.from_dict(read_json(path, "campaign manifest"))


def load_generation_receipt(path: Path) -> ACTFaultCampaignGenerationReceipt:
    return ACTFaultCampaignGenerationReceipt.from_dict(
        read_json(path, "generation receipt")
    )


def load_unsupported_contract(path: Path) -> ACTUnsupportedContractSpec:
    return ACTUnsupportedContractSpec.from_dict(
        read_json(path, "unsupported contract receipt")
    )


def load_act_fault_campaign_bundle(root: Path) -> LoadedACTFaultCampaign:
    """Fail closed on inventory, hashes, ACT requests, or typed receipts."""

    directory = Path(root).expanduser().absolute()
    receipt = load_generation_receipt(directory / GENERATION_RECEIPT_PATH)
    inventory = scan_files(
        directory,
        exclude=GENERATION_RECEIPT_PATH,
        exclude_prefixes=("artifacts", "executions"),
    )
    if inventory != dict(receipt.members):
        raise ACTFaultCampaignError("campaign member inventory mismatch")
    manifest = load_campaign_manifest(directory / CAMPAIGN_MANIFEST_PATH)
    if (
        manifest.sha256 != receipt.campaign_manifest_sha256
        or manifest.campaign_id != receipt.campaign_id
    ):
        raise ACTFaultCampaignError("generation receipt does not bind the manifest")
    _validate_counts(manifest, receipt)
    _validate_rest_bindings(directory, manifest)
    clean_pairs: set[str] = set()
    pair_runtime_dirs: dict[str, Path] = {}
    for cell in manifest.cells:
        fault = _load_cell_fault(directory, cell)
        if cell.disposition is ACTFaultCellDisposition.UNSUPPORTED_CONTRACT:
            _validate_unsupported_cell(directory, manifest.campaign_id, cell)
            continue
        request_path = directory / str(cell.request_relpath)
        if file_sha256(request_path) != cell.request_file_sha256:
            raise ACTFaultCampaignError("request file hash differs from campaign cell")
        request = load_live_univtac_request(request_path)
        loaded = load_live_univtac_run(request)
        expected_system = f"official-univtac-act.{cell.task}.{OfficialACTProfile.UNIVTAC.value}.policy_last.v1"
        expected_output = (directory / str(cell.artifact_relpath)).resolve()
        runtime_dir = request.runtime_dir.resolve(strict=False)
        pair_runtime_dir = pair_runtime_dirs.setdefault(cell.pair_key, runtime_dir)
        if (
            request.policy_kind is not LivePolicyKind.ACT
            or request.condition is not cell.condition
            or request.base_system_id != expected_system
            or request.execute_action_steps != 1
            or loaded.trial.sha256 != cell.trial_manifest_sha256
            or loaded.trial.pair_key != cell.pair_key
            or loaded.fault_manifest != fault
            or request.output_dir is None
            or request.output_dir.resolve() != expected_output
            or runtime_dir != pair_runtime_dir
        ):
            raise ACTFaultCampaignError("live ACT request differs from campaign cell")
        if cell.condition is Condition.CLEAN:
            clean_pairs.add(cell.pair_key)
    if clean_pairs != {cell.pair_key for cell in manifest.cells}:
        raise ACTFaultCampaignError("campaign Clean pairing inventory mismatch")
    _validate_reset_references(directory, manifest)
    _validate_reset_trajectories(directory, manifest)
    return LoadedACTFaultCampaign(
        root=directory,
        manifest=manifest,
        receipt=receipt,
        receipt_file_sha256=file_sha256(directory / GENERATION_RECEIPT_PATH),
    )


def _validate_rest_bindings(
    directory: Path, manifest: ACTFaultCampaignManifest
) -> None:
    for task, binding in manifest.rest_reference_bindings.items():
        loaded = load_rest_reference_artifact(directory / binding["artifact_relpath"])
        if (
            loaded.validation.task != task
            or loaded.root_receipt_sha256 != binding["artifact_root_sha256"]
            or loaded.references.sha256 != binding["rest_reference_sha256"]
        ):
            raise ACTFaultCampaignError("rest-reference binding drifted")


def _validate_reset_references(
    directory: Path,
    manifest: ACTFaultCampaignManifest,
) -> None:
    clean = tuple(
        cell
        for cell in manifest.cells
        if cell.condition is Condition.CLEAN
        and cell.disposition is ACTFaultCellDisposition.LIVE_REQUEST
    )
    paths = tuple(
        directory / act_reset_reference_relpath(cell.task, cell.pair_key)
        for cell in clean
    )
    present = tuple(path.is_file() and not path.is_symlink() for path in paths)
    if not any(present):
        # Legacy generated bundles remain loadable for reporting, but the
        # execution runner requires a source-bound reset reference.
        return
    if not all(present):
        raise ACTFaultCampaignError("reset references only partially cover pairs")
    for cell, path in zip(clean, paths):
        reference = load_act_reset_reference(path)
        request = load_live_univtac_request(directory / str(cell.request_relpath))
        loaded = load_live_univtac_run(request)
        trial = loaded.trial
        if (
            reference.task_id != trial.task
            or reference.initial_seed != trial.initial_seed
            or reference.exogenous_seed != trial.exogenous_seed
            or reference.pair_key != trial.pair_key
            or reference.dataset_sha256 != trial.dataset_sha256
            or reference.checkpoint_sha256 != trial.checkpoint_sha256
            or reference.config_sha256 != trial.config_sha256
            or reference.source_run_content_sha256 != loaded.content_sha256
        ):
            raise ACTFaultCampaignError(
                "reset reference identity differs from campaign Clean"
            )


def _validate_reset_trajectories(
    directory: Path,
    manifest: ACTFaultCampaignManifest,
) -> None:
    clean = tuple(
        cell
        for cell in manifest.cells
        if cell.condition is Condition.CLEAN
        and cell.disposition is ACTFaultCellDisposition.LIVE_REQUEST
    )
    paths = tuple(
        directory / act_reset_trajectory_relpath(cell.task, cell.pair_key)
        for cell in clean
    )
    present = tuple(path.is_file() and not path.is_symlink() for path in paths)
    if not any(present):
        # Legacy bundles remain reportable; execution requires this artifact.
        return
    if not all(present):
        raise ACTFaultCampaignError("reset trajectories only partially cover pairs")
    for cell, path in zip(clean, paths):
        trajectory = load_act_reset_trajectory(path)
        request = load_live_univtac_request(directory / str(cell.request_relpath))
        loaded = load_live_univtac_run(request)
        reference = load_act_reset_reference(
            directory / act_reset_reference_relpath(cell.task, cell.pair_key)
        )
        trial = loaded.trial
        if (
            trajectory.task_id != trial.task
            or trajectory.initial_seed != trial.initial_seed
            or trajectory.exogenous_seed != trial.exogenous_seed
            or trajectory.pair_key != trial.pair_key
            or trajectory.dataset_sha256 != trial.dataset_sha256
            or trajectory.checkpoint_sha256 != trial.checkpoint_sha256
            or trajectory.config_sha256 != trial.config_sha256
            or trajectory.source_run_content_sha256 != loaded.content_sha256
            or trajectory.reset_reference_sha256 != reference.sha256
            or trajectory.upstream_commit != loaded.backend_config.upstream_commit
            or trajectory.task_source_sha256
            != loaded.backend_config.task.task_source_sha256
        ):
            raise ACTFaultCampaignError(
                "reset trajectory identity differs from campaign Clean"
            )


def _load_cell_fault(
    directory: Path, cell: ACTFaultCampaignCellSpec
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
        raise ACTFaultCampaignError("fault manifest differs from campaign cell")
    return fault


def _validate_unsupported_cell(
    directory: Path, campaign_id: str, cell: ACTFaultCampaignCellSpec
) -> None:
    path = directory / str(cell.unsupported_receipt_relpath)
    unsupported = load_unsupported_contract(path)
    expected = (
        unsupported.campaign_id == campaign_id,
        unsupported.task == cell.task,
        unsupported.initial_seed == cell.initial_seed,
        unsupported.exogenous_seed == cell.exogenous_seed,
        unsupported.pair_key == cell.pair_key,
        unsupported.operator_id == cell.operator_id,
        unsupported.severity_level == cell.severity_level,
        unsupported.operator_template_seed == cell.operator_template_seed,
        unsupported.fault_manifest_sha256 == cell.fault_manifest_sha256,
        unsupported.trial_manifest_sha256 == cell.trial_manifest_sha256,
    )
    if file_sha256(path) != cell.unsupported_receipt_sha256 or not all(expected):
        raise ACTFaultCampaignError("unsupported receipt differs from campaign cell")


def _validate_counts(
    manifest: ACTFaultCampaignManifest, receipt: ACTFaultCampaignGenerationReceipt
) -> None:
    if (
        receipt.pair_count != manifest.pair_count
        or receipt.cell_count != manifest.cell_count
        or receipt.live_request_count != manifest.live_request_count
        or receipt.unsupported_contract_count != manifest.unsupported_contract_count
        or receipt.fault_manifest_count != manifest.cell_count - manifest.pair_count
    ):
        raise ACTFaultCampaignError("generation receipt counts disagree")
