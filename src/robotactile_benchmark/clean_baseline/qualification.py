"""Fail-closed UniVTAC all-task qualification loading and source binding."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Tuple, cast

from robotactile_benchmark.backends.univtac_registry import load_registry
from robotactile_benchmark.clean_baseline.contracts import (
    CleanCampaignError,
    CleanCampaignManifest,
)
from robotactile_benchmark.clean_baseline.io import read_canonical_json_file
from robotactile_benchmark.integrations.n0_twam.observation_parity_contracts import (
    ObservationEvidenceBinding,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity_io import (
    load_observation_parity_artifact,
)
from robotactile_benchmark.runtime_source import RuntimeSourceBinding

QUALIFICATION_EVIDENCE_LEVEL = "all_tasks_reset_action_contract_qualification_v1"
SOURCE_BOUND_QUALIFICATION_EVIDENCE_LEVEL = (
    "source_bound_all_tasks_observation_action_contract_qualification_v2"
)
TASK_SOURCE_BOUND_QUALIFICATION_EVIDENCE_LEVEL = (
    "task_source_bound_all_tasks_observation_action_contract_qualification_v3"
)
QUALIFICATION_V1_SEMANTIC_VERSION = "1.0"
QUALIFICATION_V2_SEMANTIC_VERSION = "2.0"
QUALIFICATION_V3_SEMANTIC_VERSION = "3.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BASE_FIELDS = frozenset(
    {
        "action_spec",
        "campaign_id",
        "closed_loop_episode_executed",
        "evidence_level",
        "master_seed",
        "policy_loaded",
        "repetitions",
        "recorded_at_utc",
        "seed_derivation",
        "semantic_version",
        "success_rate_claimed",
        "tasks",
    }
)
_V2_FIELDS = _BASE_FIELDS | {
    "campaign_manifest_relpath",
    "campaign_manifest_sha256",
    "installation_receipts",
    "source_binding",
}
_V3_FIELDS = _V2_FIELDS - {"source_binding"}
_TASK_BASE_FIELDS = frozenset(
    {
        "exogenous_seed",
        "import_receipt_relpath",
        "import_receipt_sha256",
        "initial_seed",
        "pairing_receipt_relpaths",
        "pairing_receipt_sha256s",
        "reset_receipt_relpaths",
        "reset_receipt_sha256s",
        "task_id",
    }
)
_TASK_V2_FIELDS = _TASK_BASE_FIELDS | {
    "observation_parity_relpath",
    "observation_parity_sha256",
}
_TASK_V3_FIELDS = _TASK_V2_FIELDS | {"source_binding"}
_INSTALLATION_KINDS = (
    "isaac_install_receipt",
    "n0_client_install_receipt",
    "n0_runtime_receipt",
    "tacex_install_receipt",
)
_SHARED_RUNTIME_SOURCE_FIELDS = (
    "robotactile_source_manifest_sha256",
    "robotactile_wheel_sha256",
    "integrations_lock_sha256",
    "univtac_source_commit",
    "n0_source_commit",
    "checkpoint_sha256",
    "config_sha256",
    "prompt_manifest_sha256",
    "input_profile_sha256",
    "action_execution_contract",
    "native_step_contract",
)


@dataclass(frozen=True)
class VerifiedAllTaskQualification:
    """Verified qualification with an explicit legacy/source-bound distinction."""

    path: Path
    sha256: str
    campaign_id: str
    action_spec: str
    repetitions: int
    tasks: Tuple[str, ...]
    semantic_version: str
    source_binding: Optional[RuntimeSourceBinding]
    task_source_bindings: Tuple[RuntimeSourceBinding, ...]
    campaign_manifest_sha256: Optional[str]
    observation_parity_sha256s: Tuple[str, ...]

    @property
    def source_bound(self) -> bool:
        return bool(self.tasks) and len(self.task_source_bindings) == len(self.tasks)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CleanCampaignError(f"{name} must be a lowercase SHA256")
    return value


def _member(root: Path, relative: object, name: str) -> Path:
    if type(relative) is not str or "\\" in relative:
        raise CleanCampaignError(f"{name} must be a POSIX path")
    member = PurePosixPath(relative)
    if (
        member.is_absolute()
        or not member.parts
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise CleanCampaignError(f"{name} is unsafe")
    if member.parts[0] not in {"artifacts", "outputs", "requests"}:
        raise CleanCampaignError(f"{name} is outside deployment evidence")
    candidate = root / Path(*member.parts)
    current = root
    for part in member.parts:
        current = current / part
        if current.is_symlink():
            raise CleanCampaignError(f"{name} cannot traverse a symlink")
    return candidate


def _verify_bound_file(
    root: Path, relative: object, digest: object, *, name: str
) -> Path:
    expected = _sha256(digest, f"{name} hash")
    path = _member(root, relative, f"{name} path")
    if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
        raise CleanCampaignError(f"{name} binding mismatch")
    return path


def _verify_campaign_manifest_binding(
    root: Path, relative: object, digest: object
) -> str:
    """Verify a typed campaign and normalize legacy file hashes to content hash."""

    expected = _sha256(digest, "qualification campaign manifest hash")
    path = _member(root, relative, "qualification campaign manifest path")
    document, raw = read_canonical_json_file(path, "qualification campaign manifest")
    manifest = CleanCampaignManifest.from_dict(document)
    content_sha256 = manifest.sha256
    if expected not in {content_sha256, _sha256_bytes(raw)}:
        raise CleanCampaignError("qualification campaign manifest binding mismatch")
    return content_sha256


def _load_receipt(path: Path, name: str) -> Mapping[str, object]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CleanCampaignError(f"{name} contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CleanCampaignError(f"{name} must be JSON") from error
    if not isinstance(value, Mapping):
        raise CleanCampaignError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _expect_receipt(
    document: Mapping[str, object], expected: Mapping[str, str], name: str
) -> None:
    if any(document.get(key) != value for key, value in expected.items()):
        raise CleanCampaignError(f"{name} semantic identity mismatch")


def _qualification_path(root: Path, qualification_path: Path) -> Path:
    path = Path(qualification_path).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise CleanCampaignError(
            "all-task qualification must remain below deployment root"
        ) from error
    if relative.parts[:2] != ("artifacts", "deployment") or path.is_symlink():
        raise CleanCampaignError(
            "all-task qualification must be a regular artifacts/deployment file"
        )
    return path


def verify_all_task_qualification(
    deployment_root: Path, qualification_path: Path
) -> VerifiedAllTaskQualification:
    """Load v1 or fully verify source-bound v2/v3 qualification evidence."""

    root = Path(deployment_root).resolve(strict=True)
    path = _qualification_path(root, qualification_path)
    document, raw = read_canonical_json_file(path, "all-task qualification")
    semantic_version = document.get("semantic_version")
    if semantic_version == QUALIFICATION_V1_SEMANTIC_VERSION:
        return _verify_v1(root, path, document, raw)
    if semantic_version == QUALIFICATION_V2_SEMANTIC_VERSION:
        return _verify_v2(root, path, document, raw)
    if semantic_version == QUALIFICATION_V3_SEMANTIC_VERSION:
        return _verify_v3(root, path, document, raw)
    raise CleanCampaignError("all-task qualification version is unsupported")


def _verify_common(
    document: Mapping[str, object],
    *,
    expected_fields: frozenset[str],
    semantic_version: str,
) -> tuple[str, str, int, Sequence[object]]:
    if set(document) != expected_fields:
        raise CleanCampaignError("all-task qualification fields mismatch")
    fixed: dict[str, object] = {
        "closed_loop_episode_executed": False,
        "policy_loaded": False,
        "seed_derivation": "sha256_signed31_v1",
        "success_rate_claimed": False,
    }
    evidence_levels = {
        QUALIFICATION_V1_SEMANTIC_VERSION: QUALIFICATION_EVIDENCE_LEVEL,
        QUALIFICATION_V2_SEMANTIC_VERSION: (SOURCE_BOUND_QUALIFICATION_EVIDENCE_LEVEL),
        QUALIFICATION_V3_SEMANTIC_VERSION: (
            TASK_SOURCE_BOUND_QUALIFICATION_EVIDENCE_LEVEL
        ),
    }
    fixed.update(
        {
            "evidence_level": evidence_levels[semantic_version],
            "semantic_version": semantic_version,
        }
    )
    if any(document.get(key) != value for key, value in fixed.items()):
        raise CleanCampaignError("all-task qualification evidence boundary mismatch")
    action_spec = document.get("action_spec")
    campaign_id = document.get("campaign_id")
    repetitions = document.get("repetitions")
    tasks = document.get("tasks")
    supported = (
        {"qpos8_next_step", "ee8_absolute"}
        if semantic_version == QUALIFICATION_V1_SEMANTIC_VERSION
        else {"ee8_absolute"}
    )
    if action_spec not in supported:
        raise CleanCampaignError("qualification action spec is unsupported")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise CleanCampaignError("qualification campaign ID is invalid")
    if type(repetitions) is not int or repetitions < 1:
        raise CleanCampaignError("qualification repetitions must be positive")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        raise CleanCampaignError("qualification tasks must be a sequence")
    if len(tasks) != 8:
        raise CleanCampaignError("qualification must contain exactly eight tasks")
    return cast(str, action_spec), campaign_id, repetitions, tasks


def _verify_v1(
    root: Path,
    path: Path,
    document: Mapping[str, object],
    raw: bytes,
) -> VerifiedAllTaskQualification:
    action_spec, campaign_id, repetitions, tasks = _verify_common(
        document,
        expected_fields=_BASE_FIELDS,
        semantic_version=QUALIFICATION_V1_SEMANTIC_VERSION,
    )
    task_ids = tuple(_verify_task_v1(root, item, repetitions) for item in tasks)
    _verify_task_inventory(task_ids)
    return VerifiedAllTaskQualification(
        path=path,
        sha256=_sha256_bytes(raw),
        campaign_id=campaign_id,
        action_spec=action_spec,
        repetitions=repetitions,
        tasks=task_ids,
        semantic_version=QUALIFICATION_V1_SEMANTIC_VERSION,
        source_binding=None,
        task_source_bindings=(),
        campaign_manifest_sha256=None,
        observation_parity_sha256s=(),
    )


def _verify_task_v1(root: Path, value: object, repetitions: int) -> str:
    if not isinstance(value, Mapping) or set(value) != _TASK_BASE_FIELDS:
        raise CleanCampaignError("qualification task fields mismatch")
    task = cast(Mapping[str, object], value)
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise CleanCampaignError("qualification task ID is invalid")
    _verify_bound_file(
        root,
        task["import_receipt_relpath"],
        task["import_receipt_sha256"],
        name="qualification receipt",
    )
    _verify_repeated_files(root, task, repetitions)
    return task_id


def _verify_repeated_files(
    root: Path, task: Mapping[str, object], repetitions: int
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    verified: dict[str, tuple[Path, ...]] = {}
    for prefix in ("reset", "pairing"):
        paths = task[f"{prefix}_receipt_relpaths"]
        hashes = task[f"{prefix}_receipt_sha256s"]
        if (
            not isinstance(paths, Sequence)
            or isinstance(paths, (str, bytes))
            or not isinstance(hashes, Sequence)
            or isinstance(hashes, (str, bytes))
            or len(paths) != repetitions
            or len(hashes) != repetitions
        ):
            raise CleanCampaignError("qualification repeated receipt count mismatch")
        verified[prefix] = tuple(
            _verify_bound_file(
                root,
                relative,
                digest,
                name=f"qualification {prefix} receipt",
            )
            for relative, digest in zip(paths, hashes)
        )
    return verified["reset"], verified["pairing"]


def _verify_task_inventory(task_ids: tuple[str, ...]) -> None:
    expected = tuple(task.task_id for task in load_registry().tasks)
    if task_ids != expected:
        raise CleanCampaignError(
            "qualification task inventory/order must match the frozen registry"
        )


def _verify_v2(
    root: Path,
    path: Path,
    document: Mapping[str, object],
    raw: bytes,
) -> VerifiedAllTaskQualification:
    action_spec, campaign_id, repetitions, tasks = _verify_common(
        document,
        expected_fields=_V2_FIELDS,
        semantic_version=QUALIFICATION_V2_SEMANTIC_VERSION,
    )
    source_binding = RuntimeSourceBinding.from_dict(document["source_binding"])
    campaign_manifest_sha256 = _verify_campaign_manifest_binding(
        root,
        document["campaign_manifest_relpath"],
        document["campaign_manifest_sha256"],
    )
    _verify_installation_receipts(
        root, document["installation_receipts"], source_binding
    )
    task_results = tuple(
        _verify_task_v2(root, item, repetitions, source_binding) for item in tasks
    )
    task_ids = tuple(item[0] for item in task_results)
    _verify_task_inventory(task_ids)
    return VerifiedAllTaskQualification(
        path=path,
        sha256=_sha256_bytes(raw),
        campaign_id=campaign_id,
        action_spec=action_spec,
        repetitions=repetitions,
        tasks=task_ids,
        semantic_version=QUALIFICATION_V2_SEMANTIC_VERSION,
        source_binding=source_binding,
        task_source_bindings=(source_binding,) * len(task_ids),
        campaign_manifest_sha256=campaign_manifest_sha256,
        observation_parity_sha256s=tuple(item[1] for item in task_results),
    )


def _verify_v3(
    root: Path,
    path: Path,
    document: Mapping[str, object],
    raw: bytes,
) -> VerifiedAllTaskQualification:
    action_spec, campaign_id, repetitions, tasks = _verify_common(
        document,
        expected_fields=_V3_FIELDS,
        semantic_version=QUALIFICATION_V3_SEMANTIC_VERSION,
    )
    campaign_manifest_sha256 = _verify_campaign_manifest_binding(
        root,
        document["campaign_manifest_relpath"],
        document["campaign_manifest_sha256"],
    )
    task_results = tuple(_verify_task_v3(root, item, repetitions) for item in tasks)
    task_ids = tuple(item[0] for item in task_results)
    task_sources = tuple(item[2] for item in task_results)
    _verify_task_inventory(task_ids)
    _verify_shared_runtime_sources(task_sources)
    _verify_installation_receipts(
        root,
        document["installation_receipts"],
        task_sources[0],
    )
    return VerifiedAllTaskQualification(
        path=path,
        sha256=_sha256_bytes(raw),
        campaign_id=campaign_id,
        action_spec=action_spec,
        repetitions=repetitions,
        tasks=task_ids,
        semantic_version=QUALIFICATION_V3_SEMANTIC_VERSION,
        source_binding=None,
        task_source_bindings=task_sources,
        campaign_manifest_sha256=campaign_manifest_sha256,
        observation_parity_sha256s=tuple(item[1] for item in task_results),
    )


def _verify_shared_runtime_sources(
    sources: tuple[RuntimeSourceBinding, ...],
) -> None:
    if not sources:
        raise CleanCampaignError("qualification task source inventory is empty")
    expected = tuple(
        getattr(sources[0], field) for field in _SHARED_RUNTIME_SOURCE_FIELDS
    )
    for source in sources[1:]:
        identity = tuple(
            getattr(source, field) for field in _SHARED_RUNTIME_SOURCE_FIELDS
        )
        if identity != expected:
            raise CleanCampaignError(
                "qualification task source bindings disagree on shared runtime"
            )


def _verify_installation_receipts(
    root: Path, value: object, source: RuntimeSourceBinding
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CleanCampaignError(
            "qualification installation receipts must be a sequence"
        )
    receipts = tuple(ObservationEvidenceBinding.from_dict(item) for item in value)
    if tuple(item.kind for item in receipts) != _INSTALLATION_KINDS:
        raise CleanCampaignError(
            "qualification installation receipt inventory mismatch"
        )
    documents: dict[str, Mapping[str, object]] = {}
    paths: dict[str, Path] = {}
    for item in receipts:
        if item.content_sha256 is not None:
            raise CleanCampaignError("installation receipt content hash must be null")
        path = _verify_bound_file(root, item.relpath, item.sha256, name=f"{item.kind}")
        paths[item.kind] = path
        documents[item.kind] = _load_receipt(path, item.kind)
    _expect_receipt(
        documents["isaac_install_receipt"],
        {
            "component": "robotactile_isaac",
            "source_manifest_sha256": source.robotactile_source_manifest_sha256,
            "status": "installed",
            "wheel_sha256": source.robotactile_wheel_sha256,
        },
        "Isaac install receipt",
    )
    _expect_receipt(
        documents["n0_client_install_receipt"],
        {
            "component": "n0_twam_robotactile_client",
            "source_manifest_sha256": source.robotactile_source_manifest_sha256,
            "status": "installed",
            "wheel_sha256": source.robotactile_wheel_sha256,
        },
        "N0 client install receipt",
    )
    _expect_receipt(
        documents["n0_runtime_receipt"],
        {
            "component": "n0_twam_runtime",
            "source_commit": source.n0_source_commit,
            "status": "installed",
        },
        "N0 runtime receipt",
    )
    _expect_receipt(
        documents["tacex_install_receipt"],
        {
            "component": "tacex",
            "status": "installed",
            "univtac_source_commit": source.univtac_source_commit,
        },
        "TacEx install receipt",
    )
    if documents["n0_client_install_receipt"].get(
        "runtime_receipt_sha256"
    ) != _sha256_file(paths["n0_runtime_receipt"]):
        raise CleanCampaignError("N0 client/runtime installation receipt mismatch")


def _verify_task_v2(
    root: Path,
    value: object,
    repetitions: int,
    source: RuntimeSourceBinding,
) -> tuple[str, str]:
    return _verify_source_bound_task(
        root,
        value,
        repetitions,
        source,
        expected_fields=_TASK_V2_FIELDS,
    )


def _verify_task_v3(
    root: Path,
    value: object,
    repetitions: int,
) -> tuple[str, str, RuntimeSourceBinding]:
    if not isinstance(value, Mapping) or set(value) != _TASK_V3_FIELDS:
        raise CleanCampaignError("source-bound qualification task fields mismatch")
    source = RuntimeSourceBinding.from_dict(value["source_binding"])
    task_id, parity_sha256 = _verify_source_bound_task(
        root,
        value,
        repetitions,
        source,
        expected_fields=_TASK_V3_FIELDS,
    )
    return task_id, parity_sha256, source


def _verify_source_bound_task(
    root: Path,
    value: object,
    repetitions: int,
    source: RuntimeSourceBinding,
    *,
    expected_fields: frozenset[str],
) -> tuple[str, str]:
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise CleanCampaignError("source-bound qualification task fields mismatch")
    task = cast(Mapping[str, object], value)
    task_id = task.get("task_id")
    initial_seed = task.get("initial_seed")
    exogenous_seed = task.get("exogenous_seed")
    if not isinstance(task_id, str) or not task_id:
        raise CleanCampaignError("qualification task ID is invalid")
    if type(initial_seed) is not int or type(exogenous_seed) is not int:
        raise CleanCampaignError("qualification task seeds must be integers")
    registry_task = load_registry().task(task_id)
    import_path = _verify_bound_file(
        root,
        task["import_receipt_relpath"],
        task["import_receipt_sha256"],
        name="qualification import receipt",
    )
    _expect_receipt(
        _load_receipt(import_path, "qualification import receipt"),
        {
            "component": "univtac_task_import",
            "status": "qualified",
            "task_id": task_id,
            "task_source_sha256": registry_task.task_source_sha256,
            "univtac_source_commit": source.univtac_source_commit,
        },
        "qualification import receipt",
    )
    reset_paths, pairing_paths = _verify_repeated_files(root, task, repetitions)
    for reset_path, pairing_path in zip(reset_paths, pairing_paths):
        reset = _load_receipt(reset_path, "qualification reset receipt")
        pairing = _load_receipt(pairing_path, "qualification pairing receipt")
        common = {
            "action_spec": "ee8_absolute",
            "exogenous_seed": str(exogenous_seed),
            "initial_seed": str(initial_seed),
            "runtime_source_manifest_sha256": (
                source.robotactile_source_manifest_sha256
            ),
            "status": "qualified",
            "task_id": task_id,
            "task_source_sha256": registry_task.task_source_sha256,
            "univtac_source_commit": source.univtac_source_commit,
        }
        _expect_receipt(
            reset,
            {**common, "component": "univtac_task_reset"},
            "qualification reset receipt",
        )
        _expect_receipt(
            pairing,
            {
                **common,
                "all_exact": "true",
                "component": "univtac_task_pairing",
            },
            "qualification pairing receipt",
        )
        _sha256(
            pairing.get("paired_reset_receipt_sha256"),
            "qualification in-process reset receipt SHA256",
        )
        if pairing.get("canonical_state_sha256") != pairing.get("replay_state_sha256"):
            raise CleanCampaignError("qualification reset/pairing semantic mismatch")
    parity_sha256 = _sha256(
        task["observation_parity_sha256"], "observation parity SHA256"
    )
    parity_path = _verify_bound_file(
        root,
        task["observation_parity_relpath"],
        parity_sha256,
        name="observation parity artifact",
    )
    parity = load_observation_parity_artifact(root, parity_path)
    if (
        not parity.passed
        or parity.task_id != task_id
        or parity.source_binding != source
    ):
        raise CleanCampaignError("observation parity does not qualify this task")
    return task_id, parity_sha256


__all__ = [
    "QUALIFICATION_EVIDENCE_LEVEL",
    "QUALIFICATION_V1_SEMANTIC_VERSION",
    "QUALIFICATION_V2_SEMANTIC_VERSION",
    "QUALIFICATION_V3_SEMANTIC_VERSION",
    "SOURCE_BOUND_QUALIFICATION_EVIDENCE_LEVEL",
    "TASK_SOURCE_BOUND_QUALIFICATION_EVIDENCE_LEVEL",
    "VerifiedAllTaskQualification",
    "verify_all_task_qualification",
]
