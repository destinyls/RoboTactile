"""No-clobber execution and resume for one official ACT fault task."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional, Tuple, cast

from robotactile_benchmark.backends.univtac_pairing import UniVTACPairingError
from robotactile_benchmark.backends.univtac_reset_trajectory import (
    UniVTACPreMoveTrajectory,
)
from robotactile_benchmark.backends.univtac_reset_witness import (
    UniVTACResetReference,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.closed_loop.runner import run_closed_loop_trial_with_evidence
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import LiveUniVTACRunRequest
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.live_univtac import LiveUniVTACExecutionResult
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.official_act import (
    OfficialACTPairedLiveResult,
    build_official_act_live_binding,
    execute_official_act_paired_live_runs,
)
from robotactile_benchmark.execution.paired_live_univtac import (
    PairedBackendSession,
    default_paired_backend_session_factory,
)
from robotactile_benchmark.execution.sequential_policy_pool import (
    SequentialPolicyPool,
)
from robotactile_benchmark.integrations.runtime_config import (
    ACTRuntimeArtifacts,
    resolve_act_runtime_artifacts,
)
from robotactile_benchmark.policies.univtac_official_act_loading import (
    load_official_univtac_act_policy,
)
from robotactile_benchmark.trials import Condition, TerminalStatus

from .contracts import (
    ACTFaultCampaignCellSpec,
    ACTFaultCampaignError,
    ACTFaultCellDisposition,
)
from .io import LoadedACTFaultCampaign, load_act_fault_campaign_bundle
from .reset_reference import (
    act_reset_reference_relpath,
    load_act_reset_reference,
)
from .reset_trajectory import (
    act_reset_trajectory_relpath,
    build_act_trajectory_replay_reference,
    load_act_reset_trajectory,
)

ACT_FAULT_TASK_RUN_SEMANTIC_VERSION = "1.0"
ACT_FAULT_TASK_RUN_EVIDENCE_LEVEL = "unqualified_act_fault_task_execution_v1"
FRESH_RESUME_MODE = "fresh_official_paired_snapshot_v1"
PARTIAL_RESUME_MODE = "missing_only_from_reconstructed_pair_snapshot_v1"
COMPLETE_RESUME_MODE = "complete_existing_artifacts_no_execution_v1"
_ABSENCE = frozenset({"A1_stream_absence", "A2_frame_erasure"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ACTCleanBaselineUnqualifiedError(ACTFaultCampaignError):
    """Stop a fault campaign whose paired Clean baseline did not succeed."""

    code = "clean_baseline_unqualified"

    def __init__(
        self,
        *,
        task: str,
        pair_key: str,
        result: ClosedLoopTrialResult,
    ) -> None:
        self.task = task
        self.pair_key = pair_key
        self.result_sha256 = result.sha256
        self.terminal_status = result.terminal_status
        self.validation_passed = result.validation_passed
        self.score_success = result.score_success
        super().__init__(
            "Clean baseline is unqualified; refusing to execute Faulted cells "
            f"(task={task}, terminal_status={result.terminal_status.value}, "
            f"score_success={result.score_success})"
        )


@dataclass(frozen=True)
class ACTFaultTaskRunReceipt:
    """Complete 13-cell inventory; false scores remain model outcomes."""

    campaign_id: str
    campaign_manifest_sha256: str
    generation_receipt_file_sha256: str
    task: str
    pair_key: str
    request_file_sha256s: Tuple[str, ...]
    artifact_root_sha256s: Tuple[str, ...]
    result_sha256s: Tuple[str, ...]
    score_successes: Tuple[Optional[bool], ...]
    capture_profile: str
    newly_executed_count: int
    reused_artifact_count: int
    resume_mode: str
    execution_group_content_sha256: Optional[str]
    reset_receipt_sha256: Optional[str]
    evidence_level: str = ACT_FAULT_TASK_RUN_EVIDENCE_LEVEL
    simulator_qualification_claimed: bool = False
    semantic_version: str = ACT_FAULT_TASK_RUN_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        hashes = (
            self.campaign_manifest_sha256,
            self.generation_receipt_file_sha256,
            self.pair_key,
            *self.request_file_sha256s,
            *self.artifact_root_sha256s,
            *self.result_sha256s,
        )
        if (
            not self.campaign_id
            or not self.task
            or any(_SHA256.fullmatch(x) is None for x in hashes)
        ):
            raise ACTFaultCampaignError("task receipt identity is invalid")
        lengths = {
            len(self.request_file_sha256s),
            len(self.artifact_root_sha256s),
            len(self.result_sha256s),
            len(self.score_successes),
        }
        if lengths != {13}:
            raise ACTFaultCampaignError("task receipt must cover 13 live cells")
        if any(
            value is not None and type(value) is not bool
            for value in self.score_successes
        ):
            raise ACTFaultCampaignError("task receipt score value is invalid")
        LiveCaptureProfile(self.capture_profile)
        if (
            type(self.newly_executed_count) is not int
            or type(self.reused_artifact_count) is not int
            or not 0 <= self.newly_executed_count <= 13
            or not 0 <= self.reused_artifact_count <= 13
            or self.newly_executed_count + self.reused_artifact_count != 13
        ):
            raise ACTFaultCampaignError("task receipt execution counts mismatch")
        modes = {FRESH_RESUME_MODE, PARTIAL_RESUME_MODE, COMPLETE_RESUME_MODE}
        group_hashes = (self.execution_group_content_sha256, self.reset_receipt_sha256)
        if self.resume_mode not in modes or (group_hashes[0] is None) != (
            group_hashes[1] is None
        ):
            raise ACTFaultCampaignError("task receipt resume contract mismatch")
        if any(
            value is not None and _SHA256.fullmatch(value) is None
            for value in group_hashes
        ):
            raise ACTFaultCampaignError("task receipt execution hash is invalid")
        no_execution = self.newly_executed_count == 0
        if no_execution != (
            self.resume_mode == COMPLETE_RESUME_MODE and group_hashes[0] is None
        ):
            raise ACTFaultCampaignError("task receipt resume semantics mismatch")
        if self.resume_mode == FRESH_RESUME_MODE and self.newly_executed_count != 13:
            raise ACTFaultCampaignError("fresh execution must cover all live cells")
        if (
            self.resume_mode == PARTIAL_RESUME_MODE
            and not 0 < self.newly_executed_count < 13
        ):
            raise ACTFaultCampaignError("partial resume must execute a strict subset")
        if (
            self.evidence_level != ACT_FAULT_TASK_RUN_EVIDENCE_LEVEL
            or self.simulator_qualification_claimed
            or self.semantic_version != ACT_FAULT_TASK_RUN_SEMANTIC_VERSION
        ):
            raise ACTFaultCampaignError("task receipt evidence contract mismatch")

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ACTFaultTaskRunReceipt:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ACTFaultCampaignError("task receipt fields mismatch")
        data = dict(value)
        fields = (
            "request_file_sha256s",
            "artifact_root_sha256s",
            "result_sha256s",
            "score_successes",
        )
        for name in fields:
            if not isinstance(data[name], list):
                raise ACTFaultCampaignError(f"{name} must be a list")
            data[name] = tuple(data[name])
        return cls(**cast(Any, data))


@dataclass(frozen=True)
class _ExecutionGroup:
    content_sha256: str
    reset_sha256: str


def select_live_task_cells(
    cells: Tuple[ACTFaultCampaignCellSpec, ...],
    *,
    task: Optional[str],
    pair_key: Optional[str],
) -> Tuple[ACTFaultCampaignCellSpec, ...]:
    """Select Clean first and the 12 live Faulted cells; never select A1/A2."""

    task_id = _select(sorted({cell.task for cell in cells}), task, "task")
    task_cells = tuple(cell for cell in cells if cell.task == task_id)
    pair = _select(sorted({cell.pair_key for cell in task_cells}), pair_key, "pair_key")
    live = tuple(
        cell
        for cell in task_cells
        if cell.pair_key == pair
        and cell.disposition is ACTFaultCellDisposition.LIVE_REQUEST
    )
    if any(cell.operator_id in _ABSENCE for cell in live):
        raise ACTFaultCampaignError("A1/A2 cannot be live ACT requests")
    clean = tuple(cell for cell in live if cell.condition is Condition.CLEAN)
    faults = tuple(cell for cell in live if cell.condition is Condition.FAULTED)
    if len(clean) != 1 or len(faults) != 12 or len(live) != 13:
        raise ACTFaultCampaignError("ACT task requires Clean plus 12 live faults")
    return (clean[0], *sorted(faults, key=_fault_order))


def run_act_fault_task(
    campaign_root: Path,
    *,
    integration_config: Path,
    task: Optional[str] = None,
    pair_key: Optional[str] = None,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
    receipt_dir: Optional[Path] = None,
) -> ACTFaultTaskRunReceipt:
    """Execute only missing artifacts using one simulator session per invocation."""

    bundle = load_act_fault_campaign_bundle(campaign_root)
    cells = select_live_task_cells(bundle.manifest.cells, task=task, pair_key=pair_key)
    capture = LiveCaptureProfile(capture_profile)
    requests = tuple(
        load_live_univtac_request(bundle.root / _request_path(cell)) for cell in cells
    )
    runtime = resolve_act_runtime_artifacts(integration_config)
    if runtime.manifest.task_id != cells[0].task:
        raise ACTFaultCampaignError("ACT runtime config task mismatch")
    existing = tuple(
        _existing(bundle, cell, request, capture)
        for cell, request in zip(cells, requests)
    )
    if existing[0] is not None:
        _require_clean_success(
            existing[0].evidence.result,
            task=cells[0].task,
            pair_key=cells[0].pair_key,
        )
    missing = tuple(
        index for index, artifact in enumerate(existing) if artifact is None
    )
    default_receipt = bundle.root / "executions" / cells[0].task / cells[0].pair_key
    receipt_root = (
        default_receipt
        if receipt_dir is None
        else Path(receipt_dir).expanduser().absolute()
    )
    receipt_path = receipt_root / "task_run_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        if missing:
            raise FileExistsError("task receipt exists while artifacts are incomplete")
        receipt = _load_receipt(receipt_path)
        _validate_receipt(receipt, bundle, cells, _complete(existing))
        return receipt
    group: Optional[_ExecutionGroup]
    if len(missing) == 13:
        reset_reference = _load_task_reset_reference(
            bundle,
            cells[0],
            requests[0],
        )
        reset_trajectory = _load_task_reset_trajectory(
            bundle,
            cells[0],
            requests[0],
            reset_reference,
        )
        replay_reference = build_act_trajectory_replay_reference(
            reset_reference,
            reset_trajectory,
        )

        def publish_fresh_before_close(result: OfficialACTPairedLiveResult) -> None:
            group = _ExecutionGroup(
                result.paired.group_content_sha256,
                result.paired.reset_receipt.sha256,
            )
            receipt = _receipt(
                bundle,
                cells,
                result.artifacts,
                capture,
                len(missing),
                FRESH_RESUME_MODE,
                group,
            )
            _publish_receipt(receipt_path, receipt)

        result = execute_official_act_paired_live_runs(
            requests,
            artifact_root=runtime.artifact_root,
            stats_sha256=runtime.stats_sha256,
            encoder_sha256=runtime.encoder_sha256,
            session_factory=partial(
                default_paired_backend_session_factory,
                reset_reference=replay_reference,
                reset_trajectory=reset_trajectory,
            ),
            capture_profile=capture,
            pre_close_publisher=publish_fresh_before_close,
            post_execution_gate=_campaign_execution_gate,
            require_shared_runtime_dir=True,
        )
        group = _ExecutionGroup(
            result.paired.group_content_sha256, result.paired.reset_receipt.sha256
        )
        mode = FRESH_RESUME_MODE
    elif missing:
        reset_reference = _load_task_reset_reference(
            bundle,
            cells[0],
            requests[0],
        )
        reset_trajectory = _load_task_reset_trajectory(
            bundle,
            cells[0],
            requests[0],
            reset_reference,
        )
        replay_reference = build_act_trajectory_replay_reference(
            reset_reference,
            reset_trajectory,
        )

        def publish_partial_before_close(group: _ExecutionGroup) -> None:
            refreshed = tuple(
                _existing(bundle, cell, request, capture)
                for cell, request in zip(cells, requests)
            )
            receipt = _receipt(
                bundle,
                cells,
                _complete(refreshed),
                capture,
                len(missing),
                PARTIAL_RESUME_MODE,
                group,
            )
            _publish_receipt(receipt_path, receipt)

        group = _execute_missing(
            tuple(requests[index] for index in missing),
            runtime,
            capture,
            replay_reference,
            reset_trajectory,
            pre_close_publisher=publish_partial_before_close,
        )
        mode = PARTIAL_RESUME_MODE
    else:
        group, mode = None, COMPLETE_RESUME_MODE
    final = tuple(
        _existing(bundle, cell, request, capture)
        for cell, request in zip(cells, requests)
    )
    receipt = _receipt(
        bundle, cells, _complete(final), capture, len(missing), mode, group
    )
    _publish_receipt(receipt_path, receipt)
    return receipt


def _load_task_reset_reference(
    bundle: LoadedACTFaultCampaign,
    clean_cell: ACTFaultCampaignCellSpec,
    clean_request: LiveUniVTACRunRequest,
) -> UniVTACResetReference:
    path = bundle.root / act_reset_reference_relpath(
        clean_cell.task,
        clean_cell.pair_key,
    )
    if not path.is_file() or path.is_symlink():
        raise ACTFaultCampaignError(
            "ACT execution requires a successful Clean-derived reset reference"
        )
    reference = load_act_reset_reference(path)
    loaded = load_live_univtac_run(clean_request)
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
            "ACT reset reference identity differs from selected Clean"
        )
    return reference


def _load_task_reset_trajectory(
    bundle: LoadedACTFaultCampaign,
    clean_cell: ACTFaultCampaignCellSpec,
    clean_request: LiveUniVTACRunRequest,
    reset_reference: UniVTACResetReference,
) -> UniVTACPreMoveTrajectory:
    path = bundle.root / act_reset_trajectory_relpath(
        clean_cell.task,
        clean_cell.pair_key,
    )
    if not path.is_file() or path.is_symlink():
        raise ACTFaultCampaignError(
            "ACT execution requires a qualified dense reset trajectory"
        )
    trajectory = load_act_reset_trajectory(path)
    loaded = load_live_univtac_run(clean_request)
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
        or trajectory.reset_reference_sha256 != reset_reference.sha256
        or trajectory.upstream_commit != loaded.backend_config.upstream_commit
        or trajectory.task_source_sha256
        != loaded.backend_config.task.task_source_sha256
    ):
        raise ACTFaultCampaignError(
            "ACT reset trajectory identity differs from selected Clean"
        )
    return trajectory


def _execute_missing(
    requests: Sequence[LiveUniVTACRunRequest],
    runtime: ACTRuntimeArtifacts,
    capture: LiveCaptureProfile,
    reset_reference: UniVTACResetReference,
    reset_trajectory: UniVTACPreMoveTrajectory,
    *,
    pre_close_publisher: Optional[Callable[[_ExecutionGroup], None]] = None,
) -> _ExecutionGroup:
    """Reconstruct the pair snapshot and execute only missing cells."""

    loaded = tuple(load_live_univtac_run(request) for request in requests)
    _validate_missing_group(loaded)
    session = default_paired_backend_session_factory(
        loaded[0],
        reset_reference=reset_reference,
        reset_trajectory=reset_trajectory,
    )
    indices: list[int] = []
    results: list[str] = []
    try:
        with SequentialPolicyPool() as policy_pool:
            for item in loaded:
                before = _witness_count(session)
                backend = session.new_backend()
                binding = build_official_act_live_binding(
                    item.request,
                    artifact_root=runtime.artifact_root,
                    stats_sha256=runtime.stats_sha256,
                    encoder_sha256=runtime.encoder_sha256,
                )
                try:
                    policy = policy_pool.acquire(
                        (binding.load_request, item.policy_identity),
                        item.policy_identity,
                        partial(
                            load_official_univtac_act_policy,
                            item.policy_identity,
                            binding.load_request,
                        ),
                    )
                except Exception:
                    backend.close()
                    raise
                evidence = run_closed_loop_trial_with_evidence(
                    item.trial,
                    item.run_spec,
                    backend,
                    policy,
                    fault_manifest=item.fault_manifest,
                    rest_references=item.rest_references,
                    initial_state_policy=item.request.initial_state_policy,
                )
                _require_reset_qualified(
                    evidence.result,
                    task=item.trial.task,
                    pair_key=item.trial.pair_key,
                )
                after = _witness_count(session)
                if after != before + 1:
                    raise UniVTACPairingError(
                        "paired_reset_incomplete",
                        "resume did not produce one reset witness",
                    )
                witness = session.reset_receipt.witnesses[before]
                if (
                    not witness.exact_match
                    or evidence.result.initial_state_sha256
                    != witness.simulator_state_sha256
                ):
                    raise UniVTACPairingError(
                        "reset_equivalence_mismatch",
                        "resume reset differs from canonical snapshot",
                    )
                output = item.request.output_dir
                if output is None:
                    raise ACTFaultCampaignError("ACT request has no output_dir")
                write_live_univtac_artifact(
                    output,
                    item,
                    evidence,
                    capture_profile=capture,
                )
                artifact = load_live_univtac_artifact(output)
                if (
                    artifact.run_content_sha256 != item.content_sha256
                    or artifact.root_receipt.result_sha256 != evidence.result.sha256
                    or artifact.capture_profile is not capture
                ):
                    raise RuntimeError("resumed ACT artifact cross-link mismatch")
                if item.trial.condition is Condition.CLEAN:
                    _require_clean_success(
                        evidence.result,
                        task=item.trial.task,
                        pair_key=item.trial.pair_key,
                    )
                indices.append(before)
                results.append(artifact.root_receipt.result_sha256)
        reset = session.reset_receipt
        if not reset.all_exact:
            raise UniVTACPairingError(
                "reset_equivalence_mismatch", "resume reset receipt is not exact"
            )
        group = canonical_hash(
            {
                "namespace": PARTIAL_RESUME_MODE,
                "run_content_sha256": tuple(item.content_sha256 for item in loaded),
                "result_sha256": tuple(results),
                "reset_receipt_sha256": reset.sha256,
                "witness_indices": tuple(indices),
            }
        )
        execution_group = _ExecutionGroup(group, reset.sha256)
        if pre_close_publisher is not None:
            pre_close_publisher(execution_group)
        return execution_group
    finally:
        session.close()


def _existing(
    bundle: LoadedACTFaultCampaign,
    cell: ACTFaultCampaignCellSpec,
    request: LiveUniVTACRunRequest,
    capture: LiveCaptureProfile,
) -> Optional[LoadedLiveUniVTACArtifact]:
    output = request.output_dir
    expected = (bundle.root / _artifact_path(cell)).absolute()
    if output is None or Path(output).resolve(strict=False) != expected.resolve(
        strict=False
    ):
        raise ACTFaultCampaignError("request output does not match campaign artifact")
    if expected.is_symlink() or (expected.exists() and not expected.is_dir()):
        raise FileExistsError("ACT artifact target is not a real directory")
    if not expected.exists() or not any(expected.iterdir()):
        return None
    artifact = load_live_univtac_artifact(expected)
    loaded = load_live_univtac_run(request)
    if (
        artifact.run_content_sha256 != loaded.content_sha256
        or artifact.trial != loaded.trial
        or artifact.capture_profile is not capture
    ):
        raise ACTFaultCampaignError("existing ACT artifact identity mismatch")
    return artifact


def _receipt(
    bundle: LoadedACTFaultCampaign,
    cells: Tuple[ACTFaultCampaignCellSpec, ...],
    artifacts: Tuple[LoadedLiveUniVTACArtifact, ...],
    capture: LiveCaptureProfile,
    executed: int,
    mode: str,
    group: Optional[_ExecutionGroup],
) -> ACTFaultTaskRunReceipt:
    return ACTFaultTaskRunReceipt(
        campaign_id=bundle.manifest.campaign_id,
        campaign_manifest_sha256=bundle.manifest.sha256,
        generation_receipt_file_sha256=bundle.receipt_file_sha256,
        task=cells[0].task,
        pair_key=cells[0].pair_key,
        request_file_sha256s=tuple(_request_hash(cell) for cell in cells),
        artifact_root_sha256s=tuple(a.external_root_sha256 for a in artifacts),
        result_sha256s=tuple(a.root_receipt.result_sha256 for a in artifacts),
        score_successes=tuple(a.evidence.result.score_success for a in artifacts),
        capture_profile=capture.value,
        newly_executed_count=executed,
        reused_artifact_count=13 - executed,
        resume_mode=mode,
        execution_group_content_sha256=None if group is None else group.content_sha256,
        reset_receipt_sha256=None if group is None else group.reset_sha256,
    )


def _validate_receipt(
    receipt: ACTFaultTaskRunReceipt,
    bundle: LoadedACTFaultCampaign,
    cells: Tuple[ACTFaultCampaignCellSpec, ...],
    artifacts: Tuple[LoadedLiveUniVTACArtifact, ...],
) -> None:
    group = (
        None
        if receipt.execution_group_content_sha256 is None
        else _ExecutionGroup(
            receipt.execution_group_content_sha256,
            cast(str, receipt.reset_receipt_sha256),
        )
    )
    expected = _receipt(
        bundle,
        cells,
        artifacts,
        LiveCaptureProfile(receipt.capture_profile),
        receipt.newly_executed_count,
        receipt.resume_mode,
        group,
    )
    if expected != receipt:
        raise ACTFaultCampaignError("stored task receipt identity mismatch")


def _publish_receipt(path: Path, receipt: ACTFaultTaskRunReceipt) -> None:
    target = Path(path).absolute()
    if target.is_symlink():
        raise FileExistsError("task receipt cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(receipt.to_dict())
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different task receipt")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    except FileExistsError:
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError("concurrent task receipt disagrees") from None
    finally:
        temporary.unlink(missing_ok=True)


def _load_receipt(path: Path) -> ACTFaultTaskRunReceipt:
    raw = Path(path).read_bytes()
    receipt = ACTFaultTaskRunReceipt.from_dict(
        strict_json_bytes(raw, "ACT fault task receipt")
    )
    if canonical_json_bytes(receipt.to_dict()) != raw:
        raise ACTFaultCampaignError("ACT task receipt is not canonical")
    return receipt


def _validate_missing_group(items: Tuple[LoadedLiveUniVTACRun, ...]) -> None:
    if not items:
        raise ACTFaultCampaignError("partial resume has no missing requests")
    first = items[0]
    if any(
        item.trial.task != first.trial.task
        or item.trial.pair_key != first.trial.pair_key
        or item.backend_config != first.backend_config
        for item in items
    ):
        raise ACTFaultCampaignError("partial ACT resume identity mismatch")
    if any(item.trial.condition is Condition.NO_TOUCH for item in items):
        raise ACTFaultCampaignError("fault resume cannot execute No-touch")
    runtime_dir = first.request.runtime_dir.resolve(strict=False)
    if any(
        item.request.runtime_dir.resolve(strict=False) != runtime_dir for item in items
    ):
        raise ACTFaultCampaignError("partial ACT resume runtime_dir mismatch")


def _campaign_execution_gate(
    index: int,
    execution: LiveUniVTACExecutionResult,
) -> None:
    _require_reset_qualified(
        execution.evidence.result,
        task=execution.loaded.trial.task,
        pair_key=execution.loaded.trial.pair_key,
    )
    if index != 0:
        return
    if execution.loaded.trial.condition is not Condition.CLEAN:
        raise ACTFaultCampaignError("first ACT campaign execution is not Clean")
    _require_clean_success(
        execution.evidence.result,
        task=execution.loaded.trial.task,
        pair_key=execution.loaded.trial.pair_key,
    )


def _require_reset_qualified(
    result: ClosedLoopTrialResult,
    *,
    task: str,
    pair_key: str,
) -> None:
    if result.failure_code != "qualification_reset_not_viable":
        return
    raise ACTFaultCampaignError(
        "ACT reset reference mismatch before policy inference: "
        f"task={task} pair_key={pair_key}"
    )


def _require_clean_success(
    result: ClosedLoopTrialResult,
    *,
    task: str,
    pair_key: str,
) -> None:
    if (
        result.terminal_status is TerminalStatus.SUCCESS
        and result.score_success is True
        and result.validation_passed is True
    ):
        return
    raise ACTCleanBaselineUnqualifiedError(
        task=task,
        pair_key=pair_key,
        result=result,
    )


def _complete(
    values: tuple[Optional[LoadedLiveUniVTACArtifact], ...],
) -> Tuple[LoadedLiveUniVTACArtifact, ...]:
    if any(value is None for value in values):
        raise RuntimeError("ACT fault task execution left a missing artifact")
    return tuple(cast(LoadedLiveUniVTACArtifact, value) for value in values)


def _select(values: list[str], selected: Optional[str], label: str) -> str:
    if selected is None and len(values) != 1:
        raise ACTFaultCampaignError(f"multi-{label} campaign requires selection")
    result = values[0] if selected is None else selected
    if result not in values:
        raise ACTFaultCampaignError(f"selected {label} is absent")
    return result


def _fault_order(cell: ACTFaultCampaignCellSpec) -> tuple[str, int]:
    if cell.operator_id is None or cell.severity_level is None:
        raise ACTFaultCampaignError("Faulted cell lost operator identity")
    return cell.operator_id, cell.severity_level


def _request_path(cell: ACTFaultCampaignCellSpec) -> str:
    if cell.request_relpath is None:
        raise ACTFaultCampaignError("live cell lost request path")
    return cast(str, cell.request_relpath)


def _artifact_path(cell: ACTFaultCampaignCellSpec) -> str:
    if cell.artifact_relpath is None:
        raise ACTFaultCampaignError("live cell lost artifact path")
    return cast(str, cell.artifact_relpath)


def _request_hash(cell: ACTFaultCampaignCellSpec) -> str:
    if cell.request_file_sha256 is None:
        raise ACTFaultCampaignError("live cell lost request hash")
    return cast(str, cell.request_file_sha256)


def _witness_count(session: PairedBackendSession) -> int:
    try:
        return len(session.reset_receipt.witnesses)
    except UniVTACPairingError as error:
        if error.code == "canonical_reset_missing":
            return 0
        raise


__all__ = [
    "ACT_FAULT_TASK_RUN_EVIDENCE_LEVEL",
    "ACT_FAULT_TASK_RUN_SEMANTIC_VERSION",
    "ACTFaultTaskRunReceipt",
    "COMPLETE_RESUME_MODE",
    "FRESH_RESUME_MODE",
    "PARTIAL_RESUME_MODE",
    "run_act_fault_task",
    "select_live_task_cells",
]
