"""Exact in-process reset pairing for live UniVTAC benchmark conditions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACBackendConfig,
    UniVTACContractError,
    validate_packaged_univtac_config,
)
from robotactile_benchmark.backends.univtac_conversion import (
    ConvertedUniVTACObservation,
)
from robotactile_benchmark.backends.univtac_isaac import (
    UniVTACIsaacBackend,
    UniVTACTaskRuntime,
)
from robotactile_benchmark.backends.univtac_reset_trajectory import (
    UniVTACPreMoveTrajectory,
)
from robotactile_benchmark.backends.univtac_reset_witness import (
    UniVTACResetReference,
)
from robotactile_benchmark.closed_loop.contracts import PolicyEpisodeContext
from robotactile_benchmark.contracts import build_evaluation_record, canonical_hash

PAIRING_EVIDENCE_LEVEL = "in_process_snapshot_replay_equivalence_v1"
_ACT_TRAJECTORY_REPLAY_QPOS_ATOL = 1e-5


class UniVTACPairingError(UniVTACContractError):
    """Stable fail-closed error for paired reset or equivalence violations."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class UniVTACResetWitness:
    """Exact observable reset witness for one condition in a paired session."""

    ordinal: int
    reset_mode: str
    native_step_id: int
    simulator_state_sha256: str
    joint_reorder_witness_sha256: str
    canonical_joint9_sha256: str
    model_visible_qpos8_sha256: str
    snapshot_state_sha256: str
    exact_match: bool

    @property
    def equivalence_key(self) -> Tuple[object, ...]:
        return (
            self.native_step_id,
            self.simulator_state_sha256,
            self.joint_reorder_witness_sha256,
            self.canonical_joint9_sha256,
            self.model_visible_qpos8_sha256,
            self.snapshot_state_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "reset_mode": self.reset_mode,
            "native_step_id": self.native_step_id,
            "simulator_state_sha256": self.simulator_state_sha256,
            "joint_reorder_witness_sha256": self.joint_reorder_witness_sha256,
            "canonical_joint9_sha256": self.canonical_joint9_sha256,
            "model_visible_qpos8_sha256": self.model_visible_qpos8_sha256,
            "snapshot_state_sha256": self.snapshot_state_sha256,
            "exact_match": self.exact_match,
        }


@dataclass(frozen=True)
class UniVTACPairedResetReceipt:
    """Path-free receipt for one canonical reset and its replay witnesses."""

    session_id: str
    task_id: str
    initial_seed: int
    exogenous_seed: int
    config_sha256: str
    witnesses: Tuple[UniVTACResetWitness, ...]
    all_exact: bool
    evidence_level: str = PAIRING_EVIDENCE_LEVEL
    simulator_qualification_claimed: bool = False

    def __post_init__(self) -> None:
        if self.evidence_level != PAIRING_EVIDENCE_LEVEL:
            raise ValueError("paired reset evidence level mismatch")
        if self.simulator_qualification_claimed is not False:
            raise ValueError("paired reset receipt cannot claim qualification")
        if not self.witnesses:
            raise ValueError("paired reset receipt requires witnesses")
        expected = tuple(range(len(self.witnesses)))
        if tuple(item.ordinal for item in self.witnesses) != expected:
            raise ValueError("paired reset witness ordinals are not contiguous")
        if self.witnesses[0].reset_mode != "canonical_reset":
            raise ValueError("first paired reset witness must be canonical")
        expected_exact = all(item.exact_match for item in self.witnesses)
        if self.all_exact != expected_exact:
            raise ValueError("paired reset exactness summary mismatch")

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "initial_seed": self.initial_seed,
            "exogenous_seed": self.exogenous_seed,
            "config_sha256": self.config_sha256,
            "witnesses": [item.to_dict() for item in self.witnesses],
            "all_exact": self.all_exact,
            "evidence_level": self.evidence_level,
            "simulator_qualification_claimed": (self.simulator_qualification_claimed),
        }


class UniVTACPairedResetCoordinator:
    """Capture once, restore before every later condition, and compare exactly."""

    def __init__(
        self,
        config: UniVTACBackendConfig,
        runtime: UniVTACTaskRuntime,
    ) -> None:
        validate_packaged_univtac_config(config)
        config.validate_handshake(runtime.handshake)
        if (
            runtime.capture_state is None
            or runtime.restore_state is None
            or runtime.snapshot_state_sha256 is None
        ):
            raise UniVTACPairingError(
                "snapshot_callbacks_unavailable",
                "live runtime does not expose paired snapshot callbacks",
            )
        self._config = config
        self._runtime = runtime
        self._snapshot: Optional[object] = None
        self._canonical: Optional[UniVTACResetWitness] = None
        self._canonical_conversion: Optional[ConvertedUniVTACObservation] = None
        self._canonical_snapshot_state_sha256: Optional[str] = None
        self._context_key: Optional[Tuple[str, int, int, str, str]] = None
        self._witnesses: list[UniVTACResetWitness] = []
        self._prepared_mode: Optional[str] = None
        self._prepared_snapshot_state_sha256: Optional[str] = None

    @property
    def witness_count(self) -> int:
        return len(self._witnesses)

    @property
    def all_exact(self) -> bool:
        return bool(self._witnesses) and all(
            item.exact_match for item in self._witnesses
        )

    def prepare(self, context: PolicyEpisodeContext) -> str:
        """Perform the sole upstream reset or restore the canonical snapshot."""

        if self._prepared_mode is not None:
            raise UniVTACPairingError(
                "paired_reset_reentry", "paired reset prepare was not accepted"
            )
        key = self._context_invariant_key(context)
        if self._context_key is None:
            self._context_key = key
        elif key != self._context_key:
            raise UniVTACPairingError(
                "paired_context_mismatch",
                "paired conditions do not share task, seeds, prompt, and action spec",
            )
        if self._snapshot is None:
            self._runtime.task.reset(
                seed=context.initial_seed,
                instructions=[self._config.task.prompt],
            )
            mode = "canonical_reset"
        else:
            assert self._runtime.restore_state is not None
            self._runtime.restore_state(self._snapshot)
            assert self._runtime.snapshot_state_sha256 is not None
            restored_state_sha256 = self._runtime.snapshot_state_sha256()
            if restored_state_sha256 != self._canonical_snapshot_state_sha256:
                canonical_conversion = self._canonical_conversion
                if canonical_conversion is None:
                    raise UniVTACPairingError(
                        "canonical_reset_missing",
                        "paired session has no canonical initial conversion",
                    )
                self._witnesses.append(
                    self._witness(
                        canonical_conversion,
                        len(self._witnesses),
                        "snapshot_replay",
                        snapshot_state_sha256=restored_state_sha256,
                        exact_match=False,
                    )
                )
                raise UniVTACPairingError(
                    "reset_equivalence_mismatch",
                    "restored UniVTAC physical snapshot state does not match exactly",
                )
            self._prepared_snapshot_state_sha256 = restored_state_sha256
            mode = "snapshot_replay"
        self._prepared_mode = mode
        return mode

    def initial_conversion(
        self, context: PolicyEpisodeContext
    ) -> Optional[ConvertedUniVTACObservation]:
        """Reuse the canonical reset frame after exact physical-state restore."""

        if self._prepared_mode == "canonical_reset":
            return None
        if self._prepared_mode != "snapshot_replay":
            raise UniVTACPairingError(
                "paired_reset_not_prepared",
                "paired initial conversion lacks snapshot replay preparation",
            )
        canonical = self._canonical_conversion
        if canonical is None:
            raise UniVTACPairingError(
                "canonical_reset_missing",
                "paired session has no canonical initial conversion",
            )
        observation = replace(
            canonical.record.observation,
            episode_id=context.episode_id,
            task=context.task,
            seed=context.initial_seed,
            step_index=0,
        )
        record = build_evaluation_record(observation, canonical.record.provenance)
        return replace(canonical, record=record)

    def accept(self, converted: ConvertedUniVTACObservation) -> None:
        """Capture the canonical state or reject any replay-state mismatch."""

        mode = self._prepared_mode
        if mode is None:
            raise UniVTACPairingError(
                "paired_reset_not_prepared", "paired reset accept lacks prepare"
            )
        ordinal = len(self._witnesses)
        if self._canonical is None:
            assert self._runtime.capture_state is not None
            assert self._runtime.snapshot_state_sha256 is not None
            self._snapshot = self._runtime.capture_state()
            snapshot_state_sha256 = self._runtime.snapshot_state_sha256()
            candidate = self._witness(
                converted,
                ordinal,
                mode,
                snapshot_state_sha256=snapshot_state_sha256,
                exact_match=True,
            )
            self._canonical = candidate
            self._canonical_conversion = converted
            self._canonical_snapshot_state_sha256 = snapshot_state_sha256
            accepted = candidate
        else:
            prepared_snapshot_sha256 = self._prepared_snapshot_state_sha256
            if prepared_snapshot_sha256 is None:
                raise UniVTACPairingError(
                    "paired_reset_not_prepared",
                    "paired replay lacks a physical snapshot witness",
                )
            candidate = self._witness(
                converted,
                ordinal,
                mode,
                snapshot_state_sha256=prepared_snapshot_sha256,
                exact_match=True,
            )
            exact = candidate.equivalence_key == self._canonical.equivalence_key
            accepted = self._witness(
                converted,
                ordinal,
                mode,
                snapshot_state_sha256=prepared_snapshot_sha256,
                exact_match=exact,
            )
        self._witnesses.append(accepted)
        self._prepared_mode = None
        self._prepared_snapshot_state_sha256 = None
        if not accepted.exact_match:
            raise UniVTACPairingError(
                "reset_equivalence_mismatch",
                "restored UniVTAC state does not exactly match the canonical reset",
            )

    def receipt(self) -> UniVTACPairedResetReceipt:
        """Freeze all witnesses without upgrading them to simulator qualification."""

        canonical = self._canonical
        key = self._context_key
        if canonical is None or key is None:
            raise UniVTACPairingError(
                "canonical_reset_missing", "paired session has no canonical reset"
            )
        task_id, initial_seed, exogenous_seed, _, _ = key
        session_id = canonical_hash(
            {
                "namespace": PAIRING_EVIDENCE_LEVEL,
                "task_id": task_id,
                "initial_seed": initial_seed,
                "exogenous_seed": exogenous_seed,
                "config_sha256": self._config.sha256,
                "canonical_equivalence_key": canonical.equivalence_key,
            }
        )
        witnesses = tuple(self._witnesses)
        return UniVTACPairedResetReceipt(
            session_id=session_id,
            task_id=str(task_id),
            initial_seed=initial_seed,
            exogenous_seed=exogenous_seed,
            config_sha256=self._config.sha256,
            witnesses=witnesses,
            all_exact=all(item.exact_match for item in witnesses),
        )

    @staticmethod
    def _context_invariant_key(
        context: PolicyEpisodeContext,
    ) -> Tuple[str, int, int, str, str]:
        return (
            context.task,
            context.initial_seed,
            context.exogenous_seed,
            context.instruction,
            context.action_spec,
        )

    @staticmethod
    def _witness(
        converted: ConvertedUniVTACObservation,
        ordinal: int,
        mode: str,
        *,
        snapshot_state_sha256: str,
        exact_match: bool,
    ) -> UniVTACResetWitness:
        return UniVTACResetWitness(
            ordinal=ordinal,
            reset_mode=mode,
            native_step_id=converted.native_step_id,
            simulator_state_sha256=converted.simulator_state_sha256,
            joint_reorder_witness_sha256=(converted.joint_reorder_witness_sha256),
            canonical_joint9_sha256=canonical_hash(converted.canonical_joint9),
            model_visible_qpos8_sha256=canonical_hash(converted.model_visible_qpos8),
            snapshot_state_sha256=snapshot_state_sha256,
            exact_match=exact_match,
        )


class UniVTACPairedBackendSession:
    """Own one live runtime while trial runners consume non-owning backends."""

    def __init__(
        self,
        config: UniVTACBackendConfig,
        runtime: UniVTACTaskRuntime,
        *,
        reset_reference: Optional[UniVTACResetReference] = None,
        reset_trajectory: Optional[UniVTACPreMoveTrajectory] = None,
        success_predicate_id: Optional[str] = None,
    ) -> None:
        self._config = config
        self._runtime = runtime
        if reset_reference is not None and type(reset_reference) is not (
            UniVTACResetReference
        ):
            raise TypeError("reset_reference must be an exact UniVTACResetReference")
        if reset_trajectory is not None and type(reset_trajectory) is not (
            UniVTACPreMoveTrajectory
        ):
            raise TypeError(
                "reset_trajectory must be an exact UniVTACPreMoveTrajectory"
            )
        if reset_trajectory is not None:
            _validate_trajectory_reset_reference(
                config,
                reset_reference,
                reset_trajectory,
            )
        self._reset_reference = reset_reference
        self._reset_reference_require_simulator_state_match = reset_trajectory is None
        self._success_predicate_id = (
            config.task.success_predicate_id
            if success_predicate_id is None
            else success_predicate_id
        )
        self._coordinator = UniVTACPairedResetCoordinator(config, runtime)
        self._closed = False

    @property
    def reset_receipt(self) -> UniVTACPairedResetReceipt:
        return self._coordinator.receipt()

    def new_backend(self) -> UniVTACIsaacBackend:
        if self._closed:
            raise UniVTACPairingError(
                "paired_session_closed", "closed paired session cannot lease a backend"
            )
        return UniVTACIsaacBackend(
            self._config,
            self._runtime,
            reset_coordinator=self._coordinator,
            reset_reference=self._reset_reference,
            reset_reference_require_simulator_state_match=(
                self._reset_reference_require_simulator_state_match
            ),
            owns_runtime=False,
            success_predicate_id=self._success_predicate_id,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime.close_runtime()


def _validate_trajectory_reset_reference(
    config: UniVTACBackendConfig,
    reference: Optional[UniVTACResetReference],
    trajectory: UniVTACPreMoveTrajectory,
) -> None:
    """Bind the ACT-v3 physical endpoint profile to one exact trajectory."""

    if reference is None:
        raise ValueError("trajectory reset qualification requires a reset reference")
    if (
        trajectory.task_id != config.task.task_id
        or trajectory.action_spec != config.action_spec
        or trajectory.upstream_commit != config.upstream_commit
        or trajectory.task_source_sha256 != config.task.task_source_sha256
    ):
        raise ValueError("reset trajectory does not match the backend source")
    if (
        reference.task_id != trajectory.task_id
        or reference.initial_seed != trajectory.initial_seed
        or reference.exogenous_seed != trajectory.exogenous_seed
        or reference.pair_key != trajectory.pair_key
        or reference.dataset_sha256 != trajectory.dataset_sha256
        or reference.checkpoint_sha256 != trajectory.checkpoint_sha256
        or reference.config_sha256 != trajectory.config_sha256
        or reference.source_run_content_sha256 != trajectory.source_run_content_sha256
    ):
        raise ValueError("reset trajectory and reference identities differ")
    if (
        reference.expected_simulator_state_sha256
        != trajectory.capture_simulator_state_sha256
        or reference.expected_native_step != trajectory.capture_native_step
        or reference.expected_qpos8 != trajectory.capture_qpos8
        or reference.qpos_atol != _ACT_TRAJECTORY_REPLAY_QPOS_ATOL
    ):
        raise ValueError("reset trajectory and reference endpoints differ")
