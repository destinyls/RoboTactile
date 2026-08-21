"""Closed-loop contracts and runtime-checkable integration protocols."""

from robotactile_benchmark.closed_loop.artifact_contracts import (
    ArtifactValidationError,
    LoadedClosedLoopBundle,
    RootReceipt,
)
from robotactile_benchmark.closed_loop.artifacts import (
    load_closed_loop_bundle,
    write_closed_loop_bundle,
)
from robotactile_benchmark.closed_loop.capture import (
    ActionTraceEntry,
    ClosedLoopExecutionEvidence,
)
from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    SEMANTIC_VERSION,
    ActionPlan,
    BackendSignal,
    BackendTransition,
    ClosedLoopRunSpec,
    ExecutionBatch,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
    ResetReceipt,
)
from robotactile_benchmark.closed_loop.delivery import (
    DeliveryFinalization,
    IdentityDeliverySession,
    OnlineFaultSession,
)
from robotactile_benchmark.closed_loop.interfaces import (
    ClosedLoopPolicy,
    SimulationBackend,
)
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.closed_loop.runner import (
    run_closed_loop_trial,
    run_closed_loop_trial_with_evidence,
)

__all__ = [
    "ACTION_SPEC",
    "SEMANTIC_VERSION",
    "ActionTraceEntry",
    "ActionPlan",
    "ArtifactValidationError",
    "BackendSignal",
    "BackendTransition",
    "ClosedLoopPolicy",
    "ClosedLoopTrialResult",
    "ClosedLoopExecutionEvidence",
    "ClosedLoopRunSpec",
    "DeliveryFinalization",
    "ExecutionBatch",
    "IdentityDeliverySession",
    "LoadedClosedLoopBundle",
    "OnlineFaultSession",
    "PolicyEpisodeContext",
    "PolicyExecution",
    "PolicyIdentity",
    "ResetReceipt",
    "RootReceipt",
    "SimulationBackend",
    "load_closed_loop_bundle",
    "run_closed_loop_trial",
    "run_closed_loop_trial_with_evidence",
    "write_closed_loop_bundle",
]
