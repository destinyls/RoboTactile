"""Runtime-checkable closed-loop backend and policy interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from robotactile_benchmark.closed_loop.contracts import (
    ActionPlan,
    ExecutionBatch,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
    ResetReceipt,
)
from robotactile_benchmark.contracts import Array, EvaluationRecord, ObservationRecord


@runtime_checkable
class SimulationBackend(Protocol):
    """A simulator adapter capable of deterministic reset, observation, and action."""

    backend_id: str
    action_spec: str
    success_predicate_id: str

    def reset(self, context: PolicyEpisodeContext) -> ResetReceipt:
        """Reset for the policy context and return an auditable receipt."""

        ...

    def observe(self) -> EvaluationRecord:
        """Return the next evaluator-visible clean record."""

        ...

    def execute(self, actions: Array) -> ExecutionBatch:
        """Execute canonical qpos actions and return clean transitions."""

        ...

    def close(self) -> None:
        """Release backend resources."""

        ...


@runtime_checkable
class ClosedLoopPolicy(Protocol):
    """A closed-loop policy operating exclusively on model-visible observations."""

    identity: PolicyIdentity

    def reset(self, context: PolicyEpisodeContext) -> None:
        """Start a fresh episode."""

        ...

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        """Propose an action plan from one delivered observation."""

        ...

    def commit(self, execution: PolicyExecution) -> None:
        """Accept the delivered execution trace."""

        ...

    def abort(self, reason_code: str) -> None:
        """Abort the active episode with a stable reason code."""

        ...

    def close(self) -> None:
        """Release policy resources."""

        ...
