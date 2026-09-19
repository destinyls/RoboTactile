"""Sequential non-owning leases for one reusable closed-loop policy runtime."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from types import TracebackType
from typing import Literal, Optional

from robotactile_benchmark.closed_loop.contracts import (
    ActionPlan,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.contracts import ObservationRecord


def _add_exception_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


class _SequentialPolicyLease:
    """Expose one episode while leaving the pooled policy runtime allocated."""

    def __init__(self, policy: ClosedLoopPolicy) -> None:
        self.identity = policy.identity
        self._policy = policy
        self._closed = False

    def _active(self) -> ClosedLoopPolicy:
        if self._closed:
            raise RuntimeError("sequential policy lease is closed")
        return self._policy

    def reset(self, context: PolicyEpisodeContext) -> None:
        self._active().reset(context)

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        return self._active().infer(observation)

    def commit(self, execution: PolicyExecution) -> None:
        self._active().commit(execution)

    def abort(self, reason_code: str) -> None:
        self._active().abort(reason_code)

    def close(self) -> None:
        self._closed = True


class SequentialPolicyPool:
    """Own policy runtimes and issue one non-overlapping lease per episode."""

    def __init__(self) -> None:
        self._policies: dict[Hashable, ClosedLoopPolicy] = {}
        self._lease_active = False
        self._closed = False

    def acquire(
        self,
        key: Hashable,
        identity: PolicyIdentity,
        factory: Callable[[], ClosedLoopPolicy],
    ) -> ClosedLoopPolicy:
        if self._closed:
            raise RuntimeError("sequential policy pool is closed")
        if self._lease_active:
            raise RuntimeError("sequential policy pool lease overlap")
        policy = self._policies.get(key)
        if policy is None:
            policy = factory()
            if policy.identity != identity:
                policy.close()
                raise ValueError("pooled policy loader changed policy identity")
            self._policies[key] = policy
        elif policy.identity != identity:
            raise ValueError("pooled policy key changed policy identity")
        self._lease_active = True
        return _PoolLease(self, policy)

    def _release(self) -> None:
        if not self._lease_active:
            raise RuntimeError("sequential policy pool lease is not active")
        self._lease_active = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        policies, self._policies = tuple(self._policies.values()), {}
        first_error: Optional[BaseException] = (
            RuntimeError("sequential policy pool closed with an active lease")
            if self._lease_active
            else None
        )
        for policy in policies:
            try:
                policy.close()
            except (Exception, SystemExit) as error:
                if first_error is None:
                    first_error = error
                else:
                    _add_exception_note(
                        first_error,
                        f"additional pooled policy close failure: {error!r}",
                    )
        self._lease_active = False
        if first_error is not None:
            raise first_error

    def __enter__(self) -> SequentialPolicyPool:
        if self._closed:
            raise RuntimeError("closed sequential policy pool cannot be entered")
        return self

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        _exception: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> Literal[False]:
        try:
            self.close()
        except (Exception, SystemExit) as close_error:
            if exception_type is None:
                raise
            if _exception is not None:
                _add_exception_note(
                    _exception, f"pooled policy close failure: {close_error!r}"
                )
        return False


class _PoolLease(_SequentialPolicyLease):
    """Lease that returns its sequential slot without closing the runtime."""

    def __init__(
        self,
        pool: SequentialPolicyPool,
        policy: ClosedLoopPolicy,
    ) -> None:
        super().__init__(policy)
        self._pool = pool

    def close(self) -> None:
        if self._closed:
            return
        super().close()
        self._pool._release()


__all__ = ["SequentialPolicyPool"]
