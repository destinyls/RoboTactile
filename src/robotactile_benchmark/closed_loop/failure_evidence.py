"""Bounded JSON-safe crash evidence retained inside live artifact diagnostics."""

from __future__ import annotations

import hashlib
from typing import Union

MAX_FAILURE_MESSAGE_CHARS = 2048


def build_runner_failure_evidence(
    stage: str,
    failure_code: str,
    error: Union[Exception, SystemExit],
) -> dict[str, object]:
    """Preserve actionable exception identity without an unbounded traceback."""

    if not stage or not failure_code:
        raise ValueError("runner failure stage and code must be non-empty")
    original = str(error)
    encoded = original.encode("utf-8", errors="replace")
    message = original[:MAX_FAILURE_MESSAGE_CHARS]
    return {
        "exception_module": type(error).__module__,
        "exception_type": type(error).__qualname__,
        "failure_code": failure_code,
        "message": message,
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
        "message_truncated": len(original) > MAX_FAILURE_MESSAGE_CHARS,
        "stage": stage,
    }


__all__ = ["MAX_FAILURE_MESSAGE_CHARS", "build_runner_failure_evidence"]
