"""Safe stderr diagnostics for failures inside the closed-loop runner."""

from __future__ import annotations

import os
import traceback


def emit_crash_marker(
    stage: str,
    failure_code: str,
    error: BaseException,
) -> None:
    """Write a searchable stack-only diagnostic without runtime payload values."""

    try:
        exception_type = type(error)
        marker = (
            "ROBOTACTILE_CLOSED_LOOP_CRASH "
            f"stage={stage} "
            f"exception_type={exception_type.__module__}."
            f"{exception_type.__qualname__} "
            f"failure_code={failure_code} "
            f"system_exit_code={_safe_system_exit_code(error)}\n"
        )
        frame_lines = tuple(
            f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}\n'
            for frame in traceback.extract_tb(error.__traceback__)
        )
        payload = (
            marker + "Traceback (most recent call last):\n" + "".join(frame_lines)
        ).encode("utf-8", errors="backslashreplace")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(2, remaining)
            if written < 1:
                raise OSError("stderr write returned no progress")
            remaining = remaining[written:]
    except (Exception, SystemExit):
        return


def _safe_system_exit_code(error: BaseException) -> str:
    if not isinstance(error, SystemExit):
        return "not_applicable"
    code = error.code
    if code is None:
        return "none"
    if type(code) is bool:
        return "true" if code else "false"
    if type(code) is int:
        return str(code)
    code_type = type(code)
    return f"redacted:{code_type.__module__}.{code_type.__qualname__}"
