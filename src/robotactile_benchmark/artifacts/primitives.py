"""Small validation primitives shared by artifact bundle families."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_SHA256 = re.compile(r"[0-9a-f]{64}")


def require_lowercase_sha256(
    value: object,
    name: str,
    error_type: type[ValueError] = ValueError,
) -> str:
    """Return one canonical SHA-256 value or raise the requested error type."""

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise error_type(f"{name} must be a lowercase SHA256")
    return value


def validate_posix_member_path(
    value: object,
    *,
    label: str,
    error_type: type[ValueError] = ValueError,
) -> str:
    """Reject absolute, platform-specific, or escaping artifact members."""

    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise error_type(f"{label} is not normalized POSIX")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise error_type(f"{label} escapes the bundle")
    if any(part in {"", "."} for part in path.parts):
        raise error_type(f"{label} is not normalized")
    return value


__all__ = ["require_lowercase_sha256", "validate_posix_member_path"]
