"""Shared primitives for immutable N0 fault campaign contracts."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from robotactile_benchmark.constants import CORE_OPERATOR_IDS

N0_FAULT_CAMPAIGN_SEMANTIC_VERSION = "1.0"
N0_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION = "1.0"
N0_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION = "1.0"
N0_FAULT_TEMPLATE_SEED_DERIVATION = "sha256_signed31_pair_operator_v1"
N0_FAULT_GENERATION_EVIDENCE_LEVEL = "request_generation_only_no_execution_v1"
N0_UNSUPPORTED_OPERATOR_IDS = frozenset({"A1_stream_absence", "A2_frame_erasure"})
N0_SUPPORTED_OPERATOR_IDS = CORE_OPERATOR_IDS - N0_UNSUPPORTED_OPERATOR_IDS

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class N0FaultCampaignError(ValueError):
    """A generated N0 fault campaign failed its frozen contract."""


class N0FaultCellDisposition(str, Enum):
    """How one campaign cell is represented before execution."""

    LIVE_REQUEST = "live_request"
    UNSUPPORTED_CONTRACT = "unsupported_contract"


def require_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise N0FaultCampaignError(f"{name} must be a non-empty string")
    return value


def require_identifier(value: object, name: str) -> str:
    result = require_nonempty(value, name)
    if _IDENTIFIER.fullmatch(result) is None:
        raise N0FaultCampaignError(f"{name} must be a safe identifier")
    return result


def require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise N0FaultCampaignError(f"{name} must be a lowercase SHA256")
    return value


def require_integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise N0FaultCampaignError(f"{name} must be an integer >= {minimum}")
    return value


def require_relative_path(value: object, name: str, prefix: str) -> str:
    path_text = require_nonempty(value, name)
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or path.as_posix() != path_text
        or not path.parts
        or path.parts[0] != prefix
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in path_text
    ):
        raise N0FaultCampaignError(f"{name} must be below {prefix}/")
    return path_text


def enum_field_dict(value: Any) -> dict[str, object]:
    """Serialize a frozen dataclass while lowering enum fields to values."""

    return {
        name: item.value if isinstance(item := getattr(value, name), Enum) else item
        for name in value.__dataclass_fields__
    }
