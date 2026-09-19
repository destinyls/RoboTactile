"""Cross-backend tensor contract for the Dream-Tac base checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

TRANSFORMER_ENGINE_FP8_EXTRA_STATE_KEYS: Final[tuple[str, ...]] = tuple(
    f"net.blocks.{index}.cross_attn.attn_op._extra_state" for index in range(28)
)


def validate_flat_contract(
    source: Mapping[str, object],
    expected: Mapping[str, object],
    allowed_missing: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate tensors and return exact source-only backend metadata keys."""

    source_keys, expected_keys = set(source), set(expected)
    unexpected = source_keys - expected_keys
    allowed_backend_metadata = set(TRANSFORMER_ENGINE_FP8_EXTRA_STATE_KEYS)
    if unexpected and unexpected != allowed_backend_metadata:
        unauthorized = sorted(unexpected - allowed_backend_metadata)
        incomplete = sorted(allowed_backend_metadata - unexpected)
        raise ValueError(
            "unexpected checkpoint keys: "
            f"unauthorized={unauthorized[:8]}, "
            f"incomplete_backend_metadata={incomplete[:8]}"
        )
    missing = expected_keys - source_keys
    allowed = set(allowed_missing)
    if missing != allowed:
        unauthorized = sorted(missing - allowed)
        unused = sorted(allowed - missing)
        raise ValueError(
            f"missing-key whitelist mismatch; unauthorized={unauthorized[:8]}, "
            f"unused={unused[:8]}"
        )
    shared = source_keys & expected_keys
    mismatched = sorted(key for key in shared if source[key] != expected[key])
    if mismatched:
        raise ValueError(f"checkpoint tensor shape/dtype mismatch: {mismatched[:8]}")
    return tuple(sorted(unexpected))


__all__ = [
    "TRANSFORMER_ENGINE_FP8_EXTRA_STATE_KEYS",
    "validate_flat_contract",
]
