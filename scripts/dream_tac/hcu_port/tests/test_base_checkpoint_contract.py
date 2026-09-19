"""Tests for the strict cross-backend flat checkpoint contract."""

from __future__ import annotations

import pytest

from scripts.dream_tac.hcu_port.base_checkpoint_contract import (
    TRANSFORMER_ENGINE_FP8_EXTRA_STATE_KEYS,
    validate_flat_contract,
)
from scripts.dream_tac.hcu_port.base_checkpoint_converter import TensorDescriptor

_TENSOR = TensorDescriptor((2, 2), "torch.bfloat16", 4)
_METADATA = TensorDescriptor((0,), "torch.uint8", 0)


def test_exact_transformer_engine_metadata_set_is_recorded() -> None:
    expected = {"net.weight": _TENSOR}
    source = {
        "net.weight": _TENSOR,
        **{key: _METADATA for key in TRANSFORMER_ENGINE_FP8_EXTRA_STATE_KEYS},
    }

    ignored = validate_flat_contract(source, expected, ())

    assert ignored == tuple(sorted(TRANSFORMER_ENGINE_FP8_EXTRA_STATE_KEYS))


def test_partial_transformer_engine_metadata_set_is_rejected() -> None:
    source = {
        "net.weight": _TENSOR,
        TRANSFORMER_ENGINE_FP8_EXTRA_STATE_KEYS[0]: _METADATA,
    }

    with pytest.raises(ValueError, match="incomplete_backend_metadata"):
        validate_flat_contract(source, {"net.weight": _TENSOR}, ())


def test_unrelated_unexpected_key_remains_rejected() -> None:
    source = {"net.weight": _TENSOR, "net.unrelated": _TENSOR}

    with pytest.raises(ValueError, match="unauthorized"):
        validate_flat_contract(source, {"net.weight": _TENSOR}, ())
